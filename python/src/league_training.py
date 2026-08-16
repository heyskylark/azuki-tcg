from __future__ import annotations

import contextlib
import os
import time
from collections import defaultdict
from dataclasses import dataclass

import azk_puffer.pytorch as azk_pytorch
import azk_puffer.trainer as pufferl
from azk_puffer.core import unroll_nested_dict
import numpy as np
import torch
import torch.nn.functional as F

from draft_prefix_outcome import (
  FrozenDraftPrefixPredictor,
  PrefixOutcomeRedistributor,
)


@dataclass
class LeagueConfig:
  enabled: bool = False
  frozen_ratio: float | None = None
  latest_ratio: float | None = 0.85
  randomize_learner_seat: bool = True
  seed: int = 0
  activate_after_steps: int = 0
  # OSFP-style windowed opponent sampling: every N epochs draw at most
  # max_distinct_frozen pool members and assign all frozen games to them.
  # Rollout then pays a fixed number of extra forwards per step regardless of
  # pool size (per-distinct-policy batch splitting is the SPS sag). 0 = legacy
  # per-game uniform draws over the whole pool.
  frozen_window_epochs: int = 0
  max_distinct_frozen: int = 1


def compute_learner_row_mask(
  env_ids: np.ndarray,
  *,
  agents_per_env: int,
  env_learner_seat: np.ndarray,
) -> np.ndarray:
  env_indices = (env_ids // agents_per_env).astype(np.int32)
  seat_indices = (env_ids % agents_per_env).astype(np.int32)
  return seat_indices == env_learner_seat[env_indices]


def compute_trainable_row_mask(
  env_ids: np.ndarray,
  *,
  agents_per_env: int,
  env_learner_seat: np.ndarray,
  env_use_latest: np.ndarray,
  league_active: bool,
) -> np.ndarray:
  if not league_active:
    return np.ones(env_ids.shape[0], dtype=np.bool_)

  env_indices = (env_ids // agents_per_env).astype(np.int32)
  learner_rows = compute_learner_row_mask(
    env_ids,
    agents_per_env=agents_per_env,
    env_learner_seat=env_learner_seat,
  )
  return np.logical_or(learner_rows, env_use_latest[env_indices])


def compute_frozen_matchup_ratio(
  *,
  frozen_row_ratio: float,
  agents_per_env: int,
) -> float:
  if agents_per_env <= 1:
    return 0.0
  scaled = float(frozen_row_ratio) * float(agents_per_env) / float(agents_per_env - 1)
  return float(max(0.0, min(1.0, scaled)))


def compensate_frozen_matchup_ratio_for_reference(
  *,
  target_frozen_matchup_ratio: float,
  reference_matchup_probability: float,
) -> float:
  """Keep total frozen matchups fixed when every reference seat is frozen."""
  target = float(target_frozen_matchup_ratio)
  reference = float(reference_matchup_probability)
  if not 0.0 <= target <= 1.0:
    raise ValueError("target_frozen_matchup_ratio must be in [0, 1]")
  if not 0.0 <= reference <= 1.0:
    raise ValueError("reference_matchup_probability must be in [0, 1]")
  if reference > target:
    raise ValueError(
      "reference_matchup_probability cannot exceed the target frozen matchup ratio"
    )
  if reference >= 1.0:
    return 0.0
  return float((target - reference) / (1.0 - reference))


def detect_reset_reference_seats(
  env_ids: np.ndarray,
  deck_modes: np.ndarray,
  *,
  pending_envs: np.ndarray,
  agents_per_env: int,
) -> dict[int, int]:
  """Resolve new episodes as normal (-1) or fixed-reference (seat 0/1)."""
  if env_ids.shape != deck_modes.shape:
    raise ValueError("env_ids and deck_modes must have matching shapes")
  if agents_per_env != 2:
    raise ValueError("reference-seat alignment currently requires two-player matches")

  env_indices = (env_ids // agents_per_env).astype(np.int32)
  seat_indices = (env_ids % agents_per_env).astype(np.int32)
  resolved: dict[int, int] = {}
  for env_index in np.unique(env_indices):
    if env_index < 0 or env_index >= pending_envs.size or not pending_envs[env_index]:
      continue
    selector = env_indices == env_index
    seats = seat_indices[selector]
    if seats.size != agents_per_env or len(set(seats.tolist())) != agents_per_env:
      continue
    modes_by_seat = np.full(agents_per_env, -1, dtype=np.int32)
    modes_by_seat[seats] = deck_modes[selector]
    incomplete = np.nonzero(modes_by_seat > 0)[0]
    complete = np.nonzero(modes_by_seat == 0)[0]
    if incomplete.size == agents_per_env:
      resolved[int(env_index)] = -1
    elif incomplete.size == 1 and complete.size == 1:
      resolved[int(env_index)] = int(complete[0])
    else:
      raise RuntimeError(
        f"cannot classify reset deck modes for env {int(env_index)}: "
        f"{modes_by_seat.tolist()}"
      )
  return resolved


def compute_battle_row_mask(
  env_ids: np.ndarray,
  deck_modes: np.ndarray,
  *,
  agents_per_env: int,
) -> np.ndarray:
  """A match is in battle only when every seat has completed its draft."""
  if env_ids.shape != deck_modes.shape:
    raise ValueError("env_ids and deck_modes must have matching shapes")
  env_indices = (env_ids // agents_per_env).astype(np.int32)
  out = np.zeros(env_ids.shape, dtype=np.bool_)
  for env_index in np.unique(env_indices):
    selector = env_indices == env_index
    if int(selector.sum()) == agents_per_env and bool((deck_modes[selector] == 0).all()):
      out[selector] = True
  return out


def compute_league_active(*, global_step: int, activate_after_steps: int) -> bool:
  threshold = int(max(0, activate_after_steps))
  return int(global_step) >= threshold


def clipped_terminal_policy_loss(
  new_logprobs: torch.Tensor,
  old_logprobs: torch.Tensor,
  advantages: torch.Tensor,
  clip_coef: float,
  mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """PPO-clipped contextual-bandit loss for delayed terminal labels."""
  if new_logprobs.shape != old_logprobs.shape or new_logprobs.shape != advantages.shape:
    raise ValueError("terminal-credit tensors must have matching shapes")
  if not 0.0 <= float(clip_coef) < 1.0:
    raise ValueError("clip_coef must be in [0, 1)")
  if mask is not None and mask.shape != advantages.shape:
    raise ValueError("terminal-credit mask must match advantage shape")
  ratio = torch.exp(new_logprobs - old_logprobs)
  unclipped = -advantages * ratio
  clipped = -advantages * torch.clamp(
    ratio,
    1.0 - float(clip_coef),
    1.0 + float(clip_coef),
  )
  loss = torch.maximum(unclipped, clipped)
  if mask is None:
    return loss.mean(), ratio
  mask_float = mask.to(device=loss.device, dtype=loss.dtype)
  return (loss * mask_float).sum() / mask_float.sum().clamp_min(1.0), ratio


DRAFT_TERMINAL_CREDIT_QUARTILES = ((1, 13), (14, 25), (26, 38), (39, 50))
DRAFT_EPISODE_MAIN_PICKS = 50
DRAFT_EPISODE_DECISIVE_LABEL_PATIENCE = 4


def terminal_only_rewards(
  total_rewards: np.ndarray,
  terminal_mask: np.ndarray,
) -> np.ndarray:
  """Keep native terminal outcomes while discarding nonterminal shaped rewards."""
  rewards = np.asarray(total_rewards, dtype=np.float32).reshape(-1)
  terminals = np.asarray(terminal_mask, dtype=np.bool_).reshape(-1)
  if rewards.shape != terminals.shape:
    raise ValueError("total_rewards and terminal_mask must have matching shapes")
  return np.where(terminals, rewards, 0.0).astype(np.float32, copy=False)


_UINT64_MASK = (1 << 64) - 1


def deterministic_draft_credit_pick(
  episode_id: int,
  seat: int,
  quartile: int,
  seed: int,
) -> int:
  """Choose one reproducible, approximately uniform main pick per quartile."""
  if not 0 <= int(quartile) < len(DRAFT_TERMINAL_CREDIT_QUARTILES):
    raise ValueError("quartile must be in [0, 4)")
  if int(episode_id) < 0 or int(seat) < 0:
    raise ValueError("episode_id and seat must be nonnegative")
  low, high = DRAFT_TERMINAL_CREDIT_QUARTILES[int(quartile)]
  value = (
    (int(seed) & _UINT64_MASK)
    ^ (((int(episode_id) + 1) * 0xD2B74407B1CE6E93) & _UINT64_MASK)
    ^ (((int(seat) + 1) * 0xCA5A826395121157) & _UINT64_MASK)
    ^ (((int(quartile) + 1) * 0x9E3779B97F4A7C15) & _UINT64_MASK)
  )
  value ^= value >> 30
  value = (value * 0xBF58476D1CE4E5B9) & _UINT64_MASK
  value ^= value >> 27
  value = (value * 0x94D049BB133111EB) & _UINT64_MASK
  value ^= value >> 31
  return int(low + value % (high - low + 1))


def deterministic_draft_episode_priority(
  episode_id: int,
  seat: int,
  seed: int,
) -> int:
  """Stable hash priority for bounded, timing-unbiased episode sampling."""
  if int(episode_id) < 0 or int(seat) < 0:
    raise ValueError("episode_id and seat must be nonnegative")
  value = (
    (int(seed) & _UINT64_MASK)
    ^ (((int(episode_id) + 1) * 0xD2B74407B1CE6E93) & _UINT64_MASK)
    ^ (((int(seat) + 1) * 0xCA5A826395121157) & _UINT64_MASK)
  )
  value ^= value >> 30
  value = (value * 0xBF58476D1CE4E5B9) & _UINT64_MASK
  value ^= value >> 27
  value = (value * 0x94D049BB133111EB) & _UINT64_MASK
  value ^= value >> 31
  return int(value)


def parse_draft_prefix_distribution(
  lengths_text: str,
  probabilities_text: str,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
  lengths = tuple(int(value.strip()) for value in lengths_text.split(","))
  probabilities = tuple(
    float(value.strip()) for value in probabilities_text.split(",")
  )
  if not lengths or len(lengths) != len(probabilities):
    raise ValueError("draft prefix lengths and probabilities must align")
  if any(length < 0 or length > DRAFT_EPISODE_MAIN_PICKS for length in lengths):
    raise ValueError("draft prefix lengths must be in [0, 50]")
  if any(probability < 0.0 for probability in probabilities):
    raise ValueError("draft prefix probabilities must be nonnegative")
  if abs(sum(probabilities) - 1.0) > 1e-9:
    raise ValueError("draft prefix probabilities must sum to one")
  return lengths, probabilities


def deterministic_draft_prefix_length(
  episode_id: int,
  seat: int,
  seed: int,
  lengths: tuple[int, ...],
  probabilities: tuple[float, ...],
) -> int:
  if len(lengths) != len(probabilities) or not lengths:
    raise ValueError("draft prefix distribution is empty or misaligned")
  unit = deterministic_draft_episode_priority(
    episode_id,
    seat,
    seed,
  ) / float(1 << 64)
  cumulative = 0.0
  for length, probability in zip(lengths, probabilities):
    cumulative += probability
    if unit < cumulative:
      return int(length)
  return int(lengths[-1])


def deterministic_draft_prefix_candidate(
  episode_id: int,
  seat: int,
  main_pick: int,
  candidate_count: int,
  seed: int,
) -> int:
  if not 1 <= int(main_pick) <= DRAFT_EPISODE_MAIN_PICKS:
    raise ValueError("main_pick must be in [1, 50]")
  if int(candidate_count) < 1:
    raise ValueError("candidate_count must be positive")
  pick_seed = (
    int(seed)
    ^ ((int(main_pick) * 0x9E3779B97F4A7C15) & _UINT64_MASK)
  )
  return int(
    deterministic_draft_episode_priority(episode_id, seat, pick_seed)
    % int(candidate_count)
  )


def pad_terminal_credit_records(
  records: list[dict[str, object]],
  batch_size: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
  """Pad one nonempty terminal-credit chunk to a stable compiled batch shape."""
  if int(batch_size) < 1:
    raise ValueError("terminal-credit batch_size must be positive")
  if not records or len(records) > int(batch_size):
    raise ValueError("terminal-credit record chunk must contain 1..batch_size rows")
  valid = np.zeros(int(batch_size), dtype=np.bool_)
  valid[:len(records)] = True
  return records + [records[0]] * (int(batch_size) - len(records)), valid


def take_terminal_credit_training_records(
  records: list[dict[str, object]],
  batch_size: int,
  *,
  flush: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
  """Take full replay batches, retaining a partial tail until the final flush."""
  if int(batch_size) < 1:
    raise ValueError("terminal-credit batch_size must be positive")
  take_count = (
    len(records)
    if flush
    else len(records) // int(batch_size) * int(batch_size)
  )
  return records[:take_count], records[take_count:]


def pad_complete_draft_episodes(
  records: list[dict[str, object]],
  batch_drafts: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
  """Pad completed 50-pick episodes to one stable compiled batch shape."""
  if int(batch_drafts) < 1:
    raise ValueError("draft episode batch size must be positive")
  if not records or len(records) > int(batch_drafts):
    raise ValueError("draft episode chunk must contain 1..batch_drafts episodes")
  valid = np.zeros(int(batch_drafts), dtype=np.bool_)
  valid[:len(records)] = True
  return records + [records[0]] * (int(batch_drafts) - len(records)), valid


def masked_tensor_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  if values.shape != mask.shape:
    raise ValueError("masked mean values and mask must have matching shapes")
  weights = mask.to(device=values.device, dtype=values.dtype)
  return (values * weights).sum() / weights.sum().clamp_min(1.0)


def masked_tensor_mean_std(
  values: torch.Tensor,
  mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute selected-row statistics without synchronizing the accelerator."""
  if values.shape != mask.shape:
    raise ValueError("masked statistics values and mask must have matching shapes")
  weights = mask.to(device=values.device, dtype=values.dtype)
  count = weights.sum()
  mean = (values * weights).sum() / count.clamp_min(1.0)
  centered = values - mean
  variance = (centered.square() * weights).sum() / (count - 1.0).clamp_min(1.0)
  return mean, variance.sqrt(), count


def draft_episode_credit_update_due(
  epoch: int,
  total_epochs: int,
  interval: int,
) -> bool:
  if int(interval) < 1:
    raise ValueError("draft episode credit interval must be positive")
  update_number = int(epoch) + 1
  return (
    update_number % int(interval) == 0
    or update_number >= int(total_epochs)
  )


class LeaguePuffeRL(pufferl.PuffeRL):
  """PuffeRL variant with dual-policy league rollouts and latest-policy updates."""

  def __init__(
    self,
    config,
    vecenv,
    policy,
    opponent_policies,
    league_cfg: LeagueConfig,
    logger=None,
    opponent_keys: list[str] | None = None,
  ):
    super().__init__(config, vecenv, policy, logger=logger)
    self.league_cfg = league_cfg
    self.opponent_policies = list(opponent_policies)
    if opponent_keys is None:
      self.opponent_keys = [f"initial:{index}" for index in range(len(self.opponent_policies))]
    else:
      self.opponent_keys = [str(key) for key in opponent_keys]
    if len(self.opponent_keys) != len(self.opponent_policies):
      raise ValueError("opponent_keys must match opponent_policies")
    if len(set(self.opponent_keys)) != len(self.opponent_keys):
      raise ValueError("opponent_keys must be unique")
    self._sampling_policy_ids = np.arange(len(self.opponent_policies), dtype=np.int32)
    self._rng = np.random.default_rng(int(league_cfg.seed))

    # Game-granular row grouping (see trainer.py): native driver envs pack
    # many games per instance, but rows are [g0_p0, g0_p1, g1_p0, ...] on
    # every path. agents_per_match=2 keeps matchup/seat assignment, frozen
    # sampling, and resampling per GAME instead of per worker instance.
    driver = vecenv.driver_env
    self._agents_per_env = int(
      getattr(driver, "agents_per_match", getattr(driver, "num_agents", 0))
    )
    if self._agents_per_env <= 1:
      raise ValueError("League mode requires at least 2 agents per env")

    self._num_envs_total = int(self.total_agents // self._agents_per_env)

    self._env_learner_seat = self._rng.integers(
      0, self._agents_per_env, size=self._num_envs_total, dtype=np.int32
    )
    self._env_opp_policy = np.zeros(self._num_envs_total, dtype=np.int32)
    self._env_use_latest = np.ones(self._num_envs_total, dtype=np.bool_)
    self._window_policy_ids: np.ndarray | None = None
    self._window_index = -1
    self._pfsp_enabled = os.environ.get("AZK_PFSP") == "1"
    self._pfsp_power = float(os.environ.get("AZK_PFSP_POWER", "2.0") or 2.0)
    self._reference_opponent_only = os.environ.get("AZK_DRAFT_REF_OPPONENT_ONLY") == "1"
    self._reference_matchup_probability = float(
      os.environ.get("AZK_DRAFT_REF_SEAT_PROB", "0") or 0.0
    )
    if not 0.0 <= self._reference_matchup_probability <= 1.0:
      raise ValueError("AZK_DRAFT_REF_SEAT_PROB must be in [0, 1]")
    if self._reference_opponent_only and int(self.league_cfg.activate_after_steps) != 0:
      raise ValueError("opponent-only reference seats require an immediately active league")
    self._reference_mode_offset: int | None = None
    if self._reference_opponent_only:
      from observation import DECKBUILD_OBSERVATION_CTYPE

      obs_dtype = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
      deck_context_dtype, deck_context_offset = obs_dtype.fields["deck_context"][:2]
      _, mode_offset = deck_context_dtype.fields["mode"][:2]
      self._reference_mode_offset = int(deck_context_offset + mode_offset)
    self._env_reference_pending = np.ones(self._num_envs_total, dtype=np.bool_)
    self._env_is_reference = np.zeros(self._num_envs_total, dtype=np.bool_)
    self._env_reference_seat = np.full(self._num_envs_total, -1, dtype=np.int8)
    if self.league_cfg.frozen_ratio is not None:
      target_frozen_matchup_ratio = compute_frozen_matchup_ratio(
        frozen_row_ratio=float(self.league_cfg.frozen_ratio),
        agents_per_env=self._agents_per_env,
      )
    else:
      latest_ratio = (
        1.0 if self.league_cfg.latest_ratio is None else float(self.league_cfg.latest_ratio)
      )
      target_frozen_matchup_ratio = float(max(0.0, min(1.0, 1.0 - latest_ratio)))
    self._target_frozen_matchup_ratio = target_frozen_matchup_ratio
    self._base_frozen_matchup_ratio = (
      compensate_frozen_matchup_ratio_for_reference(
        target_frozen_matchup_ratio=target_frozen_matchup_ratio,
        reference_matchup_probability=self._reference_matchup_probability,
      )
      if self._reference_opponent_only
      else target_frozen_matchup_ratio
    )
    self._pfsp_wins = np.zeros(len(self.opponent_policies), dtype=np.float64)
    self._pfsp_games = np.zeros(len(self.opponent_policies), dtype=np.float64)
    self._pfsp_keys = list(self.opponent_keys)
    self._refresh_frozen_window()
    self._resample_matchups(np.arange(self._num_envs_total, dtype=np.int32))

    self._segment_is_trainable = torch.zeros(self.segments, device=self.config["device"], dtype=torch.bool)

    # A-DRAFTAUX (league path): per-row prev deck_context.mode for boundary
    # detection, pending aux stash from _infer_actions, and prev-step buffer
    # coords per recv group for injection at the last pick step.
    self._draftaux_prev_mode = torch.full(
      (self.total_agents,), -999, device=self.config["device"], dtype=torch.int32
    )
    self._draftaux_pending = None
    self._draftaux_prevcoords = {}

    # S3 cross-gate replay pick-masking (AZK_XGATE_MASK=1): at a draft->battle
    # boundary whose gate id CHANGED (env swapped to the sibling), zero the
    # actor-loss mask for that row's earlier steps in the boundary segment —
    # the only steps whose GAE window crosses the swap. Slightly over-broad
    # (may include tail steps of a previous episode in the same segment).
    self._xgate_mask_enabled = os.environ.get("AZK_XGATE_MASK") == "1"
    self._xgate_prev_gate = torch.full(
      (self.total_agents,), -32768, device=self.config["device"], dtype=torch.int32
    )
    self._xgate_prev_mode = torch.full(
      (self.total_agents,), -999, device=self.config["device"], dtype=torch.int32
    )
    self._xgate_masked_steps = 0
    self.actor_loss_mask = torch.ones(
      self.segments, self.config["bptt_horizon"], device=self.config["device"]
    )

    self._use_rnn = bool(self.config.get("use_rnn", False))
    if self._use_rnn:
      hidden_size = int(policy.hidden_size)
      device = self.config["device"]
      self._learner_lstm_h = torch.zeros(self.total_agents, hidden_size, device=device)
      self._learner_lstm_c = torch.zeros(self.total_agents, hidden_size, device=device)
      self._opp_lstm_h = [
        torch.zeros(self.total_agents, hidden_size, device=device) for _ in self.opponent_policies
      ]
      self._opp_lstm_c = [
        torch.zeros(self.total_agents, hidden_size, device=device) for _ in self.opponent_policies
      ]
    else:
      self._learner_lstm_h = None
      self._learner_lstm_c = None
      self._opp_lstm_h = []
      self._opp_lstm_c = []

    self._leader_credit_coef = float(
      os.environ.get("AZK_LEADER_TERMINAL_CREDIT_COEF", "0") or 0.0
    )
    if self._leader_credit_coef < 0.0:
      raise ValueError("AZK_LEADER_TERMINAL_CREDIT_COEF must be nonnegative")
    self._leader_credit_enabled = self._leader_credit_coef > 0.0
    self._leader_credit_clip = float(
      os.environ.get("AZK_LEADER_TERMINAL_CREDIT_CLIP", str(config["clip_coef"]))
      or config["clip_coef"]
    )
    if not 0.0 <= self._leader_credit_clip < 1.0:
      raise ValueError("AZK_LEADER_TERMINAL_CREDIT_CLIP must be in [0, 1)")
    self._leader_credit_gate_codes = tuple(
      code.strip()
      for code in os.environ.get("AZK_LEADER_TERMINAL_CREDIT_GATE_CODES", "").split(",")
      if code.strip()
    )
    self._leader_credit_update_interval = int(
      os.environ.get("AZK_LEADER_TERMINAL_CREDIT_UPDATE_INTERVAL", "4") or 4
    )
    if self._leader_credit_update_interval < 1:
      raise ValueError("AZK_LEADER_TERMINAL_CREDIT_UPDATE_INTERVAL must be positive")
    self._leader_credit_layout: dict[str, object] | None = None
    self._leader_credit_pending: dict[tuple[int, int], dict[str, object]] = {}
    self._leader_credit_ready: list[dict[str, object]] = []
    self._leader_credit_captured = 0
    self._leader_credit_completed = 0
    self._leader_credit_label_warmup_epochs = int(
      os.environ.get("AZK_LEADER_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS", "10") or 10
    )
    if self._leader_credit_label_warmup_epochs < 1:
      raise ValueError(
        "AZK_LEADER_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS must be positive"
      )
    if self._leader_credit_enabled and not self._win_prob_aux_enabled():
      raise ValueError(
        "leader terminal credit requires policy.win_prob_aux_enabled=true"
      )

    self._draft_credit_coef = float(
      os.environ.get("AZK_DRAFT_TERMINAL_CREDIT_COEF", "0") or 0.0
    )
    if self._draft_credit_coef < 0.0:
      raise ValueError("AZK_DRAFT_TERMINAL_CREDIT_COEF must be nonnegative")
    self._draft_credit_enabled = self._draft_credit_coef > 0.0
    if self._draft_credit_enabled and self._leader_credit_enabled:
      raise ValueError(
        "whole-draft terminal credit and leader-only terminal credit are alternative arms"
      )
    self._draft_credit_clip = float(
      os.environ.get("AZK_DRAFT_TERMINAL_CREDIT_CLIP", str(config["clip_coef"]))
      or config["clip_coef"]
    )
    if not 0.0 <= self._draft_credit_clip < 1.0:
      raise ValueError("AZK_DRAFT_TERMINAL_CREDIT_CLIP must be in [0, 1)")
    self._draft_credit_update_interval = int(
      os.environ.get("AZK_DRAFT_TERMINAL_CREDIT_UPDATE_INTERVAL", "4") or 4
    )
    if self._draft_credit_update_interval < 1:
      raise ValueError("AZK_DRAFT_TERMINAL_CREDIT_UPDATE_INTERVAL must be positive")
    self._draft_credit_batch_size = int(
      os.environ.get("AZK_DRAFT_TERMINAL_CREDIT_BATCH_SIZE", "512") or 512
    )
    if self._draft_credit_batch_size < 1:
      raise ValueError("AZK_DRAFT_TERMINAL_CREDIT_BATCH_SIZE must be positive")
    self._draft_credit_seed = int(
      os.environ.get("AZK_DRAFT_TERMINAL_CREDIT_SEED", str(league_cfg.seed))
      or league_cfg.seed
    )
    self._draft_credit_grad_probe_enabled = (
      os.environ.get("AZK_DRAFT_TERMINAL_CREDIT_GRAD_PROBE") == "1"
    )
    self._draft_credit_grad_probe_done = False
    self._draft_credit_control_grad_norm = 0.0
    self._draft_credit_label_warmup_epochs = int(
      os.environ.get("AZK_DRAFT_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS", "10") or 10
    )
    if self._draft_credit_label_warmup_epochs < 1:
      raise ValueError(
        "AZK_DRAFT_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS must be positive"
      )
    self._draft_credit_layout: dict[str, int] | None = None
    self._draft_credit_pending: dict[tuple[int, int], dict[str, object]] = {}
    self._draft_credit_ready: list[dict[str, object]] = []
    self._draft_credit_captured = 0
    self._draft_credit_labeled = 0
    self._draft_credit_truncated = 0
    self._draft_credit_incomplete = 0
    self._draft_credit_incomplete_records = 0
    if self._draft_credit_enabled and not self._win_prob_aux_enabled():
      raise ValueError(
        "whole-draft terminal credit requires policy.win_prob_aux_enabled=true"
      )

    self._draft_episode_credit_coef = float(
      os.environ.get("AZK_DRAFT_EPISODE_CREDIT_COEF", "0") or 0.0
    )
    if self._draft_episode_credit_coef < 0.0:
      raise ValueError("AZK_DRAFT_EPISODE_CREDIT_COEF must be nonnegative")
    self._draft_episode_credit_enabled = self._draft_episode_credit_coef > 0.0
    if self._draft_episode_credit_enabled and (
      self._leader_credit_enabled or self._draft_credit_enabled
    ):
      raise ValueError(
        "full-episode draft credit cannot be combined with legacy terminal credit"
      )
    self._draft_episode_credit_clip = float(
      os.environ.get("AZK_DRAFT_EPISODE_CREDIT_CLIP", str(config["clip_coef"]))
      or config["clip_coef"]
    )
    if not 0.0 <= self._draft_episode_credit_clip < 1.0:
      raise ValueError("AZK_DRAFT_EPISODE_CREDIT_CLIP must be in [0, 1)")
    self._draft_episode_credit_batch_drafts = int(
      os.environ.get("AZK_DRAFT_EPISODE_CREDIT_BATCH_DRAFTS", "80") or 80
    )
    if self._draft_episode_credit_batch_drafts < 1:
      raise ValueError("AZK_DRAFT_EPISODE_CREDIT_BATCH_DRAFTS must be positive")
    self._draft_episode_credit_update_interval = int(
      os.environ.get("AZK_DRAFT_EPISODE_CREDIT_UPDATE_INTERVAL", "4") or 4
    )
    if self._draft_episode_credit_update_interval < 1:
      raise ValueError(
        "AZK_DRAFT_EPISODE_CREDIT_UPDATE_INTERVAL must be positive"
      )
    self._draft_episode_credit_baseline_coef = float(
      os.environ.get(
        "AZK_DRAFT_EPISODE_CREDIT_BASELINE_COEF",
        str(self._win_prob_aux_coef()),
      )
      or self._win_prob_aux_coef()
    )
    if self._draft_episode_credit_baseline_coef < 0.0:
      raise ValueError("AZK_DRAFT_EPISODE_CREDIT_BASELINE_COEF must be nonnegative")
    self._draft_episode_credit_seed = int(
      os.environ.get("AZK_DRAFT_EPISODE_CREDIT_SEED", "420052") or 420052
    )
    self._draft_episode_credit_label_warmup_epochs = int(
      os.environ.get("AZK_DRAFT_EPISODE_CREDIT_LABEL_WARMUP_EPOCHS", "10") or 10
    )
    if self._draft_episode_credit_label_warmup_epochs < 1:
      raise ValueError(
        "AZK_DRAFT_EPISODE_CREDIT_LABEL_WARMUP_EPOCHS must be positive"
      )
    self._draft_episode_credit_layout: dict[str, int] | None = None
    self._draft_episode_credit_pending: dict[
      tuple[int, int], dict[str, object]
    ] = {}
    self._draft_episode_credit_ready: list[dict[str, object]] = []
    self._draft_episode_credit_captured = 0
    self._draft_episode_credit_labeled = 0
    self._draft_episode_credit_truncated = 0
    self._draft_episode_credit_draws = 0
    self._draft_episode_credit_decisive = 0
    self._draft_episode_credit_wins = 0
    self._draft_episode_credit_losses = 0
    self._draft_episode_credit_incomplete = 0
    self._draft_episode_credit_completed = 0
    self._draft_episode_credit_sample_dropped = 0
    self._draft_episode_credit_window_completed = 0
    self._draft_episode_credit_window_sample_dropped = 0
    self._draft_episode_credit_zero_label_epochs = 0
    self._draft_episode_credit_no_decisive_epochs = 0
    self._draft_episode_credit_capture_seconds = 0.0
    self._draft_episode_phase_mask = None
    if self._draft_episode_credit_enabled:
      self._draft_episode_phase_mask = torch.zeros(
        self.segments,
        self.config["bptt_horizon"],
        device=self.config["device"],
        dtype=torch.bool,
      )
    if self._draft_episode_credit_enabled and not self._win_prob_aux_enabled():
      raise ValueError(
        "full-episode draft credit requires policy.win_prob_aux_enabled=true"
      )
    if self._draft_episode_credit_enabled and not self._use_rnn:
      raise ValueError("full-episode draft credit requires recurrent policy state")
    uniform_assignment = bool(
      getattr(
        driver,
        "_draft_uniform_assignment",
        getattr(driver, "draft_uniform_assignment", False),
      )
    )
    if self._draft_episode_credit_enabled and not uniform_assignment:
      raise ValueError(
        "full-episode draft credit requires draft_uniform_assignment=true"
      )
    if self._draft_episode_credit_enabled and self.total_minibatches < 1:
      raise ValueError(
        "full-episode draft credit requires at least one optimizer minibatch"
      )

    draft_prefix_probabilities = os.environ.get(
      "AZK_DRAFT_PREFIX_PROBS", ""
    ).strip()
    self._draft_prefix_enabled = bool(draft_prefix_probabilities)
    if self._draft_prefix_enabled:
      (
        self._draft_prefix_lengths,
        self._draft_prefix_probabilities,
      ) = parse_draft_prefix_distribution(
        os.environ.get("AZK_DRAFT_PREFIX_LENGTHS", "0,1,2,4"),
        draft_prefix_probabilities,
      )
    else:
      self._draft_prefix_lengths = (0,)
      self._draft_prefix_probabilities = (1.0,)
    self._draft_prefix_seed = int(
      os.environ.get("AZK_DRAFT_PREFIX_SEED", "420053") or 420053
    )
    self._draft_prefix_forced_rows = 0
    self._draft_prefix_episode_lengths: list[int] = []
    if self._draft_prefix_enabled and not self._draft_episode_credit_enabled:
      raise ValueError(
        "random main prefixes require full-episode draft credit to mask forced rows"
      )
    if self._draft_prefix_enabled:
      print(
        "[draft-prefix] enabled: "
        f"lengths={self._draft_prefix_lengths}, "
        f"probabilities={self._draft_prefix_probabilities}, "
        f"seed={self._draft_prefix_seed}, live-policy-seats-only=true"
      )

    prefix_outcome_model = os.environ.get(
      "AZK_DRAFT_PREFIX_OUTCOME_MODEL", ""
    ).strip()
    self._prefix_outcome_enabled = bool(prefix_outcome_model)
    self._prefix_outcome_coefficient = float(
      os.environ.get("AZK_DRAFT_PREFIX_OUTCOME_COEF", "1.0") or 1.0
    )
    if self._prefix_outcome_coefficient < 0.0:
      raise ValueError("AZK_DRAFT_PREFIX_OUTCOME_COEF must be nonnegative")
    if self._prefix_outcome_enabled and self._prefix_outcome_coefficient == 0.0:
      raise ValueError("enabled draft prefix outcome model requires a positive coefficient")
    if self._prefix_outcome_enabled and (
      self._leader_credit_enabled
      or self._draft_credit_enabled
      or self._draft_episode_credit_enabled
    ):
      raise ValueError(
        "frozen prefix outcome redistribution cannot be combined with terminal-credit arms"
      )
    if self._prefix_outcome_enabled and not uniform_assignment:
      raise ValueError(
        "frozen prefix outcome redistribution requires draft_uniform_assignment=true"
      )
    self._prefix_outcome_layout: dict[str, int] | None = None
    self._prefix_outcome_redistributor: PrefixOutcomeRedistributor | None = None
    self._prefix_outcome_inference_seconds = 0.0
    if self._prefix_outcome_enabled:
      expected_sha256 = os.environ.get(
        "AZK_DRAFT_PREFIX_OUTCOME_SHA256", ""
      ).strip()
      if not expected_sha256:
        raise ValueError(
          "AZK_DRAFT_PREFIX_OUTCOME_SHA256 is required when the predictor is enabled"
        )
      predictor = FrozenDraftPrefixPredictor(
        prefix_outcome_model,
        expected_sha256=expected_sha256,
      )
      self._prefix_outcome_redistributor = PrefixOutcomeRedistributor(
        predictor,
        total_agents=self.total_agents,
        coefficient=self._prefix_outcome_coefficient,
      )
      print(
        "[draft-prefix-outcome] enabled: "
        f"model={predictor.path}, sha256={predictor.sha256}, "
        f"coefficient={self._prefix_outcome_coefficient:.6f}, "
        "objective=successive-potential-plus-terminal-residual"
      )

    for opp in self.opponent_policies:
      opp.eval()
      for param in opp.parameters():
        param.requires_grad_(False)

  def _refresh_frozen_window(self) -> None:
    """Redraw the window's allowed frozen-policy ids when the window rolls.

    Assignments only change for envs as they finish (via _resample_matchups),
    so a window transition phases in over ~one episode and per-policy LSTM
    stores stay consistent for in-flight games.
    """
    window_epochs = int(getattr(self.league_cfg, "frozen_window_epochs", 0) or 0)
    pool_size = len(self.opponent_policies)
    available = np.asarray(
      getattr(self, "_sampling_policy_ids", np.arange(pool_size, dtype=np.int32)),
      dtype=np.int32,
    )
    if window_epochs <= 0 or available.size == 0:
      self._window_policy_ids = None
      return
    window_index = int(getattr(self, "epoch", 0)) // window_epochs
    if (
      window_index == self._window_index
      and self._window_policy_ids is not None
      and self._window_policy_ids.size > 0
      and bool(np.isin(self._window_policy_ids, available).all())
    ):
      return
    self._window_index = window_index
    k = max(1, int(getattr(self.league_cfg, "max_distinct_frozen", 1) or 1))
    k = min(k, int(available.size))
    if getattr(self, "_pfsp_enabled", False) and self._pfsp_games.size == pool_size:
      # PFSP-hard: weight opponents by (1 - learner winrate)^p with an
      # exploration floor; unplayed opponents sit at winrate 0.5.
      games = self._pfsp_games
      winrate = np.where(games >= 3, self._pfsp_wins / np.maximum(games, 1e-9), 0.5)
      weights = (1.0 - winrate[available]) ** self._pfsp_power + 0.05
      weights = weights / weights.sum()
      self._window_policy_ids = self._rng.choice(
        available, size=k, replace=False, p=weights
      ).astype(np.int32)
      picked = int(self._window_policy_ids[0])
      self.stats["league/pfsp_picked_winrate"].append(float(winrate[picked]))
      self.stats["league/pfsp_pool_min_winrate"].append(float(winrate[available].min()))
    else:
      self._window_policy_ids = self._rng.choice(available, size=k, replace=False).astype(np.int32)

  def _resample_matchups(self, env_indices: np.ndarray) -> None:
    if env_indices.size == 0:
      return

    if self.league_cfg.randomize_learner_seat:
      self._env_learner_seat[env_indices] = self._rng.integers(
        0, self._agents_per_env, size=env_indices.size, dtype=np.int32
      )

    sampling_ids = np.asarray(
      getattr(self, "_sampling_policy_ids", np.arange(len(self.opponent_policies))),
      dtype=np.int32,
    )
    if sampling_ids.size == 0:
      self._env_use_latest[env_indices] = True
      self._env_opp_policy[env_indices] = -1
      return

    active = compute_league_active(
      global_step=int(self.global_step),
      activate_after_steps=int(self.league_cfg.activate_after_steps),
    )
    if not active:
      self._env_use_latest[env_indices] = True
    else:
      base_frozen_matchup_ratio = getattr(self, "_base_frozen_matchup_ratio", None)
      if base_frozen_matchup_ratio is not None:
        frozen_matchup_ratio = float(base_frozen_matchup_ratio)
      elif self.league_cfg.frozen_ratio is not None:
        frozen_matchup_ratio = compute_frozen_matchup_ratio(
          frozen_row_ratio=float(self.league_cfg.frozen_ratio),
          agents_per_env=self._agents_per_env,
        )
      else:
        latest_ratio = 1.0 if self.league_cfg.latest_ratio is None else float(self.league_cfg.latest_ratio)
        frozen_matchup_ratio = float(max(0.0, min(1.0, 1.0 - latest_ratio)))
      frozen_draw = self._rng.random(env_indices.size)
      self._env_use_latest[env_indices] = frozen_draw >= frozen_matchup_ratio
    if self._window_policy_ids is not None and self._window_policy_ids.size > 0:
      picks = self._rng.integers(
        0, self._window_policy_ids.size, size=env_indices.size
      )
      self._env_opp_policy[env_indices] = self._window_policy_ids[picks]
    else:
      picks = self._rng.integers(0, sampling_ids.size, size=env_indices.size)
      self._env_opp_policy[env_indices] = sampling_ids[picks]

  def _decode_reference_deck_modes(self, observations: torch.Tensor) -> np.ndarray | None:
    if not self._reference_opponent_only or self._reference_mode_offset is None:
      return None
    offset = self._reference_mode_offset
    if observations.ndim != 2 or observations.shape[1] < offset + 4:
      raise RuntimeError("packed deck-building observation is too small for reference alignment")
    return (
      observations[:, offset:offset + 4]
      .contiguous()
      .view(torch.int32)
      .flatten()
      .cpu()
      .numpy()
      .astype(np.int32, copy=False)
    )

  def _init_prefix_outcome_layout(self, obs_row_bytes: int) -> dict[str, int]:
    from observation import DECKBUILD_OBSERVATION_CTYPE

    dtype = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
    if int(obs_row_bytes) != dtype.itemsize:
      raise ValueError(
        f"draft prefix outcome redistribution requires {dtype.itemsize}B packed "
        f"observations, got {obs_row_bytes}B"
      )
    deck_dtype, deck_offset = dtype.fields["deck_context"][:2]
    _, mode_offset = deck_dtype.fields["mode"][:2]
    _, gate_offset = deck_dtype.fields["gate_card_def_id"][:2]
    _, leader_offset = deck_dtype.fields["leader_card_def_id"][:2]
    main_dtype, main_offset = deck_dtype.fields["main_card_def_ids"][:2]
    _, main_count_offset = deck_dtype.fields["main_count"][:2]
    if main_dtype.itemsize != 2 * 50:
      raise ValueError("draft prefix outcome layout requires exactly 50 int16 main cards")
    layout = {
      "mode_offset": int(deck_offset + mode_offset),
      "gate_offset": int(deck_offset + gate_offset),
      "leader_offset": int(deck_offset + leader_offset),
      "main_offset": int(deck_offset + main_offset),
      "main_count_offset": int(deck_offset + main_count_offset),
    }
    self._prefix_outcome_layout = layout
    return layout

  def _compute_prefix_outcome_redistribution(
    self,
    *,
    observations_cpu: torch.Tensor,
    env_id_np: np.ndarray,
    trainable_rows_np: np.ndarray,
    done_mask: np.ndarray,
    terminal_mask: np.ndarray,
  ) -> np.ndarray:
    redistributor = self._prefix_outcome_redistributor
    if not self._prefix_outcome_enabled or redistributor is None:
      return np.zeros(env_id_np.size, dtype=np.float32)
    started = time.perf_counter()
    layout = self._prefix_outcome_layout
    if layout is None:
      layout = self._init_prefix_outcome_layout(int(observations_cpu.shape[-1]))
    raw = observations_cpu.detach().cpu().numpy()
    mode_offset = layout["mode_offset"]
    gate_offset = layout["gate_offset"]
    leader_offset = layout["leader_offset"]
    main_offset = layout["main_offset"]
    count_offset = layout["main_count_offset"]
    modes = raw[:, mode_offset:mode_offset + 4].copy().view(np.int32).reshape(-1)
    gate_ids = raw[:, gate_offset:gate_offset + 2].copy().view(np.int16).reshape(-1)
    leader_ids = (
      raw[:, leader_offset:leader_offset + 2].copy().view(np.int16).reshape(-1)
    )
    main_cards = (
      raw[:, main_offset:main_offset + 100]
      .copy()
      .view(np.int16)
      .reshape(-1, 50)
    )
    main_counts = raw[:, count_offset].astype(np.int16, copy=True)
    env_indices = (env_id_np // self._agents_per_env).astype(np.int32)
    redistribution = redistributor.step(
      agent_ids=env_id_np,
      episode_ids=self._env_episode_ids[env_indices],
      trainable=trainable_rows_np,
      done=done_mask,
      terminal=terminal_mask,
      modes=modes,
      gate_ids=gate_ids,
      leader_ids=leader_ids,
      main_card_ids=main_cards,
      main_counts=main_counts,
    )
    self._prefix_outcome_inference_seconds += time.perf_counter() - started
    return redistribution

  def _align_reference_matchups(
    self,
    env_id_np: np.ndarray,
    deck_modes: np.ndarray | None,
  ) -> None:
    if not self._reference_opponent_only or deck_modes is None:
      return
    resolved = detect_reset_reference_seats(
      env_id_np,
      deck_modes,
      pending_envs=self._env_reference_pending,
      agents_per_env=self._agents_per_env,
    )
    for env_index, reference_seat in resolved.items():
      self._env_reference_pending[env_index] = False
      self._env_is_reference[env_index] = reference_seat >= 0
      self._env_reference_seat[env_index] = reference_seat
      if reference_seat < 0:
        continue
      if not self.opponent_policies:
        raise RuntimeError("opponent-only reference seat requires a frozen policy pool")
      self._env_learner_seat[env_index] = 1 - reference_seat
      self._env_use_latest[env_index] = False

  def set_opponent_policies(
    self, opponent_policies: list[torch.nn.Module], opponent_keys: list[str] | None = None
  ) -> None:
    desired_policies = list(opponent_policies)
    desired_keys = (
      [str(key) for key in opponent_keys]
      if opponent_keys is not None
      else [f"refresh:{index}" for index in range(len(desired_policies))]
    )
    if len(desired_keys) != len(desired_policies):
      raise ValueError("opponent_keys must match opponent_policies")
    if len(set(desired_keys)) != len(desired_keys):
      raise ValueError("opponent_keys must be unique")

    prev_keys = list(getattr(self, "opponent_keys", []))
    prev_index = {key: index for index, key in enumerate(prev_keys)}
    prev_policies = list(self.opponent_policies)
    prev_h = list(self._opp_lstm_h)
    prev_c = list(self._opp_lstm_c)
    prev_wins = np.asarray(getattr(self, "_pfsp_wins", np.zeros(0)), dtype=np.float64)
    prev_games = np.asarray(getattr(self, "_pfsp_games", np.zeros(0)), dtype=np.float64)

    # Keep a retired policy resident only while a frozen game is still using
    # it. New assignments draw solely from desired_keys and phase in when each
    # environment completes, so a refresh never changes an opponent mid-game.
    inflight_keys: list[str] = []
    for env_index in np.nonzero(~self._env_use_latest)[0].tolist():
      old_index = int(self._env_opp_policy[env_index])
      if 0 <= old_index < len(prev_keys):
        key = prev_keys[old_index]
        if key not in desired_keys and key not in inflight_keys:
          inflight_keys.append(key)
    resident_keys = desired_keys + inflight_keys
    desired_by_key = dict(zip(desired_keys, desired_policies))
    resident_policies = [
      desired_by_key[key] if key in desired_by_key else prev_policies[prev_index[key]]
      for key in resident_keys
    ]
    resident_index = {key: index for index, key in enumerate(resident_keys)}

    remapped = np.full_like(self._env_opp_policy, -1)
    for env_index, old_index_raw in enumerate(self._env_opp_policy.tolist()):
      old_index = int(old_index_raw)
      if 0 <= old_index < len(prev_keys):
        remapped[env_index] = resident_index.get(prev_keys[old_index], -1)
    self._env_opp_policy = remapped
    self.opponent_policies = resident_policies
    self.opponent_keys = resident_keys
    self._sampling_policy_ids = np.arange(len(desired_keys), dtype=np.int32)

    for opp in self.opponent_policies:
      opp.eval()
      for param in opp.parameters():
        param.requires_grad_(False)

    self._opp_lstm_h = []
    self._opp_lstm_c = []
    if self._use_rnn:
      hidden_size = int(self.policy.hidden_size)
      device = self.config["device"]
      for key in resident_keys:
        old_index = prev_index.get(key)
        if old_index is not None and old_index < len(prev_h):
          self._opp_lstm_h.append(prev_h[old_index])
          self._opp_lstm_c.append(prev_c[old_index])
        else:
          self._opp_lstm_h.append(torch.zeros(self.total_agents, hidden_size, device=device))
          self._opp_lstm_c.append(torch.zeros(self.total_agents, hidden_size, device=device))

    # Force the next window draw against the desired sampling set. Existing
    # games keep their remapped assignment until their normal terminal step.
    self._window_index = -1
    self._window_policy_ids = None
    self._pfsp_enabled = os.environ.get("AZK_PFSP") == "1"
    self._pfsp_power = float(os.environ.get("AZK_PFSP_POWER", "2.0") or 2.0)
    self._pfsp_wins = np.zeros(len(self.opponent_policies), dtype=np.float64)
    self._pfsp_games = np.zeros(len(self.opponent_policies), dtype=np.float64)
    self._pfsp_keys = list(resident_keys)
    for new_index, key in enumerate(resident_keys):
      old_index = prev_index.get(key)
      if old_index is not None and old_index < prev_wins.size:
        self._pfsp_wins[new_index] = prev_wins[old_index]
        self._pfsp_games[new_index] = prev_games[old_index]
    self._refresh_frozen_window()
    invalid_envs = np.nonzero(
      np.logical_and(self._env_opp_policy < 0, ~self._env_use_latest)
    )[0].astype(np.int32)
    self._resample_matchups(invalid_envs)
    self.stats["league/opponent_sampling_pool_size"].append(float(len(desired_keys)))
    self.stats["league/opponent_resident_count"].append(float(len(resident_keys)))

  def _episode_envs_from_done_mask(self, env_id: np.ndarray, done_mask: np.ndarray) -> np.ndarray:
    if env_id.size == 0:
      return np.zeros((0,), dtype=np.int32)
    env_indices = (env_id // self._agents_per_env).astype(np.int32)
    finished_envs: list[int] = []
    for env_idx in np.unique(env_indices):
      selector = env_indices == env_idx
      if selector.any() and bool(done_mask[selector].all()):
        finished_envs.append(int(env_idx))
    return np.asarray(finished_envs, dtype=np.int32)

  def _zero_done_states(self, done_rows: np.ndarray) -> None:
    if done_rows.size == 0 or not self._use_rnn:
      return
    done_rows_np = np.asarray(done_rows, dtype=np.int64).reshape(-1)
    in_bounds = np.logical_and(done_rows_np >= 0, done_rows_np < self.total_agents)
    if not bool(in_bounds.any()):
      return
    done_rows_np = np.unique(done_rows_np[in_bounds])
    done_t = torch.as_tensor(done_rows_np, device=self.config["device"], dtype=torch.long)
    self._learner_lstm_h[done_t] = 0
    self._learner_lstm_c[done_t] = 0
    for idx in range(len(self.opponent_policies)):
      self._opp_lstm_h[idx][done_t] = 0
      self._opp_lstm_c[idx][done_t] = 0

  def _draftaux_league_stash(self, o_device, mask_t, learner_idx_t, values, env_id_np):
    """Boundary detection + counterfactual sibling value for learner rows.

    Called between the learner forward and the LSTM write-back so
    self._learner_lstm_h/_c still hold the pre-forward states the boundary
    observation was evaluated with."""
    lay = self._draftaux_layout
    if lay is None:
      lay = self._draftaux_init_layout(o_device.shape[-1], o_device.device)
      if lay is None:
        return
    device = o_device.device
    mo = lay["mode_off"]
    global_rows = torch.as_tensor(env_id_np, device=device, dtype=torch.long)
    mode_all = o_device[:, mo:mo + 4].contiguous().view(torch.int32).flatten()
    prev_all = self._draftaux_prev_mode[global_rows]
    self._draftaux_prev_mode[global_rows] = mode_all
    learner_pos = learner_idx_t
    boundary = (mode_all[learner_pos] == 0) & (prev_all[learner_pos] > 0)
    if not bool(boundary.any()):
      return
    b_local = boundary.nonzero(as_tuple=False).flatten()          # into learner rows
    b_batch = learner_pos[b_local]                                # into recv batch
    b_global = global_rows[b_batch]                               # into total_agents
    v_own = values.flatten()[b_local].detach().float()
    aux = self._draftaux_vboot * v_own
    if self._draftaux_sibdiff > 0.0:
      g_off = lay["gate_ctx_off"]
      z_off = lay["gate_zone_off"]
      rows = o_device[b_batch].clone()
      gid = rows[:, g_off:g_off + 2].contiguous().view(torch.int16).flatten().long()
      sib = lay["sibling"][gid.clamp(min=0, max=lay["sibling"].numel() - 1)]
      ok = (gid >= 0) & (sib >= 0)
      if bool(ok.any()):
        sb_bytes = sib.to(torch.int16).view(torch.uint8).reshape(-1, 2)
        rows[:, g_off:g_off + 2] = sb_bytes
        rows[:, z_off:z_off + 2] = sb_bytes
        cf_state = {
          "mask": mask_t[b_batch],
          "lstm_h": self._learner_lstm_h[b_global].clone(),
          "lstm_c": self._learner_lstm_c[b_global].clone(),
        }
        with torch.no_grad(), self.amp_context:
          _, v_sib = self._safe_forward_eval(self.policy, rows, cf_state)
        diff = (v_own - v_sib.flatten().detach().float()).clamp(
          min=0.0, max=self._draftaux_cap
        )
        aux = aux + self._draftaux_sibdiff * diff * ok.float()
    self._draftaux_pending = (b_global, aux * self._draftaux_aux_scale())

  def _init_leader_credit_layout(self, obs_row_bytes: int) -> dict[str, object]:
    from deck_building import build_deck_build_catalog
    from observation import DECKBUILD_OBSERVATION_CTYPE
    from training_deck_pool import load_training_deck_pool

    dtype = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
    if int(obs_row_bytes) != dtype.itemsize:
      raise ValueError(
        f"leader credit requires {dtype.itemsize}B packed deck observations, "
        f"got {obs_row_bytes}B"
      )

    deck_dtype, deck_offset = dtype.fields["deck_context"][:2]
    _, mode_offset = deck_dtype.fields["mode"][:2]
    _, gate_offset = deck_dtype.fields["gate_card_def_id"][:2]
    catalog = build_deck_build_catalog(load_training_deck_pool())
    gate_population = {int(value) for value in catalog.gate_def_id_population}
    if self._leader_credit_gate_codes:
      gate_ids = []
      for code in self._leader_credit_gate_codes:
        record = catalog.records_by_code.get(code)
        if record is None or int(record.card_def_id) not in gate_population:
          raise ValueError(f"unknown leader-credit gate code: {code}")
        gate_ids.append(int(record.card_def_id))
    else:
      gate_ids = sorted(gate_population)
    layout: dict[str, object] = {
      "mode_offset": int(deck_offset + mode_offset),
      "gate_offset": int(deck_offset + gate_offset),
      "gate_ids": np.asarray(sorted(set(gate_ids)), dtype=np.int16),
    }
    self._leader_credit_layout = layout
    print(
      "[leader-credit] enabled: "
      f"coef={self._leader_credit_coef:.6f}, clip={self._leader_credit_clip:.3f}, "
      f"update_interval={self._leader_credit_update_interval}, "
      f"gates={','.join(self._leader_credit_gate_codes) or 'ALL'}"
    )
    return layout

  def _decode_leader_credit_rows(
    self,
    observations_cpu: torch.Tensor,
    active_mask: np.ndarray,
  ) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not self._leader_credit_enabled:
      return None, None
    layout = self._leader_credit_layout
    if layout is None:
      layout = self._init_leader_credit_layout(int(observations_cpu.shape[-1]))
    raw = observations_cpu.detach().cpu().numpy()
    mode_offset = int(layout["mode_offset"])
    gate_offset = int(layout["gate_offset"])
    modes = raw[:, mode_offset:mode_offset + 4].copy().view(np.int32).reshape(-1)
    gate_ids = raw[:, gate_offset:gate_offset + 2].copy().view(np.int16).reshape(-1)
    eligible = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
    eligible = np.logical_and(eligible, modes == 1)
    eligible = np.logical_and(eligible, np.isin(gate_ids, layout["gate_ids"]))
    return eligible, gate_ids

  def _stash_leader_credit_rows(
    self,
    observations_cpu: torch.Tensor,
    batch_positions: np.ndarray,
    actions: torch.Tensor,
    logprobs: torch.Tensor,
    pre_lstm_h: torch.Tensor | None,
    pre_lstm_c: torch.Tensor | None,
    env_id_np: np.ndarray,
    gate_ids_np: np.ndarray,
  ) -> None:
    if batch_positions.size == 0:
      return
    positions_t = torch.as_tensor(batch_positions, dtype=torch.long)
    obs_rows = observations_cpu[positions_t].detach().cpu().clone()
    action_rows = actions.detach().cpu().to(dtype=torch.long).clone()
    logprob_rows = logprobs.detach().cpu().float().clone()
    h_rows = None if pre_lstm_h is None else pre_lstm_h.detach().cpu().float().clone()
    c_rows = None if pre_lstm_c is None else pre_lstm_c.detach().cpu().float().clone()

    for local_index, batch_position in enumerate(batch_positions.tolist()):
      agent_id = int(env_id_np[batch_position])
      env_index = int(agent_id // self._agents_per_env)
      seat = int(agent_id % self._agents_per_env)
      episode_id = int(self._env_episode_ids[env_index])
      key = (episode_id, seat)
      record: dict[str, object] = {
        "observation": obs_rows[local_index],
        "action": action_rows[local_index],
        "old_logprob": logprob_rows[local_index],
        "gate_id": int(gate_ids_np[batch_position]),
        "seat": seat,
      }
      if h_rows is not None and c_rows is not None:
        record["lstm_h"] = h_rows[local_index]
        record["lstm_c"] = c_rows[local_index]
      if key not in self._leader_credit_pending:
        self._leader_credit_captured += 1
      self._leader_credit_pending[key] = record

  def _stash_selected_leader_credit_rows(
    self,
    *,
    observations_cpu: torch.Tensor | None,
    policy_rows_np: np.ndarray,
    actions: torch.Tensor,
    logprobs: torch.Tensor,
    pre_lstm_h: torch.Tensor | None,
    pre_lstm_c: torch.Tensor | None,
    env_id_np: np.ndarray,
    leader_rows_np: np.ndarray | None,
    leader_gate_ids_np: np.ndarray | None,
  ) -> None:
    if not getattr(self, "_leader_credit_enabled", False):
      return
    if (
      observations_cpu is None
      or leader_rows_np is None
      or leader_gate_ids_np is None
    ):
      raise RuntimeError("enabled leader credit requires decoded rollout rows")
    eligible_local = np.nonzero(leader_rows_np[policy_rows_np])[0]
    if eligible_local.size == 0:
      return
    eligible_local_t = torch.as_tensor(
      eligible_local,
      device=actions.device,
      dtype=torch.long,
    )
    self._stash_leader_credit_rows(
      observations_cpu,
      policy_rows_np[eligible_local],
      actions[eligible_local_t],
      logprobs[eligible_local_t],
      None if pre_lstm_h is None else pre_lstm_h[eligible_local_t],
      None if pre_lstm_c is None else pre_lstm_c[eligible_local_t],
      env_id_np,
      leader_gate_ids_np,
    )

  def _finalize_leader_credit_rows(
    self,
    env_id_np: np.ndarray,
    done_mask: np.ndarray,
    terminal_mask: np.ndarray,
    terminal_rewards: np.ndarray,
  ) -> None:
    if not self._leader_credit_enabled or not self._leader_credit_pending:
      return
    finished_envs = self._episode_envs_from_done_mask(env_id_np, done_mask)
    if finished_envs.size == 0:
      return
    labels_by_env = pufferl.terminal_win_labels_from_rewards(
      env_id_np,
      terminal_mask,
      terminal_rewards,
      self._agents_per_env,
    )
    for env_index_raw in finished_envs:
      env_index = int(env_index_raw)
      episode_id = int(self._env_episode_ids[env_index])
      labels = labels_by_env.get(env_index, {})
      for seat in range(self._agents_per_env):
        record = self._leader_credit_pending.pop((episode_id, seat), None)
        target = labels.get(seat)
        if record is None or target is None:
          continue
        record["target"] = float(target)
        self._leader_credit_ready.append(record)
        self._leader_credit_completed += 1

  def _train_leader_terminal_credit(self) -> dict[str, float]:
    update_number = int(self.epoch) + 1
    flush_endpoint = update_number >= int(self.total_epochs)
    due = update_number % self._leader_credit_update_interval == 0
    if (
      not self._leader_credit_enabled
      or not self._leader_credit_ready
      or (not due and not flush_endpoint)
    ):
      return {
        "leader_credit_examples": 0.0,
        "leader_credit_loss": 0.0,
        "leader_credit_train_seconds": 0.0,
      }
    train_started = time.perf_counter()
    records = self._leader_credit_ready
    self._leader_credit_ready = []
    device = self.config["device"]
    observations = torch.stack(
      [record["observation"] for record in records]
    ).to(device=device, non_blocking=True)
    actions = torch.stack([record["action"] for record in records]).to(
      device=device, dtype=torch.long, non_blocking=True
    )
    old_logprobs = torch.stack([record["old_logprob"] for record in records]).to(
      device=device, dtype=torch.float32, non_blocking=True
    )
    targets = torch.tensor(
      [float(record["target"]) for record in records],
      device=device,
      dtype=torch.float32,
    )
    state: dict[str, object] = {
      "mask": torch.ones(len(records), device=device, dtype=torch.bool),
    }
    if self._use_rnn:
      state["lstm_h"] = torch.stack([record["lstm_h"] for record in records]).to(
        device=device, dtype=torch.float32, non_blocking=True
      )
      state["lstm_c"] = torch.stack([record["lstm_c"] for record in records]).to(
        device=device, dtype=torch.float32, non_blocking=True
      )

    self.optimizer.zero_grad()
    amp_cm = self.amp_context if self.amp_context is not None else contextlib.nullcontext()
    base_policy = getattr(self.uncompiled_policy, "policy", None)
    compiled_paths = None
    if (
      base_policy is not None
      and hasattr(base_policy, "_eager_encode_observations")
      and hasattr(base_policy, "_eager_decode_actions")
    ):
      # Record counts vary with episode completions. Reusing the compiled hot
      # paths here would create a new graph per batch shape and stall training.
      compiled_paths = (
        base_policy.encode_observations,
        base_policy.decode_actions,
      )
      base_policy.encode_observations = base_policy._eager_encode_observations
      base_policy.decode_actions = base_policy._eager_decode_actions
    try:
      with amp_cm:
        logits, _ = self.policy.forward_eval(observations, state)
        _, new_logprobs, _ = azk_pytorch.sample_logits(logits, action=actions)
        win_logits = state.get("_azk_win_prob_logits")
        if not torch.is_tensor(win_logits):
          raise RuntimeError("leader credit requires win-probability logits")
        win_logits = win_logits.reshape_as(targets)
        baselines = torch.sigmoid(win_logits).detach()
        advantages = targets - baselines
        policy_loss, ratios = clipped_terminal_policy_loss(
          new_logprobs.reshape_as(targets),
          old_logprobs,
          advantages,
          self._leader_credit_clip,
        )
        baseline_loss = F.binary_cross_entropy_with_logits(win_logits, targets)
        total_loss = self._leader_credit_coef * policy_loss
    finally:
      if compiled_paths is not None:
        base_policy.encode_observations, base_policy.decode_actions = compiled_paths
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config["max_grad_norm"])
    self.optimizer.step()
    self.optimizer.zero_grad()

    with torch.no_grad():
      clip_fraction = (
        (ratios - 1.0).abs() > self._leader_credit_clip
      ).float().mean()
      return {
        "leader_credit_examples": float(len(records)),
        "leader_credit_loss": float(policy_loss.detach().item()),
        "leader_credit_baseline_loss": float(baseline_loss.detach().item()),
        "leader_credit_advantage_abs_mean": float(advantages.abs().mean().item()),
        "leader_credit_importance_mean": float(ratios.mean().item()),
        "leader_credit_clipfrac": float(clip_fraction.item()),
        "leader_credit_train_seconds": float(time.perf_counter() - train_started),
      }

  def _init_draft_credit_layout(self, obs_row_bytes: int) -> dict[str, int]:
    from observation import DECKBUILD_OBSERVATION_CTYPE

    dtype = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
    if int(obs_row_bytes) != dtype.itemsize:
      raise ValueError(
        f"whole-draft credit requires {dtype.itemsize}B packed deck observations, "
        f"got {obs_row_bytes}B"
      )
    deck_dtype, deck_offset = dtype.fields["deck_context"][:2]
    _, mode_offset = deck_dtype.fields["mode"][:2]
    _, gate_offset = deck_dtype.fields["gate_card_def_id"][:2]
    _, main_count_offset = deck_dtype.fields["main_count"][:2]
    layout = {
      "mode_offset": int(deck_offset + mode_offset),
      "gate_offset": int(deck_offset + gate_offset),
      "main_count_offset": int(deck_offset + main_count_offset),
    }
    self._draft_credit_layout = layout
    print(
      "[draft-terminal-credit] enabled: "
      f"coef={self._draft_credit_coef:.6f}, clip={self._draft_credit_clip:.3f}, "
      f"update_interval={self._draft_credit_update_interval}, "
      f"batch_size={self._draft_credit_batch_size}, seed={self._draft_credit_seed}, "
      "rows=leader+one-per-main-quartile"
    )
    return layout

  def _decode_draft_credit_rows(
    self,
    observations_cpu: torch.Tensor,
    active_mask: np.ndarray,
  ) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
  ]:
    if not self._draft_credit_enabled:
      return None, None, None, None
    layout = self._draft_credit_layout
    if layout is None:
      layout = self._init_draft_credit_layout(int(observations_cpu.shape[-1]))
    raw = observations_cpu.detach().cpu().numpy()
    mode_offset = layout["mode_offset"]
    gate_offset = layout["gate_offset"]
    main_count_offset = layout["main_count_offset"]
    modes = raw[:, mode_offset:mode_offset + 4].copy().view(np.int32).reshape(-1)
    gate_ids = raw[:, gate_offset:gate_offset + 2].copy().view(np.int16).reshape(-1)
    main_counts = raw[:, main_count_offset].astype(np.int16, copy=True)
    eligible = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
    eligible = np.logical_and(eligible, np.logical_or(modes == 1, modes == 2))
    return eligible, modes, main_counts, gate_ids

  def _stash_selected_draft_credit_rows(
    self,
    *,
    observations_cpu: torch.Tensor | None,
    policy_rows_np: np.ndarray,
    actions: torch.Tensor,
    logprobs: torch.Tensor,
    pre_lstm_h: torch.Tensor | None,
    pre_lstm_c: torch.Tensor | None,
    env_id_np: np.ndarray,
    draft_rows_np: np.ndarray | None,
    draft_modes_np: np.ndarray | None,
    draft_main_counts_np: np.ndarray | None,
    draft_gate_ids_np: np.ndarray | None,
  ) -> None:
    if not getattr(self, "_draft_credit_enabled", False):
      return
    if (
      observations_cpu is None
      or draft_rows_np is None
      or draft_modes_np is None
      or draft_main_counts_np is None
      or draft_gate_ids_np is None
    ):
      raise RuntimeError("enabled whole-draft credit requires decoded rollout rows")
    eligible_local = np.nonzero(draft_rows_np[policy_rows_np])[0]
    if eligible_local.size == 0:
      return

    selected: list[tuple[int, int, tuple[int, int], int, int]] = []
    for local_index_raw in eligible_local.tolist():
      local_index = int(local_index_raw)
      batch_position = int(policy_rows_np[local_index])
      agent_id = int(env_id_np[batch_position])
      env_index = int(agent_id // self._agents_per_env)
      seat = int(agent_id % self._agents_per_env)
      episode_id = int(self._env_episode_ids[env_index])
      key = (episode_id, seat)
      pending = self._draft_credit_pending.get(key)
      if pending is None:
        pending = {
          "seat": seat,
          "gate_id": int(draft_gate_ids_np[batch_position]),
          "sampled_picks": tuple(
            deterministic_draft_credit_pick(
              episode_id,
              seat,
              quartile,
              self._draft_credit_seed,
            )
            for quartile in range(len(DRAFT_TERMINAL_CREDIT_QUARTILES))
          ),
          "records": {},
        }
        self._draft_credit_pending[key] = pending
      elif int(pending["gate_id"]) != int(draft_gate_ids_np[batch_position]):
        raise RuntimeError("gate id changed within a pending draft-credit episode")

      mode = int(draft_modes_np[batch_position])
      main_pick = 0
      if mode == 1:
        slot = 0
      else:
        main_pick = int(draft_main_counts_np[batch_position]) + 1
        quartile = next(
          (
            index
            for index, (low, high) in enumerate(DRAFT_TERMINAL_CREDIT_QUARTILES)
            if low <= main_pick <= high
          ),
          -1,
        )
        if quartile < 0 or main_pick != int(pending["sampled_picks"][quartile]):
          continue
        slot = quartile + 1

      records = pending["records"]
      if not isinstance(records, dict):
        raise RuntimeError("invalid pending draft-credit record map")
      if slot in records:
        continue
      selected.append((local_index, batch_position, key, slot, main_pick))

    if not selected:
      return
    local_indices_t = torch.as_tensor(
      [item[0] for item in selected],
      device=actions.device,
      dtype=torch.long,
    )
    batch_positions_t = torch.as_tensor(
      [item[1] for item in selected],
      dtype=torch.long,
    )
    obs_rows = observations_cpu[batch_positions_t].detach().cpu().clone()
    # These records are tiny compared with packed observations. Keep them on
    # the policy device so sampling does not introduce a GPU->CPU synchronization
    # into the rollout hot path; the fixed replay batch consumes them on the
    # same device later.
    action_rows = actions[local_indices_t].detach().to(dtype=torch.long).clone()
    logprob_rows = logprobs[local_indices_t].detach().float().clone()
    h_rows = (
      None
      if pre_lstm_h is None
      else pre_lstm_h[local_indices_t].detach().float().clone()
    )
    c_rows = (
      None
      if pre_lstm_c is None
      else pre_lstm_c[local_indices_t].detach().float().clone()
    )
    for selected_index, (_, batch_position, key, slot, main_pick) in enumerate(selected):
      pending = self._draft_credit_pending[key]
      records = pending["records"]
      if not isinstance(records, dict):
        raise RuntimeError("invalid pending draft-credit record map")
      record: dict[str, object] = {
        "observation": obs_rows[selected_index],
        "action": action_rows[selected_index],
        "old_logprob": logprob_rows[selected_index],
        "gate_id": int(draft_gate_ids_np[batch_position]),
        "seat": int(pending["seat"]),
        "slot": int(slot),
        "main_pick": int(main_pick),
      }
      if h_rows is not None and c_rows is not None:
        record["lstm_h"] = h_rows[selected_index]
        record["lstm_c"] = c_rows[selected_index]
      records[slot] = record
      self._draft_credit_captured += 1

  def _finalize_draft_credit_rows(
    self,
    env_id_np: np.ndarray,
    done_mask: np.ndarray,
    terminal_mask: np.ndarray,
    terminal_rewards: np.ndarray,
  ) -> None:
    if not self._draft_credit_enabled or not self._draft_credit_pending:
      return
    finished_envs = self._episode_envs_from_done_mask(env_id_np, done_mask)
    if finished_envs.size == 0:
      return
    labels_by_env = pufferl.terminal_win_labels_from_rewards(
      env_id_np,
      terminal_mask,
      terminal_rewards,
      self._agents_per_env,
    )
    required_slots = set(range(1 + len(DRAFT_TERMINAL_CREDIT_QUARTILES)))
    for env_index_raw in finished_envs:
      env_index = int(env_index_raw)
      episode_id = int(self._env_episode_ids[env_index])
      labels = labels_by_env.get(env_index, {})
      for seat in range(self._agents_per_env):
        pending = self._draft_credit_pending.pop((episode_id, seat), None)
        if pending is None:
          continue
        records = pending["records"]
        if not isinstance(records, dict):
          raise RuntimeError("invalid pending draft-credit record map")
        target = labels.get(seat)
        if target is None:
          self._draft_credit_truncated += len(records)
          continue
        if set(records) != required_slots:
          self._draft_credit_incomplete += 1
          self._draft_credit_incomplete_records += len(records)
          continue
        for slot in sorted(records):
          record = records[slot]
          record["target"] = float(target)
          self._draft_credit_ready.append(record)
          self._draft_credit_labeled += 1

  @staticmethod
  def _draft_credit_position_metrics(
    advantages: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    baseline_bce: np.ndarray,
    slots: np.ndarray,
  ) -> dict[str, float]:
    groups = {
      "leader": slots == 0,
      "early": slots == 1,
      "middle": np.logical_or(slots == 2, slots == 3),
      "late": slots == 4,
    }
    metrics: dict[str, float] = {}
    for name, selector in groups.items():
      count = int(selector.sum())
      metrics[f"draft_credit_{name}_examples"] = float(count)
      if count == 0:
        for suffix in (
          "advantage_mean",
          "advantage_abs_mean",
          "advantage_std",
          "sign_outcome_agreement",
          "baseline_bce",
          "baseline_brier",
          "baseline_pred_mean",
        ):
          metrics[f"draft_credit_{name}_{suffix}"] = 0.0
        continue
      selected_advantages = advantages[selector]
      selected_targets = targets[selector]
      selected_predictions = predictions[selector]
      metrics[f"draft_credit_{name}_advantage_mean"] = float(selected_advantages.mean())
      metrics[f"draft_credit_{name}_advantage_abs_mean"] = float(
        np.abs(selected_advantages).mean()
      )
      metrics[f"draft_credit_{name}_advantage_std"] = float(selected_advantages.std())
      metrics[f"draft_credit_{name}_sign_outcome_agreement"] = float(
        ((selected_advantages > 0.0) == (selected_targets > 0.5)).mean()
      )
      metrics[f"draft_credit_{name}_baseline_bce"] = float(baseline_bce[selector].mean())
      metrics[f"draft_credit_{name}_baseline_brier"] = float(
        np.square(selected_predictions - selected_targets).mean()
      )
      metrics[f"draft_credit_{name}_baseline_pred_mean"] = float(
        selected_predictions.mean()
      )
    return metrics

  def _train_draft_terminal_credit(self) -> dict[str, float]:
    empty = {
      "draft_credit_examples": 0.0,
      "draft_credit_batches": 0.0,
      "draft_credit_loss": 0.0,
      "draft_credit_gradient_norm": 0.0,
      "draft_credit_control_draft_gradient_norm": float(
        self._draft_credit_control_grad_norm
      ),
      "draft_credit_train_seconds": 0.0,
      "draft_credit_gpu_seconds": 0.0,
    }
    update_number = int(self.epoch) + 1
    flush_endpoint = update_number >= int(self.total_epochs)
    due = update_number % self._draft_credit_update_interval == 0
    if (
      not self._draft_credit_enabled
      or not self._draft_credit_ready
      or (not due and not flush_endpoint)
    ):
      return empty

    train_started = time.perf_counter()
    device = self.config["device"]
    batch_size = self._draft_credit_batch_size
    records, self._draft_credit_ready = take_terminal_credit_training_records(
      self._draft_credit_ready,
      batch_size,
      flush=flush_endpoint,
    )
    if not records:
      return empty
    use_cuda_timer = str(device).startswith("cuda") and torch.cuda.is_available()
    gpu_start = torch.cuda.Event(enable_timing=True) if use_cuda_timer else None
    gpu_end = torch.cuda.Event(enable_timing=True) if use_cuda_timer else None
    if gpu_start is not None:
      gpu_start.record()

    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    weighted_loss = 0.0
    weighted_grad_norm = 0.0
    batches = 0
    fixed_rows = 0
    for offset in range(0, len(records), batch_size):
      chunk = records[offset:offset + batch_size]
      padded, valid_np = pad_terminal_credit_records(chunk, batch_size)
      valid_count = len(chunk)
      batches += 1
      fixed_rows += batch_size
      observations = torch.stack([record["observation"] for record in padded]).to(
        device=device,
        non_blocking=True,
      )
      actions = torch.stack([record["action"] for record in padded]).to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
      )
      old_logprobs = torch.stack([record["old_logprob"] for record in padded]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
      )
      targets = torch.tensor(
        [float(record["target"]) for record in padded],
        device=device,
        dtype=torch.float32,
      )
      slots = torch.tensor(
        [int(record["slot"]) for record in padded],
        device=device,
        dtype=torch.int64,
      )
      valid = torch.as_tensor(valid_np, device=device, dtype=torch.bool)
      state: dict[str, object] = {
        "mask": torch.ones(batch_size, device=device, dtype=torch.bool),
      }
      if self._use_rnn:
        state["lstm_h"] = torch.stack([record["lstm_h"] for record in padded]).to(
          device=device,
          dtype=torch.float32,
          non_blocking=True,
        )
        state["lstm_c"] = torch.stack([record["lstm_c"] for record in padded]).to(
          device=device,
          dtype=torch.float32,
          non_blocking=True,
        )

      self.optimizer.zero_grad()
      amp_cm = self.amp_context if self.amp_context is not None else contextlib.nullcontext()
      with amp_cm:
        logits, _ = self.policy.forward_eval(observations, state)
        _, new_logprobs, entropy = azk_pytorch.sample_logits(logits, action=actions)
        win_logits = state.get("_azk_win_prob_logits")
        if not torch.is_tensor(win_logits):
          raise RuntimeError("whole-draft credit requires win-probability logits")
        win_logits = win_logits.reshape_as(targets)
        predictions = torch.sigmoid(win_logits)
        advantages = targets - predictions.detach()
        policy_loss, ratios = clipped_terminal_policy_loss(
          new_logprobs.reshape_as(targets),
          old_logprobs,
          advantages,
          self._draft_credit_clip,
          mask=valid,
        )
        total_loss = self._draft_credit_coef * policy_loss
      total_loss.backward()
      grad_norm = torch.nn.utils.clip_grad_norm_(
        self.policy.parameters(),
        self.config["max_grad_norm"],
      )
      self.optimizer.step()
      self.optimizer.zero_grad()

      with torch.no_grad():
        valid_advantages = advantages[valid].float()
        valid_targets = targets[valid].float()
        valid_predictions = predictions[valid].float()
        valid_ratios = ratios[valid].float()
        valid_entropy = entropy.reshape_as(targets)[valid].float()
        valid_logits = win_logits[valid].float()
        baseline_bce = F.binary_cross_entropy_with_logits(
          valid_logits,
          valid_targets,
          reduction="none",
        )
        arrays["advantages"].append(valid_advantages.cpu().numpy())
        arrays["targets"].append(valid_targets.cpu().numpy())
        arrays["predictions"].append(valid_predictions.cpu().numpy())
        arrays["ratios"].append(valid_ratios.cpu().numpy())
        arrays["entropy"].append(valid_entropy.cpu().numpy())
        arrays["baseline_bce"].append(baseline_bce.cpu().numpy())
        arrays["slots"].append(slots[valid].cpu().numpy())
        weighted_loss += float(policy_loss.detach().item()) * valid_count
        weighted_grad_norm += float(grad_norm.detach().item()) * valid_count

    gpu_seconds = 0.0
    if gpu_end is not None and gpu_start is not None:
      gpu_end.record()
      gpu_end.synchronize()
      gpu_seconds = float(gpu_start.elapsed_time(gpu_end) / 1000.0)
    advantages_np = np.concatenate(arrays["advantages"])
    targets_np = np.concatenate(arrays["targets"])
    predictions_np = np.concatenate(arrays["predictions"])
    ratios_np = np.concatenate(arrays["ratios"])
    entropy_np = np.concatenate(arrays["entropy"])
    baseline_bce_np = np.concatenate(arrays["baseline_bce"])
    slots_np = np.concatenate(arrays["slots"])
    example_count = int(advantages_np.size)
    metrics = {
      "draft_credit_examples": float(example_count),
      "draft_credit_batches": float(batches),
      "draft_credit_fixed_batch_rows": float(fixed_rows),
      "draft_credit_padding_fraction": float(1.0 - example_count / fixed_rows),
      "draft_credit_loss": weighted_loss / example_count,
      "draft_credit_advantage_mean": float(advantages_np.mean()),
      "draft_credit_advantage_abs_mean": float(np.abs(advantages_np).mean()),
      "draft_credit_advantage_std": float(advantages_np.std()),
      "draft_credit_sign_outcome_agreement": float(
        ((advantages_np > 0.0) == (targets_np > 0.5)).mean()
      ),
      "draft_credit_baseline_bce": float(baseline_bce_np.mean()),
      "draft_credit_baseline_brier": float(
        np.square(predictions_np - targets_np).mean()
      ),
      "draft_credit_baseline_pred_mean": float(predictions_np.mean()),
      "draft_credit_importance_mean": float(ratios_np.mean()),
      "draft_credit_clipfrac": float(
        (np.abs(ratios_np - 1.0) > self._draft_credit_clip).mean()
      ),
      "draft_credit_entropy": float(entropy_np.mean()),
      "draft_credit_gradient_norm": weighted_grad_norm / example_count,
      "draft_credit_raw_gradient_norm": (
        weighted_grad_norm / example_count / self._draft_credit_coef
      ),
      "draft_credit_train_seconds": float(time.perf_counter() - train_started),
      "draft_credit_gpu_seconds": gpu_seconds,
      "draft_credit_control_draft_gradient_norm": float(
        self._draft_credit_control_grad_norm
      ),
    }
    metrics.update(
      self._draft_credit_position_metrics(
        advantages_np,
        targets_np,
        predictions_np,
        baseline_bce_np,
        slots_np,
      )
    )
    return metrics

  def _init_draft_episode_credit_layout(self, obs_row_bytes: int) -> dict[str, int]:
    from observation import DECKBUILD_OBSERVATION_CTYPE

    dtype = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
    if int(obs_row_bytes) != dtype.itemsize:
      raise ValueError(
        f"full-episode draft credit requires {dtype.itemsize}B packed observations, "
        f"got {obs_row_bytes}B"
      )
    deck_dtype, deck_offset = dtype.fields["deck_context"][:2]
    _, mode_offset = deck_dtype.fields["mode"][:2]
    _, gate_offset = deck_dtype.fields["gate_card_def_id"][:2]
    _, leader_offset = deck_dtype.fields["leader_card_def_id"][:2]
    _, main_count_offset = deck_dtype.fields["main_count"][:2]
    action_mask_dtype, action_mask_offset = dtype.fields["action_mask"][:2]
    _, legal_count_offset = action_mask_dtype.fields["legal_action_count"][:2]
    layout = {
      "mode_offset": int(deck_offset + mode_offset),
      "gate_offset": int(deck_offset + gate_offset),
      "leader_offset": int(deck_offset + leader_offset),
      "main_count_offset": int(deck_offset + main_count_offset),
      "legal_count_offset": int(action_mask_offset + legal_count_offset),
    }
    self._draft_episode_credit_layout = layout
    print(
      "[draft-episode-credit] enabled: "
      f"coef={self._draft_episode_credit_coef:.6f}, "
      f"baseline_coef={self._draft_episode_credit_baseline_coef:.6f}, "
      f"clip={self._draft_episode_credit_clip:.3f}, "
      f"batch_drafts={self._draft_episode_credit_batch_drafts}, "
      f"update_interval={self._draft_episode_credit_update_interval}, "
      f"seed={self._draft_episode_credit_seed}, "
      "rows=all-50-main-picks, immediate-draft-actor=masked"
    )
    return layout

  def _decode_draft_episode_credit_rows(
    self,
    observations_cpu: torch.Tensor,
    active_mask: np.ndarray,
  ) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
  ]:
    if not self._draft_episode_credit_enabled:
      return None, None, None, None, None, None
    layout = self._draft_episode_credit_layout
    if layout is None:
      layout = self._init_draft_episode_credit_layout(
        int(observations_cpu.shape[-1])
      )
    raw = observations_cpu.detach().cpu().numpy()
    mode_offset = layout["mode_offset"]
    gate_offset = layout["gate_offset"]
    leader_offset = layout["leader_offset"]
    main_count_offset = layout["main_count_offset"]
    legal_count_offset = layout["legal_count_offset"]
    modes = raw[:, mode_offset:mode_offset + 4].copy().view(np.int32).reshape(-1)
    gate_ids = raw[:, gate_offset:gate_offset + 2].copy().view(np.int16).reshape(-1)
    leader_ids = (
      raw[:, leader_offset:leader_offset + 2].copy().view(np.int16).reshape(-1)
    )
    main_counts = raw[:, main_count_offset].astype(np.int16, copy=True)
    legal_counts = (
      raw[:, legal_count_offset:legal_count_offset + 2]
      .copy()
      .view(np.uint16)
      .reshape(-1)
    )
    eligible = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
    eligible = np.logical_and(eligible, modes == 2)
    eligible = np.logical_and(eligible, legal_counts > 0)
    return eligible, modes, main_counts, gate_ids, leader_ids, legal_counts

  def _apply_random_draft_prefix(
    self,
    *,
    logits: object,
    actions: torch.Tensor,
    logprobs: torch.Tensor,
    policy_rows_np: np.ndarray,
    env_id_np: np.ndarray,
    draft_rows_np: np.ndarray | None,
    draft_main_counts_np: np.ndarray | None,
    draft_legal_counts_np: np.ndarray | None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forced = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.bool)
    if not getattr(self, "_draft_prefix_enabled", False):
      return actions, logprobs, forced
    if (
      draft_rows_np is None
      or draft_main_counts_np is None
      or draft_legal_counts_np is None
    ):
      raise RuntimeError("random main prefix requires decoded draft rows")
    policy_main_counts = draft_main_counts_np[policy_rows_np]
    prefix_candidate_rows = np.logical_and(
      draft_rows_np[policy_rows_np],
      policy_main_counts < max(self._draft_prefix_lengths),
    )
    eligible_local = np.nonzero(prefix_candidate_rows)[0]
    selections: list[tuple[int, int]] = []
    for local_index_raw in eligible_local.tolist():
      local_index = int(local_index_raw)
      batch_position = int(policy_rows_np[local_index])
      agent_id = int(env_id_np[batch_position])
      env_index = int(agent_id // self._agents_per_env)
      seat = int(agent_id % self._agents_per_env)
      episode_id = int(self._env_episode_ids[env_index])
      main_count = int(draft_main_counts_np[batch_position])
      prefix_length = deterministic_draft_prefix_length(
        episode_id,
        seat,
        self._draft_prefix_seed,
        self._draft_prefix_lengths,
        self._draft_prefix_probabilities,
      )
      if main_count == 0:
        self._draft_prefix_episode_lengths.append(prefix_length)
      if main_count >= prefix_length:
        continue
      legal_count = int(draft_legal_counts_np[batch_position])
      candidate = deterministic_draft_prefix_candidate(
        episode_id,
        seat,
        main_count + 1,
        legal_count,
        self._draft_prefix_seed,
      )
      selections.append((local_index, candidate))
    if not selections:
      return actions, logprobs, forced

    local_indices = torch.as_tensor(
      [selection[0] for selection in selections],
      device=actions.device,
      dtype=torch.long,
    )
    candidates = torch.as_tensor(
      [selection[1] for selection in selections],
      device=actions.device,
      dtype=torch.long,
    )
    actions = actions.clone()
    actions[local_indices] = 0
    actions[local_indices, 0] = 3
    actions[local_indices, 1] = candidates
    forced[local_indices] = True
    amp_cm = (
      self.amp_context
      if self.amp_context is not None
      else contextlib.nullcontext()
    )
    with torch.no_grad(), amp_cm:
      _, logprobs, _ = azk_pytorch.sample_logits(logits, action=actions)
    self._draft_prefix_forced_rows += len(selections)
    return actions, logprobs, forced

  def _stash_draft_episode_credit_rows(
    self,
    *,
    observations_cpu: torch.Tensor | None,
    policy_rows_np: np.ndarray,
    actions: torch.Tensor,
    logprobs: torch.Tensor,
    pre_lstm_h: torch.Tensor | None,
    pre_lstm_c: torch.Tensor | None,
    env_id_np: np.ndarray,
    draft_rows_np: np.ndarray | None,
    draft_main_counts_np: np.ndarray | None,
    draft_gate_ids_np: np.ndarray | None,
    draft_leader_ids_np: np.ndarray | None,
    forced_rows: torch.Tensor | None = None,
  ) -> None:
    if not getattr(self, "_draft_episode_credit_enabled", False):
      return
    if (
      observations_cpu is None
      or draft_rows_np is None
      or draft_main_counts_np is None
      or draft_gate_ids_np is None
      or draft_leader_ids_np is None
      or pre_lstm_h is None
      or pre_lstm_c is None
    ):
      raise RuntimeError("full-episode draft credit requires decoded rollout rows")
    eligible_local = np.nonzero(draft_rows_np[policy_rows_np])[0]
    if eligible_local.size == 0:
      return
    started = time.perf_counter()
    local_indices_t = torch.as_tensor(
      eligible_local,
      device=actions.device,
      dtype=torch.long,
    )
    batch_positions = policy_rows_np[eligible_local]
    batch_positions_t = torch.as_tensor(batch_positions, dtype=torch.long)
    observation_rows = observations_cpu[batch_positions_t].detach().cpu()
    action_rows = actions[local_indices_t].detach().to(
      device="cpu",
      dtype=torch.long,
    )
    logprob_rows = logprobs[local_indices_t].detach().to(
      device="cpu",
      dtype=torch.float32,
    )
    h_rows = pre_lstm_h[local_indices_t].detach().to(
      device="cpu",
      dtype=torch.float32,
    )
    c_rows = pre_lstm_c[local_indices_t].detach().to(
      device="cpu",
      dtype=torch.float32,
    )
    actor_valid_rows = (
      torch.ones(eligible_local.size, dtype=torch.bool)
      if forced_rows is None
      else torch.logical_not(forced_rows[local_indices_t]).detach().to(
        device="cpu",
        dtype=torch.bool,
      )
    )

    for selected_index, batch_position_raw in enumerate(batch_positions.tolist()):
      batch_position = int(batch_position_raw)
      agent_id = int(env_id_np[batch_position])
      env_index = int(agent_id // self._agents_per_env)
      seat = int(agent_id % self._agents_per_env)
      episode_id = int(self._env_episode_ids[env_index])
      key = (episode_id, seat)
      main_pick = int(draft_main_counts_np[batch_position]) + 1
      if not 1 <= main_pick <= DRAFT_EPISODE_MAIN_PICKS:
        raise RuntimeError(f"invalid full-episode main pick {main_pick}")
      gate_id = int(draft_gate_ids_np[batch_position])
      leader_id = int(draft_leader_ids_np[batch_position])
      pending = self._draft_episode_credit_pending.get(key)
      if pending is None:
        pending = {
          "episode_id": episode_id,
          "seat": seat,
          "gate_id": gate_id,
          "leader_id": leader_id,
          "observations": [None] * DRAFT_EPISODE_MAIN_PICKS,
          "actions": [None] * DRAFT_EPISODE_MAIN_PICKS,
          "old_logprobs": [None] * DRAFT_EPISODE_MAIN_PICKS,
          "lstm_h": [None] * DRAFT_EPISODE_MAIN_PICKS,
          "lstm_c": [None] * DRAFT_EPISODE_MAIN_PICKS,
          "actor_valid": [None] * DRAFT_EPISODE_MAIN_PICKS,
          "capture_epochs": np.full(
            DRAFT_EPISODE_MAIN_PICKS,
            -1,
            dtype=np.int32,
          ),
          "seen": np.zeros(DRAFT_EPISODE_MAIN_PICKS, dtype=np.bool_),
        }
        self._draft_episode_credit_pending[key] = pending
      elif (
        int(pending["gate_id"]) != gate_id
        or int(pending["leader_id"]) != leader_id
      ):
        raise RuntimeError("gate or leader changed within a retained draft")
      seen = pending["seen"]
      if not isinstance(seen, np.ndarray):
        raise RuntimeError("invalid full-episode draft seen mask")
      pick_index = main_pick - 1
      if bool(seen[pick_index]):
        continue
      pending["observations"][pick_index] = observation_rows[selected_index]
      pending["actions"][pick_index] = action_rows[selected_index]
      pending["old_logprobs"][pick_index] = logprob_rows[selected_index]
      pending["lstm_h"][pick_index] = h_rows[selected_index]
      pending["lstm_c"][pick_index] = c_rows[selected_index]
      pending["actor_valid"][pick_index] = actor_valid_rows[selected_index]
      pending["capture_epochs"][pick_index] = int(self.epoch)
      seen[pick_index] = True
      self._draft_episode_credit_captured += 1
    self._draft_episode_credit_capture_seconds += time.perf_counter() - started

  def _offer_completed_draft_episode_credit(
    self,
    record: dict[str, object],
  ) -> None:
    """Keep a bounded deterministic sample of this epoch's complete drafts."""
    priority = deterministic_draft_episode_priority(
      int(record["episode_id"]),
      int(record["seat"]),
      self._draft_episode_credit_seed,
    )
    record["sample_priority"] = priority
    self._draft_episode_credit_completed += 1
    self._draft_episode_credit_window_completed += 1
    if len(self._draft_episode_credit_ready) < self._draft_episode_credit_batch_drafts:
      self._draft_episode_credit_ready.append(record)
      return
    worst_index = max(
      range(len(self._draft_episode_credit_ready)),
      key=lambda index: int(
        self._draft_episode_credit_ready[index]["sample_priority"]
      ),
    )
    worst_priority = int(
      self._draft_episode_credit_ready[worst_index]["sample_priority"]
    )
    if priority < worst_priority:
      self._draft_episode_credit_ready[worst_index] = record
    self._draft_episode_credit_sample_dropped += 1
    self._draft_episode_credit_window_sample_dropped += 1

  def _finalize_draft_episode_credit_rows(
    self,
    env_id_np: np.ndarray,
    done_mask: np.ndarray,
    terminal_mask: np.ndarray,
    terminal_rewards: np.ndarray,
  ) -> None:
    if not self._draft_episode_credit_enabled or not self._draft_episode_credit_pending:
      return
    finished_envs = self._episode_envs_from_done_mask(env_id_np, done_mask)
    if finished_envs.size == 0:
      return
    labels_by_env = pufferl.terminal_win_labels_from_rewards(
      env_id_np,
      terminal_mask,
      terminal_rewards,
      self._agents_per_env,
    )
    for env_index_raw in finished_envs:
      env_index = int(env_index_raw)
      episode_id = int(self._env_episode_ids[env_index])
      labels = labels_by_env.get(env_index, {})
      for seat in range(self._agents_per_env):
        pending = self._draft_episode_credit_pending.pop((episode_id, seat), None)
        if pending is None:
          continue
        seen = pending["seen"]
        if not isinstance(seen, np.ndarray):
          raise RuntimeError("invalid full-episode draft seen mask")
        target = labels.get(seat)
        if target is None:
          agent_id = env_index * self._agents_per_env + seat
          positions = np.nonzero(env_id_np == agent_id)[0]
          is_terminal_draw = bool(
            positions.size > 0
            and terminal_mask[positions[0]]
            and float(terminal_rewards[positions[0]]) == 0.0
          )
          if is_terminal_draw:
            target = 0.5
            self._draft_episode_credit_draws += 1
          else:
            self._draft_episode_credit_truncated += int(seen.sum())
            continue
        if not bool(seen.all()):
          self._draft_episode_credit_incomplete += 1
          continue
        for field in (
          "observations",
          "actions",
          "old_logprobs",
          "lstm_h",
          "lstm_c",
          "actor_valid",
        ):
          rows = pending[field]
          if not isinstance(rows, list) or any(row is None for row in rows):
            raise RuntimeError(f"full-episode draft has invalid {field}")
          pending[field] = torch.stack(rows)
        pending["target"] = float(target)
        pending["terminal_epoch"] = int(self.epoch)
        if float(target) > 0.5:
          self._draft_episode_credit_decisive += 1
          self._draft_episode_credit_wins += 1
        elif float(target) < 0.5:
          self._draft_episode_credit_decisive += 1
          self._draft_episode_credit_losses += 1
        self._offer_completed_draft_episode_credit(pending)
        self._draft_episode_credit_labeled += DRAFT_EPISODE_MAIN_PICKS

  @staticmethod
  def _draft_episode_credit_position_metrics(
    advantages: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    baseline_bce: np.ndarray,
    main_picks: np.ndarray,
  ) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for index, (low, high) in enumerate(DRAFT_TERMINAL_CREDIT_QUARTILES, start=1):
      selector = np.logical_and(main_picks >= low, main_picks <= high)
      name = f"q{index}"
      count = int(selector.sum())
      metrics[f"draft_episode_credit_{name}_examples"] = float(count)
      if count == 0:
        continue
      selected_advantages = advantages[selector]
      selected_targets = targets[selector]
      selected_predictions = predictions[selector]
      decisive = selected_targets != 0.5
      metrics[f"draft_episode_credit_{name}_advantage_mean"] = float(
        selected_advantages.mean()
      )
      metrics[f"draft_episode_credit_{name}_advantage_abs_mean"] = float(
        np.abs(selected_advantages).mean()
      )
      metrics[f"draft_episode_credit_{name}_advantage_std"] = float(
        selected_advantages.std()
      )
      metrics[f"draft_episode_credit_{name}_sign_outcome_agreement"] = (
        float(
          (
            (selected_advantages[decisive] > 0.0)
            == (selected_targets[decisive] > 0.5)
          ).mean()
        )
        if bool(decisive.any())
        else 0.0
      )
      metrics[f"draft_episode_credit_{name}_draw_fraction"] = float(
        np.logical_not(decisive).mean()
      )
      metrics[f"draft_episode_credit_{name}_baseline_bce"] = float(
        baseline_bce[selector].mean()
      )
      metrics[f"draft_episode_credit_{name}_baseline_brier"] = float(
        np.square(selected_predictions - selected_targets).mean()
      )
    return metrics

  def _train_draft_episode_credit(self) -> dict[str, float]:
    empty = {
      "draft_episode_credit_episodes": 0.0,
      "draft_episode_credit_examples": 0.0,
      "draft_episode_credit_loss": 0.0,
      "draft_episode_credit_baseline_loss": 0.0,
      "draft_episode_credit_train_seconds": 0.0,
      "draft_episode_credit_gpu_seconds": 0.0,
      "draft_episode_credit_gradient_norm": 0.0,
    }
    if (
      not self._draft_episode_credit_enabled
      or not self._draft_episode_credit_ready
      or not draft_episode_credit_update_due(
        self.epoch,
        self.total_epochs,
        self._draft_episode_credit_update_interval,
      )
    ):
      return empty
    started = time.perf_counter()
    batch_drafts = self._draft_episode_credit_batch_drafts
    records = sorted(
      self._draft_episode_credit_ready,
      key=lambda record: int(record["sample_priority"]),
    )
    self._draft_episode_credit_ready = []
    padded, valid_episode_np = pad_complete_draft_episodes(records, batch_drafts)
    device = self.config["device"]
    observations = torch.stack(
      [record["observations"] for record in padded]
    ).reshape(
      batch_drafts * DRAFT_EPISODE_MAIN_PICKS,
      -1,
    ).to(device=device, non_blocking=True)
    actions = torch.stack([record["actions"] for record in padded]).to(
      device=device,
      dtype=torch.long,
      non_blocking=True,
    ).reshape(batch_drafts * DRAFT_EPISODE_MAIN_PICKS, -1)
    old_logprobs = torch.stack(
      [record["old_logprobs"] for record in padded]
    ).to(device=device, dtype=torch.float32, non_blocking=True).reshape(-1)
    lstm_h = torch.stack([record["lstm_h"] for record in padded]).to(
      device=device, dtype=torch.float32, non_blocking=True
    ).reshape(batch_drafts * DRAFT_EPISODE_MAIN_PICKS, -1)
    lstm_c = torch.stack([record["lstm_c"] for record in padded]).to(
      device=device, dtype=torch.float32, non_blocking=True
    ).reshape(batch_drafts * DRAFT_EPISODE_MAIN_PICKS, -1)
    targets_by_episode = torch.tensor(
      [float(record["target"]) for record in padded],
      device=device,
      dtype=torch.float32,
    )
    targets = targets_by_episode[:, None].expand(
      batch_drafts, DRAFT_EPISODE_MAIN_PICKS
    ).reshape(-1)
    valid_episode = torch.as_tensor(
      valid_episode_np,
      device=device,
      dtype=torch.bool,
    )
    valid = valid_episode[:, None].expand(
      batch_drafts, DRAFT_EPISODE_MAIN_PICKS
    ).reshape(-1)
    actor_valid = torch.stack(
      [record["actor_valid"] for record in padded]
    ).to(device=device, dtype=torch.bool, non_blocking=True).reshape(-1)
    actor_valid = torch.logical_and(valid, actor_valid)
    state: dict[str, object] = {
      "mask": torch.ones_like(valid),
      "lstm_h": lstm_h,
      "lstm_c": lstm_c,
    }
    use_cuda_timer = str(device).startswith("cuda") and torch.cuda.is_available()
    gpu_start = torch.cuda.Event(enable_timing=True) if use_cuda_timer else None
    gpu_end = torch.cuda.Event(enable_timing=True) if use_cuda_timer else None
    if gpu_start is not None:
      gpu_start.record()

    self.optimizer.zero_grad()
    amp_cm = self.amp_context if self.amp_context is not None else contextlib.nullcontext()
    with amp_cm:
      differentiable_forward = getattr(
        self.policy,
        "_cg_eager_forward_eval",
        self.policy.forward_eval,
      )
      logits, _ = differentiable_forward(observations, state)
      _, new_logprobs, entropy = azk_pytorch.sample_logits(logits, action=actions)
      new_logprobs = new_logprobs.reshape_as(targets)
      entropy = entropy.reshape_as(targets)
      win_logits = state.get("_azk_win_prob_logits")
      if not torch.is_tensor(win_logits):
        raise RuntimeError("full-episode draft credit requires win-probability logits")
      win_logits = win_logits.reshape_as(targets)
      predictions = torch.sigmoid(win_logits)
      advantages = targets - predictions.detach()
      policy_loss, ratios = clipped_terminal_policy_loss(
        new_logprobs,
        old_logprobs,
        advantages,
        self._draft_episode_credit_clip,
        mask=actor_valid,
      )
      baseline_elements = F.binary_cross_entropy_with_logits(
        win_logits,
        targets,
        reduction="none",
      )
      baseline_loss = masked_tensor_mean(baseline_elements, valid)
      entropy_loss = masked_tensor_mean(entropy, actor_valid)
      total_loss = (
        self._draft_episode_credit_coef * policy_loss
        + self._draft_episode_credit_baseline_coef * baseline_loss
        - float(self.config["ent_coef"]) * entropy_loss
      )
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
      self.policy.parameters(),
      self.config["max_grad_norm"],
    )
    self.optimizer.step()
    self.optimizer.zero_grad()

    gpu_seconds = 0.0
    if gpu_end is not None and gpu_start is not None:
      gpu_end.record()
      gpu_end.synchronize()
      gpu_seconds = float(gpu_start.elapsed_time(gpu_end) / 1000.0)
    with torch.no_grad():
      valid_advantages = advantages[actor_valid].float().cpu().numpy()
      valid_targets = targets[actor_valid].float().cpu().numpy()
      valid_predictions = predictions[actor_valid].float().cpu().numpy()
      valid_bce = baseline_elements[actor_valid].float().cpu().numpy()
      valid_ratios = ratios[actor_valid].float().cpu().numpy()
      valid_entropy = entropy[actor_valid].float().cpu().numpy()
      baseline_predictions = predictions[valid].float().cpu().numpy()
    decisive = valid_targets != 0.5

    pick_grid = np.broadcast_to(
      np.arange(1, DRAFT_EPISODE_MAIN_PICKS + 1, dtype=np.int16),
      (batch_drafts, DRAFT_EPISODE_MAIN_PICKS),
    )
    actor_valid_np = actor_valid.reshape(
      batch_drafts,
      DRAFT_EPISODE_MAIN_PICKS,
    ).detach().cpu().numpy()
    valid_main_picks = pick_grid[actor_valid_np]
    capture_epochs = np.stack(
      [np.asarray(record["capture_epochs"], dtype=np.int32) for record in records]
    ).reshape(-1)
    record_ages = (int(self.epoch) - capture_epochs)[
      actor_valid_np[:len(records)].reshape(-1)
    ]
    terminal_ages = np.asarray(
      [int(self.epoch) - int(record["terminal_epoch"]) for record in records],
      dtype=np.int32,
    )
    metrics = {
      "draft_episode_credit_episodes": float(len(records)),
      "draft_episode_credit_examples": float(valid_advantages.size),
      "draft_episode_credit_baseline_examples": float(
        baseline_predictions.size
      ),
      "draft_episode_credit_forced_rows": float(
        baseline_predictions.size - valid_advantages.size
      ),
      "draft_episode_credit_fixed_rows": float(
        batch_drafts * DRAFT_EPISODE_MAIN_PICKS
      ),
      "draft_episode_credit_padding_fraction": float(
        1.0 - len(records) / batch_drafts
      ),
      "draft_episode_credit_window_completed": float(
        self._draft_episode_credit_window_completed
      ),
      "draft_episode_credit_window_sample_dropped": float(
        self._draft_episode_credit_window_sample_dropped
      ),
      "draft_episode_credit_window_retention_fraction": float(
        len(records) / max(self._draft_episode_credit_window_completed, 1)
      ),
      "draft_episode_credit_loss": float(policy_loss.detach().item()),
      "draft_episode_credit_baseline_loss": float(baseline_loss.detach().item()),
      "draft_episode_credit_advantage_mean": float(valid_advantages.mean()),
      "draft_episode_credit_advantage_abs_mean": float(
        np.abs(valid_advantages).mean()
      ),
      "draft_episode_credit_advantage_std": float(valid_advantages.std()),
      "draft_episode_credit_sign_outcome_agreement": (
        float(
          (
            (valid_advantages[decisive] > 0.0)
            == (valid_targets[decisive] > 0.5)
          ).mean()
        )
        if bool(decisive.any())
        else 0.0
      ),
      "draft_episode_credit_target_mean": float(valid_targets.mean()),
      "draft_episode_credit_baseline_pred_mean": float(valid_predictions.mean()),
      "draft_episode_credit_baseline_bce": float(valid_bce.mean()),
      "draft_episode_credit_baseline_brier": float(
        np.square(valid_predictions - valid_targets).mean()
      ),
      "draft_episode_credit_importance_mean": float(valid_ratios.mean()),
      "draft_episode_credit_importance_p95": float(
        np.quantile(valid_ratios, 0.95)
      ),
      "draft_episode_credit_clipfrac": float(
        (np.abs(valid_ratios - 1.0) > self._draft_episode_credit_clip).mean()
      ),
      "draft_episode_credit_entropy": float(valid_entropy.mean()),
      "draft_episode_credit_gradient_norm": float(grad_norm.detach().item()),
      "draft_episode_credit_record_age_mean": float(record_ages.mean()),
      "draft_episode_credit_record_age_p95": float(np.quantile(record_ages, 0.95)),
      "draft_episode_credit_record_age_max": float(record_ages.max()),
      "draft_episode_credit_terminal_age_mean": float(terminal_ages.mean()),
      "draft_episode_credit_train_seconds": float(time.perf_counter() - started),
      "draft_episode_credit_gpu_seconds": gpu_seconds,
    }
    metrics.update(
      self._draft_episode_credit_position_metrics(
        valid_advantages,
        valid_targets,
        valid_predictions,
        valid_bce,
        valid_main_picks,
      )
    )
    for context_name, key in (("gate", "gate_id"), ("leader", "leader_id")):
      ids = np.asarray([int(record[key]) for record in records], dtype=np.int32)
      episode_targets = np.asarray(
        [float(record["target"]) for record in records], dtype=np.float32
      )
      for context_id in np.unique(ids):
        selector = ids == context_id
        prefix = f"draft_episode_credit_{context_name}_{int(context_id)}"
        metrics[f"{prefix}_episodes"] = float(selector.sum())
        metrics[f"{prefix}_target_mean"] = float(episode_targets[selector].mean())
    self._draft_episode_credit_window_completed = 0
    self._draft_episode_credit_window_sample_dropped = 0
    return metrics

  def _maybe_probe_control_draft_gradient(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    normalized_advantages: torch.Tensor,
    actor_loss_mask: torch.Tensor | None,
  ) -> None:
    if (
      not self._draft_credit_enabled
      or not self._draft_credit_grad_probe_enabled
      or self._draft_credit_grad_probe_done
    ):
      return
    layout = self._draft_credit_layout
    if layout is None:
      layout = self._init_draft_credit_layout(int(observations.shape[-1]))
    mode_offset = layout["mode_offset"]
    modes = observations[..., mode_offset:mode_offset + 4].contiguous().view(torch.int32)
    modes = modes.reshape(normalized_advantages.shape)
    draft_mask = torch.logical_or(modes == 1, modes == 2)
    if not bool(draft_mask.any()):
      return

    # Recompute this one disposable calibration minibatch through the eager hot
    # paths. The compiled backward donates buffers and cannot be retained for a
    # second autograd.grad call; a separate eager graph leaves production PPO
    # and its compiled graph untouched.
    base_policy = getattr(self.uncompiled_policy, "policy", None)
    compiled_paths = None
    if (
      base_policy is not None
      and hasattr(base_policy, "_eager_encode_observations")
      and hasattr(base_policy, "_eager_decode_actions")
    ):
      compiled_paths = (
        base_policy.encode_observations,
        base_policy.decode_actions,
      )
      base_policy.encode_observations = base_policy._eager_encode_observations
      base_policy.decode_actions = base_policy._eager_decode_actions
    try:
      probe_state = dict(action=actions, lstm_h=None, lstm_c=None)
      logits, _ = self.policy(observations, probe_state)
      _, newlogprobs, _ = azk_pytorch.sample_logits(logits, action=actions)
      newlogprobs = newlogprobs.reshape(old_logprobs.shape)
      ratio = torch.exp(newlogprobs - old_logprobs)
      pg_loss1 = -normalized_advantages * ratio
      pg_loss2 = -normalized_advantages * torch.clamp(
        ratio,
        1.0 - float(self.config["clip_coef"]),
        1.0 + float(self.config["clip_coef"]),
      )
      pg_loss_elem = torch.maximum(pg_loss1, pg_loss2)
      if actor_loss_mask is not None:
        pg_loss_elem = pg_loss_elem * actor_loss_mask
      draft_mask_float = draft_mask.to(dtype=pg_loss_elem.dtype)
      draft_loss = (
        (pg_loss_elem * draft_mask_float).sum()
        / draft_mask_float.sum().clamp_min(1.0)
      )
      parameters = [
        parameter for parameter in self.policy.parameters() if parameter.requires_grad
      ]
      gradients = torch.autograd.grad(
        draft_loss,
        parameters,
        allow_unused=True,
      )
    finally:
      if compiled_paths is not None:
        base_policy.encode_observations, base_policy.decode_actions = compiled_paths
    norm_sq = torch.zeros((), device=pg_loss_elem.device, dtype=torch.float32)
    for gradient in gradients:
      if gradient is not None:
        norm_sq += gradient.detach().float().square().sum()
    self._draft_credit_control_grad_norm = float(torch.sqrt(norm_sq).item())
    self._draft_credit_grad_probe_done = True

  def _infer_actions(
    self,
    o_device: torch.Tensor,
    mask_t: torch.Tensor,
    env_id_np: np.ndarray,
    *,
    observations_cpu: torch.Tensor | None = None,
    leader_rows_np: np.ndarray | None = None,
    leader_gate_ids_np: np.ndarray | None = None,
    draft_rows_np: np.ndarray | None = None,
    draft_modes_np: np.ndarray | None = None,
    draft_main_counts_np: np.ndarray | None = None,
    draft_gate_ids_np: np.ndarray | None = None,
    draft_episode_rows_np: np.ndarray | None = None,
    draft_episode_main_counts_np: np.ndarray | None = None,
    draft_episode_gate_ids_np: np.ndarray | None = None,
    draft_episode_leader_ids_np: np.ndarray | None = None,
    draft_episode_legal_counts_np: np.ndarray | None = None,
  ):
    device = self.config["device"]
    batch_n = o_device.shape[0]
    active = compute_league_active(
      global_step=int(self.global_step),
      activate_after_steps=int(self.league_cfg.activate_after_steps),
    )

    env_indices = (env_id_np // self._agents_per_env).astype(np.int32)
    if active:
      learner_rows_np = compute_learner_row_mask(
        env_id_np,
        agents_per_env=self._agents_per_env,
        env_learner_seat=self._env_learner_seat,
      )
      trainable_rows_np = compute_trainable_row_mask(
        env_id_np,
        agents_per_env=self._agents_per_env,
        env_learner_seat=self._env_learner_seat,
        env_use_latest=self._env_use_latest,
        league_active=active,
      )
    else:
      learner_rows_np = np.ones(batch_n, dtype=np.bool_)
      trainable_rows_np = np.ones(batch_n, dtype=np.bool_)

    actions_out = torch.zeros((batch_n, *self.vecenv.single_action_space.shape), device=device, dtype=torch.int32)
    logprobs_out = torch.zeros(batch_n, device=device)
    values_out = torch.zeros(batch_n, device=device)
    terminal_values_out = torch.zeros(batch_n, device=device)
    shaped_values_out = torch.zeros(batch_n, device=device)

    learner_idx = np.nonzero(learner_rows_np)[0]
    if learner_idx.size > 0:
      learner_idx_t = torch.as_tensor(learner_idx, device=device, dtype=torch.long)
      learner_state = {"mask": mask_t[learner_idx_t]}
      learner_pre_h = None
      learner_pre_c = None
      if self._use_rnn:
        learner_pre_h = self._learner_lstm_h[learner_idx_t]
        learner_pre_c = self._learner_lstm_c[learner_idx_t]
        learner_state["lstm_h"] = learner_pre_h
        learner_state["lstm_c"] = learner_pre_c
      logits, values = self._safe_forward_eval(self.policy, o_device[learner_idx_t], learner_state)
      if self._draftaux_enabled:
        # Pre-write-back: self._learner_lstm_h still holds the states the
        # boundary forward consumed (dict entries were replaced, not storage).
        self._draftaux_league_stash(
          o_device, mask_t, learner_idx_t, values, env_id_np
        )
      with torch.no_grad(), self.amp_context:
        actions, logprobs, _ = azk_pytorch.sample_logits(logits)
      actions, logprobs, forced_rows = self._apply_random_draft_prefix(
        logits=logits,
        actions=actions,
        logprobs=logprobs,
        policy_rows_np=learner_idx,
        env_id_np=env_id_np,
        draft_rows_np=draft_episode_rows_np,
        draft_main_counts_np=draft_episode_main_counts_np,
        draft_legal_counts_np=draft_episode_legal_counts_np,
      )
      self._stash_selected_leader_credit_rows(
        observations_cpu=observations_cpu,
        policy_rows_np=learner_idx,
        actions=actions,
        logprobs=logprobs,
        pre_lstm_h=learner_pre_h,
        pre_lstm_c=learner_pre_c,
        env_id_np=env_id_np,
        leader_rows_np=leader_rows_np,
        leader_gate_ids_np=leader_gate_ids_np,
      )
      self._stash_selected_draft_credit_rows(
        observations_cpu=observations_cpu,
        policy_rows_np=learner_idx,
        actions=actions,
        logprobs=logprobs,
        pre_lstm_h=learner_pre_h,
        pre_lstm_c=learner_pre_c,
        env_id_np=env_id_np,
        draft_rows_np=draft_rows_np,
        draft_modes_np=draft_modes_np,
        draft_main_counts_np=draft_main_counts_np,
        draft_gate_ids_np=draft_gate_ids_np,
      )
      self._stash_draft_episode_credit_rows(
        observations_cpu=observations_cpu,
        policy_rows_np=learner_idx,
        actions=actions,
        logprobs=logprobs,
        pre_lstm_h=learner_pre_h,
        pre_lstm_c=learner_pre_c,
        env_id_np=env_id_np,
        draft_rows_np=draft_episode_rows_np,
        draft_main_counts_np=draft_episode_main_counts_np,
        draft_gate_ids_np=draft_episode_gate_ids_np,
        draft_leader_ids_np=draft_episode_leader_ids_np,
        forced_rows=forced_rows,
      )
      actions_out[learner_idx_t] = actions.to(dtype=torch.int32)
      logprobs_out[learner_idx_t] = logprobs.to(dtype=logprobs_out.dtype)
      values_out[learner_idx_t] = values.flatten().to(dtype=values_out.dtype)
      if self._split_value_heads_enabled():
        terminal_value, shaped_value = self._component_values_from_state(
          learner_state,
          values_out[learner_idx_t].shape,
        )
        terminal_values_out[learner_idx_t] = terminal_value.detach().float()
        shaped_values_out[learner_idx_t] = shaped_value.detach().float()
      if self._use_rnn:
        self._learner_lstm_h[learner_idx_t] = learner_state["lstm_h"].to(
          device=self._learner_lstm_h.device,
          dtype=self._learner_lstm_h.dtype,
        )
        self._learner_lstm_c[learner_idx_t] = learner_state["lstm_c"].to(
          device=self._learner_lstm_c.device,
          dtype=self._learner_lstm_c.dtype,
        )

    opp_rows_np = np.nonzero(~learner_rows_np)[0]
    if active and opp_rows_np.size > 0:
      opp_env_indices = env_indices[opp_rows_np]
      opp_use_latest = self._env_use_latest[opp_env_indices]
      latest_rows_np = opp_rows_np[opp_use_latest]
      frozen_rows_np = opp_rows_np[~opp_use_latest]

      if latest_rows_np.size > 0:
        latest_idx_t = torch.as_tensor(latest_rows_np, device=device, dtype=torch.long)
        latest_state = {"mask": mask_t[latest_idx_t]}
        latest_pre_h = None
        latest_pre_c = None
        if self._use_rnn:
          latest_pre_h = self._learner_lstm_h[latest_idx_t]
          latest_pre_c = self._learner_lstm_c[latest_idx_t]
          latest_state["lstm_h"] = latest_pre_h
          latest_state["lstm_c"] = latest_pre_c
        latest_logits, latest_values = self._safe_forward_eval(self.policy, o_device[latest_idx_t], latest_state)
        with torch.no_grad(), self.amp_context:
          latest_actions, latest_logprobs, _ = azk_pytorch.sample_logits(latest_logits)
        (
          latest_actions,
          latest_logprobs,
          latest_forced_rows,
        ) = self._apply_random_draft_prefix(
          logits=latest_logits,
          actions=latest_actions,
          logprobs=latest_logprobs,
          policy_rows_np=latest_rows_np,
          env_id_np=env_id_np,
          draft_rows_np=draft_episode_rows_np,
          draft_main_counts_np=draft_episode_main_counts_np,
          draft_legal_counts_np=draft_episode_legal_counts_np,
        )
        self._stash_selected_leader_credit_rows(
          observations_cpu=observations_cpu,
          policy_rows_np=latest_rows_np,
          actions=latest_actions,
          logprobs=latest_logprobs,
          pre_lstm_h=latest_pre_h,
          pre_lstm_c=latest_pre_c,
          env_id_np=env_id_np,
          leader_rows_np=leader_rows_np,
          leader_gate_ids_np=leader_gate_ids_np,
        )
        self._stash_selected_draft_credit_rows(
          observations_cpu=observations_cpu,
          policy_rows_np=latest_rows_np,
          actions=latest_actions,
          logprobs=latest_logprobs,
          pre_lstm_h=latest_pre_h,
          pre_lstm_c=latest_pre_c,
          env_id_np=env_id_np,
          draft_rows_np=draft_rows_np,
          draft_modes_np=draft_modes_np,
          draft_main_counts_np=draft_main_counts_np,
          draft_gate_ids_np=draft_gate_ids_np,
        )
        self._stash_draft_episode_credit_rows(
          observations_cpu=observations_cpu,
          policy_rows_np=latest_rows_np,
          actions=latest_actions,
          logprobs=latest_logprobs,
          pre_lstm_h=latest_pre_h,
          pre_lstm_c=latest_pre_c,
          env_id_np=env_id_np,
          draft_rows_np=draft_episode_rows_np,
          draft_main_counts_np=draft_episode_main_counts_np,
          draft_gate_ids_np=draft_episode_gate_ids_np,
          draft_leader_ids_np=draft_episode_leader_ids_np,
          forced_rows=latest_forced_rows,
        )
        actions_out[latest_idx_t] = latest_actions.to(dtype=torch.int32)
        logprobs_out[latest_idx_t] = latest_logprobs.to(dtype=logprobs_out.dtype)
        values_out[latest_idx_t] = latest_values.flatten().to(dtype=values_out.dtype)
        if self._split_value_heads_enabled():
          terminal_value, shaped_value = self._component_values_from_state(
            latest_state,
            values_out[latest_idx_t].shape,
          )
          terminal_values_out[latest_idx_t] = terminal_value.detach().float()
          shaped_values_out[latest_idx_t] = shaped_value.detach().float()
        if self._use_rnn:
          self._learner_lstm_h[latest_idx_t] = latest_state["lstm_h"].to(
            device=self._learner_lstm_h.device,
            dtype=self._learner_lstm_h.dtype,
          )
          self._learner_lstm_c[latest_idx_t] = latest_state["lstm_c"].to(
            device=self._learner_lstm_c.device,
            dtype=self._learner_lstm_c.dtype,
          )

      if frozen_rows_np.size > 0:
        frozen_env_indices = env_indices[frozen_rows_np]
        frozen_policy_ids = self._env_opp_policy[frozen_env_indices]
        for policy_id in np.unique(frozen_policy_ids):
          rows_np = frozen_rows_np[frozen_policy_ids == policy_id]
          if rows_np.size == 0:
            continue
          rows_t = torch.as_tensor(rows_np, device=device, dtype=torch.long)
          opp_policy = self.opponent_policies[int(policy_id)]
          opp_state = {"mask": mask_t[rows_t]}
          if self._use_rnn:
            opp_state["lstm_h"] = self._opp_lstm_h[int(policy_id)][rows_t]
            opp_state["lstm_c"] = self._opp_lstm_c[int(policy_id)][rows_t]
          opp_logits, opp_values = self._safe_forward_eval(opp_policy, o_device[rows_t], opp_state)
          with torch.no_grad(), self.amp_context:
            opp_actions, _, _ = azk_pytorch.sample_logits(opp_logits)
          actions_out[rows_t] = opp_actions.to(dtype=torch.int32)
          values_out[rows_t] = opp_values.flatten().to(dtype=values_out.dtype)
          if self._split_value_heads_enabled():
            terminal_value, shaped_value = self._component_values_from_state(
              opp_state,
              values_out[rows_t].shape,
            )
            terminal_values_out[rows_t] = terminal_value.detach().float()
            shaped_values_out[rows_t] = shaped_value.detach().float()
          if self._use_rnn:
            self._opp_lstm_h[int(policy_id)][rows_t] = opp_state["lstm_h"].to(
              device=self._opp_lstm_h[int(policy_id)].device,
              dtype=self._opp_lstm_h[int(policy_id)].dtype,
            )
            self._opp_lstm_c[int(policy_id)][rows_t] = opp_state["lstm_c"].to(
              device=self._opp_lstm_c[int(policy_id)].device,
              dtype=self._opp_lstm_c[int(policy_id)].dtype,
            )

    return (
      actions_out,
      logprobs_out,
      values_out,
      terminal_values_out,
      shaped_values_out,
      trainable_rows_np,
      learner_rows_np,
      env_indices,
    )

  def _safe_forward_eval(self, model, obs: torch.Tensor, state: dict):
    with torch.no_grad(), self.amp_context:
      return model.forward_eval(obs, state)

  def evaluate(self):
    profile = self.profile
    epoch = self.epoch
    profile("eval", epoch)
    profile("eval_misc", epoch, nest=True)
    self._refresh_frozen_window()

    config = self.config
    device = config["device"]
    reward_multiplier = self._prepare_trainer_shaped_reward_anneal()

    self.full_rows = 0
    if self._leader_credit_enabled:
      self._leader_credit_captured = 0
      self._leader_credit_completed = 0
    if self._draft_credit_enabled:
      self._draft_credit_captured = 0
      self._draft_credit_labeled = 0
      self._draft_credit_truncated = 0
      self._draft_credit_incomplete = 0
      self._draft_credit_incomplete_records = 0
    if self._draft_episode_credit_enabled:
      self._draft_episode_credit_captured = 0
      self._draft_episode_credit_labeled = 0
      self._draft_episode_credit_truncated = 0
      self._draft_episode_credit_draws = 0
      self._draft_episode_credit_decisive = 0
      self._draft_episode_credit_wins = 0
      self._draft_episode_credit_losses = 0
      self._draft_episode_credit_incomplete = 0
      self._draft_episode_credit_completed = 0
      self._draft_episode_credit_sample_dropped = 0
      self._draft_episode_credit_capture_seconds = 0.0
    if self._draft_prefix_enabled:
      self._draft_prefix_forced_rows = 0
      self._draft_prefix_episode_lengths = []
    if self._prefix_outcome_enabled:
      if self._prefix_outcome_redistributor is None:
        raise RuntimeError("enabled draft prefix outcome redistributor is missing")
      self._prefix_outcome_redistributor.reset_window_metrics()
      self._prefix_outcome_inference_seconds = 0.0
    reference_finished = 0
    reference_finished_aligned = 0
    total_finished = 0
    reference_env_steps = 0
    total_env_steps = 0
    reference_trainable_rows = 0
    total_trainable_rows = 0
    reference_battle_rows = 0
    total_battle_rows = 0
    self._segment_is_trainable.zero_()
    self._reset_win_prob_rollout_buffers()
    self._reset_split_value_rollout_buffers()
    if self._xgate_mask_enabled:
      self.actor_loss_mask.fill_(1.0)
      if self._xgate_masked_steps:
        self.stats["xgate_masked_steps"].append(float(self._xgate_masked_steps))
      self._xgate_masked_steps = 0
    if self._draft_episode_credit_enabled:
      if self._draft_episode_phase_mask is None:
        raise RuntimeError("full-episode draft phase mask is missing")
      self._draft_episode_phase_mask.zero_()
    if self._draftaux_enabled:
      # Buffer rows recycle each epoch: never inject across the boundary.
      self._draftaux_pending = None
      self._draftaux_prevcoords.clear()
      if self._draftaux_events:
        self.stats["draftaux_events"].append(float(self._draftaux_events))
        self.stats["draftaux_injected_mean"].append(
          self._draftaux_injected / max(self._draftaux_events, 1)
        )
      self._draftaux_events = 0
      self._draftaux_injected = 0.0
    while self.full_rows < self.segments:
      profile("env", epoch)
      o, r, d, t, info, env_id, mask = self.vecenv.recv()

      profile("eval_misc", epoch)
      env_id_slice = slice(env_id[0], env_id[-1] + 1)
      done_mask = (d + t).astype(np.bool_)
      terminal_mask = np.asarray(d, dtype=np.bool_)
      env_id_np = np.asarray(env_id, dtype=np.int64)

      profile("eval_copy", epoch)
      o = torch.as_tensor(o)
      deck_modes = self._decode_reference_deck_modes(o)
      self._align_reference_matchups(env_id_np, deck_modes)
      leader_rows_np, leader_gate_ids_np = self._decode_leader_credit_rows(o, mask)
      (
        draft_rows_np,
        draft_modes_np,
        draft_main_counts_np,
        draft_gate_ids_np,
      ) = self._decode_draft_credit_rows(o, mask)
      (
        draft_episode_rows_np,
        draft_episode_modes_np,
        draft_episode_main_counts_np,
        draft_episode_gate_ids_np,
        draft_episode_leader_ids_np,
        draft_episode_legal_counts_np,
      ) = self._decode_draft_episode_credit_rows(o, mask)
      o_device = o.to(device)
      r_t = torch.as_tensor(r).to(device)
      d_t = torch.as_tensor(d).to(device)
      mask_t = torch.as_tensor(mask, device=device, dtype=torch.bool)
      reward_components_terminal_np, reward_components_shaped_np = self._extract_step_reward_components(
        info,
        env_id_np,
        np.clip(np.asarray(r, dtype=np.float32), -1.0, 1.0),
      )
      reward_components_terminal = torch.as_tensor(reward_components_terminal_np, device=device)
      reward_components_shaped = torch.as_tensor(reward_components_shaped_np, device=device)

      profile("eval_forward", epoch)
      (
        actions_t,
        logprobs_t,
        values_t,
        terminal_values_t,
        shaped_values_t,
        trainable_rows_np,
        learner_rows_np,
        env_indices,
      ) = self._infer_actions(
        o_device,
        mask_t,
        env_id_np,
        observations_cpu=o,
        leader_rows_np=leader_rows_np,
        leader_gate_ids_np=leader_gate_ids_np,
        draft_rows_np=draft_rows_np,
        draft_modes_np=draft_modes_np,
        draft_main_counts_np=draft_main_counts_np,
        draft_gate_ids_np=draft_gate_ids_np,
        draft_episode_rows_np=draft_episode_rows_np,
        draft_episode_main_counts_np=draft_episode_main_counts_np,
        draft_episode_gate_ids_np=draft_episode_gate_ids_np,
        draft_episode_leader_ids_np=draft_episode_leader_ids_np,
        draft_episode_legal_counts_np=draft_episode_legal_counts_np,
      )
      prefix_outcome_redistribution_np = None
      if self._prefix_outcome_enabled:
        prefix_outcome_redistribution_np = self._compute_prefix_outcome_redistribution(
          observations_cpu=o,
          env_id_np=env_id_np,
          trainable_rows_np=trainable_rows_np,
          done_mask=done_mask,
          terminal_mask=terminal_mask,
        )

      trainable_step_mask = np.logical_and(mask.astype(np.bool_), trainable_rows_np)
      self.global_step += int(trainable_step_mask.sum())
      if self._reference_opponent_only:
        reference_rows = self._env_is_reference[env_indices]
        total_trainable_rows += int(trainable_step_mask.sum())
        reference_trainable_rows += int(
          np.logical_and(trainable_step_mask, reference_rows).sum()
        )
        unique_envs = np.unique(env_indices)
        total_env_steps += int(unique_envs.size)
        reference_env_steps += int(self._env_is_reference[unique_envs].sum())
        if deck_modes is None:
          raise RuntimeError("reference alignment is enabled without decoded deck modes")
        battle_rows = compute_battle_row_mask(
          env_id_np,
          deck_modes,
          agents_per_env=self._agents_per_env,
        )
        trainable_battle_rows = np.logical_and(trainable_step_mask, battle_rows)
        total_battle_rows += int(trainable_battle_rows.sum())
        reference_battle_rows += int(
          np.logical_and(trainable_battle_rows, reference_rows).sum()
        )

      profile("eval_copy", epoch)
      with torch.no_grad():
        l = self.ep_lengths[env_id_slice.start].item()
        batch_rows = slice(
          self.ep_indices[env_id_slice.start].item(),
          1 + self.ep_indices[env_id_slice.stop - 1].item(),
        )

        if config["cpu_offload"]:
          self.observations[batch_rows, l] = o
        else:
          self.observations[batch_rows, l] = o_device

        self.actions[batch_rows, l] = actions_t
        self.logprobs[batch_rows, l] = logprobs_t
        scaled_total, scaled_shaped = pufferl.recombine_reward_components(
          torch.clamp(r_t, -1, 1),
          reward_components_terminal,
          reward_components_shaped,
          reward_multiplier,
        )
        if prefix_outcome_redistribution_np is None:
          self.rewards[batch_rows, l] = scaled_total
          self.terminal_reward_components[batch_rows, l] = reward_components_terminal
        else:
          prefix_outcome_redistribution = torch.as_tensor(
            prefix_outcome_redistribution_np,
            device=device,
            dtype=scaled_total.dtype,
          )
          self.rewards[batch_rows, l] = scaled_total + prefix_outcome_redistribution
          self.terminal_reward_components[batch_rows, l] = (
            reward_components_terminal + prefix_outcome_redistribution
          )
        self.shaped_reward_components[batch_rows, l] = scaled_shaped
        if self._draftaux_enabled and self._draftaux_pending is not None:
          b_global, aux = self._draftaux_pending
          self._draftaux_pending = None
          prev = self._draftaux_prevcoords.get(env_id_slice.start)
          if prev is not None:
            prev_row_start, prev_l = prev
            seg_rows = prev_row_start + (b_global - env_id_slice.start)
            aux_cast = aux.to(self.rewards.dtype)
            self.rewards[seg_rows, prev_l] += aux_cast
            self.shaped_reward_components[seg_rows, prev_l] += aux_cast
            self._draftaux_injected += float(aux.sum().item())
            self._draftaux_events += int(b_global.numel())
        if self._draftaux_enabled:
          self._draftaux_prevcoords[env_id_slice.start] = (batch_rows.start, l)
        if self._xgate_mask_enabled:
          lay = self._draftaux_layout
          if lay is None:
            lay = self._draftaux_init_layout(o_device.shape[-1], o_device.device)
          if lay is not None:
            mo, go = lay["mode_off"], lay["gate_ctx_off"]
            mode_now = o_device[:, mo:mo + 4].contiguous().view(torch.int32).flatten()
            gate_now = o_device[:, go:go + 2].contiguous().view(torch.int16).flatten().to(torch.int32)
            gr = torch.as_tensor(env_id_np, device=o_device.device, dtype=torch.long)
            prev_mode = self._xgate_prev_mode[gr]
            prev_gate = self._xgate_prev_gate[gr]
            swapped = (mode_now == 0) & (prev_mode > 0) & (prev_gate > -32768) & (gate_now != prev_gate)
            self._xgate_prev_mode[gr] = mode_now
            self._xgate_prev_gate[gr] = gate_now
            if bool(swapped.any()) and l > 0:
              b = swapped.nonzero(as_tuple=False).flatten()
              seg_rows = batch_rows.start + (gr[b] - env_id_slice.start)
              self.actor_loss_mask[seg_rows, :l] = 0.0
              self._xgate_masked_steps += int(b.numel()) * l
        self.terminals[batch_rows, l] = d_t.float()
        self.values[batch_rows, l] = values_t.float()
        if self._split_value_heads_enabled():
          self.terminal_values[batch_rows, l] = terminal_values_t.float()
          self.shaped_values[batch_rows, l] = shaped_values_t.float()
        self._segment_is_trainable[batch_rows] = torch.as_tensor(trainable_rows_np, device=device, dtype=torch.bool)
        self._stamp_win_prob_rollout_metadata(batch_rows, l, env_id_np)
        if self._draft_episode_credit_enabled:
          if (
            draft_episode_modes_np is None
            or self._draft_episode_phase_mask is None
          ):
            raise RuntimeError("full-episode draft credit row mask is missing")
          self._draft_episode_phase_mask[batch_rows, l] = torch.as_tensor(
            draft_episode_modes_np == 2,
            device=device,
            dtype=torch.bool,
          )

        self.ep_lengths[env_id_slice] += 1
        if l + 1 >= config["bptt_horizon"]:
          num_full = env_id_slice.stop - env_id_slice.start
          self.ep_indices[env_id_slice] = self.free_idx + torch.arange(num_full, device=device).int()
          self.ep_lengths[env_id_slice] = 0
          self.free_idx += num_full
          self.full_rows += num_full

      terminal_rewards = np.asarray(r, dtype=np.float32)
      draft_episode_terminal_rewards = terminal_only_rewards(
        terminal_rewards,
        terminal_mask,
      )
      self._finalize_leader_credit_rows(
        env_id_np,
        done_mask,
        terminal_mask,
        terminal_rewards,
      )
      self._finalize_draft_credit_rows(
        env_id_np,
        done_mask,
        terminal_mask,
        terminal_rewards,
      )
      self._finalize_draft_episode_credit_rows(
        env_id_np,
        done_mask,
        terminal_mask,
        draft_episode_terminal_rewards,
      )
      done_rows = env_id_np[done_mask]
      self._zero_done_states(done_rows)
      finished_envs = self._assign_terminal_win_prob_targets(
        info,
        env_id_np,
        done_mask,
        terminal_rewards=terminal_rewards,
        label_mask=terminal_mask,
      )
      if self._reference_opponent_only and finished_envs.size > 0:
        total_finished += int(finished_envs.size)
        finished_reference = self._env_is_reference[finished_envs]
        reference_finished += int(finished_reference.sum())
        if bool(finished_reference.any()):
          ref_envs = finished_envs[finished_reference]
          aligned = self._env_learner_seat[ref_envs] != self._env_reference_seat[ref_envs]
          reference_finished_aligned += int(aligned.sum())
      if self._pfsp_enabled and finished_envs.size > 0:
        # Record learner result vs the frozen opponent (terminal reward sign
        # on the learner seat; truncations carry 0 and are skipped).
        r_np = np.asarray(r)
        for env_idx in finished_envs:
          if self._env_use_latest[env_idx]:
            continue
          seat = int(self._env_learner_seat[env_idx])
          row = int(env_idx) * self._agents_per_env + seat
          pos = np.nonzero(env_id_np == row)[0]
          if pos.size == 0 or not bool(done_mask[pos[0]]):
            continue
          reward = float(r_np[pos[0]])
          if reward == 0.0:
            continue
          opp = int(self._env_opp_policy[env_idx])
          if 0 <= opp < self._pfsp_games.size:
            # Light decay keeps the estimate current across meta drift.
            self._pfsp_wins[opp] *= 0.995
            self._pfsp_games[opp] *= 0.995
            self._pfsp_wins[opp] += 1.0 if reward > 0 else 0.0
            self._pfsp_games[opp] += 1.0
      self._resample_matchups(finished_envs)
      if self._reference_opponent_only and finished_envs.size > 0:
        self._env_reference_pending[finished_envs] = True

      actions_np = actions_t.cpu().numpy()

      profile("eval_misc", epoch)
      for i in info:
        for k, v in unroll_nested_dict(i):
          if isinstance(v, np.ndarray):
            v = v.tolist()
          elif isinstance(v, (list, tuple)):
            self.stats[k].extend(v)
          else:
            self.stats[k].append(v)

      trainable_fraction = float(np.mean(trainable_rows_np))
      self.stats["league/learner_row_fraction"].append(float(np.mean(learner_rows_np)))
      self.stats["league/trainable_row_fraction"].append(trainable_fraction)
      self.stats["league/frozen_row_fraction"].append(float(1.0 - trainable_fraction))
      self.stats["league/frozen_matchup_fraction"].append(float(1.0 - np.mean(self._env_use_latest[env_indices])))
      self.stats["league/latest_opponent_fraction"].append(float(np.mean(self._env_use_latest[env_indices])))
      active = compute_league_active(
        global_step=int(self.global_step),
        activate_after_steps=int(self.league_cfg.activate_after_steps),
      )
      self.stats["league/active"].append(1.0 if active else 0.0)

      profile("env", epoch)
      self.vecenv.send(actions_np)

    if self._reference_opponent_only:
      self.stats["league/reference_configured_matchup_probability"].append(
        self._reference_matchup_probability
      )
      self.stats["league/reference_target_frozen_matchup_fraction"].append(
        self._target_frozen_matchup_ratio
      )
      self.stats["league/reference_base_frozen_matchup_fraction"].append(
        self._base_frozen_matchup_ratio
      )
      self.stats["league/reference_completed_episodes"].append(float(reference_finished))
      self.stats["league/reference_completed_episode_fraction"].append(
        float(reference_finished / total_finished) if total_finished else 0.0
      )
      self.stats["league/reference_learner_drafter_fraction"].append(
        float(reference_finished_aligned / reference_finished) if reference_finished else 1.0
      )
      self.stats["league/reference_matchup_step_fraction"].append(
        float(reference_env_steps / total_env_steps) if total_env_steps else 0.0
      )
      self.stats["league/reference_trainable_row_fraction"].append(
        float(reference_trainable_rows / total_trainable_rows) if total_trainable_rows else 0.0
      )
      self.stats["league/reference_trainable_battle_row_fraction"].append(
        float(reference_battle_rows / total_battle_rows) if total_battle_rows else 0.0
      )
    if self._prefix_outcome_enabled:
      if self._prefix_outcome_redistributor is None:
        raise RuntimeError("enabled draft prefix outcome redistributor is missing")
      for name, value in self._prefix_outcome_redistributor.window_metrics().items():
        self.stats[f"draft_prefix_outcome/{name}"].append(float(value))
      self.stats["draft_prefix_outcome/inference_seconds"].append(
        float(self._prefix_outcome_inference_seconds)
      )
    if self._leader_credit_enabled:
      self.stats["leader_credit/captured"].append(float(self._leader_credit_captured))
      self.stats["leader_credit/completed"].append(float(self._leader_credit_completed))
      self.stats["leader_credit/pending"].append(float(len(self._leader_credit_pending)))
      self.stats["leader_credit/ready"].append(float(len(self._leader_credit_ready)))
    if self._draft_credit_enabled:
      pending_records = sum(
        len(pending["records"])
        for pending in self._draft_credit_pending.values()
        if isinstance(pending.get("records"), dict)
      )
      self.stats["draft_credit/captured"].append(float(self._draft_credit_captured))
      self.stats["draft_credit/labeled"].append(float(self._draft_credit_labeled))
      self.stats["draft_credit/truncated"].append(float(self._draft_credit_truncated))
      self.stats["draft_credit/incomplete_episodes"].append(
        float(self._draft_credit_incomplete)
      )
      self.stats["draft_credit/incomplete_records"].append(
        float(self._draft_credit_incomplete_records)
      )
      self.stats["draft_credit/pending_episodes"].append(
        float(len(self._draft_credit_pending))
      )
      self.stats["draft_credit/pending_records"].append(float(pending_records))
      self.stats["draft_credit/ready"].append(float(len(self._draft_credit_ready)))
    if self._draft_episode_credit_enabled:
      pending_records = sum(
        int(np.asarray(pending["seen"], dtype=np.bool_).sum())
        for pending in self._draft_episode_credit_pending.values()
      )
      self.stats["draft_episode_credit/captured"].append(
        float(self._draft_episode_credit_captured)
      )
      self.stats["draft_episode_credit/labeled"].append(
        float(self._draft_episode_credit_labeled)
      )
      self.stats["draft_episode_credit/truncated_records"].append(
        float(self._draft_episode_credit_truncated)
      )
      self.stats["draft_episode_credit/draw_episodes"].append(
        float(self._draft_episode_credit_draws)
      )
      self.stats["draft_episode_credit/decisive_episodes"].append(
        float(self._draft_episode_credit_decisive)
      )
      self.stats["draft_episode_credit/win_episodes"].append(
        float(self._draft_episode_credit_wins)
      )
      self.stats["draft_episode_credit/loss_episodes"].append(
        float(self._draft_episode_credit_losses)
      )
      self.stats["draft_episode_credit/incomplete_episodes"].append(
        float(self._draft_episode_credit_incomplete)
      )
      self.stats["draft_episode_credit/pending_episodes"].append(
        float(len(self._draft_episode_credit_pending))
      )
      self.stats["draft_episode_credit/pending_records"].append(
        float(pending_records)
      )
      self.stats["draft_episode_credit/ready_episodes"].append(
        float(len(self._draft_episode_credit_ready))
      )
      self.stats["draft_episode_credit/completed_episodes"].append(
        float(self._draft_episode_credit_completed)
      )
      self.stats["draft_episode_credit/sample_dropped_episodes"].append(
        float(self._draft_episode_credit_sample_dropped)
      )
      self.stats["draft_episode_credit/sample_retention_fraction"].append(
        float(
          len(self._draft_episode_credit_ready)
          / max(self._draft_episode_credit_window_completed, 1)
        )
      )
      self.stats["draft_episode_credit/window_completed_episodes"].append(
        float(self._draft_episode_credit_window_completed)
      )
      self.stats["draft_episode_credit/window_sample_dropped_episodes"].append(
        float(self._draft_episode_credit_window_sample_dropped)
      )
      self.stats["draft_episode_credit/capture_seconds"].append(
        float(self._draft_episode_credit_capture_seconds)
      )
      if self._draft_episode_credit_completed > 0:
        self._draft_episode_credit_zero_label_epochs = 0
        if self._draft_episode_credit_decisive > 0:
          self._draft_episode_credit_no_decisive_epochs = 0
        else:
          self._draft_episode_credit_no_decisive_epochs += 1
          if self._draft_episode_credit_no_decisive_epochs == 1:
            print(
              "[draft-episode-credit] warning: complete retained drafts had "
              "no decisive terminal outcomes"
            )
          if (
            self._draft_episode_credit_no_decisive_epochs
            >= DRAFT_EPISODE_DECISIVE_LABEL_PATIENCE
          ):
            raise RuntimeError(
              "full-episode draft credit stopped because every complete "
              "terminal label was a draw"
            )
      else:
        self._draft_episode_credit_no_decisive_epochs = 0
        self._draft_episode_credit_zero_label_epochs += 1
        if (
          self._draft_episode_credit_zero_label_epochs
          == self._draft_episode_credit_label_warmup_epochs
        ):
          print(
            "[draft-episode-credit] warning: no complete terminal-labeled drafts "
            f"for {self._draft_episode_credit_zero_label_epochs} consecutive epochs"
          )
        if (
          self._draft_episode_credit_zero_label_epochs
          >= 2 * self._draft_episode_credit_label_warmup_epochs
        ):
          raise RuntimeError(
            "full-episode draft credit stopped because complete terminal labels "
            "stayed empty"
          )
    if self._draft_prefix_enabled:
      prefix_lengths = np.asarray(
        self._draft_prefix_episode_lengths,
        dtype=np.int16,
      )
      self.stats["draft_prefix/episodes"].append(float(prefix_lengths.size))
      self.stats["draft_prefix/forced_rows"].append(
        float(self._draft_prefix_forced_rows)
      )
      self.stats["draft_prefix/mean_length"].append(
        float(prefix_lengths.mean()) if prefix_lengths.size else 0.0
      )
      for prefix_length in self._draft_prefix_lengths:
        self.stats[f"draft_prefix/length_{prefix_length}_episodes"].append(
          float((prefix_lengths == prefix_length).sum())
        )

    # Match base PuffeRL buffer lifecycle: reset row indexing state after each
    # evaluate pass so the next epoch starts with fresh contiguous row slots.
    self.free_idx = self.total_agents
    self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
    self.ep_lengths.zero_()
    self._record_effective_reward_shaping_scale()

    profile.end()
    return self.stats

  def train(self):
    profile = self.profile
    epoch = self.epoch
    profile("train", epoch)
    profile("train_misc", epoch, nest=True)
    losses = defaultdict(float)
    config = self.config
    device = config["device"]
    # Ensure learner policy is in training mode before any backward pass.
    self.policy.train()
    win_prob_enabled = self._win_prob_aux_enabled()
    split_value_enabled = self._split_value_heads_enabled()
    win_prob_correct_sum = 0.0
    win_prob_brier_sum = 0.0
    win_prob_pred_sum = 0.0
    win_prob_target_sum = 0.0
    win_prob_example_count = 0
    draft_episode_standard_actor_rows = torch.zeros((), device=device)
    draft_episode_standard_masked_rows = torch.zeros((), device=device)
    terminal_credit_enabled = (
      self._leader_credit_enabled
      or self._draft_credit_enabled
      or self._draft_episode_credit_enabled
    )
    terminal_credit_label_warmup = (
      self._draft_credit_label_warmup_epochs
      if self._draft_credit_enabled
      else (
        self._draft_episode_credit_label_warmup_epochs
        if self._draft_episode_credit_enabled
        else self._leader_credit_label_warmup_epochs
      )
    )
    if win_prob_enabled:
      labeled_rows = int(self.win_prob_target_mask.sum().item())
      losses["win_prob_aux_labeled_rows"] = float(labeled_rows)
      if labeled_rows > 0:
        self._win_prob_zero_label_epochs = 0
      else:
        self._win_prob_zero_label_epochs += 1
        if self._win_prob_zero_label_epochs == terminal_credit_label_warmup:
          print(
            "[win-prob] warning: no terminal labels for "
            f"{self._win_prob_zero_label_epochs} consecutive epochs"
          )
        if (
          terminal_credit_enabled
          and self._win_prob_zero_label_epochs
          >= 2 * terminal_credit_label_warmup
        ):
          raise RuntimeError(
            "terminal credit stopped because win-probability labels stayed empty"
          )

    b0 = config["prio_beta0"]
    a = config["prio_alpha"]
    clip_coef = config["clip_coef"]
    vf_clip = config["vf_clip_coef"]
    anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
    self.ratio[:] = 1

    trainable_idx = torch.nonzero(self._segment_is_trainable, as_tuple=False).flatten()
    if trainable_idx.numel() == 0:
      trainable_idx = torch.arange(self.segments, device=device)
    standard_minibatches = self.total_minibatches
    loss_divisor = self.total_minibatches

    for mb in range(standard_minibatches):
      profile("train_misc", epoch)
      amp_cm = self.amp_context if self.amp_context is not None else contextlib.nullcontext()
      amp_cm.__enter__()

      shape = self.values.shape
      advantages = torch.zeros(shape, device=device)
      advantages = pufferl.compute_puff_advantage(
        self.values,
        self.rewards,
        self.terminals,
        self.ratio,
        advantages,
        config["gamma"],
        config["gae_lambda"],
        config["vtrace_rho_clip"],
        config["vtrace_c_clip"],
      )
      terminal_advantages = None
      shaped_advantages = None
      if split_value_enabled:
        terminal_advantages = torch.zeros(shape, device=device)
        terminal_advantages = pufferl.compute_puff_advantage(
          self.terminal_values,
          self.terminal_reward_components,
          self.terminals,
          self.ratio,
          terminal_advantages,
          config["gamma"],
          config["gae_lambda"],
          config["vtrace_rho_clip"],
          config["vtrace_c_clip"],
        )
        shaped_advantages = torch.zeros(shape, device=device)
        shaped_advantages = pufferl.compute_puff_advantage(
          self.shaped_values,
          self.shaped_reward_components,
          self.terminals,
          self.ratio,
          shaped_advantages,
          config["gamma"],
          config["gae_lambda"],
          config["vtrace_rho_clip"],
          config["vtrace_c_clip"],
        )

      adv = advantages.abs().sum(axis=1)
      prio_weights_all = torch.nan_to_num(adv**a, 0, 0, 0)
      prio_weights = prio_weights_all[trainable_idx]
      if float(prio_weights.sum().item()) <= 0.0:
        prio_weights = torch.ones_like(prio_weights)
      prio_probs = prio_weights / (prio_weights.sum() + 1e-8)
      rel_idx = torch.multinomial(prio_probs, self.minibatch_segments, replacement=True)
      idx = trainable_idx[rel_idx]
      mb_prio = (max(trainable_idx.numel(), 1) * prio_probs[rel_idx, None]) ** -anneal_beta

      profile("train_copy", epoch)
      mb_obs = self.observations[idx]
      mb_actions = self.actions[idx]
      mb_logprobs = self.logprobs[idx]
      mb_values = self.values[idx]
      mb_returns = advantages[idx] + mb_values
      mb_advantages = advantages[idx]
      if split_value_enabled:
        mb_terminal_values = self.terminal_values[idx]
        mb_shaped_values = self.shaped_values[idx]
        mb_terminal_returns = terminal_advantages[idx] + mb_terminal_values
        mb_shaped_returns = shaped_advantages[idx] + mb_shaped_values

      profile("train_forward", epoch)
      if not config["use_rnn"]:
        mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

      actor_mask = torch.ones_like(mb_advantages)
      if self._xgate_mask_enabled:
        actor_mask = actor_mask * self.actor_loss_mask[idx]
      if self._draft_episode_credit_enabled:
        if self._draft_episode_phase_mask is None:
          raise RuntimeError("full-episode draft phase mask is missing")
        draft_phase_rows = self._draft_episode_phase_mask[idx]
        draft_episode_standard_masked_rows += draft_phase_rows.sum()
        actor_mask = actor_mask * torch.logical_not(draft_phase_rows).to(
          dtype=actor_mask.dtype
        )
        advantage_mean, advantage_std, actor_count = masked_tensor_mean_std(
          mb_advantages,
          actor_mask,
        )
        draft_episode_standard_actor_rows += actor_count
        adv_norm = (
          mb_prio
          * (mb_advantages - advantage_mean)
          / (advantage_std + 1e-8)
          * actor_mask
        )
      else:
        adv_norm = mb_prio * (
          mb_advantages - mb_advantages.mean()
        ) / (mb_advantages.std() + 1e-8)
      probe_actor_mask = actor_mask if self._xgate_mask_enabled else None
      self._maybe_probe_control_draft_gradient(
        mb_obs,
        mb_actions,
        mb_logprobs,
        adv_norm,
        probe_actor_mask,
      )

      state = dict(action=mb_actions, lstm_h=None, lstm_c=None)
      logits, newvalue = self.policy(mb_obs, state)
      _, newlogprob, entropy = azk_pytorch.sample_logits(logits, action=mb_actions)

      profile("train_misc", epoch)
      newlogprob = newlogprob.reshape(mb_logprobs.shape)
      logratio = newlogprob - mb_logprobs
      ratio = logratio.exp()
      self.ratio[idx] = ratio.detach()

      with torch.no_grad():
        if self._draft_episode_credit_enabled:
          old_approx_kl = masked_tensor_mean(-logratio, actor_mask)
          approx_kl = masked_tensor_mean((ratio - 1) - logratio, actor_mask)
          clipfrac = masked_tensor_mean(
            ((ratio - 1.0).abs() > config["clip_coef"]).float(),
            actor_mask,
          )
        else:
          old_approx_kl = (-logratio).mean()
          approx_kl = ((ratio - 1) - logratio).mean()
          clipfrac = ((ratio - 1.0).abs() > config["clip_coef"]).float().mean()

      pg_loss1 = -adv_norm * ratio
      pg_loss2 = -adv_norm * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
      pg_loss_elem = torch.max(pg_loss1, pg_loss2)
      pg_loss = masked_tensor_mean(pg_loss_elem, actor_mask)

      newvalue = newvalue.view(mb_returns.shape)
      total_v_loss = self._clipped_value_loss(newvalue, mb_values, mb_returns, vf_clip)
      component_v_loss = torch.zeros((), device=device)
      terminal_v_loss = torch.zeros((), device=device)
      shaped_v_loss = torch.zeros((), device=device)
      if split_value_enabled:
        new_terminal_value, new_shaped_value = self._component_values_from_state(
          state,
          mb_returns.shape,
        )
        terminal_v_loss = self._clipped_value_loss(
          new_terminal_value,
          mb_terminal_values,
          mb_terminal_returns,
          vf_clip,
        )
        shaped_v_loss = self._clipped_value_loss(
          new_shaped_value,
          mb_shaped_values,
          mb_shaped_returns,
          vf_clip,
        )
        component_v_loss = 0.5 * (terminal_v_loss + shaped_v_loss)
      entropy_loss = masked_tensor_mean(
        entropy.reshape_as(actor_mask),
        actor_mask,
      )
      win_prob_aux_loss, win_prob_aux_metrics = self._compute_win_prob_aux(state, idx)
      value_loss_for_optim = total_v_loss + self._split_value_component_coef() * component_v_loss
      loss = pg_loss + config["vf_coef"] * value_loss_for_optim - config["ent_coef"] * entropy_loss + win_prob_aux_loss

      self.values[idx] = newvalue.detach().float()
      if split_value_enabled:
        self.terminal_values[idx] = new_terminal_value.detach().float()
        self.shaped_values[idx] = new_shaped_value.detach().float()

      profile("train_misc", epoch)
      losses["policy_loss"] += pg_loss.item() / loss_divisor
      losses["value_loss"] += value_loss_for_optim.item() / loss_divisor
      losses["value_loss_total"] += total_v_loss.item() / loss_divisor
      if split_value_enabled:
        losses["value_loss_terminal"] += terminal_v_loss.item() / loss_divisor
        losses["value_loss_shaped"] += shaped_v_loss.item() / loss_divisor
      losses["entropy"] += entropy_loss.item() / loss_divisor
      losses["old_approx_kl"] += old_approx_kl.item() / loss_divisor
      losses["approx_kl"] += approx_kl.item() / loss_divisor
      losses["clipfrac"] += clipfrac.item() / loss_divisor
      importance = (
        masked_tensor_mean(ratio, actor_mask)
        if self._draft_episode_credit_enabled
        else ratio.mean()
      )
      losses["importance"] += importance.item() / loss_divisor
      if win_prob_enabled:
        losses["win_prob_aux_loss"] += win_prob_aux_metrics["raw_loss"] / loss_divisor
        losses["win_prob_aux_labeled_frac"] += win_prob_aux_metrics["labeled_frac"] / loss_divisor
        win_prob_correct_sum += win_prob_aux_metrics["correct_sum"]
        win_prob_brier_sum += win_prob_aux_metrics["brier_sum"]
        win_prob_pred_sum += win_prob_aux_metrics["pred_sum"]
        win_prob_target_sum += win_prob_aux_metrics["target_sum"]
        win_prob_example_count += win_prob_aux_metrics["example_count"]

      profile("learn", epoch)
      loss.backward()
      if (mb + 1) % self.accumulate_minibatches == 0:
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config["max_grad_norm"])
        self.optimizer.step()
        self.optimizer.zero_grad()

      amp_cm.__exit__(None, None, None)

    profile("train_misc", epoch)
    leader_credit_metrics = self._train_leader_terminal_credit()
    for metric_name, metric_value in leader_credit_metrics.items():
      losses[metric_name] = metric_value
    draft_credit_metrics = self._train_draft_terminal_credit()
    for metric_name, metric_value in draft_credit_metrics.items():
      losses[metric_name] = metric_value
    draft_episode_credit_metrics = self._train_draft_episode_credit()
    for metric_name, metric_value in draft_episode_credit_metrics.items():
      losses[metric_name] = metric_value
    if self._draft_episode_credit_enabled:
      losses["draft_episode_credit_standard_actor_rows"] = float(
        draft_episode_standard_actor_rows
      )
      losses["draft_episode_credit_standard_masked_rows"] = float(
        draft_episode_standard_masked_rows
      )
    if config["anneal_lr"]:
      self.scheduler.step()

    eval_mask = self._segment_is_trainable
    if not bool(eval_mask.any().item()):
      eval_mask = torch.ones_like(eval_mask, dtype=torch.bool)
    y_pred = self.values[eval_mask].flatten()
    y_true = (advantages[eval_mask] + self.values[eval_mask]).flatten()
    var_y = y_true.var()
    explained_var = torch.nan if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
    losses["explained_variance"] = explained_var
    if split_value_enabled:
      terminal_y_pred = self.terminal_values[eval_mask].flatten()
      terminal_y_true = (terminal_advantages[eval_mask] + self.terminal_values[eval_mask]).flatten()
      terminal_var_y = terminal_y_true.var()
      losses["explained_variance_terminal"] = (
        torch.nan
        if terminal_var_y == 0
        else (1 - (terminal_y_true - terminal_y_pred).var() / terminal_var_y).item()
      )
      shaped_y_pred = self.shaped_values[eval_mask].flatten()
      shaped_y_true = (shaped_advantages[eval_mask] + self.shaped_values[eval_mask]).flatten()
      shaped_var_y = shaped_y_true.var()
      losses["explained_variance_shaped"] = (
        torch.nan
        if shaped_var_y == 0
        else (1 - (shaped_y_true - shaped_y_pred).var() / shaped_var_y).item()
      )
    if win_prob_enabled:
      if win_prob_example_count > 0:
        losses["win_prob_aux_accuracy"] = win_prob_correct_sum / win_prob_example_count
        losses["win_prob_aux_brier"] = win_prob_brier_sum / win_prob_example_count
        losses["win_prob_aux_pred_mean"] = win_prob_pred_sum / win_prob_example_count
        losses["win_prob_aux_target_mean"] = win_prob_target_sum / win_prob_example_count
      else:
        losses["win_prob_aux_accuracy"] = 0.0
        losses["win_prob_aux_brier"] = 0.0
        losses["win_prob_aux_pred_mean"] = 0.0
        losses["win_prob_aux_target_mean"] = 0.0

    profile.end()
    logs = None
    self.epoch += 1
    done_training = self.global_step >= config["total_timesteps"]
    if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
      self.losses = losses
      logs = self.mean_and_log()
      self.print_dashboard()
      self.stats = defaultdict(list)
      self.last_log_time = time.time()
      self.last_log_step = self.global_step
      profile.clear()

    if self.epoch % config["checkpoint_interval"] == 0 or done_training:
      self.save_checkpoint()
      self.msg = f"Checkpoint saved at update {self.epoch}"

    return logs
