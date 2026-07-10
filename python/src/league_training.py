from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass

import azk_puffer.pytorch as azk_pytorch
import azk_puffer.trainer as pufferl
from azk_puffer.core import unroll_nested_dict
import numpy as np
from policy.tcg_distribution import TCGActionDistribution, TCGLegalActionDistribution
import torch


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


def compute_league_active(*, global_step: int, activate_after_steps: int) -> bool:
  threshold = int(max(0, activate_after_steps))
  return int(global_step) >= threshold


class LeaguePuffeRL(pufferl.PuffeRL):
  """PuffeRL variant with dual-policy league rollouts and latest-policy updates."""

  def __init__(self, config, vecenv, policy, opponent_policies, league_cfg: LeagueConfig, logger=None):
    super().__init__(config, vecenv, policy, logger=logger)
    self.league_cfg = league_cfg
    self.opponent_policies = list(opponent_policies)
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
    if window_epochs <= 0 or pool_size == 0:
      self._window_policy_ids = None
      return
    window_index = int(getattr(self, "epoch", 0)) // window_epochs
    if (
      window_index == self._window_index
      and self._window_policy_ids is not None
      and self._window_policy_ids.size > 0
      and int(self._window_policy_ids.max()) < pool_size
    ):
      return
    self._window_index = window_index
    k = max(1, int(getattr(self.league_cfg, "max_distinct_frozen", 1) or 1))
    k = min(k, pool_size)
    self._window_policy_ids = self._rng.choice(pool_size, size=k, replace=False).astype(np.int32)

  def _resample_matchups(self, env_indices: np.ndarray) -> None:
    if env_indices.size == 0:
      return

    if self.league_cfg.randomize_learner_seat:
      self._env_learner_seat[env_indices] = self._rng.integers(
        0, self._agents_per_env, size=env_indices.size, dtype=np.int32
      )

    if len(self.opponent_policies) == 0:
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
      if self.league_cfg.frozen_ratio is not None:
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
      self._env_opp_policy[env_indices] = self._rng.integers(
        0, len(self.opponent_policies), size=env_indices.size, dtype=np.int32
      )

  def set_opponent_policies(self, opponent_policies: list[torch.nn.Module]) -> None:
    self.opponent_policies = list(opponent_policies)
    for opp in self.opponent_policies:
      opp.eval()
      for param in opp.parameters():
        param.requires_grad_(False)
    self._opp_lstm_h = []
    self._opp_lstm_c = []
    if self._use_rnn:
      hidden_size = int(self.policy.hidden_size)
      device = self.config["device"]
      self._opp_lstm_h = [
        torch.zeros(self.total_agents, hidden_size, device=device) for _ in self.opponent_policies
      ]
      self._opp_lstm_c = [
        torch.zeros(self.total_agents, hidden_size, device=device) for _ in self.opponent_policies
      ]
    # Pool indices shift on refresh; force a window redraw against the new pool.
    self._window_index = -1
    self._window_policy_ids = None
    self._refresh_frozen_window()
    self._resample_matchups(np.arange(self._num_envs_total, dtype=np.int32))

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
    self._draftaux_pending = (b_global, aux)

  def _infer_actions(self, o_device: torch.Tensor, mask_t: torch.Tensor, env_id_np: np.ndarray):
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
      if self._use_rnn:
        learner_state["lstm_h"] = self._learner_lstm_h[learner_idx_t]
        learner_state["lstm_c"] = self._learner_lstm_c[learner_idx_t]
      logits, values = self._safe_forward_eval(self.policy, o_device[learner_idx_t], learner_state)
      if self._draftaux_enabled:
        # Pre-write-back: self._learner_lstm_h still holds the states the
        # boundary forward consumed (dict entries were replaced, not storage).
        self._draftaux_league_stash(
          o_device, mask_t, learner_idx_t, values, env_id_np
        )
      with torch.no_grad(), self.amp_context:
        actions, logprobs, _ = azk_pytorch.sample_logits(logits)
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
        if self._use_rnn:
          latest_state["lstm_h"] = self._learner_lstm_h[latest_idx_t]
          latest_state["lstm_c"] = self._learner_lstm_c[latest_idx_t]
        latest_logits, latest_values = self._safe_forward_eval(self.policy, o_device[latest_idx_t], latest_state)
        with torch.no_grad(), self.amp_context:
          latest_actions, latest_logprobs, _ = azk_pytorch.sample_logits(latest_logits)
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
      if obs.shape[0] > 1:
        return model.forward_eval(obs, state)

      obs_pad = torch.cat([obs, obs], dim=0)
      state_pad = {"mask": torch.cat([state["mask"], state["mask"]], dim=0)}
      if self._use_rnn:
        state_pad["lstm_h"] = torch.cat([state["lstm_h"], state["lstm_h"]], dim=0)
        state_pad["lstm_c"] = torch.cat([state["lstm_c"], state["lstm_c"]], dim=0)
      logits, values = model.forward_eval(obs_pad, state_pad)
      if self._use_rnn:
        state["lstm_h"] = state_pad["lstm_h"][:1]
        state["lstm_c"] = state_pad["lstm_c"][:1]
      for key in ("_azk_win_prob_logits", "_azk_value_terminal", "_azk_value_shaped"):
        tensor = state_pad.get(key)
        if torch.is_tensor(tensor):
          state[key] = tensor[:1]

      if isinstance(logits, TCGActionDistribution):
        logits = type(logits)(
          primary_logits=logits.primary_logits[:1],
          primary_action_mask=logits.primary_action_mask[:1],
          legal_actions=logits.legal_actions[:1],
          legal_action_count=logits.legal_action_count[:1],
          target_matrix=logits.target_matrix[:1],
          unit1_projection=logits.unit1_projection[:1],
          unit2_projection=logits.unit2_projection[:1],
          bins2_logits=logits.bins2_logits[:1],
          bins3_logits=logits.bins3_logits[:1],
          gate1_table=logits.gate1_table,
          gate2_table=logits.gate2_table,
        )
      elif isinstance(logits, TCGLegalActionDistribution):
        logits = type(logits)(
          legal_action_logits=logits.legal_action_logits[:1],
          legal_actions=logits.legal_actions[:1],
          legal_action_count=logits.legal_action_count[:1],
        )
      return logits, values[:1]

  def evaluate(self):
    profile = self.profile
    epoch = self.epoch
    profile("eval", epoch)
    profile("eval_misc", epoch, nest=True)
    self._refresh_frozen_window()

    config = self.config
    device = config["device"]

    self.full_rows = 0
    self._segment_is_trainable.zero_()
    self._reset_win_prob_rollout_buffers()
    self._reset_split_value_rollout_buffers()
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

      profile("eval_copy", epoch)
      o = torch.as_tensor(o)
      o_device = o.to(device)
      r_t = torch.as_tensor(r).to(device)
      d_t = torch.as_tensor(d).to(device)
      mask_t = torch.as_tensor(mask, device=device, dtype=torch.bool)
      env_id_np = np.asarray(env_id, dtype=np.int64)
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
        o_device, mask_t, env_id_np
      )

      trainable_step_mask = np.logical_and(mask.astype(np.bool_), trainable_rows_np)
      self.global_step += int(trainable_step_mask.sum())

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
        self.rewards[batch_rows, l] = torch.clamp(r_t, -1, 1)
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
        self.terminal_reward_components[batch_rows, l] = reward_components_terminal
        self.shaped_reward_components[batch_rows, l] = reward_components_shaped
        self.terminals[batch_rows, l] = d_t.float()
        self.values[batch_rows, l] = values_t.float()
        if self._split_value_heads_enabled():
          self.terminal_values[batch_rows, l] = terminal_values_t.float()
          self.shaped_values[batch_rows, l] = shaped_values_t.float()
        self._segment_is_trainable[batch_rows] = torch.as_tensor(trainable_rows_np, device=device, dtype=torch.bool)
        self._stamp_win_prob_rollout_metadata(batch_rows, l, env_id_np)

        self.ep_lengths[env_id_slice] += 1
        if l + 1 >= config["bptt_horizon"]:
          num_full = env_id_slice.stop - env_id_slice.start
          self.ep_indices[env_id_slice] = self.free_idx + torch.arange(num_full, device=device).int()
          self.ep_lengths[env_id_slice] = 0
          self.free_idx += num_full
          self.full_rows += num_full

      done_rows = env_id_np[done_mask]
      self._zero_done_states(done_rows)
      finished_envs = self._assign_terminal_win_prob_targets(info, env_id_np, done_mask)
      self._resample_matchups(finished_envs)

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

    # Match base PuffeRL buffer lifecycle: reset row indexing state after each
    # evaluate pass so the next epoch starts with fresh contiguous row slots.
    self.free_idx = self.total_agents
    self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
    self.ep_lengths.zero_()

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

    b0 = config["prio_beta0"]
    a = config["prio_alpha"]
    clip_coef = config["clip_coef"]
    vf_clip = config["vf_clip_coef"]
    anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
    self.ratio[:] = 1

    trainable_idx = torch.nonzero(self._segment_is_trainable, as_tuple=False).flatten()
    if trainable_idx.numel() == 0:
      trainable_idx = torch.arange(self.segments, device=device)

    for mb in range(self.total_minibatches):
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

      state = dict(action=mb_actions, lstm_h=None, lstm_c=None)
      logits, newvalue = self.policy(mb_obs, state)
      _, newlogprob, entropy = azk_pytorch.sample_logits(logits, action=mb_actions)

      profile("train_misc", epoch)
      newlogprob = newlogprob.reshape(mb_logprobs.shape)
      logratio = newlogprob - mb_logprobs
      ratio = logratio.exp()
      self.ratio[idx] = ratio.detach()

      with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > config["clip_coef"]).float().mean()

      adv_norm = mb_prio * (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
      pg_loss1 = -adv_norm * ratio
      pg_loss2 = -adv_norm * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
      pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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
      entropy_loss = entropy.mean()
      win_prob_aux_loss, win_prob_aux_metrics = self._compute_win_prob_aux(state, idx)
      value_loss_for_optim = total_v_loss + self._split_value_component_coef() * component_v_loss
      loss = pg_loss + config["vf_coef"] * value_loss_for_optim - config["ent_coef"] * entropy_loss + win_prob_aux_loss

      self.values[idx] = newvalue.detach().float()
      if split_value_enabled:
        self.terminal_values[idx] = new_terminal_value.detach().float()
        self.shaped_values[idx] = new_shaped_value.detach().float()

      profile("train_misc", epoch)
      losses["policy_loss"] += pg_loss.item() / self.total_minibatches
      losses["value_loss"] += value_loss_for_optim.item() / self.total_minibatches
      losses["value_loss_total"] += total_v_loss.item() / self.total_minibatches
      if split_value_enabled:
        losses["value_loss_terminal"] += terminal_v_loss.item() / self.total_minibatches
        losses["value_loss_shaped"] += shaped_v_loss.item() / self.total_minibatches
      losses["entropy"] += entropy_loss.item() / self.total_minibatches
      losses["old_approx_kl"] += old_approx_kl.item() / self.total_minibatches
      losses["approx_kl"] += approx_kl.item() / self.total_minibatches
      losses["clipfrac"] += clipfrac.item() / self.total_minibatches
      losses["importance"] += ratio.mean().item() / self.total_minibatches
      if win_prob_enabled:
        losses["win_prob_aux_loss"] += win_prob_aux_metrics["raw_loss"] / self.total_minibatches
        losses["win_prob_aux_labeled_frac"] += win_prob_aux_metrics["labeled_frac"] / self.total_minibatches
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
      logs = self.mean_and_log()
      self.losses = losses
      self.print_dashboard()
      self.stats = defaultdict(list)
      self.last_log_time = time.time()
      self.last_log_step = self.global_step
      profile.clear()

    if self.epoch % config["checkpoint_interval"] == 0 or done_training:
      self.save_checkpoint()
      self.msg = f"Checkpoint saved at update {self.epoch}"

    return logs
