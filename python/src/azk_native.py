"""Native PufferEnv path for Azuki TCG.

The C engine (via binding.vec_init/vec_step) writes packed
TrainingObservationData structs directly into pufferlib's shared-memory
observation rows. No PettingZoo wrappers, no dict observations, and no
per-step Python beyond a single vec_step call plus an episode-end log fetch.

Layout equivalence that makes this zero-copy: pufferlib allocates
observations as (num_agents, STRUCT_SIZE) uint8 rows with num_agents =
2 * num_envs. Viewed as (num_envs, 2 * STRUCT_SIZE), each row is exactly the
contiguous pair of per-player structs that the C side writes for one env.

Terminal and shaped rewards are separate shared-memory channels. Combined
rewards are their unclipped sum, including resolved ability outcomes and any
potential closure on terminal steps. Win labels use only the terminal channel;
truncations are bootstrap boundaries, not losses.
"""

from __future__ import annotations

import numpy as np
import gymnasium

import binding
from azk_puffer import PufferEnv
from action import build_action_space
from observation import (
  MAX_PLAYERS_PER_MATCH,
  OBSERVATION_CTYPE,
  OBSERVATION_STRUCT_SIZE,
  DECKBUILD_OBSERVATION_CTYPE,
  DECKBUILD_OBSERVATION_STRUCT_SIZE,
)

# Exact numpy mirror of the packed C observation struct (field names/offsets
# straight from ctypes). The policy builds its decode offsets from this.
NATIVE_OBS_DTYPE = np.dtype(OBSERVATION_CTYPE)
assert NATIVE_OBS_DTYPE.itemsize == OBSERVATION_STRUCT_SIZE
NATIVE_DECKBUILD_OBS_DTYPE = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
assert NATIVE_DECKBUILD_OBS_DTYPE.itemsize == DECKBUILD_OBSERVATION_STRUCT_SIZE

# Cross-check the ctypes mirrors against the compiled C structs at import so
# layout drift fails loudly instead of decoding garbage.
_C_BATTLE_SIZE, _C_DECKBUILD_SIZE = binding.obs_struct_sizes()
assert _C_BATTLE_SIZE == OBSERVATION_STRUCT_SIZE, (
  f"C battle obs struct {_C_BATTLE_SIZE}B != ctypes {OBSERVATION_STRUCT_SIZE}B"
)
assert _C_DECKBUILD_SIZE == DECKBUILD_OBSERVATION_STRUCT_SIZE, (
  f"C deckbuild obs struct {_C_DECKBUILD_SIZE}B != ctypes "
  f"{DECKBUILD_OBSERVATION_STRUCT_SIZE}B"
)


class AzukiNativeEnv(PufferEnv):
  """One process-local vector of C envs behind the PufferEnv interface."""

  def __init__(
    self,
    num_envs: int = 1,
    deck_pool=None,
    buf=None,
    seed: int = 0,
    deck_building: bool = False,
    deck_snapshot_dir=None,
    deck_snapshot_every=None,
    draft_same_element_matchup_prob=None,
    draft_cross_gate_replay_prob=None,
    deck_building_privileged_decks: bool = False,
    draft_uniform_assignment: bool = False,
    evaluation_mode: bool = False,
    reward_telemetry: bool = False,
    reward_decomposed_schedule: bool = False,
    pbrs_mode: str = "legacy",
    pbrs_gamma: float = 0.99,
    pbrs_terminal_closure: bool | None = None,
    prebuilt_deck_groups: tuple[tuple[int, ...], ...] | None = None,
    prebuilt_probability: float = 0.0,
  ) -> None:
    num_envs = int(num_envs)
    if num_envs < 1:
      raise ValueError("num_envs must be >= 1")
    self._deck_building = bool(deck_building)
    struct_size = (
      DECKBUILD_OBSERVATION_STRUCT_SIZE if self._deck_building else OBSERVATION_STRUCT_SIZE
    )
    self.single_observation_space = gymnasium.spaces.Box(
      low=0, high=255, shape=(struct_size,), dtype=np.uint8
    )
    self.single_action_space = build_action_space()
    self.num_agents = MAX_PLAYERS_PER_MATCH * num_envs
    self._num_envs = num_envs
    if self._deck_building:
      # Shadow the class-level battle dict for this instance.
      self.emulated = {
        "observation_dtype": np.dtype(np.uint8),
        "emulated_observation_dtype": NATIVE_DECKBUILD_OBS_DTYPE,
        "native_layout": True,
      }
    super().__init__(buf)
    self.reward_decomposed_schedule = bool(reward_decomposed_schedule)
    if self.reward_decomposed_schedule and buf is not None:
      reward_scales = buf.get("reward_scales")
      if not isinstance(reward_scales, np.ndarray):
        raise ValueError(
          "reward_decomposed_schedule requires a reward_scales buffer"
        )
      if reward_scales.shape != (2,) or reward_scales.dtype != np.float32:
        raise ValueError("reward_scales must be a float32 vector of length 2")
      self._reward_scales = reward_scales
    else:
      self._reward_scales = np.ones(2, dtype=np.float32)
    self.prebuilt_curriculum = prebuilt_deck_groups is not None
    self.initial_prebuilt_probability = float(prebuilt_probability)
    if not np.isfinite(self.initial_prebuilt_probability) or not 0.0 <= self.initial_prebuilt_probability <= 1.0:
      raise ValueError("prebuilt_probability must be finite and in [0, 1]")
    if self.prebuilt_curriculum and not self._deck_building:
      raise ValueError("prebuilt decks require deck_building=True")
    if not self.prebuilt_curriculum and self.initial_prebuilt_probability != 0.0:
      raise ValueError("prebuilt_probability requires prebuilt_deck_groups")
    self._prebuilt_deck_groups = prebuilt_deck_groups
    probability_buffer = buf.get("prebuilt_probability") if buf is not None else None
    if probability_buffer is None:
      probability_buffer = np.full(1, self.initial_prebuilt_probability, dtype=np.float32)
    if (
      not isinstance(probability_buffer, np.ndarray)
      or probability_buffer.shape != (1,)
      or probability_buffer.dtype != np.float32
      or not probability_buffer.flags["C_CONTIGUOUS"]
    ):
      raise ValueError("prebuilt_probability buffer must be a contiguous float32 vector of length 1")
    self.prebuilt_probability = probability_buffer

    for name in ("observations", "actions", "rewards", "terminals", "truncations"):
      if not getattr(self, name).flags["C_CONTIGUOUS"]:
        raise ValueError(f"Native path requires contiguous {name} buffer")

    # Per-env views over the per-agent puffer buffers (see module docstring).
    self._c_obs = self.observations.reshape(
      num_envs, MAX_PLAYERS_PER_MATCH * struct_size
    )
    self._c_actions = self.actions.reshape(num_envs, MAX_PLAYERS_PER_MATCH, -1)
    self._c_rewards = self.rewards.reshape(num_envs, MAX_PLAYERS_PER_MATCH)
    self._c_terminals = self.terminals.reshape(num_envs, MAX_PLAYERS_PER_MATCH)
    self._c_truncations = self.truncations.reshape(num_envs, MAX_PLAYERS_PER_MATCH)
    # Preserve both channels across vector workers: terminal rows can contain
    # an outcome and a potential settlement at the same time.
    for name in ("terminal_rewards", "shaped_rewards"):
      component = buf.get(name) if buf is not None else None
      if component is None:
        component = np.zeros(self.num_agents, dtype=np.float32)
      if (
        not isinstance(component, np.ndarray)
        or component.shape != (self.num_agents,)
        or component.dtype != np.float32
        or not component.flags["C_CONTIGUOUS"]
      ):
        raise ValueError(f"Native {name} must be a contiguous per-agent float32 buffer")
      setattr(self, name, component)
    self._terminal_rewards = self.terminal_rewards.reshape(num_envs, MAX_PLAYERS_PER_MATCH)
    self._shaped_rewards = self.shaped_rewards.reshape(num_envs, MAX_PLAYERS_PER_MATCH)

    self._deck_pool = deck_pool
    self._seed = int(seed or 0)
    self._draft_same_element_matchup_prob = float(draft_same_element_matchup_prob or 0.0)
    if not 0.0 <= self._draft_same_element_matchup_prob <= 1.0:
      raise ValueError("draft_same_element_matchup_prob must be in [0, 1]")
    self._draft_cross_gate_replay_prob = float(draft_cross_gate_replay_prob or 0.0)
    if not 0.0 <= self._draft_cross_gate_replay_prob <= 1.0:
      raise ValueError("draft_cross_gate_replay_prob must be in [0, 1]")
    self._deck_building_privileged_decks = bool(deck_building_privileged_decks)
    self._draft_uniform_assignment = bool(draft_uniform_assignment)
    self._evaluation_mode = bool(evaluation_mode)
    self._reward_telemetry = bool(reward_telemetry)
    if self._reward_telemetry and not self._deck_building:
      raise ValueError("reward_telemetry requires deck_building=True")
    self._pbrs_mode = str(pbrs_mode)
    if self._pbrs_mode not in {"legacy", "discounted"}:
      raise ValueError("pbrs_mode must be 'legacy' or 'discounted'")
    self._pbrs_gamma = float(pbrs_gamma)
    if not 0.0 <= self._pbrs_gamma <= 1.0:
      raise ValueError("pbrs_gamma must be in [0, 1]")
    self._pbrs_terminal_closure = (
      self._pbrs_mode == "discounted"
      if pbrs_terminal_closure is None
      else bool(pbrs_terminal_closure)
    )
    if self._pbrs_mode == "discounted" and not self._pbrs_terminal_closure:
      raise ValueError("discounted PBRS requires terminal closure")
    if self._pbrs_terminal_closure and self._pbrs_mode != "discounted":
      raise ValueError("pbrs_terminal_closure requires pbrs_mode='discounted'")
    if self._evaluation_mode and not self._deck_building:
      raise ValueError("evaluation_mode requires deck_building=True")
    self._pending_evaluation_records: list[dict] = []
    self._handle = None
    self._deckbuild_helper = None
    if self._deck_building:
      from deckbuild_metrics import NativeDeckbuildHelper

      self._deckbuild_helper = NativeDeckbuildHelper(
        deck_pool=deck_pool,
        snapshot_dir=deck_snapshot_dir,
        snapshot_every=deck_snapshot_every,
        uniform_assignment=self._draft_uniform_assignment,
      )

  # PufferEnv defines `emulated` as a property returning False; shadow it so
  # the policy can build its decode table from the packed struct dtype.
  # (Instance attribute overrides this for deck-building mode.)
  emulated = {
    "observation_dtype": np.dtype(np.uint8),
    "emulated_observation_dtype": NATIVE_OBS_DTYPE,
    "native_layout": True,
  }

  # Rows are [game0_p0, game0_p1, game1_p0, ...]; trainers group rows by game
  # via this, independent of how many games one instance packs.
  agents_per_match = MAX_PLAYERS_PER_MATCH

  def _ensure_handle(self) -> None:
    if self._handle is not None:
      return
    kwargs = {}
    if self._deck_pool is not None:
      kwargs["deck_pool"] = self._deck_pool
    if self._deck_building:
      kwargs["deck_building"] = 1
      kwargs.update(self._deckbuild_helper.catalog_kwargs())
      if self._draft_same_element_matchup_prob > 0.0:
        kwargs["draft_same_element_matchup_prob"] = self._draft_same_element_matchup_prob
      if self._draft_cross_gate_replay_prob > 0.0:
        kwargs["draft_cross_gate_replay_prob"] = self._draft_cross_gate_replay_prob
      if self._deck_building_privileged_decks:
        kwargs["deck_building_privileged_decks"] = 1
      if self._draft_uniform_assignment:
        kwargs["draft_uniform_assignment"] = 1
      if self._evaluation_mode:
        kwargs["evaluation_pause_on_done"] = 1
      if self._reward_telemetry:
        kwargs["reward_telemetry"] = 1
    kwargs["pbrs_mode"] = self._pbrs_mode
    kwargs["pbrs_gamma"] = self._pbrs_gamma
    if self._pbrs_terminal_closure:
      kwargs["pbrs_terminal_closure"] = 1
    if self.reward_decomposed_schedule:
      kwargs["reward_scales"] = self._reward_scales
    if self.prebuilt_curriculum:
      kwargs["prebuilt_deck_groups"] = self._prebuilt_deck_groups
      kwargs["prebuilt_probability"] = self.prebuilt_probability
    self._handle = binding.vec_init(
      self._c_obs,
      self._c_actions,
      self._c_rewards,
      self._terminal_rewards,
      self._shaped_rewards,
      self._c_terminals,
      self._c_truncations,
      self._num_envs,
      self._seed,
      **kwargs,
    )

  def reset(self, seed=None):
    self._ensure_handle()
    binding.vec_reset(self._handle, int(seed if seed is not None else self._seed))
    return self.observations, []

  def step(self, actions=None):
    # Actions were already written into the shared buffer by the vec layer.
    binding.vec_step(self._handle)
    infos = []
    if self.terminals.any() or self.truncations.any():
      log = binding.vec_log(self._handle)
      if self._deckbuild_helper is not None:
        records = binding.vec_drain_deck_records(self._handle)
        if records:
          if self._evaluation_mode:
            self._pending_evaluation_records.extend(records)
          else:
            metrics = self._deckbuild_helper.process_records(records)
            if metrics:
              if not log:
                log = {}
              log.update(metrics)
      if log:
        infos.append(log)
    return (
      self.observations,
      self.rewards,
      self.terminals,
      self.truncations,
      infos,
    )

  def notify(self):
    pass

  def draft_snapshot(self, player_index: int, env_index: int = 0) -> dict:
    """Return an owned copy of a completed drafted deck."""
    if not self._deck_building:
      raise RuntimeError("draft_snapshot requires deck_building=True")
    if not 0 <= int(env_index) < self._num_envs:
      raise IndexError(f"env_index {env_index} is out of range")
    if not 0 <= int(player_index) < MAX_PLAYERS_PER_MATCH:
      raise IndexError(f"player_index {player_index} is out of range")
    self._ensure_handle()
    snapshot = binding.vec_draft_snapshot(
      self._handle, int(env_index), int(player_index)
    )
    if not isinstance(snapshot, dict):
      raise RuntimeError("Native draft snapshot returned a non-object result")
    if set(snapshot) != {"gate", "leader", "main_count", "main"}:
      raise RuntimeError("Native draft snapshot returned invalid fields")
    gate = snapshot["gate"]
    leader = snapshot["leader"]
    main_count = snapshot["main_count"]
    main = snapshot["main"]
    if (
      not isinstance(gate, int)
      or isinstance(gate, bool)
      or gate < 0
      or not isinstance(leader, int)
      or isinstance(leader, bool)
      or leader < 0
      or not isinstance(main_count, int)
      or isinstance(main_count, bool)
      or main_count != 50
      or not isinstance(main, list)
      or len(main) != 50
      or any(
        not isinstance(card_id, int) or isinstance(card_id, bool) or card_id < 0
        for card_id in main
      )
    ):
      raise RuntimeError("Native draft snapshot returned invalid deck data")
    return {
      "gate": gate,
      "leader": leader,
      "main_count": main_count,
      "main": list(main),
    }

  def reset_evaluation_games(self, games: list[dict]) -> None:
    if not self._evaluation_mode:
      raise RuntimeError("reset_evaluation_games requires evaluation_mode=True")
    self._ensure_handle()
    binding.vec_reset_evaluation_games(
      self._handle,
      [int(game["env_index"]) for game in games],
      [int(game["seed"]) for game in games],
      [int(game.get("gate0", -1)) for game in games],
      [int(game.get("gate1", -1)) for game in games],
      [int(game.get("leader0", -1)) for game in games],
      [int(game.get("leader1", -1)) for game in games],
      [int(game.get("reference_seat", -1)) for game in games],
      [int(game.get("reference_deck_index", -1)) for game in games],
    )

  def active_players(self) -> np.ndarray:
    if not self._evaluation_mode:
      raise RuntimeError("active_players requires evaluation_mode=True")
    self._ensure_handle()
    return np.asarray(binding.vec_active_players(self._handle), dtype=np.int8)

  def force_evaluation_truncations(self, env_indices: list[int]) -> None:
    if not self._evaluation_mode:
      raise RuntimeError("force_evaluation_truncations requires evaluation_mode=True")
    self._ensure_handle()
    binding.vec_force_evaluation_truncations(
      self._handle, [int(index) for index in env_indices]
    )
    records = binding.vec_drain_deck_records(self._handle)
    if records:
      self._pending_evaluation_records.extend(records)

  def drain_evaluation_records(self) -> list[dict]:
    if not self._evaluation_mode:
      raise RuntimeError("drain_evaluation_records requires evaluation_mode=True")
    self._ensure_handle()
    records = binding.vec_drain_deck_records(self._handle)
    if records:
      self._pending_evaluation_records.extend(records)
    pending = self._pending_evaluation_records
    self._pending_evaluation_records = []
    return pending

  def close(self):
    if self._handle is not None:
      binding.vec_close(self._handle)
      self._handle = None
