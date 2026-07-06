"""Native PufferEnv path for Azuki TCG.

The C engine (via binding.vec_init/vec_step) writes packed
TrainingObservationData structs directly into pufferlib's shared-memory
observation rows. No PettingZoo wrappers, no dict observations, and no
per-step Python beyond a single vec_step call plus an episode-end log fetch.

Layout equivalence that makes this zero-copy: pufferlib allocates
observations as (num_agents, STRUCT_SIZE) uint8 rows with num_agents =
2 * num_envs. Viewed as (num_envs, 2 * STRUCT_SIZE), each row is exactly the
contiguous pair of per-player structs that the C side writes for one env.

Reward components are NOT shipped separately: c_step guarantees that on
episode-end steps rewards hold only the terminal component (shaped zeroed)
and on all other steps only the shaped component. The trainer reconstructs
both components and win labels from (rewards, terminals, truncations).
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
    # The binding requires these channels; the trainer derives the components
    # from (rewards, done) instead, so these stay worker-local.
    self._terminal_rewards = np.zeros((num_envs, MAX_PLAYERS_PER_MATCH), np.float32)
    self._shaped_rewards = np.zeros((num_envs, MAX_PLAYERS_PER_MATCH), np.float32)

    self._deck_pool = deck_pool
    self._seed = int(seed or 0)
    self._handle = None
    self._deckbuild_helper = None
    if self._deck_building:
      from deckbuild_metrics import NativeDeckbuildHelper

      self._deckbuild_helper = NativeDeckbuildHelper(
        deck_pool=deck_pool,
        snapshot_dir=deck_snapshot_dir,
        snapshot_every=deck_snapshot_every,
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

  def close(self):
    if self._handle is not None:
      binding.vec_close(self._handle)
      self._handle = None
