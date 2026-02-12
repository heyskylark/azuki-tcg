from __future__ import annotations

import numpy as np
import pufferlib
import gymnasium as gym

import binding
from action import ACTION_COMPONENT_COUNT, build_action_space
from observation import MAX_PLAYERS_PER_MATCH, OBSERVATION_CTYPE, OBSERVATION_STRUCT_SIZE


class AzukiNativePufferEnv(pufferlib.PufferEnv):
  """Native PufferEnv path that bypasses PettingZoo AEC/parallel conversion."""

  def __init__(
    self,
    *,
    num_matches: int = 1,
    seed: int | None = None,
    buf=None,
    log_interval: int = 128,
  ) -> None:
    if num_matches < 1:
      raise ValueError(f"num_matches must be >= 1 (got {num_matches})")

    self.num_matches = int(num_matches)
    self.num_agents = MAX_PLAYERS_PER_MATCH * self.num_matches
    self.render_mode = "ansi"
    self.log_interval = int(log_interval)

    # Keep the same action encoding as the PettingZoo path.
    self.single_action_space = build_action_space()
    self.single_observation_space = gym.spaces.Box(
      low=0,
      high=255,
      shape=(OBSERVATION_STRUCT_SIZE,),
      dtype=np.uint8,
    )

    self._rng = np.random.default_rng(seed)
    self._seed = int(seed or 0)
    self._tick = 0

    # Expose struct metadata for policy-side nativize.
    self._emulated = {
      "observation_dtype": self.single_observation_space.dtype,
      "emulated_observation_dtype": np.dtype(OBSERVATION_CTYPE),
    }

    super().__init__(buf)

    if self.num_matches == 1:
      self._observations_view = self.observations.reshape(
        MAX_PLAYERS_PER_MATCH,
        OBSERVATION_STRUCT_SIZE,
      )
      self._actions_view = self.actions.reshape(
        MAX_PLAYERS_PER_MATCH,
        ACTION_COMPONENT_COUNT,
      )
      self._env_handle = binding.env_init(
        self._observations_view,
        self._actions_view,
        self.rewards,
        self.terminals,
        self.truncations,
        self._seed,
      )
      self._vec_handle = None
      self._rewards_view = self.rewards
      self._terminals_view = self.terminals
      self._truncations_view = self.truncations
    else:
      self._observations_view = self.observations.reshape(
        self.num_matches,
        MAX_PLAYERS_PER_MATCH,
        OBSERVATION_STRUCT_SIZE,
      )
      self._actions_view = self.actions.reshape(
        self.num_matches,
        MAX_PLAYERS_PER_MATCH,
        ACTION_COMPONENT_COUNT,
      )
      self._rewards_view = self.rewards.reshape(
        self.num_matches,
        MAX_PLAYERS_PER_MATCH,
      )
      self._terminals_view = self.terminals.reshape(
        self.num_matches,
        MAX_PLAYERS_PER_MATCH,
      )
      self._truncations_view = self.truncations.reshape(
        self.num_matches,
        MAX_PLAYERS_PER_MATCH,
      )
      self._vec_handle = binding.vec_init(
        self._observations_view,
        self._actions_view,
        self._rewards_view,
        self._terminals_view,
        self._truncations_view,
        self.num_matches,
        self._seed,
      )
      self._env_handle = None

  @property
  def emulated(self):
    return self._emulated

  @property
  def done(self):
    # Serial vectorization checks this before step() only for one env instance.
    # For packed mode, vec_step handles per-match resets internally.
    if self.num_matches > 1:
      return False
    return bool(np.all(self.terminals | self.truncations))

  def _next_seed(self, seed: int | None):
    if seed is None:
      return int(self._rng.integers(0, 2**31 - 1))
    return int(seed)

  def reset(self, seed: int | None = None):
    self._seed = self._next_seed(seed)
    self._tick = 0
    self.actions[:] = 0

    if self._env_handle is not None:
      binding.env_reset(self._env_handle, self._seed)
    else:
      binding.vec_reset(self._vec_handle, self._seed)

    self.rewards[:] = 0.0
    self.terminals[:] = False
    self.truncations[:] = False
    self.masks[:] = True
    return self.observations, []

  def step(self, actions):
    self._tick += 1
    self.actions[:] = actions

    if self._env_handle is not None:
      binding.env_step(self._env_handle)
      info = binding.env_log(self._env_handle)
    else:
      binding.vec_step(self._vec_handle)
      info = binding.vec_log(self._vec_handle)

    infos = []
    if info:
      infos.append(info)

    return self.observations, self.rewards, self.terminals, self.truncations, infos

  def render(self):
    if self._env_handle is not None:
      return binding.env_render(self._env_handle)
    return binding.vec_render(self._vec_handle, 0)

  def close(self):
    if self._env_handle is not None:
      binding.env_close(self._env_handle)
      self._env_handle = None
      return
    if self._vec_handle is not None:
      binding.vec_close(self._vec_handle)
      self._vec_handle = None
