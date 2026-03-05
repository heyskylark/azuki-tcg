from __future__ import annotations

import numpy as np

from action import ACTION_COMPONENT_COUNT
from tcg_parallel import AzukiTCGParallel as AzukiTCGParallelV1
from v2.observation import (
  ACTION_TYPE_COUNT,
  SUBACTION_SELECTION_COUNT,
  build_observation_space as build_observation_space_v2,
)


def _clamp_int(value: int, low: int, high: int) -> int:
  if value < low:
    return low
  if value > high:
    return high
  return value


class AzukiTCGParallel(AzukiTCGParallelV1):
  """V2 Parallel wrapper that augments observations with previous sampled action."""

  def __init__(self, seed: int | None = None) -> None:
    self._previous_actions = {}
    self._previous_action_valid = {}
    super().__init__(seed=seed)
    self._reset_previous_action_memory()

  def _reset_previous_action_memory(self):
    zero = np.zeros(ACTION_COMPONENT_COUNT, dtype=np.int32)
    self._previous_actions = {
      agent: zero.copy()
      for agent in self.possible_agents
    }
    self._previous_action_valid = {
      agent: False
      for agent in self.possible_agents
    }

  def _record_previous_action(self, agent, action):
    encoded = np.asarray(action, dtype=np.int32).reshape(-1)
    if encoded.shape != (ACTION_COMPONENT_COUNT,):
      return
    self._previous_actions[agent] = encoded.copy()
    self._previous_action_valid[agent] = True

  def _augment_previous_action(self, agent, observation: dict):
    out = dict(observation)
    prev = self._previous_actions.get(agent)
    has_action = bool(self._previous_action_valid.get(agent, False) and prev is not None)
    if prev is None:
      prev = np.zeros(ACTION_COMPONENT_COUNT, dtype=np.int32)

    primary = _clamp_int(int(prev[0]), 0, ACTION_TYPE_COUNT - 1)
    sub1 = _clamp_int(int(prev[1]), 0, SUBACTION_SELECTION_COUNT - 1)
    sub2 = _clamp_int(int(prev[2]), 0, SUBACTION_SELECTION_COUNT - 1)
    sub3 = _clamp_int(int(prev[3]), 0, SUBACTION_SELECTION_COUNT - 1)
    if not has_action:
      primary = 0
      sub1 = 0
      sub2 = 0
      sub3 = 0

    out["previous_action"] = {
      "has_action": int(has_action),
      "primary": primary,
      "sub1": sub1,
      "sub2": sub2,
      "sub3": sub3,
      "was_noop": int(has_action and primary == 0),
    }
    return out

  def observation_space(self, agent):
    return build_observation_space_v2()

  def _collect_observations(self):
    base = super()._collect_observations()
    return {
      agent: self._augment_previous_action(agent, obs)
      for agent, obs in base.items()
    }

  def reset(self, seed=None, options=None):
    self._reset_previous_action_memory()
    return super().reset(seed=seed, options=options)

  def step(self, actions):
    active_agent = self.possible_agents[self._active_player_index]
    if isinstance(actions, dict) and active_agent in actions:
      self._record_previous_action(active_agent, actions[active_agent])
    return super().step(actions)
