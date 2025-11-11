import ctypes
from enum import IntEnum

import numpy as np
import gymnasium as gym
from pettingzoo import AECEnv
from pettingzoo.utils.conversions import turn_based_aec_to_parallel
import pufferlib
from pufferlib import emulation, MultiagentEpisodeStats
from pufferlib.emulation import nativize

from observation import (
    ALLEY_SIZE,
    GARDEN_SIZE,
    IKZ_AREA_SIZE,
    IKZ_PILE_SIZE,
    MAX_DECK_SIZE,
    MAX_HAND_SIZE,
    MAX_PLAYERS_PER_MATCH,
    OBSERVATION_CTYPE,
    OBSERVATION_STRUCT_SIZE,
    build_observation_space,
    observation_to_dict,
)
import binding


class ActionType(IntEnum):
  """Mirror include/components.h::ActionType."""
  NOOP = 0
  PLAY_ENTITY_TO_GARDEN = 1
  PLAY_ENTITY_TO_ALLEY = 2
  ATTACK = 6
  ATTACH_WEAPON_FROM_HAND = 7
  DECLARE_DEFENDER = 8
  GATE_PORTAL = 9
  END_TURN = 12
  MULLIGAN_KEEP = 13
  MULLIGAN_SHUFFLE = 14


ACTION_COMPONENT_COUNT = 4  # ACTION_TYPE + three subactions; keep in sync with AZK_USER_ACTION_VALUE_COUNT.
_SUBACTION_MAX = max(
  MAX_DECK_SIZE,
  MAX_HAND_SIZE,
  GARDEN_SIZE,
  ALLEY_SIZE,
  IKZ_AREA_SIZE,
  IKZ_PILE_SIZE,
)
_ACTION_N_VEC = np.array(
  [
    ActionType.MULLIGAN_SHUFFLE + 1,  # Inclusive of highest ActionType enum.
    _SUBACTION_MAX,
    _SUBACTION_MAX,
    _SUBACTION_MAX,
  ],
  dtype=np.int64,
)


class AzukiTCG(AECEnv):
  """
  Azuki TCG environment using PettingZoo's AEC (Agent Environment Cycle) API.
  This is the natural fit for turn-based games where agents act sequentially.
  """
  metadata = {
    'render_modes': ['human', 'ansi'],
    'name': 'azuki_tcg_v0',
  }
  def __init__(
    self,
    seed: int | None = None,
  ) -> None:
    super().__init__()
    self.np_random = np.random.default_rng(seed)
    self.possible_agents = list(range(MAX_PLAYERS_PER_MATCH))
    self._agent_count = len(self.possible_agents)
    self._action_space = gym.spaces.MultiDiscrete(_ACTION_N_VEC.copy())

    # C binding arrays
    self._observations = np.zeros(
      OBSERVATION_STRUCT_SIZE * self._agent_count,
      dtype=np.uint8,
    )
    self._observation_struct = ctypes.cast(
      self._observations.ctypes.data, ctypes.POINTER(OBSERVATION_CTYPE)
    )
    self._actions = np.zeros(
      (self._agent_count, ACTION_COMPONENT_COUNT),
      dtype=np.int32,
    )
    self._rewards = np.zeros(self._agent_count, dtype=np.float32)
    self._terminals = np.zeros(self._agent_count, dtype=np.bool_)
    self._truncations = np.zeros(self._agent_count, dtype=np.bool_)

    # AEC state
    self.agents = []
    self._agent_selection = None
    self.rewards = {}
    self._cumulative_rewards = {}
    self.terminations = {}
    self.truncations = {}
    self.infos = {}

    self.c_envs = binding.env_init(
      self._observations,
      self._actions,
      self._rewards,
      self._terminals,
      self._truncations,
      int(seed or 0),
    )

  def observation_space(self, agent: str):
    return build_observation_space()

  def action_space(self, agent):
    return self._action_space

  def _player_index(self, agent) -> int:
    if isinstance(agent, str):
      if agent.startswith("player_"):
        agent = agent.split("_")[-1]
      agent = int(agent)
    return int(agent)

  def _raw_observation(self, agent_index: int):
    return self._observation_struct[agent_index]

  def _collect_observations(self):
    return {
      agent: observation_to_dict(self._raw_observation(idx))
      for idx, agent in enumerate(self.possible_agents)
    }

  def observe(self, agent):
    idx = self._player_index(agent)
    return observation_to_dict(self._raw_observation(idx))

  def reset(self, seed=None, options=None):
    binding.env_reset(self.c_envs, int(seed or 0))
    self._actions.fill(0)

    self.agents = self.possible_agents[:]
    self._agent_selection = self.agents[0]

    self.rewards = {agent: 0.0 for agent in self.possible_agents}
    self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
    self.terminations = {agent: False for agent in self.possible_agents}
    self.truncations = {agent: False for agent in self.possible_agents}
    self.infos = {agent: {} for agent in self.possible_agents}

    observation = self.observe(self._agent_selection)
    return observation, self.infos[self._agent_selection]

  def step(self, action):
    if not self.agents:
      raise RuntimeError("step() called on finished environment")

    acting_agent = self._agent_selection
    agent_idx = self._player_index(acting_agent)

    encoded_action = np.asarray(action, dtype=np.int32)
    if encoded_action.shape != (ACTION_COMPONENT_COUNT,):
      raise ValueError(
        f"AzukiTCG expects {ACTION_COMPONENT_COUNT} integers (type, subaction_1..3); got shape {encoded_action.shape}"
      )

    # Both agents cannot act at the same time, so we assume action[0] is always the acting agent.
    self._actions[0] = encoded_action
    binding.env_step(self.c_envs)

    reward = float(self._rewards[agent_idx])
    termination = bool(self._terminals[agent_idx])
    truncation = bool(self._truncations[agent_idx])
    info = {}

    self.rewards[acting_agent] = reward
    self._cumulative_rewards[acting_agent] += reward
    self.terminations[acting_agent] = termination
    self.truncations[acting_agent] = truncation
    self.infos[acting_agent] = info

    observation = self.observe(acting_agent)

    return observation, reward, termination, truncation, info

def _decode_observations(puffer_env, observations):
  """Convert flattened buffers back into structured dicts for debugging."""
  space = getattr(puffer_env, "env_single_observation_space", None)
  dtype = getattr(puffer_env, "obs_dtype", None)

  if space is None or dtype is None:
    # Fall back to any space/dtype combo we can find, or return the raw data.
    space = getattr(puffer_env, "single_observation_space", None)
    dtype = getattr(puffer_env, "obs_dtype", None)
    if space is None or dtype is None:
      return observations

  return {
    agent: nativize(obs, space, dtype)
    for agent, obs in observations.items()
  }

if __name__ == '__main__':
    env = AzukiTCG(seed=42)
    env = turn_based_aec_to_parallel(env)
    env = MultiagentEpisodeStats(env)
    env = emulation.PettingZooPufferEnv(env)
    observations, infos = env.reset()
    print(_decode_observations(env, observations))
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminals, truncations, infos = env.step(actions)
    print(_decode_observations(env, observations))
