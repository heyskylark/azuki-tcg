import ctypes
import numpy as np
import gymnasium as gym
from pettingzoo import AECEnv
from pettingzoo.utils.conversions import turn_based_aec_to_parallel
import pufferlib
from pufferlib import emulation, MultiagentEpisodeStats
from pufferlib.emulation import nativize

from observation import (
    MAX_PLAYERS_PER_MATCH,
    OBSERVATION_CTYPE,
    OBSERVATION_STRUCT_SIZE,
    build_observation_space,
    observation_to_dict,
)
import binding

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

    # C binding arrays
    self._observations = np.zeros(
      OBSERVATION_STRUCT_SIZE * self._agent_count,
      dtype=np.uint8,
    )
    self._observation_struct = ctypes.cast(
      self._observations.ctypes.data, ctypes.POINTER(OBSERVATION_CTYPE)
    )
    self._actions = np.zeros(1, dtype=np.int32)
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
    # TODO: Implement action space
    return gym.spaces.Discrete(9)

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

    # TODO: Wire real action handling once the C env consumes action buffers.
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

    current_index = self.agents.index(acting_agent)
    if termination or truncation:
      self.agents.remove(acting_agent)
      if self.agents:
        next_agent = self.agents[current_index % len(self.agents)]
      else:
        next_agent = None
    else:
      next_agent = self.agents[(current_index + 1) % len(self.agents)]

    # TODO: Fix: Agents aren't taking turns one action at a time. sometimes they take multiple actions before the next agent's turn.
    if next_agent is not None:
      self._agent_selection = next_agent
      observation = self.observe(next_agent)
    else:
      self._agent_selection = None
      observation = None

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
