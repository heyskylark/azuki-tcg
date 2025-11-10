import numpy as np
import gymnasium as gym
from pettingzoo import AECEnv
from pettingzoo.utils.conversions import turn_based_aec_to_parallel
import pufferlib
from pufferlib import emulation

from observation import build_observation_space
import binding

class AzukiTCG(AECEnv):
  """
  Azuki TCG environment using PettingZoo's AEC (Agent Environment Cycle) API.
  This is the natural fit for turn-based games where agents act sequentially.
  """
  def __init__(
    self,
    seed: int | None = None,
  ) -> None:
    super().__init__()
    self.np_random = np.random.default_rng(seed)
    self.possible_agents = [f"player_{idx}" for idx in range(2)]

    # C binding arrays
    self._observations = np.zeros(9, dtype=np.float32)
    self._actions = np.zeros(1, dtype=np.int32)
    self._rewards = np.zeros(2, dtype=np.float32)
    self._terminals = np.zeros(2, dtype=np.bool_)
    self._truncations = np.zeros(2, dtype=np.bool_)

    # AEC state
    self.agents = []
    self._agent_selection = None
    # self.rewards = {}
    # self._cumulative_rewards = {}
    # self.terminations = {}
    # self.truncations = {}
    # self.infos = {}

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

  @property
  def agent_selection(self):
    """Current agent whose turn it is to act."""
    # TODO: Need to implement this based on the current game state
    return self._agent_selection

  def observe(self, agent):
    obs = self._observations.copy()
    return obs

  def reset(self, seed=None, options=None):
    binding.env_reset(self.c_envs, int(seed or 0))

    self.agents = self.possible_agents[:]
    # TODO: Do i need to carefully select who goes first?
    # self._agent_selection = starting_agent

    self.rewards = {agent: 0.0 for agent in self.possible_agents}
    self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
    self.terminations = {agent: False for agent in self.possible_agents}
    self.truncations = {agent: False for agent in self.possible_agents}
    self.infos = {agent: {} for agent in self.possible_agents}

  def step(self, action):
    self._actions[0] = int(action)
    binding.env_step(self.c_envs)

if __name__ == '__main__':
    env = AzukiTCG()
    puffer_env = emulation.PettingZooPufferEnv(env)
    observations, infos = puffer_env.reset()
    actions = {agent: puffer_env.action_space(agent).sample() for agent in puffer_env.agents}
    observations, rewards, terminals, truncations, infos = puffer_env.step(actions)
    print(observations, rewards, terminals, truncations, infos)
