import numpy as np
import gymnasium as gym
from pettingzoo import AECEnv
from pettingzoo.utils.conversions import turn_based_aec_to_parallel
import pufferlib
from pufferlib import emulation

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

if __name__ == '__main__':
    env = AzukiTCG()
    puffer_env = emulation.PettingZooPufferEnv(env)
    observations, infos = puffer_env.reset()
    actions = {agent: puffer_env.action_space(agent).sample() for agent in puffer_env.agents}
    observations, rewards, terminals, truncations, infos = puffer_env.step(actions)
    print(observations, rewards, terminals, truncations, infos)
