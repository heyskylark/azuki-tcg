import numpy as np
import gymnasium

try:
    import gym
except ImportError:  # pragma: no cover - gym is optional for Azuki training
    gym = None

_gym_box = tuple() if gym is None else (gym.spaces.Box,)
_gym_dict = tuple() if gym is None else (gym.spaces.Dict,)
_gym_discrete = tuple() if gym is None else (gym.spaces.Discrete,)
_gym_multi_binary = tuple() if gym is None else (gym.spaces.MultiBinary,)
_gym_multi_discrete = tuple() if gym is None else (gym.spaces.MultiDiscrete,)
_gym_tuple = tuple() if gym is None else (gym.spaces.Tuple,)

Box = _gym_box + (gymnasium.spaces.Box,)
Dict = _gym_dict + (gymnasium.spaces.Dict,)
Discrete = _gym_discrete + (gymnasium.spaces.Discrete,)
MultiBinary = _gym_multi_binary + (gymnasium.spaces.MultiBinary,)
MultiDiscrete = _gym_multi_discrete + (gymnasium.spaces.MultiDiscrete,)
Tuple = _gym_tuple + (gymnasium.spaces.Tuple,)

def joint_space(space, n):
    if isinstance(space, Discrete):
        return gymnasium.spaces.MultiDiscrete([space.n] * n)
    elif isinstance(space, MultiDiscrete):
        return gymnasium.spaces.Box(low=0,
            high=np.repeat(space.nvec[None] - 1, n, axis=0),
            shape=(n, len(space)), dtype=space.dtype)
    elif isinstance(space, Box):
        low = np.repeat(space.low[None], n, axis=0)
        high = np.repeat(space.high[None], n, axis=0)
        return gymnasium.spaces.Box(low=low, high=high,
            shape=(n, *space.shape), dtype=space.dtype)
    else:
        raise ValueError(f'Unsupported space: {space}')
