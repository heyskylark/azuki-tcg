from __future__ import annotations

from enum import IntEnum

import numpy as np
import gymnasium as gym

from observation import (
  ALLEY_SIZE,
  GARDEN_SIZE,
  IKZ_AREA_SIZE,
  IKZ_PILE_SIZE,
  MAX_DECK_SIZE,
  MAX_HAND_SIZE,
)


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


def build_action_space() -> gym.Space:
  """Return the PettingZoo/Puffer action space definition."""
  return gym.spaces.MultiDiscrete(_ACTION_N_VEC.copy())
