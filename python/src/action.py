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
  """Mirror include/components/components.h::ActionType."""

  NOOP = 0
  PLAY_ENTITY_TO_GARDEN = 1
  PLAY_ENTITY_TO_ALLEY = 2
  ATTACK = 6
  ATTACH_WEAPON_FROM_HAND = 7
  PLAY_SPELL_FROM_HAND = 8
  DECLARE_DEFENDER = 9
  GATE_PORTAL = 10
  ACTIVATE_GARDEN_OR_LEADER_ABILITY = 11
  ACTIVATE_ALLEY_ABILITY = 12
  SELECT_COST_TARGET = 13
  SELECT_EFFECT_TARGET = 14
  CONFIRM_ABILITY = 16
  SELECT_FROM_SELECTION = 18
  BOTTOM_DECK_CARD = 19
  BOTTOM_DECK_ALL = 20
  SELECT_TO_ALLEY = 21
  SELECT_TO_EQUIP = 22
  MULLIGAN_SHUFFLE = 23


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
