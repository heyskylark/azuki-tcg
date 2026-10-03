"""Constants and enums mirroring the C engine.

Sources: include/constants/game.h, include/components/components.h,
include/validation/action_enumerator.h, python/src/observation.py.
"""
from __future__ import annotations

import enum

MAX_PLAYERS = 2
MAX_DECK_SIZE = 50
MAX_HAND_SIZE = 30
GARDEN_SIZE = 5
ALLEY_SIZE = 5
IKZ_PILE_SIZE = 10
IKZ_AREA_SIZE = 10
MAX_ATTACHED_WEAPONS = 10
INITIAL_DRAW_COUNT = 7
MAX_ABILITY_SELECTION = 8
# Mask/selection-zone enumeration bound. The C engine allows up to MAX_DECK_SIZE
# (50), but the largest selection any production deck (training distribution) or
# parity-suite crafted deck actually surfaces is 11 (probe_max_selection_all.py;
# production-pool max is 8). Capping at 16 covers that with margin and keeps the
# engine_step XLA compile tractable (50 → multi-hour, 43GB RAM). Raise toward 50
# only if a deck legitimately surfaces a larger selection (would re-inflate the
# compile). State arrays (ab_sel_cards) and the mask both use this bound.
MAX_SELECTION_ZONE_SIZE = 16
MAX_TIMED_TAG_GRANTS = 8
REQUIRED_DECK_SIZE = 50
REQUIRED_LEADER_SIZE = 1
REQUIRED_GATE_SIZE = 1
REQUIRED_IKZ_PILE_SIZE = 10
DECK_CARD_COUNT = 62  # 50 main + leader + gate + 10 IKZ
NUM_INSTANCES = DECK_CARD_COUNT + 1  # +1 = IKZ token slot
TOKEN_INSTANCE = DECK_CARD_COUNT  # instance index of the IKZ token

MAX_LEGAL_ACTIONS = 1024
ACTION_TYPE_COUNT = 26
RECENT_ACTION_HISTORY_LEN = 4
MAX_TRIGGERED_EFFECTS = 16

TERMINAL_REWARD = 5.0
TRUNCATION_TIMEOUT_PENALTY = 0.35
TRUNCATION_AUTO_TICK_PENALTY = 0.60
TRUNCATION_LEADER_EDGE_WEIGHT = 1.25
TRUNCATION_BOARD_EDGE_WEIGHT = 0.45
SHAPED_LEADER_DELTA_WEIGHT = 1.25
SHAPED_BOARD_DELTA_WEIGHT = 0.35
SHAPED_NOOP_PENALTY = 0.02
PBRS_LEADER_WEIGHT = 4.0
PBRS_GARDEN_ATTACK_WEIGHT = 0.7
PBRS_UNTAPPED_GARDEN_WEIGHT = 0.15
PBRS_UNTAPPED_IKZ_WEIGHT = 0.15
PBRS_GARDEN_ATTACK_CAP = 10.0
PBRS_UNTAPPED_GARDEN_CAP = 5.0
PBRS_UNTAPPED_IKZ_CAP = 10.0
PBRS_TIME_DECAY = 0.95

STARTER_SEED_XOR = 0xA511E9B3
DECK_SEED_XOR = 0x6D2B79F5


class Phase(enum.IntEnum):
  PREGAME_MULLIGAN = 0
  START_OF_TURN = 1
  MAIN = 2
  RESPONSE_WINDOW = 3
  COMBAT_RESOLVE = 4
  END_TURN_ACTION = 5  # unused by engine
  END_TURN = 6
  END_MATCH = 7


class Act(enum.IntEnum):
  NOOP = 0
  PLAY_ENTITY_TO_GARDEN = 1
  PLAY_ENTITY_TO_ALLEY = 2
  DECK_PICK_CARD = 3  # deck-building wrapper only (never hits the engine)
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
  SELECT_TO_GARDEN = 23
  TOP_DECK_CARD = 24
  MULLIGAN_SHUFFLE = 25


class Zone(enum.IntEnum):
  """Card instance location. List zones keep compacting order in zpos;
  garden/alley keep slot index in zpos; ATTACHED keeps weapon order in zpos."""

  DECK = 0
  HAND = 1
  LEADER = 2
  GATE = 3
  GARDEN = 4
  ALLEY = 5
  IKZ_PILE = 6
  IKZ_AREA = 7
  DISCARD = 8
  SELECTION = 9
  ATTACHED = 10
  TOKEN = 11  # IKZ token held by player (not yet played this turn)
  ABSENT = 12  # not in game (no token / deleted)


class CardType(enum.IntEnum):
  LEADER = 0
  GATE = 1
  ENTITY = 2
  WEAPON = 3
  SPELL = 4
  IKZ = 5
  EXTRA_IKZ = 6


class AbilityPhase(enum.IntEnum):
  NONE = 0
  CONFIRMATION = 1
  COST_SELECTION = 2
  EFFECT_SELECTION = 3
  SELECTION_PICK = 4
  BOTTOM_DECK = 5
