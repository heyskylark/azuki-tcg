"""Derived per-card ability tables (numpy, indexed by CardDefId).

Built from registry_generated.ENTRIES (only primary slot 0 exists in the C
registry today). Function-pointer behaviors are implemented per card in
azuki_jax.abilities.effects and dispatched by card_def_id.
"""
from __future__ import annotations

import numpy as np

from azuki_jax import cards
from azuki_jax.abilities import registry_generated as rg

N = cards.CARD_DEF_COUNT

HAS_ABILITY = np.zeros(N, np.bool_)
IS_OPTIONAL = np.zeros(N, np.bool_)
ONCE_PER_TURN = np.zeros(N, np.bool_)
RESPONSE_PLAY_FROM_HAND = np.zeros(N, np.bool_)
SEL_TO_GARDEN = np.zeros(N, np.bool_)
SEL_TO_ALLEY = np.zeros(N, np.bool_)
SEL_TO_EQUIP = np.zeros(N, np.bool_)
SEL_EQUIP_REEQUIP = np.zeros(N, np.bool_)
SEL_TO_HAND = np.zeros(N, np.bool_)
SEL_PICK_OPTIONAL = np.zeros(N, np.bool_)
CAN_TOPDECK = np.zeros(N, np.bool_)
ABILITY_IKZ_COST = np.zeros(N, np.int8)

TIMING_NAMES = [
    "AOnPlay", "AStartOfTurn", "AStartOfEachTurn", "AEndOfTurn",
    "AWhenEquipping", "AWhenEquipped", "AWhenAttacking", "AWhenAttacked",
    "AWhenReturnedToHand", "AOnGatePortal", "AAfterAttacking",
    "AWhenTakesDamage", "AWhenDealsDamage", "AWhenDestroyed",
    "AWhenSacrificed", "AWhenEntersGarden", "AMain", "AResponse",
]
TIMING_INDEX = {name: i for i, name in enumerate(TIMING_NAMES)}
# (N, len(TIMING_NAMES)) bool: card's ability carries this timing tag
TIMING = np.zeros((N, len(TIMING_NAMES)), np.bool_)

COST_TARGET_TYPE = np.zeros(N, np.uint8)
COST_MIN = np.zeros(N, np.uint8)
COST_MAX = np.zeros(N, np.uint8)
EFFECT_TARGET_TYPE = np.zeros(N, np.uint8)
EFFECT_MIN = np.zeros(N, np.uint8)
EFFECT_MAX = np.zeros(N, np.uint8)

TARGET_TYPE_NAMES = [
    "NONE",
    "FRIENDLY_GARDEN_ENTITY",
    "ENEMY_GARDEN_ENTITY",
    "ANY_GARDEN_ENTITY",
    "FRIENDLY_HAND",
    "FRIENDLY_HAND_WEAPON",
    "FRIENDLY_ALLEY_ENTITY",
    "FRIENDLY_GARDEN_OR_ALLEY_ENTITY",
    "ANY_LEADER",
    "ANY_LEADER_OR_GARDEN_ENTITY",
    "ENEMY_LEADER_OR_GARDEN_ENTITY",
]
_TARGET_INDEX: dict[str, int] = {name: i for i, name in enumerate(TARGET_TYPE_NAMES)}


def _target_index(name: str) -> int:
  if name not in _TARGET_INDEX:
    _TARGET_INDEX[name] = len(TARGET_TYPE_NAMES)
    TARGET_TYPE_NAMES.append(name)
  return _TARGET_INDEX[name]


# C function-pointer names per card (for the effects dispatch layer)
FN_VALIDATE: dict[int, str] = {}
FN_APPLY_COSTS: dict[int, str] = {}
FN_APPLY_EFFECTS: dict[int, str] = {}
FN_ON_COST_PAID: dict[int, str] = {}
FN_ON_SELECTION_COMPLETE: dict[int, str] = {}
FN_VALIDATE_COST_TARGET: dict[int, str] = {}
FN_VALIDATE_EFFECT_TARGET: dict[int, str] = {}
FN_VALIDATE_SELECTION_TARGET: dict[int, str] = {}
FN_PASSIVE_INIT: dict[int, str] = {}

for entry in rg.ENTRIES:
  if entry["slot"] != 0:
    raise ValueError("additional ability slots are not supported yet")
  code = entry["code"].replace("_", "-")
  def_id = cards.CODE_TO_ID[code]
  HAS_ABILITY[def_id] = entry["has_ability"]
  IS_OPTIONAL[def_id] = entry["is_optional"]
  ONCE_PER_TURN[def_id] = entry["is_once_per_turn"]
  RESPONSE_PLAY_FROM_HAND[def_id] = entry["can_play_as_response_from_hand"]
  SEL_TO_GARDEN[def_id] = entry["can_select_to_garden"]
  SEL_TO_ALLEY[def_id] = entry["can_select_to_alley"]
  SEL_TO_EQUIP[def_id] = entry["can_select_to_equip"]
  SEL_EQUIP_REEQUIP[def_id] = entry["selection_to_equip_is_reequip"]
  SEL_TO_HAND[def_id] = entry["can_select_to_hand"]
  SEL_PICK_OPTIONAL[def_id] = entry["selection_pick_is_optional"]
  CAN_TOPDECK[def_id] = entry["can_topdeck_selection"]
  ABILITY_IKZ_COST[def_id] = entry["ikz_cost"]
  for tag_field in ("timing_tag", "secondary_timing_tag"):
    tag = entry[tag_field]
    if tag is not None:
      TIMING[def_id, TIMING_INDEX[tag]] = True
  cost_type, cost_min, cost_max = entry["cost_req"]
  COST_TARGET_TYPE[def_id] = _target_index(cost_type)
  COST_MIN[def_id] = cost_min
  COST_MAX[def_id] = cost_max
  eff_type, eff_min, eff_max = entry["effect_req"]
  EFFECT_TARGET_TYPE[def_id] = _target_index(eff_type)
  EFFECT_MIN[def_id] = eff_min
  EFFECT_MAX[def_id] = eff_max
  for fn_field, registry in (
      ("validate", FN_VALIDATE),
      ("apply_costs", FN_APPLY_COSTS),
      ("apply_effects", FN_APPLY_EFFECTS),
      ("on_cost_paid", FN_ON_COST_PAID),
      ("on_selection_complete", FN_ON_SELECTION_COMPLETE),
      ("validate_cost_target", FN_VALIDATE_COST_TARGET),
      ("validate_effect_target", FN_VALIDATE_EFFECT_TARGET),
      ("validate_selection_target", FN_VALIDATE_SELECTION_TARGET),
      ("init_passive_observers", FN_PASSIVE_INIT),
  ):
    if entry[fn_field]:
      registry[def_id] = entry[fn_field]

TIMING_IS_MAIN = TIMING[:, TIMING_INDEX["AMain"]].copy()
TIMING_IS_RESPONSE = TIMING[:, TIMING_INDEX["AResponse"]].copy()
TIMING_ON_PLAY = TIMING[:, TIMING_INDEX["AOnPlay"]].copy()
TIMING_WHEN_ATTACKING = TIMING[:, TIMING_INDEX["AWhenAttacking"]].copy()
TIMING_WHEN_ATTACKED = TIMING[:, TIMING_INDEX["AWhenAttacked"]].copy()
TIMING_AFTER_ATTACKING = TIMING[:, TIMING_INDEX["AAfterAttacking"]].copy()
TIMING_ON_GATE_PORTAL = TIMING[:, TIMING_INDEX["AOnGatePortal"]].copy()
TIMING_WHEN_EQUIPPED = TIMING[:, TIMING_INDEX["AWhenEquipped"]].copy()
TIMING_END_OF_TURN = TIMING[:, TIMING_INDEX["AEndOfTurn"]].copy()
TIMING_START_OF_TURN = TIMING[:, TIMING_INDEX["AStartOfTurn"]].copy()
TIMING_START_OF_EACH_TURN = TIMING[:, TIMING_INDEX["AStartOfEachTurn"]].copy()
TIMING_WHEN_ENTERS_GARDEN = TIMING[:, TIMING_INDEX["AWhenEntersGarden"]].copy()
TIMING_WHEN_DESTROYED = TIMING[:, TIMING_INDEX["AWhenDestroyed"]].copy()
TIMING_WHEN_RETURNED_TO_HAND = TIMING[:, TIMING_INDEX["AWhenReturnedToHand"]].copy()

# C ABILITY_TARGET_* numeric values per our TARGET_TYPE_NAMES index (for the
# ability_context observation fields, which expose the C enum values).
_C_TARGET_VALUES = {
    "NONE": 0,
    "SELF": 1,
    "FRIENDLY_HAND": 2,
    "FRIENDLY_IKZ": 3,
    "FRIENDLY_GARDEN_ENTITY": 4,
    "FRIENDLY_ALLEY_ENTITY": 5,
    "FRIENDLY_ENTITY_WITH_WEAPON": 6,
    "FRIENDLY_LEADER": 7,
    "ENEMY_GARDEN_ENTITY": 8,
    "ENEMY_LEADER": 9,
    "ENEMY_LEADER_OR_GARDEN_ENTITY": 10,
    "ANY_LEADER_OR_GARDEN_ENTITY": 11,
    "ANY_GARDEN_ENTITY": 12,
    "FRIENDLY_SELECTION": 13,
    "FRIENDLY_SELECTION_WEAPON": 14,
    "FRIENDLY_HAND_WEAPON": 15,
    "ANY_LEADER": 16,
    "FRIENDLY_GARDEN_OR_ALLEY_ENTITY": 17,
}
TARGET_TYPE_C_VALUE = np.array(
    [_C_TARGET_VALUES[name] for name in TARGET_TYPE_NAMES], np.uint8
)
# per-card C-enum target type values (obs parity)
COST_TARGET_TYPE_C = TARGET_TYPE_C_VALUE[COST_TARGET_TYPE]
EFFECT_TARGET_TYPE_C = TARGET_TYPE_C_VALUE[EFFECT_TARGET_TYPE]

# Cards whose ability layer is implemented in JAX (grows as cards are ported).
IMPLEMENTED = np.zeros(N, np.bool_)
