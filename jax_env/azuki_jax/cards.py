"""Card definition tables as JAX constant arrays (broadcast via in_axes=None)."""
from __future__ import annotations

import numpy as np

from azuki_jax import cards_generated as cg

CARD_DEF_COUNT = cg.CARD_DEF_COUNT
CARD_CODES: list[str] = list(cg.CARD_CODES)
CODE_TO_ID = {code: i for i, code in enumerate(CARD_CODES)}
IKZ_001_ID = CODE_TO_ID["IKZ-001"]
IKZ_002_ID = CODE_TO_ID["IKZ-002"]

# numpy host tables (wrapped to jnp lazily by jit closure capture)
TYPE = np.asarray(cg.CARD_TYPE, np.int8)
ELEMENT = np.asarray(cg.CARD_ELEMENT, np.int8)
BASE_ATK = np.asarray(cg.CARD_BASE_ATK, np.int8)
BASE_HP = np.asarray(cg.CARD_BASE_HP, np.int8)
HAS_BASE_STATS = np.asarray(cg.CARD_HAS_BASE_STATS, np.bool_)
GATE_POINTS = np.asarray(cg.CARD_GATE_POINTS, np.int8)
HAS_GATE_POINTS = np.asarray(cg.CARD_HAS_GATE_POINTS, np.bool_)
IKZ_COST = np.asarray(cg.CARD_IKZ_COST, np.int8)
HAS_IKZ_COST = np.asarray(cg.CARD_HAS_IKZ_COST, np.bool_)
SUBTYPES: list[str] = list(cg.SUBTYPES)
# (CARD_DEF_COUNT, num_subtypes) bool membership matrix (>64 subtypes, so no bitmask)
SUBTYPE_MATRIX = np.array(
    [[(mask >> bit) & 1 for bit in range(len(SUBTYPES))] for mask in cg.CARD_SUBTYPE_MASK],
    dtype=np.bool_,
)

INHERENT_CHARGE = np.asarray(cg.CARD_INHERENT_CHARGE, np.bool_)
INHERENT_DEFENDER = np.asarray(cg.CARD_INHERENT_DEFENDER, np.bool_)
INHERENT_EFFECT_IMMUNE = np.asarray(cg.CARD_INHERENT_EFFECTIMMUNE, np.bool_)
INHERENT_GODMODE = np.asarray(cg.CARD_INHERENT_GODMODE, np.bool_)
INHERENT_TAUNT = np.asarray(cg.CARD_INHERENT_TAUNT, np.bool_)
INHERENT_ROOTED = np.asarray(cg.CARD_INHERENT_ROOTED, np.bool_)
INHERENT_INFILTRATE = np.asarray(cg.CARD_INHERENT_INFILTRATE, np.bool_)
ATTR_TARGET_TAPPED_UNTAPPED_ALLEY = np.asarray(
    cg.CARD_CANTARGETTAPPEDANDUNTAPPEDALLEY, np.bool_
)
ATTR_GARDEN_FORCE_TAPPED = np.asarray(cg.CARD_GARDENFORCETAPPED, np.bool_)
ATTR_COUNTS_AS_IKZ_SOURCE = np.asarray(cg.CARD_COUNTSASIKZSOURCE, np.bool_)
ATTR_TARGET_LEADER_ONLY = np.asarray(cg.CARD_CANTARGETLEADERONLY, np.bool_)
COND_EFFECT_IMMUNE = np.asarray(cg.CARD_COND_EFFECT_IMMUNE, np.int8)


def subtype_index(name: str) -> int:
  return SUBTYPES.index(name)
