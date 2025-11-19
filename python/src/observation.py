from __future__ import annotations

from typing import Any, Sequence

import ctypes

import numpy as np
from gymnasium import spaces

# Keep these values in sync with include/constants/game.h.
MAX_PLAYERS_PER_MATCH = 2
MAX_DECK_SIZE = 50
MAX_HAND_SIZE = 30
GARDEN_SIZE = 5
ALLEY_SIZE = 5
IKZ_PILE_SIZE = 10
IKZ_AREA_SIZE = 10
MAX_ATTACHED_WEAPONS = 10
ACTION_TYPE_COUNT = 13  # Keep in sync with include/components.h::AZK_ACTION_TYPE_COUNT.
SUBACTION_SELECTION_COUNT = MAX_DECK_SIZE  # Keep in sync with AzkActionParamSpec upper bounds.
MAX_LEGAL_ACTIONS_COUNT = 1024
ACTION_COMPONENT_COUNT = 4  # Keep in sync with python/src/action.py::ACTION_COMPONENT_COUNT.
LEGAL_ACTION_UNUSED = 0

# Phase constants mirror include/components.h.
PHASE_COUNT = 8


class _Type(ctypes.Structure):
    _fields_ = [("value", ctypes.c_int32)]


class _CardId(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int32), ("code", ctypes.c_char_p)]


class _TapState(ctypes.Structure):
    _fields_ = [("tapped", ctypes.c_uint8), ("cooldown", ctypes.c_uint8)]


class _IKZCost(ctypes.Structure):
    _fields_ = [("ikz_cost", ctypes.c_int8)]


class _CurStats(ctypes.Structure):
    _fields_ = [("cur_atk", ctypes.c_int8), ("cur_hp", ctypes.c_int8)]


class _BaseStats(ctypes.Structure):
    _fields_ = [("attack", ctypes.c_int8), ("health", ctypes.c_int8)]


class _GatePoints(ctypes.Structure):
    _fields_ = [("gate_points", ctypes.c_uint8)]


class _IKZCardObservationData(ctypes.Structure):
    _fields_ = [
        ("type", _Type),
        ("id", _CardId),
        ("tap_state", _TapState),
        ("zone_index", ctypes.c_uint8),
    ]


class _WeaponObservationData(ctypes.Structure):
    _fields_ = [
        ("type", _Type),
        ("id", _CardId),
        ("base_stats", _BaseStats),
        ("ikz_cost", _IKZCost),
    ]


class _LeaderCardObservationData(ctypes.Structure):
    _fields_ = [
        ("type", _Type),
        ("id", _CardId),
        ("cur_stats", _CurStats),
        ("tap_state", _TapState),
        ("weapon_count", ctypes.c_uint8),
        ("weapons", _WeaponObservationData * MAX_ATTACHED_WEAPONS),
    ]


class _GateCardObservationData(ctypes.Structure):
    _fields_ = [
        ("type", _Type),
        ("id", _CardId),
        ("tap_state", _TapState),
    ]


class _CardObservationData(ctypes.Structure):
    _fields_ = [
        ("type", _Type),
        ("id", _CardId),
        ("tap_state", _TapState),
        ("ikz_cost", _IKZCost),
        ("zone_index", ctypes.c_uint8),
        ("has_cur_stats", ctypes.c_bool),
        ("cur_stats", _CurStats),
        ("has_gate_points", ctypes.c_bool),
        ("gate_points", _GatePoints),
        ("weapon_count", ctypes.c_uint8),
        ("weapons", _WeaponObservationData * MAX_ATTACHED_WEAPONS),
    ]


class _MyObservationData(ctypes.Structure):
    _fields_ = [
        ("leader", _LeaderCardObservationData),
        ("gate", _GateCardObservationData),
        ("hand", _CardObservationData * MAX_HAND_SIZE),
        ("alley", _CardObservationData * ALLEY_SIZE),
        ("garden", _CardObservationData * GARDEN_SIZE),
        ("discard", _CardObservationData * MAX_DECK_SIZE),
        ("ikz_area", _IKZCardObservationData * IKZ_AREA_SIZE),
        ("deck_count", ctypes.c_uint8),
        ("ikz_pile_count", ctypes.c_uint8),
        ("has_ikz_token", ctypes.c_bool),
    ]


class _OpponentObservationData(ctypes.Structure):
    _fields_ = [
        ("leader", _LeaderCardObservationData),
        ("gate", _GateCardObservationData),
        ("alley", _CardObservationData * ALLEY_SIZE),
        ("garden", _CardObservationData * GARDEN_SIZE),
        ("discard", _CardObservationData * MAX_DECK_SIZE),
        ("ikz_area", _IKZCardObservationData * IKZ_AREA_SIZE),
        ("hand_count", ctypes.c_uint8),
        ("deck_count", ctypes.c_uint8),
        ("ikz_pile_count", ctypes.c_uint8),
        ("has_ikz_token", ctypes.c_bool),
    ]


class _ActionMaskObs(ctypes.Structure):
    _fields_ = [
        ("primary_action_mask", ctypes.c_bool * ACTION_TYPE_COUNT),
        ("legal_action_count", ctypes.c_uint16),
        ("legal_primary", ctypes.c_uint8 * MAX_LEGAL_ACTIONS_COUNT),
        ("legal_sub1", ctypes.c_uint8 * MAX_LEGAL_ACTIONS_COUNT),
        ("legal_sub2", ctypes.c_uint8 * MAX_LEGAL_ACTIONS_COUNT),
        ("legal_sub3", ctypes.c_uint8 * MAX_LEGAL_ACTIONS_COUNT),
    ]

class _ObservationData(ctypes.Structure):
    _fields_ = [
        ("my_observation_data", _MyObservationData),
        ("opponent_observation_data", _OpponentObservationData),
        ("phase", ctypes.c_int32),
        ("action_mask", _ActionMaskObs),
    ]

OBSERVATION_CTYPE = _ObservationData
OBSERVATION_STRUCT_SIZE = ctypes.sizeof(_ObservationData)

# Broad ranges for enum-backed integer ids. Using the underlying C storage size
# keeps Gym boxed spaces simple while still allowing future expansion.
CARD_TYPE_MAX = 6  # CARD_TYPE_EXTRA_IKZ
CARD_DEF_MAX = 255  # Fits in uint8_t/card catalog size.
ZONE_INDEX_MAX = 255
STAT_MIN = -128
STAT_MAX = 127
IKZ_COST_MIN = -128
IKZ_COST_MAX = 127
GATE_POINT_MAX = 255

def _coerce_bool_array(values: Any, length: int) -> np.ndarray:
    """Convert ctypes arrays or iterables into fixed-length boolean numpy arrays."""
    array = np.zeros(length, dtype=np.bool_)
    if values is None:
        return array
    src = np.asarray(values, dtype=np.bool_).reshape(-1)
    count = min(src.size, length)
    if count > 0:
        array[:count] = src[:count]
    return array


def _bool_space() -> spaces.Space:
    return spaces.Discrete(2)


def _scalar_box(low: int, high: int, *, dtype: np.dtype) -> spaces.Box:
    return spaces.Box(low, high, shape=(), dtype=dtype)


def _card_space(include_zone_index: bool) -> spaces.Dict:
    card_dict: dict[str, spaces.Space] = {
        "type_id": _scalar_box(0, CARD_TYPE_MAX, dtype=np.uint8),
        "card_id": _scalar_box(0, CARD_DEF_MAX, dtype=np.uint16),
        "tapped": _bool_space(),
        "cooldown": _bool_space(),
        "ikz_cost": _scalar_box(IKZ_COST_MIN, IKZ_COST_MAX, dtype=np.int16),
        "zone_index": _scalar_box(0, ZONE_INDEX_MAX, dtype=np.uint8),
        "has_cur_stats": _bool_space(),
        "attack": _scalar_box(STAT_MIN, STAT_MAX, dtype=np.int16),
        "health": _scalar_box(STAT_MIN, STAT_MAX, dtype=np.int16),
        "has_gate_points": _bool_space(),
        "gate_points": _scalar_box(0, GATE_POINT_MAX, dtype=np.uint8),
        "weapon_count": _scalar_box(0, MAX_ATTACHED_WEAPONS, dtype=np.uint8),
        "weapons": spaces.Tuple(tuple(_weapon_space() for _ in range(MAX_ATTACHED_WEAPONS))),
    }

    if not include_zone_index:
        card_dict["zone_index"] = _scalar_box(0, 0, dtype=np.uint8)

    return spaces.Dict(card_dict)


def _ikz_card_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "type_id": _scalar_box(0, CARD_TYPE_MAX, dtype=np.uint8),
            "card_id": _scalar_box(0, CARD_DEF_MAX, dtype=np.uint16),
            "tapped": _bool_space(),
            "cooldown": _bool_space(),
            "zone_index": _scalar_box(0, ZONE_INDEX_MAX, dtype=np.uint8),
        }
    )


def _weapon_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "type_id": _scalar_box(0, CARD_TYPE_MAX, dtype=np.uint8),
            "card_id": _scalar_box(0, CARD_DEF_MAX, dtype=np.uint16),
            "attack": _scalar_box(STAT_MIN, STAT_MAX, dtype=np.int16),
            "health": _scalar_box(STAT_MIN, STAT_MAX, dtype=np.int16),
            "ikz_cost": _scalar_box(IKZ_COST_MIN, IKZ_COST_MAX, dtype=np.int16),
        }
    )


def _leader_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "type_id": _scalar_box(0, CARD_TYPE_MAX, dtype=np.uint8),
            "card_id": _scalar_box(0, CARD_DEF_MAX, dtype=np.uint16),
            "tapped": _bool_space(),
            "cooldown": _bool_space(),
            "attack": _scalar_box(STAT_MIN, STAT_MAX, dtype=np.int16),
            "health": _scalar_box(STAT_MIN, STAT_MAX, dtype=np.int16),
            "weapon_count": _scalar_box(0, MAX_ATTACHED_WEAPONS, dtype=np.uint8),
            "weapons": spaces.Tuple(tuple(_weapon_space() for _ in range(MAX_ATTACHED_WEAPONS))),
        }
    )


def _gate_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "type_id": _scalar_box(0, CARD_TYPE_MAX, dtype=np.uint8),
            "card_id": _scalar_box(0, CARD_DEF_MAX, dtype=np.uint16),
            "tapped": _bool_space(),
            "cooldown": _bool_space(),
        }
    )

def _card_zone_space(count: int, *, include_zone_index: bool) -> spaces.Tuple:
    single = _card_space(include_zone_index=include_zone_index)
    return spaces.Tuple(tuple(single for _ in range(count)))

def _legal_actions_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "legal_primary": spaces.MultiDiscrete(np.full(MAX_LEGAL_ACTIONS_COUNT, ACTION_TYPE_COUNT, dtype=np.uint8)),
            "legal_sub1": spaces.MultiDiscrete(np.full(MAX_LEGAL_ACTIONS_COUNT, SUBACTION_SELECTION_COUNT, dtype=np.uint8)),
            "legal_sub2": spaces.MultiDiscrete(np.full(MAX_LEGAL_ACTIONS_COUNT, SUBACTION_SELECTION_COUNT, dtype=np.uint8)),
            "legal_sub3": spaces.MultiDiscrete(np.full(MAX_LEGAL_ACTIONS_COUNT, SUBACTION_SELECTION_COUNT, dtype=np.uint8)),
        }
    )


def _action_mask_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "primary_action_mask": spaces.MultiBinary(ACTION_TYPE_COUNT),
            "legal_action_count": spaces.Discrete(MAX_LEGAL_ACTIONS_COUNT),
            "legal_actions": _legal_actions_space(),
        }
    )


def _coerce_legal_actions(values: Any, count: int) -> dict[str, np.ndarray]:
    if values is None:
        raise ValueError("Legal action masking error: values is None")

    limit = min(max(int(count), 0), MAX_LEGAL_ACTIONS_COUNT)

    def _extract(key: str) -> np.ndarray:
        array = np.full(MAX_LEGAL_ACTIONS_COUNT, LEGAL_ACTION_UNUSED, dtype=np.int8)
        src = None
        if isinstance(values, dict):
            src = values.get(key)
        if src is None:
            src = getattr(values, key, None)
        if src is None:
            return array
        src_array = np.asarray(src, dtype=np.int16).reshape(-1)
        copy_count = min(limit, src_array.size)
        if copy_count > 0:
            array[:copy_count] = src_array[:copy_count]
        return array

    return {
        "legal_primary": _extract("legal_primary"),
        "legal_sub1": _extract("legal_sub1"),
        "legal_sub2": _extract("legal_sub2"),
        "legal_sub3": _extract("legal_sub3"),
    }

def build_observation_space() -> spaces.Dict:
    """Return the Gymnasium observation space shared by every agent."""

    player_space = spaces.Dict(
        {
            "leader": _leader_space(),
            "gate": _gate_space(),
            "hand": _card_zone_space(MAX_HAND_SIZE, include_zone_index=True),
            "alley": _card_zone_space(ALLEY_SIZE, include_zone_index=True),
            "garden": _card_zone_space(GARDEN_SIZE, include_zone_index=True),
            "discard": _card_zone_space(MAX_DECK_SIZE, include_zone_index=True),
            "ikz_area": spaces.Tuple(tuple(_ikz_card_space() for _ in range(IKZ_AREA_SIZE))),
            "deck_count": _scalar_box(0, MAX_DECK_SIZE, dtype=np.uint8),
            "ikz_pile_count": _scalar_box(0, IKZ_PILE_SIZE, dtype=np.uint8),
            "has_ikz_token": _bool_space(),
        }
    )

    opponent_space = spaces.Dict(
        {
            "leader": _leader_space(),
            "gate": _gate_space(),
            "alley": _card_zone_space(ALLEY_SIZE, include_zone_index=True),
            "garden": _card_zone_space(GARDEN_SIZE, include_zone_index=True),
            "discard": _card_zone_space(MAX_DECK_SIZE, include_zone_index=True),
            "ikz_area": spaces.Tuple(tuple(_ikz_card_space() for _ in range(IKZ_AREA_SIZE))),
            "hand_count": _scalar_box(0, MAX_HAND_SIZE, dtype=np.uint8),
            "deck_count": _scalar_box(0, MAX_DECK_SIZE, dtype=np.uint8),
            "ikz_pile_count": _scalar_box(0, IKZ_PILE_SIZE, dtype=np.uint8),
            "has_ikz_token": _bool_space(),
        }
    )

    return spaces.Dict(
        {
            "phase": spaces.Discrete(PHASE_COUNT),
            "player": player_space,
            "opponent": opponent_space,
            "action_mask": _action_mask_space(),
        }
    )


def _enum_value(struct: Any, attr: str = "value") -> int:
    if struct is None:
        return 0
    return int(getattr(struct, attr, 0))


def _tap_fields(tap_state: Any) -> tuple[int, int]:
    if tap_state is None:
        return 0, 0
    tapped = int(bool(getattr(tap_state, "tapped", 0)))
    cooldown = int(bool(getattr(tap_state, "cooldown", 0)))
    return tapped, cooldown


def _weapon_to_dict(weapon: Any) -> dict[str, int]:
    base_stats = getattr(weapon, "base_stats", None)
    return {
        "type_id": _enum_value(getattr(weapon, "type", None)),
        "card_id": _enum_value(getattr(weapon, "id", None), attr="id"),
        "attack": int(getattr(base_stats, "attack", 0)) if base_stats else 0,
        "health": int(getattr(base_stats, "health", 0)) if base_stats else 0,
        "ikz_cost": int(getattr(getattr(weapon, "ikz_cost", None), "ikz_cost", 0)),
    }


def _weapon_array_to_tuple(weapons: Sequence[Any]) -> tuple[dict[str, int], ...]:
    return tuple(_weapon_to_dict(weapon) for weapon in weapons)


def _card_to_dict(card: Any) -> dict[str, int]:
    tapped, cooldown = _tap_fields(getattr(card, "tap_state", None))
    has_cur_stats = int(bool(getattr(card, "has_cur_stats", 0)))
    cur_stats = getattr(card, "cur_stats", None)
    has_gate_points = int(bool(getattr(card, "has_gate_points", 0)))
    weapon_count = int(getattr(card, "weapon_count", 0))
    weapon_entries = _weapon_array_to_tuple(getattr(card, "weapons", ()))

    return {
        "type_id": _enum_value(getattr(card, "type", None)),
        "card_id": _enum_value(getattr(card, "id", None), attr="id"),
        "tapped": tapped,
        "cooldown": cooldown,
        "ikz_cost": int(getattr(getattr(card, "ikz_cost", None), "ikz_cost", 0)),
        "zone_index": int(getattr(card, "zone_index", 0)),
        "has_cur_stats": has_cur_stats,
        "attack": int(getattr(cur_stats, "cur_atk", 0)) if has_cur_stats else 0,
        "health": int(getattr(cur_stats, "cur_hp", 0)) if has_cur_stats else 0,
        "has_gate_points": has_gate_points,
        "gate_points": int(getattr(getattr(card, "gate_points", None), "gate_points", 0))
        if has_gate_points
        else 0,
        "weapon_count": weapon_count,
        "weapons": weapon_entries,
    }


def _ikz_card_to_dict(card: Any) -> dict[str, int]:
    tapped, cooldown = _tap_fields(getattr(card, "tap_state", None))
    return {
        "type_id": _enum_value(getattr(card, "type", None)),
        "card_id": _enum_value(getattr(card, "id", None), attr="id"),
        "tapped": tapped,
        "cooldown": cooldown,
        "zone_index": int(getattr(card, "zone_index", 0)),
    }

def _action_mask_to_dict(action_mask: Any) -> dict[str, Any]:
    legal_action_count = int(getattr(action_mask, "legal_action_count", 0))

    return {
        "primary_action_mask": _coerce_bool_array(
            getattr(action_mask, "primary_action_mask", None),
            ACTION_TYPE_COUNT,
        ),
        "legal_action_count": legal_action_count,
        "legal_actions": _coerce_legal_actions(action_mask, legal_action_count),
    }

def _leader_to_dict(card: Any) -> dict[str, int]:
    tapped, cooldown = _tap_fields(getattr(card, "tap_state", None))
    cur_stats = getattr(card, "cur_stats", None)
    weapon_count = int(getattr(card, "weapon_count", 0))
    weapon_entries = _weapon_array_to_tuple(getattr(card, "weapons", ()))
    return {
        "type_id": _enum_value(getattr(card, "type", None)),
        "card_id": _enum_value(getattr(card, "id", None), attr="id"),
        "tapped": tapped,
        "cooldown": cooldown,
        "attack": int(getattr(cur_stats, "cur_atk", 0)) if cur_stats else 0,
        "health": int(getattr(cur_stats, "cur_hp", 0)) if cur_stats else 0,
        "weapon_count": weapon_count,
        "weapons": weapon_entries,
    }


def _gate_to_dict(card: Any) -> dict[str, int]:
    tapped, cooldown = _tap_fields(getattr(card, "tap_state", None))
    return {
        "type_id": _enum_value(getattr(card, "type", None)),
        "card_id": _enum_value(getattr(card, "id", None), attr="id"),
        "tapped": tapped,
        "cooldown": cooldown,
    }


def _cards_to_tuple(cards: Sequence[Any], *, include_zone_index: bool) -> tuple[dict[str, int], ...]:
    card_dicts = tuple(_card_to_dict(card) for card in cards)
    if include_zone_index:
        return card_dicts

    # Force zone index fields to zero for zones that never expose ordering.
    sanitized: list[dict[str, int]] = []
    for entry in card_dicts:
        copy = dict(entry)
        copy["zone_index"] = 0
        sanitized.append(copy)
    return tuple(sanitized)


def _ikz_cards_to_tuple(cards: Sequence[Any]) -> tuple[dict[str, int], ...]:
    return tuple(_ikz_card_to_dict(card) for card in cards)


def observation_to_dict(observation: Any) -> dict[str, Any]:
    """Convert a ctypes/cffi ObservationData into pure Python primitives."""

    my_obs = getattr(observation, "my_observation_data", None)
    opp_obs = getattr(observation, "opponent_observation_data", None)

    player_dict = {
        "leader": _leader_to_dict(getattr(my_obs, "leader", None)),
        "gate": _gate_to_dict(getattr(my_obs, "gate", None)),
        "hand": _cards_to_tuple(getattr(my_obs, "hand", ()), include_zone_index=True),
        "alley": _cards_to_tuple(getattr(my_obs, "alley", ()), include_zone_index=True),
        "garden": _cards_to_tuple(getattr(my_obs, "garden", ()), include_zone_index=True),
        "discard": _cards_to_tuple(getattr(my_obs, "discard", ()), include_zone_index=True),
        "ikz_area": _ikz_cards_to_tuple(getattr(my_obs, "ikz_area", ())),
        "deck_count": int(getattr(my_obs, "deck_count", 0)),
        "ikz_pile_count": int(getattr(my_obs, "ikz_pile_count", 0)),
        "has_ikz_token": int(bool(getattr(my_obs, "has_ikz_token", 0))),
    }

    opponent_dict = {
        "leader": _leader_to_dict(getattr(opp_obs, "leader", None)),
        "gate": _gate_to_dict(getattr(opp_obs, "gate", None)),
        "alley": _cards_to_tuple(getattr(opp_obs, "alley", ()), include_zone_index=True),
        "garden": _cards_to_tuple(getattr(opp_obs, "garden", ()), include_zone_index=True),
        "discard": _cards_to_tuple(getattr(opp_obs, "discard", ()), include_zone_index=True),
        "ikz_area": _ikz_cards_to_tuple(getattr(opp_obs, "ikz_area", ())),
        "hand_count": int(getattr(opp_obs, "hand_count", 0)),
        "deck_count": int(getattr(opp_obs, "deck_count", 0)),
        "ikz_pile_count": int(getattr(opp_obs, "ikz_pile_count", 0)),
        "has_ikz_token": int(bool(getattr(opp_obs, "has_ikz_token", 0))),
    }

    action_mask_dict = _action_mask_to_dict(getattr(observation, "action_mask", None))
        
    return {
        "phase": int(getattr(getattr(observation, "phase", None), "value", getattr(observation, "phase", 0))),
        "player": player_dict,
        "opponent": opponent_dict,
        "action_mask": action_mask_dict,
    }


__all__ = ["build_observation_space", "observation_to_dict", "OBSERVATION_CTYPE", "OBSERVATION_STRUCT_SIZE"]
