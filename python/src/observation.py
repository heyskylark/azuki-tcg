from __future__ import annotations

from typing import Any, Sequence

import ctypes

import numpy as np
from gymnasium import spaces

# Keep these values in sync with include/constants/game.h and include/components/components.h.
MAX_PLAYERS_PER_MATCH = 2
MAX_DECK_SIZE = 50
MAX_HAND_SIZE = 30
GARDEN_SIZE = 5
ALLEY_SIZE = 5
IKZ_PILE_SIZE = 10
IKZ_AREA_SIZE = 10
MAX_ATTACHED_WEAPONS = 10
MAX_SELECTION_ZONE_SIZE = 5

ACTION_TYPE_COUNT = 24
SUBACTION_SELECTION_COUNT = MAX_DECK_SIZE
MAX_LEGAL_ACTIONS_COUNT = 1024
ACTION_COMPONENT_COUNT = 4
LEGAL_ACTION_UNUSED = 0

PHASE_COUNT = 8
ABILITY_PHASE_COUNT = 6
ABILITY_TARGET_TYPE_MAX = 31


class _TapState(ctypes.Structure):
    _fields_ = [
        ("tapped", ctypes.c_uint8),
        ("cooldown", ctypes.c_uint8),
    ]


class _CurStats(ctypes.Structure):
    _fields_ = [
        ("cur_atk", ctypes.c_int8),
        ("cur_hp", ctypes.c_int8),
    ]


class _TrainingWeaponObservationData(ctypes.Structure):
    _fields_ = [
        ("card_def_id", ctypes.c_int16),
        ("cur_atk", ctypes.c_int8),
    ]


class _TrainingLeaderObservationData(ctypes.Structure):
    _fields_ = [
        ("card_def_id", ctypes.c_int16),
        ("tap_state", _TapState),
        ("cur_stats", _CurStats),
        ("weapon_count", ctypes.c_uint8),
        ("weapons", _TrainingWeaponObservationData * MAX_ATTACHED_WEAPONS),
        ("has_charge", ctypes.c_bool),
        ("has_defender", ctypes.c_bool),
        ("has_infiltrate", ctypes.c_bool),
    ]


class _TrainingGateObservationData(ctypes.Structure):
    _fields_ = [
        ("card_def_id", ctypes.c_int16),
        ("tap_state", _TapState),
    ]


class _TrainingHandCardObservationData(ctypes.Structure):
    _fields_ = [
        ("card_def_id", ctypes.c_int16),
        ("zone_index", ctypes.c_uint8),
    ]


class _TrainingDiscardCardObservationData(ctypes.Structure):
    _fields_ = [
        ("card_def_id", ctypes.c_int16),
        ("zone_index", ctypes.c_uint8),
    ]


class _TrainingBoardCardObservationData(ctypes.Structure):
    _fields_ = [
        ("card_def_id", ctypes.c_int16),
        ("tap_state", _TapState),
        ("zone_index", ctypes.c_uint8),
        ("has_cur_stats", ctypes.c_bool),
        ("cur_stats", _CurStats),
        ("weapon_count", ctypes.c_uint8),
        ("weapons", _TrainingWeaponObservationData * MAX_ATTACHED_WEAPONS),
        ("has_charge", ctypes.c_bool),
        ("has_defender", ctypes.c_bool),
        ("has_infiltrate", ctypes.c_bool),
        ("is_frozen", ctypes.c_bool),
        ("is_shocked", ctypes.c_bool),
        ("is_effect_immune", ctypes.c_bool),
    ]


class _TrainingIKZCardObservationData(ctypes.Structure):
    _fields_ = [
        ("card_def_id", ctypes.c_int16),
        ("tap_state", _TapState),
        ("zone_index", ctypes.c_uint8),
    ]


class _TrainingMyObservationData(ctypes.Structure):
    _fields_ = [
        ("leader", _TrainingLeaderObservationData),
        ("gate", _TrainingGateObservationData),
        ("hand", _TrainingHandCardObservationData * MAX_HAND_SIZE),
        ("alley", _TrainingBoardCardObservationData * ALLEY_SIZE),
        ("garden", _TrainingBoardCardObservationData * GARDEN_SIZE),
        ("discard", _TrainingDiscardCardObservationData * MAX_DECK_SIZE),
        ("selection", _TrainingBoardCardObservationData * MAX_SELECTION_ZONE_SIZE),
        ("ikz_area", _TrainingIKZCardObservationData * IKZ_AREA_SIZE),
        ("hand_count", ctypes.c_uint8),
        ("deck_count", ctypes.c_uint8),
        ("ikz_pile_count", ctypes.c_uint8),
        ("selection_count", ctypes.c_uint8),
        ("has_ikz_token", ctypes.c_bool),
    ]


class _TrainingOpponentObservationData(ctypes.Structure):
    _fields_ = [
        ("leader", _TrainingLeaderObservationData),
        ("gate", _TrainingGateObservationData),
        ("alley", _TrainingBoardCardObservationData * ALLEY_SIZE),
        ("garden", _TrainingBoardCardObservationData * GARDEN_SIZE),
        ("discard", _TrainingDiscardCardObservationData * MAX_DECK_SIZE),
        ("ikz_area", _TrainingIKZCardObservationData * IKZ_AREA_SIZE),
        ("hand_count", ctypes.c_uint8),
        ("deck_count", ctypes.c_uint8),
        ("ikz_pile_count", ctypes.c_uint8),
        ("has_ikz_token", ctypes.c_bool),
    ]


class _TrainingActionMaskObs(ctypes.Structure):
    _fields_ = [
        ("primary_action_mask", ctypes.c_bool * ACTION_TYPE_COUNT),
        ("legal_action_count", ctypes.c_uint16),
        ("legal_primary", ctypes.c_uint8 * MAX_LEGAL_ACTIONS_COUNT),
        ("legal_sub1", ctypes.c_uint8 * MAX_LEGAL_ACTIONS_COUNT),
        ("legal_sub2", ctypes.c_uint8 * MAX_LEGAL_ACTIONS_COUNT),
        ("legal_sub3", ctypes.c_uint8 * MAX_LEGAL_ACTIONS_COUNT),
    ]


class _TrainingAbilityContextObservationData(ctypes.Structure):
    _fields_ = [
        ("phase", ctypes.c_int32),
        ("pending_confirmation_count", ctypes.c_uint8),
        ("has_source_card_def_id", ctypes.c_bool),
        ("source_card_def_id", ctypes.c_int16),
        ("cost_target_type", ctypes.c_uint8),
        ("effect_target_type", ctypes.c_uint8),
        ("selection_count", ctypes.c_uint8),
        ("selection_picked", ctypes.c_uint8),
        ("selection_pick_max", ctypes.c_uint8),
        ("active_player_index", ctypes.c_int8),
    ]


class _TrainingObservationData(ctypes.Structure):
    _fields_ = [
        ("my_observation_data", _TrainingMyObservationData),
        ("opponent_observation_data", _TrainingOpponentObservationData),
        ("phase", ctypes.c_int32),
        ("ability_context", _TrainingAbilityContextObservationData),
        ("action_mask", _TrainingActionMaskObs),
    ]


OBSERVATION_CTYPE = _TrainingObservationData
OBSERVATION_STRUCT_SIZE = ctypes.sizeof(_TrainingObservationData)

CARD_DEF_MIN = -1
CARD_DEF_MAX = 255
ZONE_INDEX_MAX = 255
STAT_MIN = -128
STAT_MAX = 127
ACTIVE_PLAYER_MIN = -1
ACTIVE_PLAYER_MAX = 1


def _coerce_bool_array(values: Any, length: int) -> np.ndarray:
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


def _weapon_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "card_def_id": _scalar_box(CARD_DEF_MIN, CARD_DEF_MAX, dtype=np.int16),
            "cur_atk": _scalar_box(STAT_MIN, STAT_MAX, dtype=np.int16),
        }
    )


def _leader_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "card_def_id": _scalar_box(CARD_DEF_MIN, CARD_DEF_MAX, dtype=np.int16),
            "tapped": _bool_space(),
            "cooldown": _bool_space(),
            "cur_atk": _scalar_box(STAT_MIN, STAT_MAX, dtype=np.int16),
            "cur_hp": _scalar_box(STAT_MIN, STAT_MAX, dtype=np.int16),
            "weapon_count": _scalar_box(0, MAX_ATTACHED_WEAPONS, dtype=np.uint8),
            "weapons": spaces.Tuple(tuple(_weapon_space() for _ in range(MAX_ATTACHED_WEAPONS))),
            "has_charge": _bool_space(),
            "has_defender": _bool_space(),
            "has_infiltrate": _bool_space(),
        }
    )


def _gate_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "card_def_id": _scalar_box(CARD_DEF_MIN, CARD_DEF_MAX, dtype=np.int16),
            "tapped": _bool_space(),
            "cooldown": _bool_space(),
        }
    )


def _hand_card_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "card_def_id": _scalar_box(CARD_DEF_MIN, CARD_DEF_MAX, dtype=np.int16),
            "zone_index": _scalar_box(0, ZONE_INDEX_MAX, dtype=np.uint8),
        }
    )


def _discard_card_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "card_def_id": _scalar_box(CARD_DEF_MIN, CARD_DEF_MAX, dtype=np.int16),
            "zone_index": _scalar_box(0, ZONE_INDEX_MAX, dtype=np.uint8),
        }
    )


def _board_card_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "card_def_id": _scalar_box(CARD_DEF_MIN, CARD_DEF_MAX, dtype=np.int16),
            "zone_index": _scalar_box(0, ZONE_INDEX_MAX, dtype=np.uint8),
            "tapped": _bool_space(),
            "cooldown": _bool_space(),
            "has_cur_stats": _bool_space(),
            "cur_atk": _scalar_box(STAT_MIN, STAT_MAX, dtype=np.int16),
            "cur_hp": _scalar_box(STAT_MIN, STAT_MAX, dtype=np.int16),
            "weapon_count": _scalar_box(0, MAX_ATTACHED_WEAPONS, dtype=np.uint8),
            "weapons": spaces.Tuple(tuple(_weapon_space() for _ in range(MAX_ATTACHED_WEAPONS))),
            "has_charge": _bool_space(),
            "has_defender": _bool_space(),
            "has_infiltrate": _bool_space(),
            "is_frozen": _bool_space(),
            "is_shocked": _bool_space(),
            "is_effect_immune": _bool_space(),
        }
    )


def _ikz_card_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "card_def_id": _scalar_box(CARD_DEF_MIN, CARD_DEF_MAX, dtype=np.int16),
            "zone_index": _scalar_box(0, ZONE_INDEX_MAX, dtype=np.uint8),
            "tapped": _bool_space(),
            "cooldown": _bool_space(),
        }
    )


def _ability_context_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "phase": spaces.Discrete(ABILITY_PHASE_COUNT),
            "pending_confirmation_count": _scalar_box(0, MAX_LEGAL_ACTIONS_COUNT, dtype=np.uint16),
            "has_source_card_def_id": _bool_space(),
            "source_card_def_id": _scalar_box(CARD_DEF_MIN, CARD_DEF_MAX, dtype=np.int16),
            "cost_target_type": _scalar_box(0, ABILITY_TARGET_TYPE_MAX, dtype=np.uint8),
            "effect_target_type": _scalar_box(0, ABILITY_TARGET_TYPE_MAX, dtype=np.uint8),
            "selection_count": _scalar_box(0, MAX_SELECTION_ZONE_SIZE, dtype=np.uint8),
            "selection_picked": _scalar_box(0, MAX_SELECTION_ZONE_SIZE, dtype=np.uint8),
            "selection_pick_max": _scalar_box(0, MAX_SELECTION_ZONE_SIZE, dtype=np.uint8),
            "active_player_index": _scalar_box(ACTIVE_PLAYER_MIN, ACTIVE_PLAYER_MAX, dtype=np.int8),
        }
    )


def _legal_actions_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "legal_primary": spaces.MultiDiscrete(np.full(MAX_LEGAL_ACTIONS_COUNT, ACTION_TYPE_COUNT, dtype=np.int16)),
            "legal_sub1": spaces.MultiDiscrete(np.full(MAX_LEGAL_ACTIONS_COUNT, SUBACTION_SELECTION_COUNT, dtype=np.int16)),
            "legal_sub2": spaces.MultiDiscrete(np.full(MAX_LEGAL_ACTIONS_COUNT, SUBACTION_SELECTION_COUNT, dtype=np.int16)),
            "legal_sub3": spaces.MultiDiscrete(np.full(MAX_LEGAL_ACTIONS_COUNT, SUBACTION_SELECTION_COUNT, dtype=np.int16)),
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


def build_observation_space() -> spaces.Dict:
    player_space = spaces.Dict(
        {
            "leader": _leader_space(),
            "gate": _gate_space(),
            "hand": spaces.Tuple(tuple(_hand_card_space() for _ in range(MAX_HAND_SIZE))),
            "alley": spaces.Tuple(tuple(_board_card_space() for _ in range(ALLEY_SIZE))),
            "garden": spaces.Tuple(tuple(_board_card_space() for _ in range(GARDEN_SIZE))),
            "discard": spaces.Tuple(tuple(_discard_card_space() for _ in range(MAX_DECK_SIZE))),
            "selection": spaces.Tuple(tuple(_board_card_space() for _ in range(MAX_SELECTION_ZONE_SIZE))),
            "ikz_area": spaces.Tuple(tuple(_ikz_card_space() for _ in range(IKZ_AREA_SIZE))),
            "hand_count": _scalar_box(0, MAX_HAND_SIZE, dtype=np.uint8),
            "deck_count": _scalar_box(0, MAX_DECK_SIZE, dtype=np.uint8),
            "ikz_pile_count": _scalar_box(0, IKZ_PILE_SIZE, dtype=np.uint8),
            "selection_count": _scalar_box(0, MAX_SELECTION_ZONE_SIZE, dtype=np.uint8),
            "has_ikz_token": _bool_space(),
        }
    )

    opponent_space = spaces.Dict(
        {
            "leader": _leader_space(),
            "gate": _gate_space(),
            "alley": spaces.Tuple(tuple(_board_card_space() for _ in range(ALLEY_SIZE))),
            "garden": spaces.Tuple(tuple(_board_card_space() for _ in range(GARDEN_SIZE))),
            "discard": spaces.Tuple(tuple(_discard_card_space() for _ in range(MAX_DECK_SIZE))),
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
            "ability_context": _ability_context_space(),
            "player": player_space,
            "opponent": opponent_space,
            "action_mask": _action_mask_space(),
        }
    )


def _tap_fields(tap_state: Any) -> tuple[int, int]:
    if tap_state is None:
        return 0, 0
    tapped = int(bool(getattr(tap_state, "tapped", 0)))
    cooldown = int(bool(getattr(tap_state, "cooldown", 0)))
    return tapped, cooldown


def _weapon_to_dict(weapon: Any) -> dict[str, int]:
    return {
        "card_def_id": int(getattr(weapon, "card_def_id", -1)),
        "cur_atk": int(getattr(weapon, "cur_atk", 0)),
    }


def _weapon_array_to_tuple(weapons: Sequence[Any]) -> tuple[dict[str, int], ...]:
    return tuple(_weapon_to_dict(weapon) for weapon in weapons)


def _board_card_to_dict(card: Any) -> dict[str, Any]:
    tapped, cooldown = _tap_fields(getattr(card, "tap_state", None))
    has_cur_stats = int(bool(getattr(card, "has_cur_stats", 0)))
    cur_stats = getattr(card, "cur_stats", None)

    return {
        "card_def_id": int(getattr(card, "card_def_id", -1)),
        "zone_index": int(getattr(card, "zone_index", 0)),
        "tapped": tapped,
        "cooldown": cooldown,
        "has_cur_stats": has_cur_stats,
        "cur_atk": int(getattr(cur_stats, "cur_atk", 0)) if has_cur_stats else 0,
        "cur_hp": int(getattr(cur_stats, "cur_hp", 0)) if has_cur_stats else 0,
        "weapon_count": int(getattr(card, "weapon_count", 0)),
        "weapons": _weapon_array_to_tuple(getattr(card, "weapons", ())),
        "has_charge": int(bool(getattr(card, "has_charge", 0))),
        "has_defender": int(bool(getattr(card, "has_defender", 0))),
        "has_infiltrate": int(bool(getattr(card, "has_infiltrate", 0))),
        "is_frozen": int(bool(getattr(card, "is_frozen", 0))),
        "is_shocked": int(bool(getattr(card, "is_shocked", 0))),
        "is_effect_immune": int(bool(getattr(card, "is_effect_immune", 0))),
    }


def _hand_card_to_dict(card: Any) -> dict[str, int]:
    return {
        "card_def_id": int(getattr(card, "card_def_id", -1)),
        "zone_index": int(getattr(card, "zone_index", 0)),
    }


def _discard_card_to_dict(card: Any) -> dict[str, int]:
    return {
        "card_def_id": int(getattr(card, "card_def_id", -1)),
        "zone_index": int(getattr(card, "zone_index", 0)),
    }


def _ikz_card_to_dict(card: Any) -> dict[str, int]:
    tapped, cooldown = _tap_fields(getattr(card, "tap_state", None))
    return {
        "card_def_id": int(getattr(card, "card_def_id", -1)),
        "zone_index": int(getattr(card, "zone_index", 0)),
        "tapped": tapped,
        "cooldown": cooldown,
    }


def _leader_to_dict(leader: Any) -> dict[str, Any]:
    tapped, cooldown = _tap_fields(getattr(leader, "tap_state", None))
    cur_stats = getattr(leader, "cur_stats", None)
    return {
        "card_def_id": int(getattr(leader, "card_def_id", -1)),
        "tapped": tapped,
        "cooldown": cooldown,
        "cur_atk": int(getattr(cur_stats, "cur_atk", 0)) if cur_stats else 0,
        "cur_hp": int(getattr(cur_stats, "cur_hp", 0)) if cur_stats else 0,
        "weapon_count": int(getattr(leader, "weapon_count", 0)),
        "weapons": _weapon_array_to_tuple(getattr(leader, "weapons", ())),
        "has_charge": int(bool(getattr(leader, "has_charge", 0))),
        "has_defender": int(bool(getattr(leader, "has_defender", 0))),
        "has_infiltrate": int(bool(getattr(leader, "has_infiltrate", 0))),
    }


def _gate_to_dict(gate: Any) -> dict[str, int]:
    tapped, cooldown = _tap_fields(getattr(gate, "tap_state", None))
    return {
        "card_def_id": int(getattr(gate, "card_def_id", -1)),
        "tapped": tapped,
        "cooldown": cooldown,
    }


def _coerce_legal_actions(values: Any, count: int) -> dict[str, np.ndarray]:
    if values is None:
        raise ValueError("Legal action masking error: values is None")

    limit = min(max(int(count), 0), MAX_LEGAL_ACTIONS_COUNT)

    def _extract(key: str) -> np.ndarray:
        array = np.full(MAX_LEGAL_ACTIONS_COUNT, LEGAL_ACTION_UNUSED, dtype=np.int16)
        src = getattr(values, key, None)
        if src is None:
            return array
        src_array = np.asarray(src, dtype=np.int32).reshape(-1)
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


def _action_mask_to_dict(action_mask: Any) -> dict[str, Any]:
    legal_action_count = int(getattr(action_mask, "legal_action_count", 0))

    return {
        "primary_action_mask": _coerce_bool_array(
            getattr(action_mask, "primary_action_mask", None), ACTION_TYPE_COUNT
        ),
        "legal_action_count": legal_action_count,
        "legal_actions": _coerce_legal_actions(action_mask, legal_action_count),
    }


def _ability_context_to_dict(ability_context: Any) -> dict[str, int]:
    return {
        "phase": int(getattr(ability_context, "phase", 0)),
        "pending_confirmation_count": int(
            getattr(ability_context, "pending_confirmation_count", 0)
        ),
        "has_source_card_def_id": int(
            bool(getattr(ability_context, "has_source_card_def_id", 0))
        ),
        "source_card_def_id": int(getattr(ability_context, "source_card_def_id", -1)),
        "cost_target_type": int(getattr(ability_context, "cost_target_type", 0)),
        "effect_target_type": int(getattr(ability_context, "effect_target_type", 0)),
        "selection_count": int(getattr(ability_context, "selection_count", 0)),
        "selection_picked": int(getattr(ability_context, "selection_picked", 0)),
        "selection_pick_max": int(getattr(ability_context, "selection_pick_max", 0)),
        "active_player_index": int(getattr(ability_context, "active_player_index", -1)),
    }

def observation_to_dict(observation: Any) -> dict[str, Any]:
    """Convert a ctypes/cffi TrainingObservationData into pure Python primitives."""

    my_obs = getattr(observation, "my_observation_data", None)
    opp_obs = getattr(observation, "opponent_observation_data", None)

    player_dict = {
        "leader": _leader_to_dict(getattr(my_obs, "leader", None)),
        "gate": _gate_to_dict(getattr(my_obs, "gate", None)),
        "hand": tuple(_hand_card_to_dict(card) for card in getattr(my_obs, "hand", ())),
        "alley": tuple(_board_card_to_dict(card) for card in getattr(my_obs, "alley", ())),
        "garden": tuple(_board_card_to_dict(card) for card in getattr(my_obs, "garden", ())),
        "discard": tuple(_discard_card_to_dict(card) for card in getattr(my_obs, "discard", ())),
        "selection": tuple(_board_card_to_dict(card) for card in getattr(my_obs, "selection", ())),
        "ikz_area": tuple(_ikz_card_to_dict(card) for card in getattr(my_obs, "ikz_area", ())),
        "hand_count": int(getattr(my_obs, "hand_count", 0)),
        "deck_count": int(getattr(my_obs, "deck_count", 0)),
        "ikz_pile_count": int(getattr(my_obs, "ikz_pile_count", 0)),
        "selection_count": int(getattr(my_obs, "selection_count", 0)),
        "has_ikz_token": int(bool(getattr(my_obs, "has_ikz_token", 0))),
    }

    opponent_dict = {
        "leader": _leader_to_dict(getattr(opp_obs, "leader", None)),
        "gate": _gate_to_dict(getattr(opp_obs, "gate", None)),
        "alley": tuple(_board_card_to_dict(card) for card in getattr(opp_obs, "alley", ())),
        "garden": tuple(_board_card_to_dict(card) for card in getattr(opp_obs, "garden", ())),
        "discard": tuple(_discard_card_to_dict(card) for card in getattr(opp_obs, "discard", ())),
        "ikz_area": tuple(_ikz_card_to_dict(card) for card in getattr(opp_obs, "ikz_area", ())),
        "hand_count": int(getattr(opp_obs, "hand_count", 0)),
        "deck_count": int(getattr(opp_obs, "deck_count", 0)),
        "ikz_pile_count": int(getattr(opp_obs, "ikz_pile_count", 0)),
        "has_ikz_token": int(bool(getattr(opp_obs, "has_ikz_token", 0))),
    }

    return {
        "phase": int(getattr(observation, "phase", 0)),
        "ability_context": _ability_context_to_dict(
            getattr(observation, "ability_context", None)
        ),
        "player": player_dict,
        "opponent": opponent_dict,
        "action_mask": _action_mask_to_dict(getattr(observation, "action_mask", None)),
    }


__all__ = [
    "build_observation_space",
    "observation_to_dict",
    "OBSERVATION_CTYPE",
    "OBSERVATION_STRUCT_SIZE",
]
