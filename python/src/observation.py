from __future__ import annotations

from typing import Any, Iterable, Sequence

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

# Phase constants mirror include/components.h.
PHASE_COUNT = 8

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


def _bool_space() -> spaces.Space:
    return spaces.Discrete(2)


def _card_space(include_zone_index: bool) -> spaces.Dict:
    card_dict: dict[str, spaces.Space] = {
        "type_id": spaces.Box(0, CARD_TYPE_MAX, dtype=np.uint8),
        "card_id": spaces.Box(0, CARD_DEF_MAX, dtype=np.uint16),
        "tapped": _bool_space(),
        "cooldown": _bool_space(),
        "ikz_cost": spaces.Box(IKZ_COST_MIN, IKZ_COST_MAX, dtype=np.int16),
        "has_zone_index": _bool_space(),
        "zone_index": spaces.Box(0, ZONE_INDEX_MAX, dtype=np.uint8),
        "has_cur_stats": _bool_space(),
        "attack": spaces.Box(STAT_MIN, STAT_MAX, dtype=np.int16),
        "health": spaces.Box(STAT_MIN, STAT_MAX, dtype=np.int16),
        "has_gate_points": _bool_space(),
        "gate_points": spaces.Box(0, GATE_POINT_MAX, dtype=np.uint8),
    }

    if not include_zone_index:
        # Hand cards never expose zone indices; force observers to treat the
        # value as padding by constraining it to zero.
        card_dict["zone_index"] = spaces.Box(0, 0, dtype=np.uint8)

    return spaces.Dict(card_dict)


def _ikz_card_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "type_id": spaces.Box(0, CARD_TYPE_MAX, dtype=np.uint8),
            "card_id": spaces.Box(0, CARD_DEF_MAX, dtype=np.uint16),
            "tapped": _bool_space(),
            "cooldown": _bool_space(),
        }
    )


def _leader_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "type_id": spaces.Box(0, CARD_TYPE_MAX, dtype=np.uint8),
            "card_id": spaces.Box(0, CARD_DEF_MAX, dtype=np.uint16),
            "tapped": _bool_space(),
            "cooldown": _bool_space(),
            "attack": spaces.Box(STAT_MIN, STAT_MAX, dtype=np.int16),
            "health": spaces.Box(STAT_MIN, STAT_MAX, dtype=np.int16),
        }
    )


def _gate_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "type_id": spaces.Box(0, CARD_TYPE_MAX, dtype=np.uint8),
            "card_id": spaces.Box(0, CARD_DEF_MAX, dtype=np.uint16),
            "tapped": _bool_space(),
            "cooldown": _bool_space(),
        }
    )


def _card_zone_space(count: int, *, include_zone_index: bool) -> spaces.Tuple:
    single = _card_space(include_zone_index=include_zone_index)
    return spaces.Tuple(tuple(single for _ in range(count)))


def build_observation_space() -> spaces.Dict:
    """Return the Gymnasium observation space shared by every agent."""

    player_space = spaces.Dict(
        {
            "leader": _leader_space(),
            "gate": _gate_space(),
            "hand": _card_zone_space(MAX_HAND_SIZE, include_zone_index=False),
            "alley": _card_zone_space(ALLEY_SIZE, include_zone_index=True),
            "garden": _card_zone_space(GARDEN_SIZE, include_zone_index=True),
            "ikz_area": spaces.Tuple(tuple(_ikz_card_space() for _ in range(IKZ_AREA_SIZE))),
            "deck_count": spaces.Box(0, MAX_DECK_SIZE, dtype=np.uint8),
            "ikz_pile_count": spaces.Box(0, IKZ_PILE_SIZE, dtype=np.uint8),
            "discard_count": spaces.Box(0, MAX_DECK_SIZE, dtype=np.uint8),
            "has_ikz_token": _bool_space(),
        }
    )

    opponent_space = spaces.Dict(
        {
            "leader": _leader_space(),
            "gate": _gate_space(),
            "alley": _card_zone_space(ALLEY_SIZE, include_zone_index=True),
            "garden": _card_zone_space(GARDEN_SIZE, include_zone_index=True),
            "ikz_area": spaces.Tuple(tuple(_ikz_card_space() for _ in range(IKZ_AREA_SIZE))),
            "hand_count": spaces.Box(0, MAX_HAND_SIZE, dtype=np.uint8),
            "deck_count": spaces.Box(0, MAX_DECK_SIZE, dtype=np.uint8),
            "ikz_pile_count": spaces.Box(0, IKZ_PILE_SIZE, dtype=np.uint8),
            "discard_count": spaces.Box(0, MAX_DECK_SIZE, dtype=np.uint8),
            "has_ikz_token": _bool_space(),
        }
    )

    return spaces.Dict(
        {
            "phase": spaces.Discrete(PHASE_COUNT),
            "player": player_space,
            "opponent": opponent_space,
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


def _card_to_dict(card: Any) -> dict[str, int]:
    tapped, cooldown = _tap_fields(getattr(card, "tap_state", None))
    has_cur_stats = int(bool(getattr(card, "has_cur_stats", 0)))
    cur_stats = getattr(card, "cur_stats", None)
    has_zone_index = int(bool(getattr(card, "has_zone_index", 0)))
    has_gate_points = int(bool(getattr(card, "has_gate_points", 0)))

    return {
        "type_id": _enum_value(getattr(card, "type", None)),
        "card_id": _enum_value(getattr(card, "id", None), attr="id"),
        "tapped": tapped,
        "cooldown": cooldown,
        "ikz_cost": int(getattr(getattr(card, "ikz_cost", None), "ikz_cost", 0)),
        "has_zone_index": has_zone_index,
        "zone_index": int(getattr(card, "zone_index", 0)) if has_zone_index else 0,
        "has_cur_stats": has_cur_stats,
        "attack": int(getattr(cur_stats, "cur_atk", 0)) if has_cur_stats else 0,
        "health": int(getattr(cur_stats, "cur_hp", 0)) if has_cur_stats else 0,
        "has_gate_points": has_gate_points,
        "gate_points": int(getattr(getattr(card, "gate_points", None), "gate_points", 0))
        if has_gate_points
        else 0,
    }


def _ikz_card_to_dict(card: Any) -> dict[str, int]:
    tapped, cooldown = _tap_fields(getattr(card, "tap_state", None))
    return {
        "type_id": _enum_value(getattr(card, "type", None)),
        "card_id": _enum_value(getattr(card, "id", None), attr="id"),
        "tapped": tapped,
        "cooldown": cooldown,
    }


def _leader_to_dict(card: Any) -> dict[str, int]:
    tapped, cooldown = _tap_fields(getattr(card, "tap_state", None))
    cur_stats = getattr(card, "cur_stats", None)
    return {
        "type_id": _enum_value(getattr(card, "type", None)),
        "card_id": _enum_value(getattr(card, "id", None), attr="id"),
        "tapped": tapped,
        "cooldown": cooldown,
        "attack": int(getattr(cur_stats, "cur_atk", 0)) if cur_stats else 0,
        "health": int(getattr(cur_stats, "cur_hp", 0)) if cur_stats else 0,
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
        copy["has_zone_index"] = 0
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
        "hand": _cards_to_tuple(getattr(my_obs, "hand", ()), include_zone_index=False),
        "alley": _cards_to_tuple(getattr(my_obs, "alley", ()), include_zone_index=True),
        "garden": _cards_to_tuple(getattr(my_obs, "garden", ()), include_zone_index=True),
        "ikz_area": _ikz_cards_to_tuple(getattr(my_obs, "ikz_area", ())),
        "deck_count": int(getattr(my_obs, "deck_count", 0)),
        "ikz_pile_count": int(getattr(my_obs, "ikz_pile_count", 0)),
        "discard_count": int(getattr(my_obs, "discard_count", 0)),
        "has_ikz_token": int(bool(getattr(my_obs, "has_ikz_token", 0))),
    }

    opponent_dict = {
        "leader": _leader_to_dict(getattr(opp_obs, "leader", None)),
        "gate": _gate_to_dict(getattr(opp_obs, "gate", None)),
        "alley": _cards_to_tuple(getattr(opp_obs, "alley", ()), include_zone_index=True),
        "garden": _cards_to_tuple(getattr(opp_obs, "garden", ()), include_zone_index=True),
        "ikz_area": _ikz_cards_to_tuple(getattr(opp_obs, "ikz_area", ())),
        "hand_count": int(getattr(opp_obs, "hand_count", 0)),
        "deck_count": int(getattr(opp_obs, "deck_count", 0)),
        "ikz_pile_count": int(getattr(opp_obs, "ikz_pile_count", 0)),
        "discard_count": int(getattr(opp_obs, "discard_count", 0)),
        "has_ikz_token": int(bool(getattr(opp_obs, "has_ikz_token", 0))),
    }

    return {
        "phase": int(getattr(getattr(observation, "phase", None), "value", getattr(observation, "phase", 0))),
        "player": player_dict,
        "opponent": opponent_dict,
    }


__all__ = ["build_observation_space", "observation_to_dict"]
