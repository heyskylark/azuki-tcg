#!/usr/bin/env python3
"""Build a versioned, evaluator-only strategy descriptor from self-play traces."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

from action import ActionType
from analyze_selfplay_games import annotate_turns, load_games


SCHEMA_ID = "azuki.strategy_descriptor"
SCHEMA_VERSION = 3
TRACE_SCHEMA_VERSION = 2
FUNNEL_SEMANTICS_VERSION = 2
SEQUENCE_SEMANTICS_VERSION = 3
GARDEN_SIZE = 5
CARD_METADATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "python"
    / "config"
    / "policy_card_metadata_v1.json"
)
PLAY_ACTION_NAMES = frozenset(
    {
        "PLAY_ENTITY_TO_GARDEN",
        "PLAY_ENTITY_TO_ALLEY",
        "PLAY_SPELL_FROM_HAND",
        "ATTACH_WEAPON_FROM_HAND",
    }
)
ENTITY_PLAY_NAMES = frozenset({"PLAY_ENTITY_TO_GARDEN", "PLAY_ENTITY_TO_ALLEY"})
REALIZATION_FIELDS = {
    "ATTACK": "attacker",
    "DECLARE_DEFENDER": "card",
    "GATE_PORTAL": "card",
    "ACTIVATE_GARDEN_OR_LEADER_ABILITY": "card",
    "ACTIVATE_ALLEY_ABILITY": "card",
}
SEQUENCE_REGISTRY = (
    ("lightning.surge_weapon_recovery_attack", "LIGHTNING"),
    ("lightning.stormchain_weapon_requip_attack", "LIGHTNING"),
    ("water.echoed_waves_spell_replay", "WATER"),
    ("water.healing_flutter_timing", "WATER"),
    ("water.shao_attacker_target", "WATER"),
    ("earth.devotion_sacrifice_conversion", "EARTH"),
    ("earth.bobu_before_destruction", "EARTH"),
    ("earth.stone_defender_portal_attack", "EARTH"),
    ("fire.rushfire_charge_conversion", "FIRE"),
    ("fire.zero_before_attack", "FIRE"),
    ("fire.kagoro_after_multi_play", "FIRE"),
)

# These are interpretable trace observations, not counterfactual action values.
# Missing physical-copy IDs and deferred effect IDs must not become successes.


def _metadata() -> dict[str, dict[str, object]]:
    payload = json.loads(CARD_METADATA_PATH.read_text(encoding="utf-8"))
    return {str(record["card_code"]): record for record in payload["records"]}


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _legal(step: dict) -> list[tuple[int, int, int, int]]:
    raw = step.get("legal")
    if not isinstance(raw, list):
        return []
    rows: list[tuple[int, int, int, int]] = []
    for row in raw:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"Malformed legal action row: {row!r}")
        rows.append(tuple(int(value) for value in row))
    return rows


def _card_at_hand(step: dict, index: int) -> str | None:
    hand = step.get("hand", [])
    if not isinstance(hand, list) or index < 0 or index >= len(hand):
        return None
    return str(hand[index])


def _board_entry(value: object) -> tuple[str, int, int, int] | None:
    if not isinstance(value, str) or "@" not in value or ":" not in value:
        return None
    code, rest = value.split("@", 1)
    slot_text, stats = rest.split(":", 1)
    stats = stats.split("T", 1)[0].split("D", 1)[0].split("+", 1)[0]
    if "/" not in stats:
        return None
    attack, health = stats.split("/", 1)
    try:
        return code, int(slot_text), int(attack), int(health)
    except ValueError:
        return None


def _board(step: dict, *fields: str) -> list[tuple[str, int, int, int]]:
    entities: list[tuple[str, int, int, int]] = []
    for field in fields:
        values = step.get(field, [])
        if not isinstance(values, list):
            continue
        entities.extend(parsed for value in values if (parsed := _board_entry(value)))
    return entities


def _board_codes(step: dict, *fields: str) -> Counter[str]:
    return Counter(entity[0] for entity in _board(step, *fields))


def _equipped_weapon_targets(
    step: dict, weapon_codes: set[str]
) -> dict[str, set[tuple[str, int]]]:
    targets: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for weapon in step.get("my_leader_weapons", []):
        code = str(weapon)
        if code in weapon_codes:
            targets[code].add(("MY_LEADER", GARDEN_SIZE))
    for value in step.get("my_garden", []):
        parsed = _board_entry(value)
        if parsed is None or not isinstance(value, str) or "+" not in value:
            continue
        entity, slot, _, _ = parsed
        for weapon in value.split("+", 1)[1].split(","):
            if weapon in weapon_codes:
                targets[weapon].add((entity, slot))
    return targets


def _equip_destination(step: dict) -> str | None:
    raw = step.get("d", {}).get("arg")
    if not isinstance(raw, int):
        return None
    if raw == GARDEN_SIZE:
        return "MY_LEADER"
    return next(
        (code for code, slot, _, _ in _board(step, "my_garden") if slot == raw),
        None,
    )


def _legal_portal_gate_powers(
    step: dict, metadata: dict[str, dict[str, object]]
) -> list[int]:
    return [
        int(metadata.get(source, {}).get("gate_points", 0) or 0)
        for row in _legal(step)
        if row[0] == int(ActionType.GATE_PORTAL)
        if (source := _legal_source(step, row)) is not None
    ]


def _selected_portal_gate_power(
    step: dict, metadata: dict[str, dict[str, object]]
) -> int:
    if not _selected_type(step, "GATE_PORTAL"):
        return 0
    source = str(step.get("d", {}).get("card", ""))
    return int(metadata.get(source, {}).get("gate_points", 0) or 0)


def _surge_portal_weapons(
    step: dict,
    metadata: dict[str, dict[str, object]],
    weapon_codes: set[str],
    *,
    selected_portal: bool = False,
) -> set[str]:
    gate_powers = (
        [_selected_portal_gate_power(step, metadata)]
        if selected_portal
        else _legal_portal_gate_powers(step, metadata)
    )
    if not gate_powers:
        return set()
    max_gate_power = max(gate_powers)
    return {
        weapon
        for weapon in set(str(code) for code in step.get("my_discard", []))
        & weapon_codes
        if int(metadata.get(weapon, {}).get("ikz_cost", 0) or 0)
        <= max_gate_power
    }


def _stormchain_portal_weapons(
    step: dict,
    metadata: dict[str, dict[str, object]],
    weapon_codes: set[str],
    *,
    selected_portal: bool = False,
) -> set[str]:
    gate_powers = (
        [_selected_portal_gate_power(step, metadata)]
        if selected_portal
        else _legal_portal_gate_powers(step, metadata)
    )
    if not gate_powers:
        return set()
    max_gate_power = max(gate_powers)
    equipped = _equipped_weapon_targets(step, weapon_codes)
    return {
        weapon
        for weapon, targets in equipped.items()
        if any(target != "MY_LEADER" for target, _ in targets)
        and int(metadata.get(weapon, {}).get("ikz_cost", 0) or 0)
        <= max_gate_power
    }



def _immediate_gate_equip(
    events: list[dict],
    portal_index: int,
    gate: str,
    eligible_weapons: set[str],
) -> int | None:
    equip_index = portal_index + 1
    if equip_index >= len(events):
        return None
    decoded = events[equip_index].get("d", {})
    if decoded.get("src") != gate:
        return None
    if decoded.get("t") != "SELECT_TO_EQUIP":
        return None
    if decoded.get("card") not in eligible_weapons:
        return None
    return equip_index

def _post(step: dict) -> dict | None:
    value = step.get("post")
    return value if isinstance(value, dict) else None


def _material_change(step: dict) -> bool | None:
    post = _post(step)
    if post is None:
        return None
    comparisons = (
        ("hand", "hand"),
        ("my_garden", "my_garden"),
        ("my_alley", "my_alley"),
        ("opp_garden", "opp_garden"),
        ("opp_alley", "opp_alley"),
        ("my_discard", "my_discard"),
        ("opp_discard", "opp_discard"),
        ("my_leader_weapons", "my_leader_weapons"),
        ("opp_leader_weapons", "opp_leader_weapons"),
        ("ikz", "ikz"),
    )
    if any(step.get(before) != post.get(after) for before, after in comparisons):
        return True
    actor = int(step["p"])
    hp = step.get("hp")
    if isinstance(hp, list) and len(hp) == 2:
        return int(hp[actor]) != int(post.get("my_hp", hp[actor])) or int(
            hp[1 - actor]
        ) != int(post.get("opp_hp", hp[1 - actor]))
    return False


def _effective_attack_damage(step: dict) -> int | None:
    """Observed non-overkill target damage, never inferred from the game winner.

    An open response window does not establish damage. Trace v2 has no combat
    event identifiers, so deferred outcomes remain unmeasured here.
    """
    post = _post(step)
    if post is None or post.get("combat"):
        return None
    actor = int(step["p"])
    target = str(step.get("d", {}).get("target", ""))
    if target == "OPP_LEADER":
        hp = step.get("hp")
        if not isinstance(hp, list) or "opp_hp" not in post:
            return None
        before = max(0, int(hp[1 - actor]))
        return max(0, before - max(0, int(post["opp_hp"])))
    raw = step.get("a", [])
    if len(raw) != 4:
        return None
    slot = int(raw[2])
    field = "opp_garden"
    if target.startswith("ALLEY:"):
        field = "opp_alley"
        slot -= GARDEN_SIZE + 1
    before = next((entry for entry in _board(step, field) if entry[1] == slot), None)
    if before is None or field not in post:
        return None
    after = next((entry for entry in _board(post, field) if entry[1] == slot), None)
    if after is not None and after[0] != before[0]:
        return None
    return max(0, max(0, before[3]) - max(0, after[3] if after else 0))


def _selected_resolved(step: dict, *, winner: int) -> bool | None:
    post = _post(step)
    if post is None:
        return None
    action = str(step.get("d", {}).get("t", ""))
    decoded = step.get("d", {})
    code = decoded.get("card")
    if action in PLAY_ACTION_NAMES and isinstance(code, str):
        before_hand = Counter(str(value) for value in step.get("hand", []))
        after_hand = Counter(str(value) for value in post.get("hand", []))
        if after_hand[code] >= before_hand[code]:
            return False
        if action in ENTITY_PLAY_NAMES:
            fields = (
                ("my_garden",)
                if action == "PLAY_ENTITY_TO_GARDEN"
                else ("my_alley",)
            )
            return _board_codes(post, *fields)[code] > _board_codes(step, *fields)[code]
        if action == "ATTACH_WEAPON_FROM_HAND":
            weapons = [str(value) for value in post.get("my_leader_weapons", [])]
            weapons.extend(
                weapon
                for entity in post.get("my_garden", [])
                if isinstance(entity, str) and "+" in entity
                for weapon in entity.split("+", 1)[1].split(",")
            )
            return code in weapons
        return True
    if action == "GATE_PORTAL" and isinstance(code, str):
        return (
            _board_codes(post, "my_garden")[code]
            > _board_codes(step, "my_garden")[code]
            and _board_codes(post, "my_alley")[code]
            < _board_codes(step, "my_alley")[code]
        )
    if action == "ATTACK":
        damage = _effective_attack_damage(step)
        return None if damage is None else damage > 0
    if action == "SELECT_TO_EQUIP" and isinstance(code, str):
        destination = decoded.get("arg")
        target = _equip_destination(step)
        if target is None or not isinstance(destination, int):
            return False
        before_targets = _equipped_weapon_targets(step, {code}).get(code, set())
        after_targets = _equipped_weapon_targets(post, {code}).get(code, set())
        return (target, destination) in after_targets - before_targets
    if action == "SELECT_TO_GARDEN" and isinstance(code, str):
        return _board_codes(post, "my_garden")[code] > _board_codes(step, "my_garden")[code]
    return _material_change(step)


def _spell_effect_observed(step: dict) -> bool | None:
    """Separate spell consumption/payment from an observed useful state effect."""
    post = _post(step)
    if post is None or post.get("ability_src"):
        return None
    actor = int(step["p"])
    hp = step.get("hp", [0, 0])
    if int(post.get("my_hp", hp[actor])) > int(hp[actor]):
        return True
    if int(post.get("opp_hp", hp[1 - actor])) < int(hp[1 - actor]):
        return True
    for field in ("my_garden", "my_alley", "opp_garden", "opp_alley"):
        before_board = {entry[1]: entry for entry in _board(step, field)}
        after_board = {entry[1]: entry for entry in _board(post, field)}
        beneficial_from, beneficial_to = (
            (before_board, after_board) if field.startswith("my_")
            else (after_board, before_board)
        )
        for slot, entry in beneficial_to.items():
            prior = beneficial_from.get(slot)
            if prior is None or (
                prior[0] == entry[0] and (entry[2] > prior[2] or entry[3] > prior[3])
            ):
                return True
    before = step.get("ikz", [0, 0])
    after = post.get("ikz", before)
    return any(int(after[i]) > int(before[i]) for i in (0, 1))


def _unambiguous_recovery(before: dict, after: dict) -> set[str]:
    """Recoveries identifiable without confusing another in-hand copy."""
    old_hand = Counter(before.get("hand", []))
    new_hand = Counter(after.get("hand", []))
    removed = Counter(before.get("my_discard", [])) - Counter(after.get("my_discard", []))
    return {
        code for code, count in removed.items()
        if count == 1 and old_hand[code] == 0 and new_hand[code] == 1
    }


def _record_card_transition(
    before: dict,
    after: dict,
    lifecycle: dict[str, Counter[str]],
) -> None:
    hand_added = Counter(str(code) for code in after.get("hand", [])) - Counter(
        str(code) for code in before.get("hand", [])
    )
    discard_removed = Counter(
        str(code) for code in before.get("my_discard", [])
    ) - Counter(str(code) for code in after.get("my_discard", []))
    recovered = Counter(
        {
            code: min(count, discard_removed[code])
            for code, count in hand_added.items()
            if discard_removed[code] > 0
        }
    )
    for code, count in recovered.items():
        lifecycle[code]["recovered"] += count
    deck_delta = max(
        0,
        int(before.get("my_deck_n", 0)) - int(after.get("my_deck_n", 0)),
    )
    if deck_delta <= 0:
        return
    for code, count in (hand_added - recovered).items():
        drawn = min(count, deck_delta)
        lifecycle[code]["drawn"] += drawn
        deck_delta -= drawn
        if deck_delta == 0:
            break


def _portal_usable_result(step: dict) -> bool | None:
    post = _post(step)
    if post is None:
        return None
    if not _selected_resolved(step, winner=-1):
        return False
    for field in (
        "hand",
        "my_discard",
        "opp_discard",
        "my_leader_weapons",
        "opp_leader_weapons",
        "ikz",
        "opp_garden",
        "opp_alley",
    ):
        if step.get(field, []) != post.get(field, []):
            return True
    before_combined = _board_codes(step, "my_garden", "my_alley")
    after_combined = _board_codes(post, "my_garden", "my_alley")
    if before_combined != after_combined:
        return True
    actor = int(step["p"])
    hp = step.get("hp", [0, 0])
    return (
        int(post.get("my_hp", hp[actor])) != int(hp[actor])
        or int(post.get("opp_hp", hp[1 - actor])) != int(hp[1 - actor])
    )


def _response_reserve_turns(
    game: dict,
    metadata: dict[str, dict[str, object]],
) -> list[tuple[bool, bool]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for step in game["steps"]:
        if str(step.get("ph")) == "MAIN":
            grouped[(int(step["p"]), int(step.get("turn", 0)))].append(step)
    outcomes = []
    for steps in grouped.values():
        affordable = False
        minimum_cost: int | None = None
        for step in steps:
            untapped = int(step.get("ikz", [0, 0])[0])
            for code in step.get("hand", []):
                record = metadata.get(str(code), {})
                if int(record.get("ability_timing_id", -1)) != 9:
                    continue
                cost = int(record.get("ikz_cost", 0) or 0)
                if untapped >= cost:
                    affordable = True
                    minimum_cost = cost if minimum_cost is None else min(
                        minimum_cost, cost
                    )
        if not affordable or minimum_cost is None:
            continue
        final = steps[-1]
        post = _post(final)
        remaining = int(
            (post.get("ikz", [0, 0]) if post is not None else final.get("ikz", [0, 0]))[
                0
            ]
        )
        outcomes.append((True, remaining >= minimum_cost))
    return outcomes


def _new_funnel() -> Counter[str]:
    return Counter(
        opportunities=0,
        selected=0,
        resolution_observed=0,
        resolved=0,
        conversion_observed=0,
        converted=0,
    )


def _record(
    funnel: Counter[str],
    *,
    opportunity: bool,
    selected: bool,
    resolved: bool | None = None,
    converted: bool | None = None,
) -> None:
    funnel["opportunities"] += int(opportunity)
    funnel["selected"] += int(selected)
    if selected and resolved is not None:
        funnel["resolution_observed"] += 1
        funnel["resolved"] += int(resolved)
    if resolved and converted is not None:
        funnel["conversion_observed"] += 1
        funnel["converted"] += int(converted)


def _finish_funnel(raw: Counter[str]) -> dict[str, object]:
    out = {key: int(raw[key]) for key in (
        "opportunities",
        "selected",
        "resolution_observed",
        "resolved",
        "conversion_observed",
        "converted",
    )}
    out.update(
        {
            "selected_per_opportunity": _ratio(raw["selected"], raw["opportunities"]),
            "resolved_per_observed_selection": _ratio(
                raw["resolved"], raw["resolution_observed"]
            ),
            "converted_per_observed_resolution": _ratio(
                raw["converted"], raw["conversion_observed"]
            ),
            "valid": raw["opportunities"] > 0,
            "invalid_reason": None if raw["opportunities"] > 0 else "no_semantic_opportunities",
        }
    )
    return out


def _legal_card_codes(step: dict, action_types: Iterable[ActionType]) -> set[str]:
    accepted = {int(action) for action in action_types}
    return {
        code
        for row in _legal(step)
        if row[0] in accepted
        if (code := _card_at_hand(step, row[1])) is not None
    }


def _legal_source(step: dict, row: tuple[int, int, int, int]) -> str | None:
    action = row[0]
    if action == int(ActionType.GATE_PORTAL):
        slot = row[1]
        return next(
            (code for code, zone, _, _ in _board(step, "my_alley") if zone == slot),
            None,
        )
    if action == int(ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY):
        if row[1] == GARDEN_SIZE:
            return "MY_LEADER"
        return next(
            (code for code, zone, _, _ in _board(step, "my_garden") if zone == row[1]),
            None,
        )
    if action == int(ActionType.ACTIVATE_ALLEY_ABILITY):
        return next(
            (code for code, zone, _, _ in _board(step, "my_alley") if zone == row[2]),
            None,
        )
    return None


def _sequence_result(
    sequence_id: str,
    element: str,
    eligible: bool,
    completed: bool,
    converted: bool,
    won: bool,
    indices: list[int],
    *,
    stages: dict[str, bool] | None = None,
) -> dict[str, object]:
    return {
        "id": sequence_id,
        "element": element,
        "semantic_version": SEQUENCE_SEMANTICS_VERSION,
        "eligible": int(eligible),
        "completed": int(completed),
        "converted": int(converted),
        "won_after_completion": int(completed and won),
        "stages": {
            name: int(reached) for name, reached in (stages or {}).items()
        },
        "examples": [indices] if completed else [],
    }


def _first_after(events: list[dict], start: int, predicate: Callable[[dict], bool]) -> int | None:
    for index in range(start + 1, len(events)):
        if predicate(events[index]):
            return index
    return None


def _selected_type(event: dict, action: str) -> bool:
    return str(event.get("d", {}).get("t")) == action


def _sequence_rows(game: dict, player: int, metadata: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    player_indices = [index for index, step in enumerate(game["steps"]) if int(step["p"]) == player]
    all_steps = [game["steps"][index] for index in player_indices]
    events = all_steps
    winner = int(game["outcome"]["winner"])
    won = winner == player
    deck = game["decks"][player]
    gate = str(deck["gate"])
    leader = str(deck["leader"])
    main = Counter(str(code) for code in deck["main"])
    rows: list[dict[str, object]] = []

    weapon_codes = {
        code for code in main if metadata.get(code, {}).get("card_type") == "WEAPON"
    }

    surge_eligible = gate == "STT01-002" and any(
        _surge_portal_weapons(step, metadata, weapon_codes) for step in all_steps
    )
    surge_stages = {
        "legal_portal_with_eligible_discard_weapon": surge_eligible,
        "portal_selected": False,
        "weapon_recovered_equipped": False,
        "equipped_target_attacked": False,
        "attack_resolved": False,
        "prior_attach_then_discard_observed": False,
    }
    surge_indices: list[int] = []
    if gate == "STT01-002":
        for portal, event in enumerate(all_steps):
            eligible_weapons = _surge_portal_weapons(
                event, metadata, weapon_codes, selected_portal=True
            )
            if not eligible_weapons:
                continue
            surge_stages["portal_selected"] = True
            equipped = _immediate_gate_equip(
                all_steps, portal, "STT01-002", eligible_weapons
            )
            if equipped is None or not _selected_resolved(
                all_steps[equipped], winner=winner
            ):
                continue
            weapon = str(all_steps[equipped].get("d", {}).get("card", ""))
            surge_stages["weapon_recovered_equipped"] = True
            surge_stages["prior_attach_then_discard_observed"] |= any(
                _selected_type(prior, "ATTACH_WEAPON_FROM_HAND")
                and prior.get("d", {}).get("card") == weapon
                and bool(_selected_resolved(prior, winner=winner))
                for prior in all_steps[:portal]
            )
            destination = _equip_destination(all_steps[equipped])
            if destination is None:
                continue
            attack = _first_after(
                all_steps,
                equipped,
                lambda item: _selected_type(item, "ATTACK")
                and item.get("d", {}).get("attacker") == destination
                and int(item["a"][1]) == int(all_steps[equipped]["a"][2])
                and (destination, int(item["a"][1])) in _equipped_weapon_targets(item, {weapon}).get(weapon, set()),
            )
            if attack is None:
                continue
            surge_stages["equipped_target_attacked"] = True
            resolved = bool(_selected_resolved(all_steps[attack], winner=winner))
            surge_stages["attack_resolved"] |= resolved
            if not surge_indices:
                surge_indices = [portal, equipped, attack]
            if resolved:
                break
    surge_complete = bool(surge_indices)
    rows.append(
        _sequence_result(
            SEQUENCE_REGISTRY[0][0],
            "LIGHTNING",
            surge_eligible,
            surge_complete,
            surge_stages["attack_resolved"],
            won,
            surge_indices,
            stages=surge_stages,
        )
    )

    stormchain_eligible = gate == "AZK01-120" and any(
        _stormchain_portal_weapons(step, metadata, weapon_codes)
        for step in all_steps
    )
    stormchain_stages = {
        "legal_portal_with_equipped_garden_weapon": stormchain_eligible,
        "portal_selected": False,
        "weapon_re_equipped": False,
        "destination_attacked": False,
        "attack_resolved": False,
        "prior_manual_attach_observed": False,
    }
    stormchain_indices: list[int] = []
    if gate == "AZK01-120":
        for portal, event in enumerate(all_steps):
            eligible_weapons = _stormchain_portal_weapons(
                event, metadata, weapon_codes, selected_portal=True
            )
            if not eligible_weapons:
                continue
            stormchain_stages["portal_selected"] = True
            reequipped = _immediate_gate_equip(
                all_steps, portal, "AZK01-120", eligible_weapons
            )
            if reequipped is None or not _selected_resolved(
                all_steps[reequipped], winner=winner
            ):
                continue
            weapon = str(all_steps[reequipped].get("d", {}).get("card", ""))
            stormchain_stages["weapon_re_equipped"] = True
            stormchain_stages["prior_manual_attach_observed"] |= any(
                _selected_type(prior, "ATTACH_WEAPON_FROM_HAND")
                and prior.get("d", {}).get("card") == weapon
                and bool(_selected_resolved(prior, winner=winner))
                for prior in all_steps[:portal]
            )
            destination = _equip_destination(all_steps[reequipped])
            if destination is None:
                continue
            attack = _first_after(
                all_steps,
                reequipped,
                lambda item: _selected_type(item, "ATTACK")
                and item.get("d", {}).get("attacker") == destination
                and int(item["a"][1]) == int(all_steps[reequipped]["a"][2])
                and (destination, int(item["a"][1])) in _equipped_weapon_targets(item, {weapon}).get(weapon, set()),
            )
            if attack is None:
                continue
            stormchain_stages["destination_attacked"] = True
            resolved = bool(_selected_resolved(all_steps[attack], winner=winner))
            stormchain_stages["attack_resolved"] |= resolved
            if not stormchain_indices:
                stormchain_indices = [portal, reequipped, attack]
            if resolved:
                break
    stormchain_complete = bool(stormchain_indices)
    rows.append(
        _sequence_result(
            SEQUENCE_REGISTRY[1][0],
            "LIGHTNING",
            stormchain_eligible,
            stormchain_complete,
            stormchain_stages["attack_resolved"],
            won,
            stormchain_indices,
            stages=stormchain_stages,
        )
    )

    spell_codes = {code for code in main if metadata.get(code, {}).get("card_type") == "SPELL"}
    spell_eligible = gate == "AZK01-126" and any(
        any(
            int(metadata.get(code, {}).get("ikz_cost", 0)) <= power
            for power in _legal_portal_gate_powers(step, metadata)
            for code in set(step.get("my_discard", [])) & spell_codes
        )
        for step in all_steps
    )
    replay_indices: list[int] = []
    if spell_eligible:
        for portal, event in enumerate(all_steps):
            if not _selected_type(event, "GATE_PORTAL"):
                continue
            recovery = portal
            recovery_step = event
            if not _unambiguous_recovery(event, _post(event) or {}):
                recovery += 1
                if recovery >= len(all_steps):
                    continue
                recovery_step = all_steps[recovery]
                if recovery_step.get("d", {}).get("src") != gate:
                    continue
            recovered = _unambiguous_recovery(event, _post(recovery_step) or {}) & spell_codes
            for spell in recovered:
                for index in range(recovery + 1, len(all_steps)):
                    candidate = all_steps[index]
                    # A new copy or loss of the tracked copy makes identity ambiguous.
                    if Counter(candidate.get("hand", []))[spell] != 1:
                        break
                    if _selected_type(candidate, "PLAY_SPELL_FROM_HAND") and candidate.get("d", {}).get("card") == spell:
                        if _selected_resolved(candidate, winner=winner):
                            replay_indices = [portal, recovery, index]
                        break
                if replay_indices:
                    break
            if replay_indices:
                break
    replay_complete = bool(replay_indices)
    replay_converted = replay_complete and bool(
        _spell_effect_observed(all_steps[replay_indices[-1]])
    )
    rows.append(_sequence_result(SEQUENCE_REGISTRY[2][0], "WATER", spell_eligible, replay_complete, replay_converted, won, replay_indices))

    heal_eligible = False
    heal_indices: list[int] = []
    for i, event in enumerate(all_steps if metadata.get(gate, {}).get("element") == "WATER" else []):
        legal_heals = _legal_card_codes(event, (ActionType.PLAY_SPELL_FROM_HAND,))
        actor = int(event["p"])
        hp = event.get("hp", [20, 20])
        if "AZK01-002" in legal_heals and int(hp[actor]) < 20:
            heal_eligible = True
        if _selected_type(event, "PLAY_SPELL_FROM_HAND") and event.get("d", {}).get("card") == "AZK01-002" and int(hp[actor]) < 20:
            heal_indices = [i]
            break
    heal_complete = heal_eligible and bool(heal_indices)
    heal_post = _post(all_steps[heal_indices[0]]) if heal_complete else None
    heal_converted = bool(heal_post) and int(heal_post.get("my_hp", 0)) > int(all_steps[heal_indices[0]]["hp"][player])
    rows.append(_sequence_result(SEQUENCE_REGISTRY[3][0], "WATER", heal_eligible, heal_complete, heal_converted, won, heal_indices))

    shao_eligible = False
    shao_indices: list[int] = []
    if leader == "STT02-001":
        for i, event in enumerate(events):
            legal_leader = any(
                row[0] == int(ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY) and row[1] == GARDEN_SIZE
                for row in _legal(event)
            )
            combat = event.get("combat", {})
            shao_eligible |= legal_leader and isinstance(combat, dict) and bool(combat)
            if not (_selected_type(event, "ACTIVATE_GARDEN_OR_LEADER_ABILITY") and event.get("d", {}).get("card") == "MY_LEADER"):
                continue
            attacker = combat.get("attacker") if isinstance(combat, dict) else None
            target = _first_after(
                events,
                i,
                lambda item: _selected_type(item, "SELECT_EFFECT_TARGET")
                and item.get("d", {}).get("target") == attacker,
            )
            if target is not None:
                shao_indices = [i, target]
                break
    shao_complete = shao_eligible and bool(shao_indices)
    rows.append(_sequence_result(SEQUENCE_REGISTRY[4][0], "WATER", shao_eligible, shao_complete, shao_complete and bool(_selected_resolved(events[shao_indices[-1]], winner=winner)), won, shao_indices))

    devotion_eligible = gate == "AZK01-124" and any(
        any(row[0] == int(ActionType.GATE_PORTAL) for row in _legal(step)) for step in all_steps
    )
    devotion_indices: list[int] = []
    for i, event in enumerate(events):
        if not _selected_type(event, "GATE_PORTAL"):
            continue
        post = _post(event)
        if post is None:
            continue
        own_before = _board_codes(event, "my_garden", "my_alley")
        own_after = _board_codes(post, "my_garden", "my_alley")
        opp_before = _board(event, "opp_garden")
        opp_after = _board(post, "opp_garden")
        own_loss = sum((own_before - own_after).values()) > 0
        opp_damage = opp_before != opp_after
        if own_loss and opp_damage:
            devotion_indices = [i]
            break
    devotion_complete = devotion_eligible and bool(devotion_indices)
    rows.append(_sequence_result(SEQUENCE_REGISTRY[5][0], "EARTH", devotion_eligible, devotion_complete, devotion_complete, won, devotion_indices))

    bobu_eligible = leader == "STT03-001" and any(
        any(row[0] == int(ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY) and row[1] == GARDEN_SIZE for row in _legal(step))
        for step in all_steps
    )
    bobu_indices: list[int] = []
    bobu_healed = False
    if bobu_eligible:
        for i, event in enumerate(events):
            if not (_selected_type(event, "ACTIVATE_GARDEN_OR_LEADER_ABILITY") and event.get("d", {}).get("card") == "MY_LEADER"):
                continue
            turn = int(event.get("turn", 0))
            for j in range(player_indices[i] + 1, len(game["steps"])):
                later = game["steps"][j]
                if int(later.get("turn", 0)) > turn + 1:
                    break
                post = _post(later)
                if post is None:
                    continue
                prefix = "my" if int(later["p"]) == player else "opp"
                lost = _board_codes(later, f"{prefix}_garden", f"{prefix}_alley") - _board_codes(post, f"{prefix}_garden", f"{prefix}_alley")
                earth_lost = any(metadata.get(code, {}).get("element") == "EARTH" for code in lost)
                if earth_lost:
                    bobu_indices = [player_indices[i], j]
                    bobu_healed = int(post.get(f"{prefix}_hp", 0)) > int(later.get("hp", [20, 20])[player])
                    if bobu_healed:
                        break
            if bobu_healed:
                break
    bobu_complete = bool(bobu_indices)
    rows.append(_sequence_result(SEQUENCE_REGISTRY[6][0], "EARTH", bobu_eligible, bobu_complete, bobu_healed, won, bobu_indices))

    stone_eligible = gate == "STT03-002" and any(
        any(row[0] == int(ActionType.DECLARE_DEFENDER) for row in _legal(step)) for step in all_steps
    )
    stone_indices: list[int] = []
    for i, event in enumerate(events):
        if not _selected_type(event, "DECLARE_DEFENDER"):
            continue
        portal = _first_after(events, i, lambda item: _selected_type(item, "GATE_PORTAL"))
        if portal is None:
            continue
        portaled = str(events[portal].get("d", {}).get("card", ""))
        attack = _first_after(
            events,
            portal,
            lambda item: _selected_type(item, "ATTACK") and item.get("d", {}).get("attacker") == portaled,
        )
        if attack is not None:
            stone_indices = [i, portal, attack]
            break
    stone_complete = stone_eligible and bool(stone_indices)
    rows.append(_sequence_result(SEQUENCE_REGISTRY[7][0], "EARTH", stone_eligible, stone_complete, stone_complete and bool(_selected_resolved(events[stone_indices[-1]], winner=winner)), won, stone_indices))

    rush_eligible = gate == "AZK01-122" and any(
        any(row[0] == int(ActionType.GATE_PORTAL) for row in _legal(step)) for step in all_steps
    )
    rush_indices: list[int] = []
    for i, event in enumerate(events):
        if not _selected_type(event, "GATE_PORTAL") or not _selected_resolved(event, winner=winner):
            continue
        extra = _first_after(
            events,
            i,
            lambda item: _selected_type(item, "SELECT_TO_GARDEN")
            and item.get("d", {}).get("src") == "AZK01-122"
            and int(item.get("turn", 0)) == int(event.get("turn", 0)),
        )
        if extra != i + 1 or not _selected_resolved(events[extra], winner=winner):
            continue
        entity = str(events[extra].get("d", {}).get("card", ""))
        attack = _first_after(
            events,
            extra,
            lambda item: _selected_type(item, "ATTACK")
            and item.get("d", {}).get("attacker") == entity
            and int(item["a"][1]) == int(events[extra]["a"][2])
            and int(item.get("turn", 0)) == int(event.get("turn", 0)),
        )
        if attack is not None:
            rush_indices = [i, extra, attack]
            break
    rush_complete = rush_eligible and bool(rush_indices)
    rows.append(_sequence_result(SEQUENCE_REGISTRY[8][0], "FIRE", rush_eligible, rush_complete, rush_complete and bool(_selected_resolved(events[rush_indices[-1]], winner=winner)), won, rush_indices))

    zero_eligible = leader == "STT04-001" and any(
        any(row[0] == int(ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY) and row[1] == GARDEN_SIZE for row in _legal(step))
        for step in all_steps
    )
    zero_indices: list[int] = []
    for i, event in enumerate(events):
        if not (_selected_type(event, "ACTIVATE_GARDEN_OR_LEADER_ABILITY") and event.get("d", {}).get("card") == "MY_LEADER"):
            continue
        post = _post(event)
        if post is None or int(post.get("my_hp", 20)) >= int(event.get("hp", [20, 20])[player]):
            continue
        target_selection = _first_after(
            events,
            i,
            lambda item: _selected_type(item, "SELECT_EFFECT_TARGET")
            and item.get("d", {}).get("src") == "STT04-001"
            and isinstance(item.get("d", {}).get("target"), str),
        )
        if target_selection != i + 1:
            continue
        target = events[target_selection]["d"]["target"]
        selection = events[target_selection]
        before_target = [entry for entry in _board(selection, "my_garden", "my_alley") if entry[0] == target]
        after_target = [entry for entry in _board(_post(selection) or {}, "my_garden", "my_alley") if entry[0] == target]
        if len(before_target) != 1 or len(after_target) != 1:
            continue
        if after_target[0][2] <= before_target[0][2] or after_target[0][3] >= before_target[0][3]:
            continue
        attack = _first_after(
            events,
            target_selection,
            lambda item: _selected_type(item, "ATTACK")
            and item.get("d", {}).get("attacker") == target
            and int(item.get("turn", 0)) == int(event.get("turn", 0)),
        )
        if attack is not None:
            zero_indices = [i, target_selection, attack]
            break
    zero_complete = zero_eligible and bool(zero_indices)
    rows.append(_sequence_result(SEQUENCE_REGISTRY[9][0], "FIRE", zero_eligible, zero_complete, zero_complete and bool(_selected_resolved(events[zero_indices[-1]], winner=winner)), won, zero_indices))

    kagoro_eligible = leader == "AZK01-121" and any(
        any(row[0] == int(ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY) and row[1] == GARDEN_SIZE for row in _legal(step))
        for step in all_steps
    )
    kagoro_indices: list[int] = []
    for i, event in enumerate(events):
        if not (_selected_type(event, "ACTIVATE_GARDEN_OR_LEADER_ABILITY") and event.get("d", {}).get("card") == "MY_LEADER"):
            continue
        turn = int(event.get("turn", 0))
        if int((_post(event) or {}).get("my_leader_atk", 0)) <= int(event.get("my_leader_atk", 0)):
            continue
        prior_plays = [
            index for index in range(i) if int(events[index].get("turn", 0)) == turn and str(events[index].get("d", {}).get("t")) in ENTITY_PLAY_NAMES
        ]
        if len(prior_plays) < 2:
            continue
        attack = _first_after(
            events,
            i,
            lambda item: _selected_type(item, "ATTACK")
            and item.get("d", {}).get("attacker") == "MY_LEADER"
            and int(item.get("turn", 0)) == turn,
        )
        if attack is not None:
            kagoro_indices = [prior_plays[-2], prior_plays[-1], i, attack]
            break
    kagoro_complete = kagoro_eligible and bool(kagoro_indices)
    rows.append(_sequence_result(SEQUENCE_REGISTRY[10][0], "FIRE", kagoro_eligible, kagoro_complete, kagoro_complete and bool(_selected_resolved(events[kagoro_indices[-1]], winner=winner)), won, kagoro_indices))
    for row in rows:
        if row["id"] != SEQUENCE_REGISTRY[6][0]:
            row["examples"] = [[player_indices[index] for index in example] for example in row["examples"]]
    return rows


def _aggregate_sequences(rows: Iterable[dict[str, object]]) -> dict[str, dict[str, object]]:
    aggregate: dict[str, Counter] = defaultdict(Counter)
    stage_counts: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list[list[int]]] = defaultdict(list)
    elements: dict[str, str] = {}
    for row in rows:
        sequence_id = str(row["id"])
        elements[sequence_id] = str(row["element"])
        for field in ("eligible", "completed", "converted", "won_after_completion"):
            aggregate[sequence_id][field] += int(row[field])
        for stage, reached in row["stages"].items():
            stage_counts[sequence_id][str(stage)] += int(reached)
        examples[sequence_id].extend(row["examples"][: 5 - len(examples[sequence_id])])
    return {
        sequence_id: {
            "element": elements[sequence_id],
            "semantic_version": SEQUENCE_SEMANTICS_VERSION,
            **{field: int(counts[field]) for field in ("eligible", "completed", "converted", "won_after_completion")},
            "completion_per_eligible": _ratio(counts["completed"], counts["eligible"]),
            "conversion_per_completed": _ratio(counts["converted"], counts["completed"]),
            "win_rate_after_completion": _ratio(counts["won_after_completion"], counts["completed"]),
            "stages": {
                stage: {
                    "reached": int(reached),
                    "reached_per_eligible": _ratio(reached, counts["eligible"]),
                }
                for stage, reached in stage_counts[sequence_id].items()
            },
            "valid": counts["eligible"] > 0,
            "examples": examples[sequence_id],
        }
        for sequence_id, counts in sorted(aggregate.items())
    }


def evaluate_strategy_events(games: list[dict]) -> dict[str, object]:
    metadata = _metadata()
    funnels: dict[str, Counter] = defaultdict(_new_funnel)
    card_funnels: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(_new_funnel))
    card_lifecycle: dict[str, Counter[str]] = defaultdict(Counter)
    sequence_rows: list[dict[str, object]] = []
    resource = Counter()
    face_choices = Counter()
    trace_versions = Counter()
    action_modes = Counter()

    for game_index, raw_game in enumerate(games):
        game = annotate_turns(raw_game)
        winner = int(game["outcome"]["winner"])
        trace_versions[int(game.get("trace_schema_version", 1))] += 1
        action_modes[str(game.get("policy_action_mode", "unknown"))] += 1
        for draft in game.get("draft", []):
            offered = {str(code) for code in draft.get("offered", [])}
            selected = str(draft.get("selected", ""))
            if selected not in offered:
                raise ValueError(f"Draft selection absent from offer: game={game_index}, selected={selected}")
            for code in offered:
                _record(card_funnels[code]["draft"], opportunity=True, selected=code == selected)
                card_lifecycle[code]["offered"] += 1
                card_lifecycle[code]["drafted"] += int(code == selected)
        resolved_available: Counter[tuple[int, str]] = Counter()
        recovered_available: list[set[str]] = [set(), set()]
        portaled_available: Counter[tuple[int, str]] = Counter()
        last_post: list[dict | None] = [None, None]
        last_action: list[dict | None] = [None, None]
        pending_portals = [0, 0]
        for step_index, step in enumerate(game["steps"]):
            player = int(step["p"])
            if last_post[player] is not None:
                _record_card_transition(
                    last_post[player] or {}, step, card_lifecycle
                )
                recovered_available[player].intersection_update(
                    code for code in recovered_available[player]
                    if Counter(step.get("hand", []))[code] == 1
                )
                recovered_available[player].update(
                    _unambiguous_recovery(last_post[player] or {}, step)
                )
            legal = _legal(step)
            selected_raw = tuple(int(value) for value in step["a"])
            if legal and selected_raw not in legal:
                raise ValueError(
                    f"Selected action absent from legal trace: game={game_index}, step={step_index}"
                )
            selected_type = str(step.get("d", {}).get("t", ""))
            resolved = _selected_resolved(step, winner=winner)

            play_codes = _legal_card_codes(
                step,
                (
                    ActionType.PLAY_ENTITY_TO_GARDEN,
                    ActionType.PLAY_ENTITY_TO_ALLEY,
                    ActionType.PLAY_SPELL_FROM_HAND,
                    ActionType.ATTACH_WEAPON_FROM_HAND,
                ),
            )
            for code in play_codes:
                chosen = selected_type in PLAY_ACTION_NAMES and step.get("d", {}).get("card") == code
                _record(card_funnels[code]["play"], opportunity=True, selected=chosen, resolved=resolved if chosen else None)
            selected_code = step.get("d", {}).get("card")
            if selected_type in PLAY_ACTION_NAMES and isinstance(selected_code, str) and resolved:
                card_type = str(metadata.get(selected_code, {}).get("card_type", ""))
                if selected_code in recovered_available[player]:
                    if card_type == "SPELL":
                        card_lifecycle[selected_code]["replayed"] += 1
                    elif card_type == "WEAPON":
                        card_lifecycle[selected_code]["re_equipped"] += 1
                    recovered_available[player].discard(selected_code)
                if card_type == "SPELL":
                    effect = _spell_effect_observed(step)
                    if effect is not None:
                        card_funnels[selected_code]["play"]["conversion_observed"] += 1
                        card_funnels[selected_code]["play"]["converted"] += int(effect)
                elif card_type == "ENTITY":
                    resolved_available[(player, selected_code)] += 1

            if (
                selected_type == "SELECT_TO_EQUIP"
                and isinstance(selected_code, str)
                and resolved
            ):
                source = step.get("d", {}).get("src")
                if source == "STT01-002":
                    card_lifecycle[selected_code]["recovered"] += 1
                if source in {"STT01-002", "AZK01-120"}:
                    card_lifecycle[selected_code]["re_equipped"] += 1

            realization_field = REALIZATION_FIELDS.get(selected_type)
            realized_code = step.get("d", {}).get(realization_field) if realization_field else None
            if isinstance(realized_code, str) and resolved_available[(player, realized_code)] > 0:
                card_funnels[realized_code]["play"]["conversion_observed"] += 1
                card_funnels[realized_code]["play"]["converted"] += int(bool(resolved))
                resolved_available[(player, realized_code)] -= 1

            attack_rows = [row for row in legal if row[0] == int(ActionType.ATTACK)]
            face_rows = [row for row in attack_rows if row[2] == GARDEN_SIZE]
            entity_rows = [row for row in attack_rows if row[2] != GARDEN_SIZE]
            selected_face = selected_type == "ATTACK" and step.get("d", {}).get("target") == "OPP_LEADER"
            selected_entity = selected_type == "ATTACK" and not selected_face
            _record(funnels["attack.face"], opportunity=bool(face_rows), selected=selected_face, resolved=resolved if selected_face else None, converted=(winner == player) if resolved and selected_face else None)
            _record(funnels["attack.entity"], opportunity=bool(entity_rows), selected=selected_entity, resolved=resolved if selected_entity else None, converted=resolved if selected_entity and resolved is not None else None)
            if face_rows and entity_rows:
                _record(funnels["attack.both_legal_face_choice"], opportunity=True, selected=selected_face, resolved=resolved if selected_face else None)
                _record(funnels["attack.both_legal_entity_choice"], opportunity=True, selected=selected_entity, resolved=resolved if selected_entity else None)
                face_choices["both_legal"] += 1
                face_choices["face"] += int(selected_face)
                face_choices["entity"] += int(selected_entity)
            hp = step.get("hp", [0, 0])
            lethal_rows = []
            for row in face_rows:
                if row[1] == GARDEN_SIZE:
                    attack = int(step.get("my_leader_atk", 0))
                else:
                    attack = next(
                        (
                            entity[2]
                            for entity in _board(step, "my_garden")
                            if entity[1] == row[1]
                        ),
                        0,
                    )
                if attack >= int(hp[1 - player]):
                    lethal_rows.append(row)
            selected_lethal = selected_face and selected_raw in lethal_rows
            _record(funnels["attack.nominal_lethal_face"], opportunity=bool(lethal_rows), selected=selected_lethal, resolved=resolved if selected_lethal else None, converted=(winner == player) if resolved and selected_lethal else None)

            portal_usable = (
                _portal_usable_result(step)
                if selected_type == "GATE_PORTAL"
                else None
            )
            action_specs = (
                ("portal", ActionType.GATE_PORTAL, selected_type == "GATE_PORTAL"),
                ("ability.leader", ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY, selected_type == "ACTIVATE_GARDEN_OR_LEADER_ABILITY" and selected_raw[1] == GARDEN_SIZE),
                ("ability.garden", ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY, selected_type == "ACTIVATE_GARDEN_OR_LEADER_ABILITY" and selected_raw[1] < GARDEN_SIZE),
                ("ability.alley", ActionType.ACTIVATE_ALLEY_ABILITY, selected_type == "ACTIVATE_ALLEY_ABILITY"),
                ("defender", ActionType.DECLARE_DEFENDER, selected_type == "DECLARE_DEFENDER"),
            )
            for name, action_type, selected in action_specs:
                matching = [row for row in legal if row[0] == int(action_type)]
                if name == "ability.leader":
                    matching = [row for row in matching if row[1] == GARDEN_SIZE]
                elif name == "ability.garden":
                    matching = [row for row in matching if row[1] < GARDEN_SIZE]
                converted = None
                if name == "defender" and selected and resolved is not None:
                    post = _post(step) or {}
                    combat = post.get("combat", step.get("combat", {}))
                    converted = bool(
                        isinstance(combat, dict) and combat.get("intercepted")
                    )
                elif name == "portal" and selected and resolved is not None:
                    converted = portal_usable
                _record(
                    funnels[name],
                    opportunity=bool(matching),
                    selected=selected,
                    resolved=resolved if selected else None,
                    converted=converted,
                )
            if selected_type == "GATE_PORTAL" and isinstance(selected_code, str) and resolved:
                if portal_usable:
                    pending_portals[player] = max(0, pending_portals[player] - 1)
                else:
                    portaled_available[(player, selected_code)] += 1
                    pending_portals[player] += 1
            source = step.get("d", {}).get("src")
            gate = str(game["decks"][player]["gate"])
            if (
                isinstance(source, str)
                and source == gate
                and pending_portals[player] > 0
                and resolved
            ):
                funnels["portal"]["conversion_observed"] += 1
                funnels["portal"]["converted"] += 1
                pending_portals[player] -= 1
                for key in tuple(portaled_available):
                    if key[0] == player:
                        portaled_available[key] = 0
            attacker = step.get("d", {}).get("attacker")
            if (
                selected_type == "ATTACK"
                and isinstance(attacker, str)
                and portaled_available[(player, attacker)] > 0
            ):
                funnels["portal"]["conversion_observed"] += 1
                funnels["portal"]["converted"] += int(bool(resolved))
                portaled_available[(player, attacker)] -= 1
                pending_portals[player] = max(0, pending_portals[player] - 1)

            if str(step.get("ph")) == "RESPONSE":
                nonpass = [row for row in legal if row[0] != int(ActionType.NOOP)]
                selected_response = selected_type != "NOOP"
                _record(funnels["response.affordable"], opportunity=bool(nonpass), selected=selected_response, resolved=resolved if selected_response else None, converted=resolved if selected_response and resolved is not None else None)

            post = _post(step)
            if post is not None:
                _record_card_transition(step, post, card_lifecycle)
                recovered_available[player].update(_unambiguous_recovery(step, post))
                previous = last_action[player]
                if (
                    previous is not None
                    and str(game["decks"][player]["gate"]) == "AZK01-126"
                    and _selected_type(previous, "GATE_PORTAL")
                    and selected_type == "SELECT_FROM_SELECTION"
                    and step.get("d", {}).get("src") == "AZK01-126"
                    and int(previous.get("turn", 0)) == int(step.get("turn", 0))
                    and not Counter((_post(previous) or {}).get("hand", []))[selected_code]
                ):
                    recovered = (
                        _unambiguous_recovery(previous, post)
                        - _unambiguous_recovery(step, post)
                    ) & {selected_code}
                    recovered_available[player].update(recovered)
                    for code in recovered:
                        card_lifecycle[code]["recovered"] += 1
                before_untapped, before_total = (int(value) for value in step.get("ikz", [0, 0]))
                after_untapped, after_total = (int(value) for value in post.get("ikz", [0, 0]))
                added = max(0, after_total - before_total)
                readied = max(0, after_untapped - before_untapped - added)
                spent = max(0, before_untapped - after_untapped)
                resource["net_added"] += added
                resource["net_readied"] += readied
                resource["spent"] += spent
                resource["held_step_sum"] += after_untapped
                resource["observed_transitions"] += 1
                last_post[player] = post
            last_action[player] = step
        for opportunity, reserved in _response_reserve_turns(game, metadata):
            _record(
                funnels["response.reserved"],
                opportunity=opportunity,
                selected=reserved,
                resolved=reserved,
                converted=reserved,
            )
        for player in range(2):
            sequence_rows.extend(_sequence_rows(game, player, metadata))

    sequences = _aggregate_sequences(sequence_rows)
    rush = sequences["fire.rushfire_charge_conversion"]
    charge = funnels["temporary.charge"]
    charge["opportunities"] = int(rush["eligible"])
    charge["selected"] = int(rush["completed"])
    charge["resolution_observed"] = int(rush["completed"])
    charge["resolved"] = int(rush["completed"])
    charge["conversion_observed"] = int(rush["completed"])
    charge["converted"] = int(rush["converted"])
    temporary_attack_ids = (
        "fire.zero_before_attack",
        "fire.kagoro_after_multi_play",
    )
    for sequence_id in temporary_attack_ids:
        sequence = sequences[sequence_id]
        for _ in range(int(sequence["eligible"])):
            _record(
                funnels["temporary.attack"],
                opportunity=True,
                selected=False,
            )
        for _ in range(int(sequence["completed"])):
            funnels["temporary.attack"]["selected"] += 1
            funnels["temporary.attack"]["resolution_observed"] += 1
            funnels["temporary.attack"]["resolved"] += 1
            funnels["temporary.attack"]["conversion_observed"] += 1
            # Completion does not imply the temporary attack was realized.
        funnels["temporary.attack"]["converted"] += int(sequence["converted"])
    return {
        "trace_capabilities": {
            "trace_schema_versions": {str(key): value for key, value in sorted(trace_versions.items())},
            "policy_action_modes": dict(sorted(action_modes.items())),
            "legal_actions": all("legal" in step for game in games for step in game.get("steps", [])),
            "post_action_state": all("post" in step for game in games for step in game.get("steps", [])),
            "draft_events": all("draft" in game for game in games),
        },
        "funnels": {name: _finish_funnel(raw) for name, raw in sorted(funnels.items())},
        "card_funnels": {
            code: {
                "name": str(metadata.get(code, {}).get("name", code)),
                "card_type": str(metadata.get(code, {}).get("card_type", "?")),
                "lifecycle": {
                    field: int(card_lifecycle[code][field])
                    for field in (
                        "offered",
                        "drafted",
                        "drawn",
                        "recovered",
                        "replayed",
                        "re_equipped",
                    )
                },
                **{
                    stage: _finish_funnel(raw)
                    for stage, raw in sorted(card_funnels[code].items())
                },
            }
            for code in sorted(set(card_funnels) | set(card_lifecycle))
        },
        "sequences": sequences,
        "game_profile": {
            "both_face_entity_legal": int(face_choices["both_legal"]),
            "face_selected_when_both_legal": int(face_choices["face"]),
            "entity_selected_when_both_legal": int(face_choices["entity"]),
            "face_share_when_both_legal": _ratio(face_choices["face"], face_choices["both_legal"]),
        },
        "resource_profile": {
            "observed_transitions": int(resource["observed_transitions"]),
            "ikz_net_added": int(resource["net_added"]),
            "ikz_net_readied": int(resource["net_readied"]),
            "ikz_net_spent": int(resource["spent"]),
            "generated_ikz_conversion": None,
            "attribution_status": "unmeasured: snapshots do not identify generated resource tokens or separate automatic turn refresh",
            "mean_untapped_ikz_after_step": _ratio(resource["held_step_sum"], resource["observed_transitions"]),
        },
    }


def payoff_vector_distance(left: list[dict], right: list[dict]) -> tuple[int, float]:
    left_cells = {str(cell["context_key"]): float(cell["score"]) for cell in left}
    right_cells = {str(cell["context_key"]): float(cell["score"]) for cell in right}
    common = sorted(left_cells.keys() & right_cells.keys())
    if not common:
        return 0, 0.0
    squared = sum((left_cells[key] - right_cells[key]) ** 2 for key in common)
    return len(common), math.sqrt(squared / len(common))


def _deck_profile(games: list[dict], metadata: dict[str, dict[str, object]]) -> dict[str, object]:
    contexts: dict[str, list[Counter[str]]] = defaultdict(list)
    for game in games:
        for deck in game["decks"]:
            key = f"{deck['gate']}:{deck['leader']}"
            contexts[key].append(Counter(str(code) for code in deck["main"]))
    out = {}
    for context, decks in sorted(contexts.items()):
        total_slots = sum(sum(deck.values()) for deck in decks)
        type_slots: Counter[str] = Counter()
        element_slots: Counter[str] = Counter()
        costs: Counter[int] = Counter()
        signatures: Counter[tuple[tuple[str, int], ...]] = Counter()
        for deck in decks:
            signatures[tuple(sorted(deck.items()))] += 1
            for code, count in deck.items():
                record = metadata.get(code, {})
                type_slots[str(record.get("card_type", "?"))] += count
                element_slots[str(record.get("element", "?"))] += count
                costs[int(record.get("ikz_cost", 0) or 0)] += count
        representative, representative_count = max(
            signatures.items(), key=lambda item: (item[1], item[0])
        )
        out[context] = {
            "seat_decks": len(decks),
            "main_slots": total_slots,
            "type_slot_share": {key: round(value / max(total_slots, 1), 6) for key, value in sorted(type_slots.items())},
            "element_slot_share": {key: round(value / max(total_slots, 1), 6) for key, value in sorted(element_slots.items())},
            "ikz_cost_histogram": {str(key): value for key, value in sorted(costs.items())},
            "exact_collision_decks": sum(value for value in signatures.values() if value > 1),
            "unique_signatures": len(signatures),
            "representative_signature": [
                {"code": code, "copies": count}
                for code, count in representative
            ],
            "representative_frequency": representative_count,
        }
    return out

def _deck_relationships(
    profiles: dict[str, object],
    metadata: dict[str, dict[str, object]],
) -> dict[str, object]:
    pairs = []
    for left, right in combinations(sorted(profiles), 2):
        left_gate, left_leader = left.split(":", 1)
        right_gate, right_leader = right.split(":", 1)
        left_element = str(metadata.get(left_gate, {}).get("element", "?"))
        right_element = str(metadata.get(right_gate, {}).get("element", "?"))
        if left_element != right_element or left_gate == right_gate or left_leader != right_leader:
            continue
        left_profile = profiles[left]
        right_profile = profiles[right]
        if not isinstance(left_profile, dict) or not isinstance(right_profile, dict):
            raise TypeError("Deck profiles must be mappings")
        left_deck = Counter(
            {
                str(item["code"]): int(item["copies"])
                for item in left_profile["representative_signature"]
            }
        )
        right_deck = Counter(
            {
                str(item["code"]): int(item["copies"])
                for item in right_profile["representative_signature"]
            }
        )
        cards = set(left_deck) | set(right_deck)
        intersection = sum(min(left_deck[code], right_deck[code]) for code in cards)
        union = sum(max(left_deck[code], right_deck[code]) for code in cards)
        similarity = intersection / union if union else 1.0
        pairs.append(
            {
                "left_context": left,
                "right_context": right,
                "multiset_jaccard": similarity,
                "multiset_distance": 1.0 - similarity,
                "exact_collision": similarity == 1.0,
            }
        )
    return {
        "sibling_pairs": pairs,
        "sibling_exact_collision_pairs": sum(
            bool(pair["exact_collision"]) for pair in pairs
        ),
    }


def _elemental_strategy(games: list[dict], metadata: dict[str, dict[str, object]]) -> dict:
    """Context-sliced mechanic evidence without a pooled strategy score.

    Resource conversion is a conservative same-turn lower bound: pre-existing
    IKZ is spent first. Weapon damage is damage while equipped, not marginal
    damage attributed to the weapon. Those distinctions are part of the output.
    """
    contexts: dict[str, dict] = {}
    weapons = {code for code, data in metadata.items() if data.get("card_type") == "WEAPON"}
    for game in games:
        starter = next((step["p"] for step in game["steps"] if step.get("ph") == "MAIN"), "unknown")
        for player, deck in enumerate(game["decks"]):
            gate, leader = str(deck["gate"]), str(deck["leader"])
            element = str(metadata.get(gate, {}).get("element", "UNKNOWN"))
            opponent = game["decks"][1 - player]
            key = f"{gate}|{leader}|opp={opponent['gate']}:{opponent['leader']}|seat={player}|starter={starter}"
            context = contexts.setdefault(key, {
                "element": element, "gate": gate, "leader": leader,
                "opponent_gate": opponent["gate"], "opponent_leader": opponent["leader"],
                "seat": player, "starter": starter, "counts": Counter(),
            })
            counts = context["counts"]
            counts["player_games"] += 1
            for card_type in ("SPELL", "WEAPON"):
                slots = sum(metadata.get(code, {}).get("card_type") == card_type for code in deck["main"])
                counts[f"{card_type.lower()}_deck_slots"] += slots
                counts[f"decks_with_{card_type.lower()}"] += int(slots > 0)
            native_turn = None
            recovered_budget = 0
            ordinary_budget = 0
            for step in game["steps"]:
                post = _post(step)
                if element == "EARTH" and post is not None:
                    hp_field = "my_hp" if int(step["p"]) == player else "opp_hp"
                    heal = max(0, int(post.get(hp_field, 0)) - int(step.get("hp", [20, 20])[player]))
                    counts["leader_healing_events"] += int(heal > 0)
                    counts["leader_hp_restored"] += heal
                if int(step["p"]) != player:
                    continue
                turn = int(step.get("turn", 0))
                if turn != native_turn:
                    recovered_budget = 0
                    ordinary_budget = 0
                    native_turn = turn
                action = str(step.get("d", {}).get("t", ""))
                for card_type, action_type in (
                    ("spell", ActionType.PLAY_SPELL_FROM_HAND),
                    ("weapon", ActionType.ATTACH_WEAPON_FROM_HAND),
                ):
                    legal = _legal_card_codes(step, (action_type,))
                    counts[f"{card_type}_legal_decisions"] += int(bool(legal))
                    if action == action_type.name:
                        counts[f"{card_type}_selected"] += 1
                        counts[f"{card_type}_resolved"] += int(bool(_selected_resolved(step, winner=-1)))
                        if card_type == "spell":
                            effect = _spell_effect_observed(step)
                            counts["spell_effect_observed"] += int(effect is not None)
                            counts["spell_effect_positive"] += int(bool(effect))
                if element == "LIGHTNING" and action == "ATTACK":
                    attacker = str(step.get("d", {}).get("attacker", ""))
                    slot = int(step["a"][1])
                    equipped = any((attacker, slot) in targets for targets in _equipped_weapon_targets(step, weapons).values())
                    counts["weapon_equipped_attacks"] += int(equipped)
                    if equipped:
                        damage = _effective_attack_damage(step)
                        counts["weapon_attack_outcomes_observed"] += int(damage is not None)
                        counts["weapon_attacks_with_damage"] += int(bool(damage))
                        counts["damage_while_weapon_equipped"] += damage or 0
                if element == "EARTH":
                    counts["defender_legal_decisions"] += int(any(row[0] == int(ActionType.DECLARE_DEFENDER) for row in _legal(step)))
                    counts["defender_selected"] += int(action == "DECLARE_DEFENDER")
                if gate == "STT02-002":
                    before_ikz = step.get("ikz", [0, 0])
                    counts["hydromancy_legal_with_tapped_ikz"] += int(
                        int(before_ikz[1]) > int(before_ikz[0]) and
                        any(power > 0 for power in _legal_portal_gate_powers(step, metadata))
                    )
                    if post is not None:
                        after_ikz = post.get("ikz", before_ikz)
                        ordinary_budget = max(ordinary_budget, int(before_ikz[0]) - recovered_budget)
                        spent = max(0, int(before_ikz[0]) - int(after_ikz[0]))
                        from_ordinary = min(ordinary_budget, spent)
                        ordinary_budget -= from_ordinary
                        from_recovered = min(recovered_budget, spent - from_ordinary)
                        recovered_budget -= from_recovered
                        if action == "PLAY_SPELL_FROM_HAND" and _selected_resolved(step, winner=-1):
                            counts["hydromancy_ikz_to_spell_lower_bound"] += from_recovered
                        gain = max(0, int(after_ikz[0]) - int(before_ikz[0]))
                        if action == "GATE_PORTAL" and _selected_resolved(step, winner=-1):
                            counts["hydromancy_selected"] += 1
                            counts["hydromancy_untap_events"] += int(gain > 0)
                            counts["hydromancy_ikz_readied"] += gain
                            ordinary_budget = max(ordinary_budget, int(before_ikz[0]))
                            recovered_budget += gain
                        elif gain:
                            ordinary_budget += gain
            for sequence in _sequence_rows(game, player, metadata):
                if sequence["element"] == element:
                    prefix = str(sequence["id"])
                    for field in ("eligible", "completed", "converted"):
                        counts[f"{prefix}/{field}"] += int(sequence[field])
    return {
        "semantics_version": 1,
        "contexts": {key: {**value, "counts": dict(sorted(value["counts"].items()))} for key, value in sorted(contexts.items())},
        "interpretation": {
            "water": "resource readiness and spell effects; same-turn resource-to-spell lower bound, not total generated-token attribution",
            "earth": "defensive opportunities and observed leader HP restoration; Bobu needs an Earth loss plus healing inside its active window",
            "fire": "Zero self-damage, target damage and ATK increase followed by same-turn attack; Kagoro and Rushfire are additional lines, not substitutes for self-damage evidence",
            "lightning": "weapon deck support, attachment and effective damage while the destination remains equipped; not a marginal weapon payoff estimate",
            "deck_fit": "composition and same-leader sibling distance are diagnostics; causal gate/leader fit requires matched fixed-deck counterfactual probes",
            "novel_lines": "unregistered coherent lines remain admissible through trace review and matched payoff probes; named sequences are not reward targets",
            "limitations": [
                "physical card-copy and effect IDs unavailable",
                "deferred combat outcomes are unmeasured rather than inferred from eventual winner",
                "legal setup/control alternatives have no counterfactual value ranking",
                "HP restoration alone does not establish correct healing timing",
            ],
        },
    }


def build_strategy_descriptor(
    games: list[dict],
    *,
    label: str,
    checkpoint_sha256: str | None = None,
    payoff_cells: list[dict] | None = None,
) -> dict[str, object]:
    if not games:
        raise ValueError("At least one game is required")
    evaluated = evaluate_strategy_events(games)
    metadata = _metadata()
    deck_profiles = _deck_profile(games, metadata)
    payload: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluator": {
            "funnel_semantics_version": FUNNEL_SEMANTICS_VERSION,
            "sequence_semantics_version": SEQUENCE_SEMANTICS_VERSION,
            "denominator_policy": "actual legal semantic opportunities, including declined opportunities",
            "policy_input": False,
            "reward_input": False,
            "quality_claim": "observational support, not counterfactual action quality",
        },
        "games": len(games),
        "player_games": 2 * len(games),
        "deck": {
            "contexts": deck_profiles,
            "relationships": _deck_relationships(deck_profiles, metadata),
        },
        **evaluated,
        "elemental_strategy": _elemental_strategy(games, metadata),
        "payoff_vector": sorted(payoff_cells or [], key=lambda cell: str(cell["context_key"])),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["descriptor_id"] = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--payoff-cells", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    payoff_cells = None
    if args.payoff_cells is not None:
        raw = json.loads(args.payoff_cells.read_text(encoding="utf-8"))
        payoff_cells = raw if isinstance(raw, list) else raw["payoff_vector"]
    payload = build_strategy_descriptor(
        load_games(args.inputs),
        label=args.label,
        checkpoint_sha256=args.checkpoint_sha256,
        payoff_cells=payoff_cells,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
