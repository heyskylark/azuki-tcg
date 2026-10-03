#!/usr/bin/env python3
"""Opportunity-normalized strategy diagnostics for legal-mask self-play logs.

Inputs must be produced by play_selfplay_games.py --log-legal-actions. The
report separates card/deck availability from legal opportunities and selected
actions, and reconstructs the two overlapping action bonuses in the production
recipe: early tempo and flat portal Gate Power.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from action import ActionType
from analyze_selfplay_games import ELEMENT_OF_GATE, annotate_turns, load_games


REPO_ROOT = Path(__file__).resolve().parents[1]
CARD_METADATA_PATH = REPO_ROOT / "python" / "config" / "policy_card_metadata_v1.json"
GARDEN_CONDITIONAL_ON_PLAY = {
    "AZK01-063": "Enzo",
    "AZK01-080": "Bladebound Ally",
    "AZK01-093": "Naiyara the Tideweaver",
    "AZK01-104": "Sanzu's Envoy",
    "STT03-011": "Koyama Farm Plowman",
}
EARLY_TEMPO_ACTIONS = {
    "PLAY_ENTITY_TO_GARDEN",
    "PLAY_ENTITY_TO_ALLEY",
    "PLAY_SPELL_FROM_HAND",
    "ATTACH_WEAPON_FROM_HAND",
    "GATE_PORTAL",
    "ACTIVATE_GARDEN_OR_LEADER_ABILITY",
    "ACTIVATE_ALLEY_ABILITY",
    "CONFIRM_ABILITY",
    "ATTACK",
}
GARDEN_SIZE = 5
EARLY_TEMPO_BONUS = 0.1
EARLY_TEMPO_CAP = 4
EARLY_TEMPO_GLOBAL_TURNS = 4
PORTAL_GP_BONUS = 0.3
MATURE_SHAPING_FLOOR = 0.15
EARLY_TEMPO_DEDUP_EXCLUDED_ACTIONS = {
    "GATE_PORTAL",
    "ACTIVATE_GARDEN_OR_LEADER_ABILITY",
    "ACTIVATE_ALLEY_ABILITY",
    "CONFIRM_ABILITY",
}


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _load_card_metadata() -> dict[str, dict[str, object]]:
    payload = json.loads(CARD_METADATA_PATH.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Invalid card metadata: {CARD_METADATA_PATH}")
    return {
        str(record["card_code"]): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("card_code"), str)
    }


def _legal_actions(step: dict) -> list[tuple[int, int, int, int]]:
    raw = step.get("legal")
    if not isinstance(raw, list):
        raise ValueError(
            "Input is missing legal-action traces; rerun play_selfplay_games.py "
            "with --log-legal-actions"
        )
    actions: list[tuple[int, int, int, int]] = []
    for entry in raw:
        if not isinstance(entry, list) or len(entry) != 4:
            raise ValueError(f"Malformed legal action: {entry!r}")
        actions.append(tuple(int(value) for value in entry))
    return actions


def _groups_for(game: dict, player: int) -> tuple[str, str, str]:
    gate = str(game["decks"][player]["gate"])
    element = ELEMENT_OF_GATE[gate]
    return "ALL", f"ELEMENT/{element}", f"GATE/{gate}"


def _card_at_hand(step: dict, hand_index: int) -> str | None:
    hand = step.get("hand", [])
    if not isinstance(hand, list) or hand_index < 0 or hand_index >= len(hand):
        return None
    card = hand[hand_index]
    return str(card) if isinstance(card, str) else None


def _ability_followup(steps: list[dict], start_index: int, source: str) -> tuple[bool, bool, bool]:
    actor = int(steps[start_index]["p"])
    saw_source = False
    optional_offer = False
    engaged = False
    for next_step in steps[start_index + 1 : start_index + 13]:
        if int(next_step["p"]) != actor:
            break
        next_source = next_step.get("ability_src")
        if next_source != source:
            break
        saw_source = True
        legal = _legal_actions(next_step)
        legal_types = {entry[0] for entry in legal}
        optional_offer = optional_offer or (
            int(ActionType.NOOP) in legal_types and len(legal_types - {int(ActionType.NOOP)}) > 0
        )
        if next_step["d"]["t"] != "NOOP":
            engaged = True
    return saw_source, optional_offer, engaged


def _summarize_counter(counter: Counter) -> dict[str, object]:
    out: dict[str, object] = {key: value for key, value in sorted(counter.items())}
    out["win_rate"] = _ratio(counter["wins"], counter["seat_games"])
    out["spell_slot_share"] = _ratio(counter["deck_spell_slots"], counter["deck_main_slots"])
    out["spell_observed_in_hand_seat_rate"] = _ratio(
        counter["spell_observed_in_hand_seats"], counter["seat_games"]
    )
    out["spell_legal_per_in_hand_window"] = _ratio(
        counter["spell_legal_windows"], counter["spell_in_hand_windows"]
    )
    out["spell_legal_per_in_hand_turn_card"] = _ratio(
        counter["spell_legal_turn_cards"], counter["spell_in_hand_turn_cards"]
    )
    out["spell_selected_per_legal_window"] = _ratio(
        counter["spell_selected"], counter["spell_legal_windows"]
    )
    out["spell_selected_per_legal_turn_card"] = _ratio(
        counter["spell_selected_turn_cards"], counter["spell_legal_turn_cards"]
    )
    out["main_spell_selected_per_legal_window"] = _ratio(
        counter["main_spell_selected"], counter["main_spell_legal_windows"]
    )
    out["response_spell_selected_per_legal_window"] = _ratio(
        counter["response_spell_selected"], counter["response_spell_legal_windows"]
    )
    out["direct_garden_per_comparable_window"] = _ratio(
        counter["direct_garden_from_comparable"], counter["comparable_destination_windows"]
    )
    out["garden_share_when_comparable_entity_selected"] = _ratio(
        counter["direct_garden_from_comparable"], counter["comparable_entity_choices"]
    )
    out["leader_ability_selected_per_legal_window"] = _ratio(
        counter["leader_ability_selected"], counter["leader_ability_legal_windows"]
    )
    out["garden_ability_selected_per_legal_window"] = _ratio(
        counter["garden_ability_selected"], counter["garden_ability_legal_windows"]
    )
    out["alley_ability_selected_per_legal_window"] = _ratio(
        counter["alley_ability_selected"], counter["alley_ability_legal_windows"]
    )
    return out


def summarize_games(
    games: list[dict],
    *,
    label: str,
    early_tempo_bonus: float = EARLY_TEMPO_BONUS,
    early_tempo_cap: int = EARLY_TEMPO_CAP,
    portal_gp_bonus: float = PORTAL_GP_BONUS,
    shaping_scale: float = MATURE_SHAPING_FLOOR,
    early_tempo_dedup_portal_abilities: bool = False,
) -> dict[str, object]:
    if early_tempo_bonus < 0.0:
        raise ValueError("early_tempo_bonus must be nonnegative")
    if early_tempo_cap < 0:
        raise ValueError("early_tempo_cap must be nonnegative")
    if portal_gp_bonus < 0.0:
        raise ValueError("portal_gp_bonus must be nonnegative")
    if shaping_scale < 0.0:
        raise ValueError("shaping_scale must be nonnegative")
    metadata = _load_card_metadata()
    games = [annotate_turns(game) for game in games]
    grouped: dict[str, Counter] = defaultdict(Counter)
    group_events: dict[str, dict[str, set[tuple]]] = defaultdict(
        lambda: defaultdict(set)
    )
    garden_cards: dict[str, Counter] = {
        code: Counter() for code in GARDEN_CONDITIONAL_ON_PLAY
    }
    garden_events: dict[str, dict[str, set[tuple]]] = {
        code: defaultdict(set) for code in GARDEN_CONDITIONAL_ON_PLAY
    }
    destination_by_card: dict[str, Counter] = defaultdict(Counter)
    rider_by_gate: dict[str, Counter] = defaultdict(Counter)
    leaders_by_element: dict[str, Counter] = defaultdict(Counter)
    leaders_by_gate: dict[str, Counter] = defaultdict(Counter)
    leader_outcomes_by_element: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    reward = Counter()
    reward_by_action = Counter()

    for game_index, game in enumerate(games):
        winner = int(game["outcome"]["winner"])
        seen_by_player = [set(), set()]
        for player in range(2):
            groups = _groups_for(game, player)
            main = [str(code) for code in game["decks"][player]["main"]]
            gate = str(game["decks"][player]["gate"])
            leader = str(game["decks"][player]["leader"])
            element = ELEMENT_OF_GATE[gate]
            leaders_by_element[element][leader] += 1
            leaders_by_gate[gate][leader] += 1
            leader_outcomes_by_element[element][leader]["seat_games"] += 1
            leader_outcomes_by_element[element][leader]["wins"] += int(
                winner == player
            )
            spell_slots = sum(
                1 for code in main if metadata.get(code, {}).get("card_type") == "SPELL"
            )
            for group in groups:
                grouped[group]["seat_games"] += 1
                grouped[group]["wins"] += int(winner == player)
                grouped[group]["deck_main_slots"] += len(main)
                grouped[group]["deck_spell_slots"] += spell_slots
            main_counts = Counter(main)
            for code in GARDEN_CONDITIONAL_ON_PLAY:
                garden_cards[code]["drafted_copies"] += main_counts[code]
                garden_cards[code]["seat_decks"] += int(main_counts[code] > 0)

        early_count_by_turn: Counter = Counter()
        early_alley_slots: set[tuple[int, int, int]] = set()
        steps = game["steps"]
        for step_index, step in enumerate(steps):
            player = int(step["p"])
            groups = _groups_for(game, player)
            selected_type = str(step["d"]["t"])
            selected_raw = tuple(int(value) for value in step["a"])
            legal = _legal_actions(step)
            if selected_raw not in legal:
                raise ValueError(
                    f"Selected action is absent from legal trace: game={game_index}, "
                    f"step={step_index}, selected={selected_raw}"
                )
            phase = str(step["ph"])
            turn = int(step.get("turn", 0))
            seen_by_player[player].update(str(code) for code in step.get("hand", []))

            for group in groups:
                grouped[group]["decision_windows"] += 1
                grouped[group][f"selected/{selected_type}"] += 1

            hand_spell_codes = {
                str(code)
                for code in step.get("hand", [])
                if metadata.get(str(code), {}).get("card_type") == "SPELL"
            }
            phase_key = "response" if phase == "RESPONSE" else "main"
            if hand_spell_codes:
                for group in groups:
                    grouped[group]["spell_in_hand_windows"] += 1
                    for code in hand_spell_codes:
                        group_events[group]["spell_in_hand_turn_cards"].add(
                            (game_index, player, turn, phase_key, code)
                        )

            spell_rows = [row for row in legal if row[0] == int(ActionType.PLAY_SPELL_FROM_HAND)]
            if spell_rows:
                spell_codes = {
                    code
                    for row in spell_rows
                    if (code := _card_at_hand(step, row[1])) is not None
                }
                for group in groups:
                    grouped[group]["spell_legal_windows"] += 1
                    grouped[group][f"{phase_key}_spell_legal_windows"] += 1
                    grouped[group]["spell_legal_rows"] += len(spell_rows)
                    for code in spell_codes:
                        group_events[group]["spell_legal_turn_cards"].add(
                            (game_index, player, turn, phase_key, code)
                        )
                if selected_type == "PLAY_SPELL_FROM_HAND":
                    selected_code = str(step["d"].get("card", "?"))
                    for group in groups:
                        grouped[group]["spell_selected"] += 1
                        grouped[group][f"{phase_key}_spell_selected"] += 1
                        group_events[group]["spell_selected_turn_cards"].add(
                            (game_index, player, turn, phase_key, selected_code)
                        )

            garden_rows = [row for row in legal if row[0] == int(ActionType.PLAY_ENTITY_TO_GARDEN)]
            alley_rows = [row for row in legal if row[0] == int(ActionType.PLAY_ENTITY_TO_ALLEY)]
            garden_hand = {row[1] for row in garden_rows}
            alley_hand = {row[1] for row in alley_rows}
            comparable_hand = garden_hand & alley_hand
            if comparable_hand:
                for group in groups:
                    grouped[group]["comparable_destination_windows"] += 1
                if selected_type in {"PLAY_ENTITY_TO_GARDEN", "PLAY_ENTITY_TO_ALLEY"}:
                    selected_hand = selected_raw[1]
                    if selected_hand in comparable_hand:
                        selected_code = _card_at_hand(step, selected_hand)
                        if selected_code is not None:
                            destination_by_card[selected_code]["choices"] += 1
                            destination_by_card[selected_code][
                                "garden" if selected_type == "PLAY_ENTITY_TO_GARDEN" else "alley"
                            ] += 1
                        for group in groups:
                            grouped[group]["comparable_entity_choices"] += 1
                            if selected_type == "PLAY_ENTITY_TO_GARDEN":
                                grouped[group]["direct_garden_from_comparable"] += 1

            garden_legal_codes = {
                code
                for row in garden_rows
                if (code := _card_at_hand(step, row[1])) is not None
            }
            for code in garden_legal_codes & GARDEN_CONDITIONAL_ON_PLAY.keys():
                garden_cards[code]["legal_garden_windows"] += 1
                garden_events[code]["legal_garden_turn_cards"].add(
                    (game_index, player, turn, code)
                )

            if selected_type in {"PLAY_ENTITY_TO_GARDEN", "PLAY_ENTITY_TO_ALLEY"}:
                selected_code = str(step["d"].get("card", "?"))
                if selected_code in GARDEN_CONDITIONAL_ON_PLAY:
                    destination = "garden_plays" if selected_type == "PLAY_ENTITY_TO_GARDEN" else "alley_plays"
                    garden_cards[selected_code][destination] += 1
                    if selected_raw[1] in comparable_hand:
                        garden_cards[selected_code]["comparable_destination_choices"] += 1
                        if selected_type == "PLAY_ENTITY_TO_GARDEN":
                            garden_cards[selected_code]["comparable_garden_choices"] += 1
                    if selected_type == "PLAY_ENTITY_TO_GARDEN":
                        offered, optional, engaged = _ability_followup(
                            steps, step_index, selected_code
                        )
                        garden_cards[selected_code]["interactive_effect_offers"] += int(offered)
                        garden_cards[selected_code]["optional_effect_offers"] += int(optional)
                        garden_cards[selected_code]["interactive_effect_engaged"] += int(engaged)

            leader_rows = [
                row
                for row in legal
                if row[0] == int(ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
                and row[1] == GARDEN_SIZE
            ]
            garden_ability_rows = [
                row
                for row in legal
                if row[0] == int(ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
                and row[1] < GARDEN_SIZE
            ]
            alley_ability_rows = [
                row for row in legal if row[0] == int(ActionType.ACTIVATE_ALLEY_ABILITY)
            ]
            for group in groups:
                grouped[group]["leader_ability_legal_windows"] += int(bool(leader_rows))
                grouped[group]["garden_ability_legal_windows"] += int(bool(garden_ability_rows))
                grouped[group]["alley_ability_legal_windows"] += int(bool(alley_ability_rows))
                grouped[group]["leader_ability_selected"] += int(
                    selected_type == "ACTIVATE_GARDEN_OR_LEADER_ABILITY"
                    and selected_raw[1] == GARDEN_SIZE
                )
                grouped[group]["garden_ability_selected"] += int(
                    selected_type == "ACTIVATE_GARDEN_OR_LEADER_ABILITY"
                    and selected_raw[1] < GARDEN_SIZE
                )
                grouped[group]["alley_ability_selected"] += int(
                    selected_type == "ACTIVATE_ALLEY_ABILITY"
                )

            early_paid = False
            early_key = (player, turn)
            if (
                early_tempo_bonus > 0.0
                and selected_type in EARLY_TEMPO_ACTIONS
                and 0 < turn <= EARLY_TEMPO_GLOBAL_TURNS
                and (early_tempo_cap == 0 or early_count_by_turn[early_key] < early_tempo_cap)
                and not (
                    early_tempo_dedup_portal_abilities
                    and selected_type in EARLY_TEMPO_DEDUP_EXCLUDED_ACTIONS
                )
            ):
                early_count_by_turn[early_key] += 1
                early_paid = True
                reward["early_tempo_paid_actions"] += 1
                reward["early_tempo_gross_milli"] += round(early_tempo_bonus * 1000)
                reward_by_action[selected_type] += 1

            if selected_type == "PLAY_ENTITY_TO_ALLEY" and early_paid:
                early_alley_slots.add((player, turn, selected_raw[2]))

            if selected_type == "GATE_PORTAL":
                reward["portals"] += 1
                portaled_code = str(step["d"].get("card", "?"))
                gate_points_raw = metadata.get(portaled_code, {}).get("gate_points", 0)
                gate_points = int(gate_points_raw) if isinstance(gate_points_raw, int) else 0
                portal_gp = portal_gp_bonus * min(max(gate_points, 0), 4) / 4.0
                reward["portal_gp_gross_milli"] += round(portal_gp * 1000)
                if portal_gp > 0 and early_paid:
                    reward["portal_actions_paid_both"] += 1
                    reward["portal_both_gross_milli"] += round(
                        (early_tempo_bonus + portal_gp) * 1000
                    )
                    if (player, turn, selected_raw[1]) in early_alley_slots:
                        reward["alley_portal_lines_paid_three_terms"] += 1
                        reward["alley_portal_three_term_gross_milli"] += round(
                            (2 * early_tempo_bonus + portal_gp) * 1000
                        )

                gate = str(game["decks"][player]["gate"])
                rider_by_gate[gate]["portals"] += 1
                offered, optional, engaged = _ability_followup(steps, step_index, gate)
                rider_by_gate[gate]["interactive_offers"] += int(offered)
                rider_by_gate[gate]["optional_offers"] += int(optional)
                rider_by_gate[gate]["engaged"] += int(engaged)

        for player in range(2):
            groups = _groups_for(game, player)
            observed_spell = any(
                metadata.get(code, {}).get("card_type") == "SPELL"
                for code in seen_by_player[player]
            )
            for group in groups:
                grouped[group]["spell_observed_in_hand_seats"] += int(observed_spell)
            for code in GARDEN_CONDITIONAL_ON_PLAY:
                garden_cards[code]["observed_in_hand_seats"] += int(
                    code in seen_by_player[player]
                )

    for group, events in group_events.items():
        grouped[group]["spell_in_hand_turn_cards"] = len(
            events["spell_in_hand_turn_cards"]
        )
        grouped[group]["spell_legal_turn_cards"] = len(events["spell_legal_turn_cards"])
        grouped[group]["spell_selected_turn_cards"] = len(events["spell_selected_turn_cards"])

    garden_out: dict[str, object] = {}
    for code, stats in garden_cards.items():
        stats["legal_garden_turn_cards"] = len(
            garden_events[code]["legal_garden_turn_cards"]
        )
        payload = {key: value for key, value in sorted(stats.items())}
        payload["name"] = GARDEN_CONDITIONAL_ON_PLAY[code]
        payload["garden_share_when_comparable_selected"] = _ratio(
            stats["comparable_garden_choices"], stats["comparable_destination_choices"]
        )
        payload["effect_engagement_per_offer"] = _ratio(
            stats["interactive_effect_engaged"], stats["interactive_effect_offers"]
        )
        garden_out[code] = payload

    rider_out: dict[str, object] = {}
    for gate, stats in sorted(rider_by_gate.items()):
        payload = {key: value for key, value in sorted(stats.items())}
        payload["offer_per_portal"] = _ratio(stats["interactive_offers"], stats["portals"])
        payload["engagement_per_offer"] = _ratio(stats["engaged"], stats["interactive_offers"])
        rider_out[gate] = payload

    destination_out = {}
    for code, stats in sorted(
        destination_by_card.items(), key=lambda item: (-item[1]["choices"], item[0])
    ):
        card_record = metadata.get(code, {})
        destination_out[code] = {
            "name": card_record.get("name", code),
            "choices": stats["choices"],
            "garden": stats["garden"],
            "alley": stats["alley"],
            "garden_share": _ratio(stats["garden"], stats["choices"]),
        }

    seat_games = max(1, grouped["ALL"]["seat_games"])
    games_count = max(1, len(games))
    early_gross = reward["early_tempo_gross_milli"] / 1000.0
    portal_gross = reward["portal_gp_gross_milli"] / 1000.0
    for key in (
        "early_tempo_paid_actions",
        "early_tempo_gross_milli",
        "portals",
        "portal_gp_gross_milli",
        "portal_actions_paid_both",
        "portal_both_gross_milli",
        "alley_portal_lines_paid_three_terms",
        "alley_portal_three_term_gross_milli",
    ):
        reward.setdefault(key, 0)
    reward_out = {}
    for key, value in sorted(reward.items()):
        output_key = key.removesuffix("_milli") if key.endswith("_milli") else key
        reward_out[output_key] = value / 1000.0 if key.endswith("_milli") else value
    reward_out.update(
        {
            "early_tempo_gross_per_game": round(early_gross / games_count, 6),
            "portal_gp_gross_per_game": round(portal_gross / games_count, 6),
            "combined_gross_per_game": round((early_gross + portal_gross) / games_count, 6),
            "combined_gross_per_seat_game": round(
                (early_gross + portal_gross) / seat_games, 6
            ),
            "combined_at_mature_floor_per_game": round(
                shaping_scale * (early_gross + portal_gross) / games_count,
                6,
            ),
            "early_tempo_paid_by_action": dict(sorted(reward_by_action.items())),
            "config": {
                "early_tempo_bonus": early_tempo_bonus,
                "early_tempo_cap": early_tempo_cap,
                "early_tempo_dedup_portal_abilities": (
                    early_tempo_dedup_portal_abilities
                ),
                "portal_gp_bonus": portal_gp_bonus,
                "shaping_scale": shaping_scale,
            },
        }
    )

    return {
        "label": label,
        "n_games": len(games),
        "groups": {
            group: _summarize_counter(stats)
            for group, stats in sorted(grouped.items())
        },
        "garden_conditional_on_play": garden_out,
        "comparable_destination_by_card": destination_out,
        "gate_rider_opportunities": rider_out,
        "leaders_by_element": {
            element: dict(sorted(counts.items()))
            for element, counts in sorted(leaders_by_element.items())
        },
        "leaders_by_gate": {
            gate: dict(sorted(counts.items()))
            for gate, counts in sorted(leaders_by_gate.items())
        },
        "leader_outcomes_by_element": {
            element: {
                leader: {
                    "seat_games": stats["seat_games"],
                    "wins": stats["wins"],
                    "win_rate": _ratio(stats["wins"], stats["seat_games"]),
                }
                for leader, stats in sorted(leaders.items())
            }
            for element, leaders in sorted(leader_outcomes_by_element.items())
        },
        "reward_overlap": reward_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--early-tempo-bonus", type=float, default=EARLY_TEMPO_BONUS)
    parser.add_argument("--early-tempo-cap", type=int, default=EARLY_TEMPO_CAP)
    parser.add_argument("--portal-gp-bonus", type=float, default=PORTAL_GP_BONUS)
    parser.add_argument("--shaping-scale", type=float, default=MATURE_SHAPING_FLOOR)
    parser.add_argument(
        "--early-tempo-dedup-portal-abilities",
        action="store_true",
    )
    args = parser.parse_args()

    games = load_games(args.inputs)
    result = summarize_games(
        games,
        label=args.label,
        early_tempo_bonus=args.early_tempo_bonus,
        early_tempo_cap=args.early_tempo_cap,
        portal_gp_bonus=args.portal_gp_bonus,
        shaping_scale=args.shaping_scale,
        early_tempo_dedup_portal_abilities=(
            args.early_tempo_dedup_portal_abilities
        ),
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    overall = result["groups"]["ALL"]
    water = result["groups"].get("ELEMENT/WATER", {})
    print(
        f"[{args.label}] games={result['n_games']} "
        f"spell_slots={overall.get('spell_slot_share')} "
        f"spell_given_legal={overall.get('spell_selected_per_legal_window')} "
        f"water_spell_given_legal={water.get('spell_selected_per_legal_window')} "
        f"garden_given_comparable={overall.get('garden_share_when_comparable_entity_selected')}"
    )
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
