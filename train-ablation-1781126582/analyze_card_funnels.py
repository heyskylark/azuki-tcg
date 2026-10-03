#!/usr/bin/env python3
"""Summarize per-card exposure and use from legal-action self-play traces."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from action import ActionType
from analyze_opportunity_rates import _card_at_hand, _legal_actions, _load_card_metadata
from analyze_selfplay_games import annotate_turns, load_games


PLAY_ACTIONS = frozenset((
    int(ActionType.PLAY_ENTITY_TO_GARDEN),
    int(ActionType.PLAY_ENTITY_TO_ALLEY),
    int(ActionType.ATTACH_WEAPON_FROM_HAND),
    int(ActionType.PLAY_SPELL_FROM_HAND),
))
PLAY_ACTION_NAMES = frozenset((
    "PLAY_ENTITY_TO_GARDEN",
    "PLAY_ENTITY_TO_ALLEY",
    "ATTACH_WEAPON_FROM_HAND",
    "PLAY_SPELL_FROM_HAND",
))
ENTITY_PLAY_NAMES = frozenset((
    "PLAY_ENTITY_TO_GARDEN",
    "PLAY_ENTITY_TO_ALLEY",
))
REALIZATION_ACTION_FIELDS = {
    "ATTACK": "attacker",
    "DECLARE_DEFENDER": "card",
    "GATE_PORTAL": "card",
    "ACTIVATE_GARDEN_OR_LEADER_ABILITY": "card",
    "ACTIVATE_ALLEY_ABILITY": "card",
}
MAIN_CARD_TYPES = frozenset(("ENTITY", "SPELL", "WEAPON"))


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _valid_card(code: object, metadata: dict[str, dict[str, object]]) -> str | None:
    if not isinstance(code, str):
        return None
    record = metadata.get(code)
    if record is None or record.get("card_type") not in MAIN_CARD_TYPES:
        return None
    return code


def summarize_card_funnels(games: list[dict], *, label: str) -> dict[str, object]:
    metadata = _load_card_metadata()
    counters: dict[str, Counter] = defaultdict(Counter)
    events: dict[str, dict[str, set[tuple]]] = defaultdict(lambda: defaultdict(set))
    seat_games = 0

    for game_index, raw_game in enumerate(games):
        game = annotate_turns(raw_game)
        winner = int(game["outcome"]["winner"])
        initial_hand: list[set[str] | None] = [None, None]
        observed: list[set[str]] = [set(), set()]
        later_drawn: list[set[str]] = [set(), set()]

        for player in range(2):
            seat_games += 1
            main = [
                code
                for raw_code in game["decks"][player]["main"]
                if (code := _valid_card(raw_code, metadata)) is not None
            ]
            for code, copies in Counter(main).items():
                counters[code]["drafted_copies"] += copies
                counters[code]["deck_seats"] += 1
                counters[code]["deck_wins"] += int(winner == player)

        for step_index, step in enumerate(game["steps"]):
            player = int(step["p"])
            turn = int(step.get("turn", 0))
            hand_codes = {
                code
                for raw_code in step.get("hand", [])
                if (code := _valid_card(raw_code, metadata)) is not None
            }
            if initial_hand[player] is None:
                initial_hand[player] = set(hand_codes)
                for code in hand_codes:
                    events[code]["opening_seats"].add((game_index, player))
            else:
                later_drawn[player].update(hand_codes - observed[player])
            observed[player].update(hand_codes)
            for code in hand_codes:
                counters[code]["in_hand_windows"] += 1
                events[code]["in_hand_turn_cards"].add(
                    (game_index, player, turn, code)
                )

            legal = _legal_actions(step)
            selected = tuple(int(value) for value in step["a"])
            if selected not in legal:
                raise ValueError(
                    "selected action absent from legal trace: "
                    f"game={game_index}, step={step_index}, selected={selected}"
                )
            legal_codes = {
                code
                for row in legal
                if row[0] in PLAY_ACTIONS
                if (code := _valid_card(_card_at_hand(step, row[1]), metadata))
                is not None
            }
            for code in legal_codes:
                counters[code]["legal_play_windows"] += 1
                events[code]["legal_play_turn_cards"].add(
                    (game_index, player, turn, code)
                )

            selected_type = str(step["d"]["t"])
            selected_code = _valid_card(step["d"].get("card"), metadata)
            if selected_type in PLAY_ACTION_NAMES and selected_code is not None:
                counters[selected_code]["selected_plays"] += 1
                events[selected_code]["selected_play_seats"].add(
                    (game_index, player)
                )
                if selected_type not in ENTITY_PLAY_NAMES:
                    counters[selected_code]["realized_effect_events"] += 1
                    events[selected_code]["realized_effect_seats"].add(
                        (game_index, player)
                    )

            realization_field = REALIZATION_ACTION_FIELDS.get(selected_type)
            realization_code = (
                _valid_card(step["d"].get(realization_field), metadata)
                if realization_field is not None
                else None
            )
            if realization_code is not None:
                counters[realization_code]["realized_effect_events"] += 1
                events[realization_code]["realized_effect_seats"].add(
                    (game_index, player)
                )

            source_code = _valid_card(step.get("ability_src"), metadata)
            if (
                source_code is not None
                and selected_type not in {"NOOP", *REALIZATION_ACTION_FIELDS}
            ):
                counters[source_code]["realized_effect_events"] += 1
                events[source_code]["realized_effect_seats"].add(
                    (game_index, player)
                )

        for player in range(2):
            for code in observed[player]:
                events[code]["observed_in_hand_seats"].add((game_index, player))
            for code in later_drawn[player]:
                events[code]["drawn_after_opening_seats"].add((game_index, player))

    cards: dict[str, dict[str, object]] = {}
    for code in sorted(counters):
        stats = counters[code]
        card_events = events[code]
        for name in (
            "opening_seats",
            "observed_in_hand_seats",
            "drawn_after_opening_seats",
            "in_hand_turn_cards",
            "legal_play_turn_cards",
            "selected_play_seats",
            "realized_effect_seats",
        ):
            stats[name] = len(card_events[name])
        record = metadata[code]
        cards[code] = {
            "name": record.get("name", code),
            "card_type": record.get("card_type"),
            "element": record.get("element"),
            "ikz_cost": record.get("ikz_cost"),
            **dict(sorted(stats.items())),
            "deck_inclusion_rate": _ratio(stats["deck_seats"], seat_games),
            "copies_per_included_deck": _ratio(
                stats["drafted_copies"], stats["deck_seats"]
            ),
            "observed_given_included_deck": _ratio(
                stats["observed_in_hand_seats"], stats["deck_seats"]
            ),
            "drawn_after_opening_given_included_deck": _ratio(
                stats["drawn_after_opening_seats"], stats["deck_seats"]
            ),
            "legal_turn_card_given_in_hand_turn_card": _ratio(
                stats["legal_play_turn_cards"], stats["in_hand_turn_cards"]
            ),
            "selected_play_given_legal_window": _ratio(
                stats["selected_plays"], stats["legal_play_windows"]
            ),
            "realized_seat_given_selected_play_seat": _ratio(
                stats["realized_effect_seats"], stats["selected_play_seats"]
            ),
            "deck_win_rate": _ratio(stats["deck_wins"], stats["deck_seats"]),
        }

    stages = (
        ("drafted", "deck_seats"),
        ("opening", "opening_seats"),
        ("drawn_after_opening", "drawn_after_opening_seats"),
        ("observed", "observed_in_hand_seats"),
        ("legal", "legal_play_windows"),
        ("selected", "selected_plays"),
        ("realized", "realized_effect_seats"),
    )
    coverage = {
        name: sum(int(card.get(field, 0)) > 0 for card in cards.values())
        for name, field in stages
    }
    return {
        "schema_version": 1,
        "label": label,
        "games": len(games),
        "seat_games": seat_games,
        "coverage": coverage,
        "cards": cards,
        "definitions": {
            "drawn_after_opening": (
                "Card code first observed after that seat's first logged hand; "
                "duplicates already visible in the opening hand are not recoverable."
            ),
            "legal": "Decision windows with at least one legal hand-play row for the card.",
            "selected": "Entity, spell, or weapon hand-play actions selected for the card.",
            "realized": (
                "A selected spell/weapon, or a card later used to attack, defend, "
                "portal, activate an ability, or drive an ability follow-up."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_card_funnels(load_games(args.inputs), label=args.label)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{args.label}] games={result['games']} "
        + " ".join(
            f"{name}={value}" for name, value in result["coverage"].items()
        )
    )
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
