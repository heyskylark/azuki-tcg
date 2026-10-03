#!/usr/bin/env python3
"""Analyze paired supplied-deck battles for the frozen p2930 policy."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from action import ActionType
from analyze_opportunity_rates import (
    GARDEN_CONDITIONAL_ON_PLAY,
    GARDEN_SIZE,
    _ability_followup,
    _card_at_hand,
    _legal_actions,
    _load_card_metadata,
    _ratio,
)
from analyze_selfplay_games import ELEMENT_OF_GATE, annotate_turns, load_games


PROACTIVE_HAND_ACTIONS = frozenset(
    (
        int(ActionType.PLAY_ENTITY_TO_GARDEN),
        int(ActionType.PLAY_ENTITY_TO_ALLEY),
        int(ActionType.PLAY_SPELL_FROM_HAND),
        int(ActionType.ATTACH_WEAPON_FROM_HAND),
    )
)


def _mean(counter: Counter, total_key: str, count_key: str) -> float | None:
    return _ratio(counter[total_key], counter[count_key])


def _group_summary(counter: Counter) -> dict[str, object]:
    return {
        **dict(sorted(counter.items())),
        "score": _ratio(counter["score_milli"], 1000 * counter["games"]),
        "mean_battle_steps": _mean(counter, "battle_steps", "games"),
        "mean_turns": _mean(counter, "turns", "games"),
        "spell_slot_share": _ratio(counter["deck_spell_slots"], counter["deck_main_slots"]),
        "main_spell_selected_per_legal_window": _ratio(
            counter["main_spell_selected"], counter["main_spell_legal_windows"]
        ),
        "response_spell_selected_per_legal_window": _ratio(
            counter["response_spell_selected"], counter["response_spell_legal_windows"]
        ),
        "garden_share_when_comparable_selected": _ratio(
            counter["direct_garden_from_comparable"], counter["comparable_entity_choices"]
        ),
        "leader_ability_selected_per_legal_window": _ratio(
            counter["leader_ability_selected"], counter["leader_ability_legal_windows"]
        ),
        "first_main_no_hand_deploy_rate": _ratio(
            counter["first_main_no_hand_deploy"], counter["first_main_observed"]
        ),
        "first_two_mains_both_no_hand_deploy_rate": _ratio(
            counter["first_two_mains_both_no_hand_deploy"],
            counter["first_two_mains_observed"],
        ),
        "spell_observed_in_hand_game_rate": _ratio(
            counter["spell_observed_in_hand_games"], counter["games"]
        ),
        "spell_plays_per_game": _ratio(counter["selected/PLAY_SPELL_FROM_HAND"], counter["games"]),
        "alley_plays_per_game": _ratio(counter["selected/PLAY_ENTITY_TO_ALLEY"], counter["games"]),
        "garden_plays_per_game": _ratio(counter["selected/PLAY_ENTITY_TO_GARDEN"], counter["games"]),
        "portals_per_game": _ratio(counter["selected/GATE_PORTAL"], counter["games"]),
        "attacks_per_game": _ratio(counter["selected/ATTACK"], counter["games"]),
    }


def _summarize_arm(games: list[dict], metadata: dict[str, dict[str, object]]) -> dict:
    grouped: dict[str, Counter] = defaultdict(Counter)
    garden_cards = {code: Counter() for code in GARDEN_CONDITIONAL_ON_PLAY}
    garden_legal_events = {code: set() for code in GARDEN_CONDITIONAL_ON_PLAY}
    riders: dict[str, Counter] = defaultdict(Counter)
    by_reference: dict[str, Counter] = defaultdict(Counter)

    for game_index, raw_game in enumerate(games):
        game = annotate_turns(raw_game)
        context = game["counterfactual"]
        player = int(context["candidate_seat"])
        gate = str(context["target_gate"])
        element = ELEMENT_OF_GATE[gate]
        groups = ("ALL", f"ELEMENT/{element}", f"GATE/{gate}")
        score = float(context["candidate_score"])
        main = [str(code) for code in game["decks"][player]["main"]]
        spell_slots = sum(
            metadata.get(code, {}).get("card_type") == "SPELL" for code in main
        )
        for group in groups:
            grouped[group]["games"] += 1
            grouped[group]["score_milli"] += round(1000 * score)
            grouped[group]["battle_steps"] += int(game["battle_steps"])
            grouped[group]["turns"] += int(game["n_turns"])
            grouped[group]["timeouts"] += int(bool(game["outcome"].get("truncated")))
            grouped[group]["deck_main_slots"] += len(main)
            grouped[group]["deck_spell_slots"] += spell_slots
        reference = str(context["reference_label"])
        by_reference[reference]["games"] += 1
        by_reference[reference]["score_milli"] += round(1000 * score)
        for code, quantity in Counter(main).items():
            if code in garden_cards:
                garden_cards[code]["drafted_copies"] += quantity
                garden_cards[code]["game_decks"] += 1

        seen: set[str] = set()
        first_main_deploy: dict[int, bool] = {}
        for step_index, step in enumerate(game["steps"]):
            if int(step["p"]) != player:
                continue
            selected = tuple(int(value) for value in step["a"])
            legal = _legal_actions(step)
            if selected not in legal:
                raise ValueError(
                    f"Selected action missing from legal trace: game={game_index} "
                    f"step={step_index} selected={selected}"
                )
            selected_type = str(step["d"]["t"])
            phase_key = "response" if step["ph"] == "RESPONSE" else "main"
            turn = int(step.get("turn", 0))
            seen.update(str(code) for code in step.get("hand", []))
            for group in groups:
                grouped[group][f"selected/{selected_type}"] += 1

            if (
                step["ph"] == "MAIN"
                and int(step.get("turn_owner", -1)) == player
                and turn not in first_main_deploy
            ):
                first_main_deploy[turn] = any(row[0] in PROACTIVE_HAND_ACTIONS for row in legal)

            spell_rows = [
                row for row in legal if row[0] == int(ActionType.PLAY_SPELL_FROM_HAND)
            ]
            if spell_rows:
                for group in groups:
                    grouped[group][f"{phase_key}_spell_legal_windows"] += 1
                if selected_type == "PLAY_SPELL_FROM_HAND":
                    for group in groups:
                        grouped[group][f"{phase_key}_spell_selected"] += 1

            garden_rows = [
                row for row in legal if row[0] == int(ActionType.PLAY_ENTITY_TO_GARDEN)
            ]
            alley_rows = [
                row for row in legal if row[0] == int(ActionType.PLAY_ENTITY_TO_ALLEY)
            ]
            comparable_hand = {row[1] for row in garden_rows} & {
                row[1] for row in alley_rows
            }
            if (
                selected_type in {"PLAY_ENTITY_TO_GARDEN", "PLAY_ENTITY_TO_ALLEY"}
                and selected[1] in comparable_hand
            ):
                for group in groups:
                    grouped[group]["comparable_entity_choices"] += 1
                    grouped[group]["direct_garden_from_comparable"] += int(
                        selected_type == "PLAY_ENTITY_TO_GARDEN"
                    )

            garden_legal_codes = {
                code
                for row in garden_rows
                if (code := _card_at_hand(step, row[1])) is not None
            }
            for code in garden_legal_codes & GARDEN_CONDITIONAL_ON_PLAY.keys():
                garden_cards[code]["legal_garden_windows"] += 1
                garden_legal_events[code].add((game_index, turn, code))
            if selected_type in {"PLAY_ENTITY_TO_GARDEN", "PLAY_ENTITY_TO_ALLEY"}:
                code = str(step["d"].get("card", "?"))
                if code in garden_cards:
                    garden_cards[code][
                        "garden_plays"
                        if selected_type == "PLAY_ENTITY_TO_GARDEN"
                        else "alley_plays"
                    ] += 1
                    if selected[1] in comparable_hand:
                        garden_cards[code]["comparable_choices"] += 1
                        garden_cards[code]["comparable_garden"] += int(
                            selected_type == "PLAY_ENTITY_TO_GARDEN"
                        )
                    if selected_type == "PLAY_ENTITY_TO_GARDEN":
                        offered, optional, engaged = _ability_followup(
                            game["steps"], step_index, code
                        )
                        garden_cards[code]["interactive_offers"] += int(offered)
                        garden_cards[code]["optional_offers"] += int(optional)
                        garden_cards[code]["interactive_engaged"] += int(engaged)

            leader_rows = [
                row
                for row in legal
                if row[0] == int(ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
                and row[1] == GARDEN_SIZE
            ]
            for group in groups:
                grouped[group]["leader_ability_legal_windows"] += int(bool(leader_rows))
                grouped[group]["leader_ability_selected"] += int(
                    selected_type == "ACTIVATE_GARDEN_OR_LEADER_ABILITY"
                    and selected[1] == GARDEN_SIZE
                )

            if selected_type == "GATE_PORTAL":
                riders[gate]["portals"] += 1
                offered, optional, engaged = _ability_followup(game["steps"], step_index, gate)
                riders[gate]["interactive_offers"] += int(offered)
                riders[gate]["optional_offers"] += int(optional)
                riders[gate]["engaged"] += int(engaged)

        early = [first_main_deploy[turn] for turn in sorted(first_main_deploy)[:2]]
        for group in groups:
            grouped[group]["spell_observed_in_hand_games"] += int(
                any(metadata.get(code, {}).get("card_type") == "SPELL" for code in seen)
            )
            if early:
                grouped[group]["first_main_observed"] += 1
                grouped[group]["first_main_no_hand_deploy"] += int(not early[0])
            if len(early) >= 2:
                grouped[group]["first_two_mains_observed"] += 1
                grouped[group]["first_two_mains_both_no_hand_deploy"] += int(
                    not early[0] and not early[1]
                )
        for code in garden_cards:
            garden_cards[code]["observed_in_hand_games"] += int(code in seen)

    garden_out = {}
    for code, stats in garden_cards.items():
        stats["legal_garden_turn_cards"] = len(garden_legal_events[code])
        garden_out[code] = {
            **dict(sorted(stats.items())),
            "name": GARDEN_CONDITIONAL_ON_PLAY[code],
            "garden_share_when_comparable": _ratio(
                stats["comparable_garden"], stats["comparable_choices"]
            ),
            "effect_engagement_per_offer": _ratio(
                stats["interactive_engaged"], stats["interactive_offers"]
            ),
        }
    rider_out = {
        gate: {
            **dict(sorted(stats.items())),
            "offer_per_portal": _ratio(stats["interactive_offers"], stats["portals"]),
            "engagement_per_offer": _ratio(stats["engaged"], stats["interactive_offers"]),
        }
        for gate, stats in sorted(riders.items())
    }
    return {
        "groups": {
            group: _group_summary(counter) for group, counter in sorted(grouped.items())
        },
        "by_reference": {
            label: {
                "games": stats["games"],
                "score": _ratio(stats["score_milli"], 1000 * stats["games"]),
            }
            for label, stats in sorted(by_reference.items())
        },
        "garden_conditional_on_play": garden_out,
        "gate_rider_opportunities": rider_out,
    }


def _paired_delta(
    games_by_arm: dict[str, list[dict]], arm: str, *, baseline_arm: str,
    water_only: bool, gate_filter: frozenset[str] | None = None,
) -> dict[str, object]:
    baseline = {
        (
            str(game["counterfactual"]["target_gate"]),
            int(game["counterfactual"]["reference_index"]),
            int(game["counterfactual"]["candidate_seat"]),
        ): float(game["counterfactual"]["candidate_score"])
        for game in games_by_arm[baseline_arm]
    }
    candidate = {
        (
            str(game["counterfactual"]["target_gate"]),
            int(game["counterfactual"]["reference_index"]),
            int(game["counterfactual"]["candidate_seat"]),
        ): float(game["counterfactual"]["candidate_score"])
        for game in games_by_arm[arm]
    }
    if set(candidate) != set(baseline):
        raise ValueError(f"Arm {arm} does not have the {baseline_arm} pairing keys")
    by_block: dict[tuple[str, int], list[float]] = defaultdict(list)
    for key, score in candidate.items():
        gate, reference_index, _ = key
        if water_only and ELEMENT_OF_GATE[gate] != "WATER":
            continue
        if gate_filter is not None and gate not in gate_filter:
            continue
        by_block[(gate, reference_index)].append(score - baseline[key])
    block_deltas = np.asarray(
        [
            float(sum(values) / len(values))
            for values in by_block.values()
            if len(values) == 2
        ],
        dtype=np.float64,
    )
    malformed_blocks = [key for key, values in by_block.items() if len(values) != 2]
    if malformed_blocks:
        raise ValueError(f"Paired comparison has blocks without both seats: {malformed_blocks[:5]}")
    if block_deltas.size == 0:
        return {"blocks": 0, "delta": None, "ci80": [None, None]}
    filter_seed = sum(
        sum(ord(char) for char in gate) for gate in (gate_filter or frozenset())
    )
    rng = np.random.default_rng(
        91_337 + sum(ord(char) for char in arm) + int(water_only) + filter_seed
    )
    samples = rng.choice(
        block_deltas, size=(10_000, block_deltas.size), replace=True
    ).mean(axis=1)
    return {
        "blocks": int(block_deltas.size),
        "seat_games": int(2 * block_deltas.size),
        "delta": round(float(block_deltas.mean()), 6),
        "ci80": [
            round(float(np.quantile(samples, 0.10)), 6),
            round(float(np.quantile(samples, 0.90)), 6),
        ],
        "improved_blocks": int(np.sum(block_deltas > 0.0)),
        "tied_blocks": int(np.sum(block_deltas == 0.0)),
        "regressed_blocks": int(np.sum(block_deltas < 0.0)),
    }


def _validate_identical_control(games_by_arm: dict[str, list[dict]]) -> int:
    native = {
        (
            str(game["counterfactual"]["target_gate"]),
            int(game["counterfactual"]["reference_index"]),
            int(game["counterfactual"]["candidate_seat"]),
        ): game
        for game in games_by_arm["native_p2930"]
    }
    mismatches = 0
    for game in games_by_arm["water_spell_restored"]:
        gate = str(game["counterfactual"]["target_gate"])
        if ELEMENT_OF_GATE[gate] == "WATER":
            continue
        key = (
            gate,
            int(game["counterfactual"]["reference_index"]),
            int(game["counterfactual"]["candidate_seat"]),
        )
        baseline = native[key]
        signature = (
            game["outcome"],
            game["final_hp_before_last_action"],
            [step["a"] for step in game["steps"]],
        )
        baseline_signature = (
            baseline["outcome"],
            baseline["final_hp_before_last_action"],
            [step["a"] for step in baseline["steps"]],
        )
        mismatches += int(signature != baseline_signature)
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--deck-arms", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--expected-games-per-arm", type=int, default=144)
    parser.add_argument("--baseline-arm", default="native_p2930")
    parser.add_argument("--skip-identical-control-validation", action="store_true")
    args = parser.parse_args()
    games = load_games(args.inputs)
    deck_arm_payload = json.loads(args.deck_arms.read_text(encoding="utf-8"))
    games_by_arm: dict[str, list[dict]] = defaultdict(list)
    task_indices = set()
    for game in games:
        context = game.get("counterfactual")
        if not isinstance(context, dict):
            raise ValueError("Input game is missing counterfactual metadata")
        task_index = int(context["global_task_index"])
        if task_index in task_indices:
            raise ValueError(f"Duplicate global task index {task_index}")
        task_indices.add(task_index)
        games_by_arm[str(context["arm"])].append(game)
    if args.baseline_arm not in games_by_arm:
        raise ValueError(f"Input has no {args.baseline_arm} baseline arm")
    if args.expected_games_per_arm > 0:
        unexpected = {
            arm: len(selected)
            for arm, selected in games_by_arm.items()
            if len(selected) != args.expected_games_per_arm
        }
        if unexpected:
            raise ValueError(
                f"Incomplete or oversized counterfactual arms; expected "
                f"{args.expected_games_per_arm} games each, got {unexpected}"
            )

    metadata = _load_card_metadata()
    arms = {
        arm: _summarize_arm(selected, metadata)
        for arm, selected in sorted(games_by_arm.items())
    }
    comparison_gates = sorted(
        {
            str(game["counterfactual"]["target_gate"])
            for game in games_by_arm[args.baseline_arm]
        }
    )
    comparison_elements = sorted({ELEMENT_OF_GATE[gate] for gate in comparison_gates})
    comparisons = {}
    for arm in sorted(games_by_arm):
        comparisons[arm] = {
            "all_gates": _paired_delta(
                games_by_arm, arm, baseline_arm=args.baseline_arm, water_only=False
            ),
            "water_only": _paired_delta(
                games_by_arm, arm, baseline_arm=args.baseline_arm, water_only=True
            ),
            "by_element": {
                element: _paired_delta(
                    games_by_arm,
                    arm,
                    baseline_arm=args.baseline_arm,
                    water_only=False,
                    gate_filter=frozenset(
                        gate
                        for gate in comparison_gates
                        if ELEMENT_OF_GATE[gate] == element
                    ),
                )
                for element in comparison_elements
            },
            "by_gate": {
                gate: _paired_delta(
                    games_by_arm,
                    arm,
                    baseline_arm=args.baseline_arm,
                    water_only=False,
                    gate_filter=frozenset((gate,)),
                )
                for gate in comparison_gates
            },
        }
    identical_mismatches = None
    if not args.skip_identical_control_validation:
        identical_mismatches = _validate_identical_control(games_by_arm)
        if identical_mismatches:
            raise ValueError(
                "Paired schedule is not reproducible: "
                f"{identical_mismatches} unchanged non-Water games diverged"
            )
    payload = {
        "schema_version": 1,
        "deck_arms": {
            "path": str(args.deck_arms),
            "sha256": hashlib.sha256(args.deck_arms.read_bytes()).hexdigest(),
            "sources": deck_arm_payload.get("sources", {}),
            "definitions": deck_arm_payload.get("arm_definitions", {}),
        },
        "games": len(games),
        "games_per_arm": {
            arm: len(selected) for arm, selected in sorted(games_by_arm.items())
        },
        "validation": {
            "unique_global_tasks": len(task_indices),
            "water_arm_nonwater_semantic_mismatches": identical_mismatches,
            "identical_control_validation_skipped": bool(
                args.skip_identical_control_validation
            ),
        },
        "arms": arms,
        "paired_vs_native": comparisons,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for arm in sorted(arms):
        overall = arms[arm]["groups"]["ALL"]
        water = arms[arm]["groups"].get("ELEMENT/WATER")
        delta = comparisons[arm]["all_gates"]
        water_main_spell = (
            water["main_spell_selected_per_legal_window"] if water is not None else None
        )
        print(
            f"{arm}: score={overall['score']:.3f} delta={delta['delta']:+.3f} "
            f"main_spell={overall['main_spell_selected_per_legal_window']} "
            f"water_main_spell={water_main_spell} "
            f"garden={overall['garden_share_when_comparable_selected']}"
        )
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
