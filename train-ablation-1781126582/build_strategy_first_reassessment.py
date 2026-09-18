#!/usr/bin/env python3
"""Reassess short ablations using strategy evidence instead of early win rate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "train-ablation-1781126582/results"
ELEMENTS = ("EARTH", "FIRE", "LIGHTNING", "WATER")
FAMILIES = (
    "reward_screens",
    "draft_screens",
    "curriculum_screens",
    "league_screens",
    "shaping_screens",
    "terminal_closure_screen",
)
BASELINE = RESULTS / "strategy_baseline_v1/baseline_packet.json"
OUTPUT = RESULTS / "strategy_first_reassessment.json"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def relative_result_path(path: str) -> Path:
    direct = ROOT / path
    if direct.is_file():
        return direct
    nested = ROOT / "train-ablation-1781126582" / path
    if nested.is_file():
        return nested
    raise FileNotFoundError(path)


def element_sequence_metrics(descriptor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    sequences = descriptor["sequences"]
    for element in ELEMENTS:
        matching = {
            name.split(".", 1)[1]: sequence
            for name, sequence in sequences.items()
            if sequence["element"] == element and sequence.get("valid", True)
        }
        eligible = sum(int(sequence["eligible"]) for sequence in matching.values())
        completed = sum(int(sequence["completed"]) for sequence in matching.values())
        converted = sum(int(sequence["converted"]) for sequence in matching.values())
        result[element] = {
            "eligible": eligible,
            "completed": completed,
            "converted": converted,
            "completion_per_eligible": completed / eligible if eligible else None,
            "conversion_per_eligible": converted / eligible if eligible else None,
            "by_sequence": {
                name: {
                    "eligible": int(sequence["eligible"]),
                    "completed": int(sequence["completed"]),
                    "converted": int(sequence["converted"]),
                }
                for name, sequence in matching.items()
            },
        }
    return result


def descriptor_metrics(path: str) -> dict[str, Any]:
    descriptor = read_json(relative_result_path(path))
    contexts = list(descriptor["deck"]["contexts"].values())
    main_slots = sum(int(context["main_slots"]) for context in contexts)
    sibling_pairs = descriptor["deck"]["relationships"]["sibling_pairs"]
    return {
        "games": int(descriptor["games"]),
        "elements": element_sequence_metrics(descriptor),
        "deck_element_slot_share": {
            element: sum(
                float(context["element_slot_share"].get(element, 0.0))
                * int(context["main_slots"])
                for context in contexts
            )
            / main_slots
            for element in (*ELEMENTS, "NORMAL")
        },
        "type_slot_share": {
            card_type: sum(
                float(context["type_slot_share"].get(card_type, 0.0))
                * int(context["main_slots"])
                for context in contexts
            )
            / main_slots
            for card_type in ("ENTITY", "SPELL", "WEAPON")
        },
        "face_share_when_both_legal": descriptor["funnels"]
        ["attack.both_legal_face_choice"]["selected_per_opportunity"],
        "entity_share_when_both_legal": descriptor["funnels"]
        ["attack.both_legal_entity_choice"]["selected_per_opportunity"],
        "sibling_mean_multiset_distance": sum(
            float(pair["multiset_distance"]) for pair in sibling_pairs
        )
        / len(sibling_pairs),
    }


def curated_score(path: str | None) -> float | None:
    if path is None:
        return None
    panel = read_json(relative_result_path(path))
    cells = list(panel["by_gate"].values())
    games = sum(int(cell["games"]) for cell in cells)
    return sum(float(cell["score"]) * int(cell["games"]) for cell in cells) / games


def late_delta(windows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(windows) < 2:
        return None
    previous = windows[-2]
    endpoint = windows[-1]
    result: dict[str, Any] = {
        "from_update": previous["update"],
        "to_update": endpoint["update"],
    }
    for mode in ("sample", "argmax"):
        completion_delta: dict[str, dict[str, Any]] = {}
        for element in ELEMENTS:
            previous_element = previous[mode]["elements"][element]
            endpoint_element = endpoint[mode]["elements"][element]
            previous_eligible = int(previous_element["eligible"])
            endpoint_eligible = int(endpoint_element["eligible"])
            delta = None
            if previous_eligible > 0 and endpoint_eligible > 0:
                delta = (
                    float(endpoint_element["completion_per_eligible"])
                    - float(previous_element["completion_per_eligible"])
                )
            completion_delta[element] = {
                "previous_eligible": previous_eligible,
                "endpoint_eligible": endpoint_eligible,
                "delta": delta,
            }
        result[mode] = {
            "completion_per_eligible": completion_delta,
            "face_share_when_both_legal": (
                endpoint[mode]["face_share_when_both_legal"]
                - previous[mode]["face_share_when_both_legal"]
            ),
        }
    return result


def ablation_arms() -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for family in FAMILIES:
        index_path = RESULTS / family / "evaluation_index.json"
        index = read_json(index_path)
        for arm in index["arms"]:
            windows = [
                {
                    "update": int(window["update"]),
                    "sample": descriptor_metrics(window["sample_descriptor"]),
                    "argmax": descriptor_metrics(window["argmax_descriptor"]),
                }
                for window in arm["windows"]
            ]
            endpoint = arm["windows"][-1]
            arms.append(
                {
                    "family": family,
                    "id": arm["id"],
                    "name": arm["name"],
                    "source_index": str(index_path.relative_to(ROOT)),
                    "windows": windows,
                    "late_delta": late_delta(windows),
                    "curated_strategy_score_supporting_only": curated_score(
                        endpoint.get("curated_strategy_panel")
                    ),
                }
            )
    return arms


def production_baseline() -> list[dict[str, Any]]:
    packet = read_json(BASELINE)
    return [
        {
            "update": int(checkpoint["update"]),
            "source_descriptor": checkpoint["descriptor"],
            "sample": descriptor_metrics(checkpoint["descriptor"]),
        }
        for checkpoint in packet["checkpoints"]
    ]


def main() -> None:
    payload = {
        "schema_id": "azuki.strategy_first_reassessment",
        "schema_version": 2,
        "decision_policy": {
            "short_run_primary": [
                "opportunity-normalized ordered-sequence completion and conversion by element",
                "sampled and deterministic structural presence",
                "multiple learned lines per element",
                "sibling differentiation",
                "conditional face-versus-entity attack mix (descriptive unless paired with a curated value-labeled probe)",
                "later-window persistence or positive slope",
            ],
            "short_run_win_rate": "deferred conversion evidence; never a ranker or tiebreaker against mature aggressive policies",
            "short_run_strength_veto": "only catastrophic broken-learning evidence when strategy is also absent or degrading",
            "curated_strategy_score": "supporting evidence only; never a pooled ranker",
        },
        "production_baseline": production_baseline(),
        "ablation_arms": ablation_arms(),
        "strategy_first_findings": {
            "reward": {
                "balanced_leader": "R1",
                "fire_and_low_face_specialist": "R3",
                "interpretation": "R1 shows the clearest balanced Earth/Fire/Water acquisition and positive late slopes. R3 is the strongest Fire candidate; its conditional face-versus-entity mix is reported descriptively and should not be discarded solely for early strength.",
            },
            "draft_credit": {
                "broad_sampled_leader": "D2",
                "deterministic_breadth_candidate": "D3",
                "interpretation": "D2 strongly restores Water while retaining Earth. D3 supplies the clearest deterministic Water replay and sibling differentiation; its early strength disposition is insufficient to discard that signal.",
            },
            "curriculum": {
                "strategy_candidate": "C1",
                "interpretation": "C1 improves sampled Fire, opportunity-normalized deterministic Water, and the curated interaction panel, while its conditional face-versus-entity mix shifts toward face attacks and it does not repair Lightning.",
            },
            "league": {
                "best_strategy_signal": "L1",
                "water_specialist": "L4",
                "interpretation": "L1 is the most balanced league strategy result and L4 is strongest for Water. Both miss the registered runtime floor. L5 is fast but loses late element strategy, so win rate must not make it the strategy parent.",
            },
            "shaping_tail": {
                "earth_and_deterministic_water": "S0",
                "fire_and_water_growth": "S1",
                "water_specialist": "S2",
                "fire_and_low_face_specialist": "S3",
                "interpretation": "No S0-S3 arm dominates the strategy Pareto front. S1, S2, and S3 each contain strategy evidence that the prior strength-first disposition undervalued.",
            },
            "terminal_closure": {
                "candidate": "S4",
                "interpretation": "S4 has mixed strategy evidence: curated interactions improved, endpoint Water breadth fell, and Lightning remained absent. Its conditional face-versus-entity mix shifted toward face attacks, which is not itself positive or negative strategy evidence.",
            },
            "elements": {
                "EARTH": "Consistently present through Bobu-before-destruction and stone-defender portal lines. Devotion sacrifice conversion is absent in every inspected arm, so Earth is only partially learned.",
                "FIRE": "Real but sparse. R3 and S3 are the clearest Fire acquisitions; R1 and C1 show smaller balanced gains. Rushfire charge conversion is absent in every inspected arm.",
                "LIGHTNING": "Structurally absent: zero weapon-recovery portal-attack completions in every baseline, arm, checkpoint endpoint, and policy mode despite many eligible opportunities and roughly 9-10% Lightning deck slots.",
                "WATER": "Strongest evidence of strategic learning. R1, D2, D3, L4, S1, and S2 recover healing/Shao timing and some Echoed Waves replay that disappeared in the late production baseline. Deterministic spell replay remains fragile.",
            },
        },
        "revised_action": {
            "canary_status": "remain held",
            "reason": "No arm covers all four elements; Lightning is structurally absent. The hold is strategy-based, not a consequence of 15M win rate.",
            "next_comparison": "Use longer strategy-first confirmation for Pareto candidates R1/R3, D2/D3, and S0/S1/S2/S3, or first run a focused Lightning acquisition intervention. Do not select a single parent from 15M win rate.",
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
