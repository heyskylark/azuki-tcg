#!/usr/bin/env python3
"""Reassess short ablations using strategy evidence instead of early win rate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from strategy_descriptor import SCHEMA_VERSION


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
                    "stages": sequence.get("stages", {}),
                }
                for name, sequence in matching.items()
            },
        }
    return result


def descriptor_metrics(path: str) -> dict[str, Any]:
    descriptor = read_json(relative_result_path(path))
    if descriptor.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"stale strategy descriptor: {path}")
    contexts = list(descriptor["deck"]["contexts"].values())
    main_slots = sum(int(context["main_slots"]) for context in contexts)
    sibling_pairs = descriptor["deck"]["relationships"]["sibling_pairs"]
    return {
        "descriptor": path,
        "descriptor_id": descriptor["descriptor_id"],
        "elemental_strategy": descriptor["elemental_strategy"],
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
        result[mode] = {
            "completion_per_eligible": {
                element: (
                    endpoint[mode]["elements"][element]["completion_per_eligible"]
                    - previous[mode]["elements"][element]["completion_per_eligible"]
                    if endpoint[mode]["elements"][element]["completion_per_eligible"] is not None
                    and previous[mode]["elements"][element]["completion_per_eligible"] is not None
                    else None
                )
                for element in ELEMENTS
            },
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
        "schema_version": 3,
        "decision_policy": {
            "short_run_primary": [
                "opportunity-normalized ordered-sequence stage, completion, and conversion by element",
                "sampled and deterministic structural presence",
                "multiple learned lines per element",
                "sibling differentiation",
                "conditional face-versus-setup/control choice",
                "later-window persistence or positive slope",
            ],
            "short_run_win_rate": "deferred conversion evidence; never a ranker or tiebreaker against mature aggressive policies",
            "short_run_strength_veto": "only catastrophic broken-learning evidence when strategy is also absent or degrading",
            "curated_strategy_score": "supporting evidence only; never a pooled ranker",
        },
        "production_baseline": production_baseline(),
        "ablation_arms": ablation_arms(),
        "strategy_first_findings": {
            "status": "observations_rebuilt_rankings_withheld",
            "reason": "Schema-v2 hardcoded rankings and numeric conclusions are superseded. Corrected per-context elemental effects above are evidence, not a pooled strategy rank.",
            "element_objectives": {
                "WATER": "Gate/leader-fit resource readiness, spell effects and recovery-to-replay conversion.",
                "EARTH": "Defensive opportunities, damage prevention and properly timed leader health restoration.",
                "FIRE": "Self-damage converted into same-turn attack value, alongside other coherent tempo and multi-play lines.",
                "LIGHTNING": "Weapon-aware deck construction, recovery/re-equip and effective attacks while equipped.",
            },
            "novel_strategies": "Allow coherent unregistered lines supported by reviewed traces and matched payoff probes; no named-card rewards.",
        },
        "revised_action": {
            "canary_status": "blocked",
            "reason": "Recipe boundaries remain unqualified under corrected elemental semantics. Legal alternatives and deck differences do not prove causal strategy quality.",
            "next_comparison": "Requalify R14, then matched D2/D3 and replay-off; exposure; performance-qualified league with realized diversity; separate S1/S3. R3 direct-edge removal is redundant with R14.",
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
