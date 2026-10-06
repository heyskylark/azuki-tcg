#!/usr/bin/env python3
"""Build the R5/R6 15M decision packet from registered evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from build_shaping_screens import ROOT, sha256
from strategy_descriptor import SCHEMA_VERSION


RESULTS = ROOT / "train-ablation-1781126582/results/terminal_safe_reward_screens"
REGISTRATION = RESULTS / "registration.json"
EVALUATION_INDEX = RESULTS / "evaluation_index.json"
REWARD_DECISION = ROOT / "train-ablation-1781126582/results/reward_screens/decision_packet.json"
S4_DECISION = ROOT / "train-ablation-1781126582/results/terminal_closure_screen/decision_packet.json"


def superseded_decision() -> dict[str, object]:
    """Historical reward screens retain observations, never active selections."""
    return {
        "selected_arm": None,
        "production_qualified": False,
        "reward_foundation_qualified": False,
        "status": "superseded_strategy_semantics_require_requalification",
        "disposition": "historical_decision_superseded_by_strategy_semantics",
        "reason": [
            "Current descriptor observations do not reinstate historical rankings or rejection-to-followup conclusions.",
            "Only the current R14 final decision computes reward-foundation qualification.",
        ],
        "next_action": "Hold historical reward brackets and qualifiers. Resolve the registered R14 hold/fallback before any downstream comparison; do not launch 1B.",
    }


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _descriptor_summary(path: Path) -> dict[str, object]:
    descriptor = _read_json(path)
    if descriptor.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"stale strategy descriptor: {path}")
    sequences = descriptor["sequences"]
    if not isinstance(sequences, dict):
        raise ValueError(f"invalid sequence payload: {path}")
    by_element = {}
    for element in ("LIGHTNING", "WATER", "FIRE", "EARTH"):
        entries = [entry for entry in sequences.values() if entry["element"] == element]
        by_element[element] = {
            "eligible": sum(int(entry["eligible"]) for entry in entries),
            "completed": sum(int(entry["completed"]) for entry in entries),
            "converted": sum(int(entry["converted"]) for entry in entries),
            "active_sequences": sum(int(entry["completed"]) > 0 for entry in entries),
            "converted_sequences": sum(int(entry["converted"]) > 0 for entry in entries),
            "registered_sequences": len(entries),
        }
    relationships = descriptor["deck"]["relationships"]
    sibling_pairs = relationships["sibling_pairs"]
    game_profile = descriptor["game_profile"]
    both_legal = int(game_profile["both_face_entity_legal"])
    return {
        "descriptor": str(path.relative_to(ROOT)),
        "descriptor_id": descriptor["descriptor_id"],
        "descriptor_schema_version": descriptor["schema_version"],
        "elemental_strategy": descriptor["elemental_strategy"],
        "by_element": by_element,
        "face_share_when_both_legal": game_profile["face_share_when_both_legal"],
        "entity_share_when_both_legal": (
            int(game_profile["entity_selected_when_both_legal"]) / both_legal
            if both_legal
            else None
        ),
        "sibling_mean_multiset_distance": mean(
            float(pair["multiset_distance"]) for pair in sibling_pairs
        ),
        "sibling_exact_collision_pairs": relationships["sibling_exact_collision_pairs"],
    }


def _anchor_scores(window: dict[str, object]) -> dict[str, float]:
    scores = {
        label: float(_read_json(ROOT / path)["summary"]["score"])
        for label, path in window["h2h"].items()
    }
    return {"mean": mean(scores.values()), **scores}


def _curated_summary(path: Path) -> dict[str, object]:
    panel = _read_json(path)
    records = panel["records"]
    return {
        "artifact": str(path.relative_to(ROOT)),
        "games": panel["games"],
        "completed": sum(bool(record["completed"]) for record in records),
        "score": mean(float(record["score"]) for record in records),
        "by_gate": panel["by_gate"],
    }


def _reward_component_profile(result_root: Path) -> dict[str, float]:
    metrics = [
        json.loads(line)
        for line in (result_root / "logs/production.jsonl").read_text().splitlines()
        if line.strip() and '"_step"' in line
    ]
    if not metrics:
        raise ValueError(f"no training metrics: {result_root}")
    components = (
        "potential_leader_health",
        "potential_garden_attack",
        "potential_untapped_garden",
        "potential_untapped_ikz",
        "direct_leader_edge",
        "direct_board_edge",
    )
    return {
        component: max(
            float(
                row.get(
                    f"environment/reward_component/all/{component}/raw_abs_sum",
                    0.0,
                )
            )
            for row in metrics
        )
        for component in components
    }


def _arm_summary(
    arm: dict[str, object],
    registration_arm: dict[str, object],
) -> dict[str, object]:
    windows = []
    for window in arm["windows"]:
        windows.append(
            {
                "update": window["update"],
                "checkpoint": window["checkpoint"],
                "checkpoint_sha256": window["checkpoint_sha256"],
                "anchor_scores": _anchor_scores(window),
                "sample": _descriptor_summary(ROOT / window["sample_descriptor"]),
                "argmax": _descriptor_summary(ROOT / window["argmax_descriptor"]),
            }
        )
    curated_path = ROOT / arm["windows"][-1]["curated_strategy_panel"]
    return {
        "id": arm["id"],
        "potential_weights": registration_arm["potential_weights"],
        "training": arm["training"],
        "reward_component_raw_abs_max": _reward_component_profile(
            ROOT / registration_arm["result_root"]
        ),
        "windows": windows,
        "endpoint_curated": _curated_summary(curated_path),
    }


def _prior_references() -> dict[str, dict[str, object]]:
    reward_decision = _read_json(REWARD_DECISION)
    s4_decision = _read_json(S4_DECISION)
    references = {}
    for arm_id in ("R1", "R4"):
        endpoint = reward_decision["arms"][arm_id]["windows"][-1]
        anchor_scores = {
            label: float(result["score"])
            for label, result in endpoint["h2h"].items()
        }
        references[arm_id] = {
            "anchor_scores": {"mean": mean(anchor_scores.values()), **anchor_scores},
            "curated_score": endpoint["curated_strategy_panel"]["score"],
            "decision_disposition": reward_decision["decision"]["disposition"][arm_id],
        }
    s4_anchor_scores = {
        label: float(score)
        for label, score in s4_decision["treatment"]["endpoint_anchor_scores"].items()
    }
    references["S4"] = {
        "anchor_scores": {"mean": mean(s4_anchor_scores.values()), **s4_anchor_scores},
        "curated_score": s4_decision["treatment"]["endpoint_curated_strategy_score"],
        "decision_disposition": s4_decision["decision"]["disposition"],
    }
    return references


def main() -> None:
    registration = _read_json(REGISTRATION)
    evaluation = _read_json(EVALUATION_INDEX)
    if evaluation.get("registration_sha256") != sha256(REGISTRATION):
        raise ValueError("evaluation index does not match current registration")
    if evaluation.get("completed_at") is None:
        raise ValueError("evaluation index is incomplete")

    registered_by_id = {arm["id"]: arm for arm in registration["arms"]}
    evaluated_by_id = {arm["id"]: arm for arm in evaluation["arms"]}
    if set(registered_by_id) != {"R5", "R6"} or set(evaluated_by_id) != {"R5", "R6"}:
        raise ValueError("decision requires complete R5 and R6 artifacts")

    arms = {
        arm_id: _arm_summary(evaluated_by_id[arm_id], registered_by_id[arm_id])
        for arm_id in ("R5", "R6")
    }
    direct_path = ROOT / evaluated_by_id["R6"]["windows"][-1]["h2h_vs_control"]
    direct = _read_json(direct_path)
    direct_summary = direct["summary"]
    r6_score = float(direct_summary["score"])
    prior_references = _prior_references()

    packet = {
        "schema_id": "azuki.terminal_safe_reward_decision",
        "schema_version": 2,
        "registration": str(REGISTRATION.relative_to(ROOT)),
        "registration_sha256": sha256(REGISTRATION),
        "evaluation_index": str(EVALUATION_INDEX.relative_to(ROOT)),
        "evaluation_index_sha256": sha256(EVALUATION_INDEX),
        "contract": {
            "arms": ["R5", "R6"],
            "sampled_rows_per_arm": registration["sampled_rows_per_arm"],
            "seed": registration["seed"],
            "evaluation_updates": evaluation["updates"],
            "trace_games_per_mode_window": evaluation["trace_games_per_mode_window"],
            "selection_rules": registration["selection_rules"],
        },
        "arms": arms,
        "prior_references": prior_references,
        "direct_comparison": {
            "artifact": str(direct_path.relative_to(ROOT)),
            "R6_score_vs_R5": r6_score,
            "R6_paired_lcb_80": direct_summary["paired_lcb_80"],
            "R5_score_vs_R6": 1.0 - r6_score,
            "episodes": direct_summary["episodes"],
            "timeout_rate": direct_summary["timeout_rate"],
            "R6_by_candidate_gate": direct_summary["by_candidate_gate"],
        },
        "decision": superseded_decision(),
    }
    output = RESULTS / "decision_packet.json"
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
