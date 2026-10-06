#!/usr/bin/env python3
"""Build the final R14 45M terminal-safe reward qualification decision."""

from __future__ import annotations

import json
import math
from collections import Counter

from build_shaping_screens import ROOT, sha256
from build_terminal_safe_reward_decision import _arm_summary, _prior_references, _read_json


RESULTS = ROOT / "train-ablation-1781126582/results/terminal_safe_reward_horizon_followup"
REGISTRATION = RESULTS / "fallback_qualifier_registration.json"
EVALUATION_INDEX = RESULTS / "fallback_qualifier_evaluation_index.json"
HORIZON_DECISION = RESULTS / "decision_packet.json"
R13_DECISION = RESULTS / "qualifier_decision_packet.json"


def _active_sequences(window: dict[str, object], mode: str) -> dict[str, int]:
    return {
        element: int(values["active_sequences"])
        for element, values in window[mode]["by_element"].items()
    }


def qualification_checks(arm: dict) -> tuple[dict, dict]:
    training = arm["training"]
    checks = {
        "integrity": all(float(value) == 0 for value in training["integrity_maxima"].values())
        and float(training["max_invalid_metric"]) == 0,
        "reward_reconstruction": all(
            math.isfinite(float(training[key])) and float(training[key]) <= tolerance
            for key, tolerance in (
                ("reward_raw_reconstruction_max_abs_error", 2e-6),
                ("reward_scaled_reconstruction_max_abs_error", 2e-6),
                ("ppo_component_reconstruction_max_abs_error", 1e-5),
            )
        ),
        "complete_training": training["last_epoch"] == 2930,
        "curated_complete": arm["endpoint_curated"]["completed"] == arm["endpoint_curated"]["games"],
    }
    windows = arm["windows"]
    for mode in ("sample", "argmax"):
        checks[f"{mode}_late_converted_breadth"] = all(
            windows[-1][mode]["by_element"][element]["converted_sequences"] > 0
            and windows[-1][mode]["by_element"][element]["converted_sequences"]
            >= windows[-2][mode]["by_element"][element]["converted_sequences"]
            for element in ("LIGHTNING", "WATER", "EARTH", "FIRE")
        )
    core = {}
    for mode in ("sample", "argmax"):
        core[mode] = []
        for window in windows:
            counts = {}
            for context in window[mode]["elemental_strategy"]["contexts"].values():
                counts.setdefault(context["element"], Counter()).update(context["counts"])
            core[mode].append({
                "update": window["update"],
                "by_element": {element: dict(values) for element, values in sorted(counts.items())},
            })
    return checks, core


def main() -> None:
    registration = _read_json(REGISTRATION)
    evaluation = _read_json(EVALUATION_INDEX)
    if evaluation.get("registration_sha256") != sha256(REGISTRATION):
        raise ValueError("evaluation index does not match current registration")
    if evaluation.get("completed_at") is None:
        raise ValueError("evaluation index is incomplete")
    if [arm["id"] for arm in registration["arms"]] != ["R14"]:
        raise ValueError("final decision requires the registered R14 fallback")
    if [arm["id"] for arm in evaluation["arms"]] != ["R14"]:
        raise ValueError("final decision requires complete R14 artifacts")

    arm = _arm_summary(evaluation["arms"][0], registration["arms"][0])
    windows = arm["windows"]
    anchors = [float(window["anchor_scores"]["mean"]) for window in windows]
    sample_breadth = [_active_sequences(window, "sample") for window in windows]
    argmax_breadth = [_active_sequences(window, "argmax") for window in windows]
    references = _prior_references()
    horizon = _read_json(HORIZON_DECISION)
    r13 = _read_json(R13_DECISION)
    checks, core = qualification_checks(arm)
    unresolved = [
        "Causal gate/leader deck fit and stronger legal alternative values require matched counterfactual probes; deck distance and face share cannot establish them.",
        "Trace v2 lacks physical-copy/effect IDs and deferred combat attribution; reported conversions are observed lower bounds, not complete mechanic payoffs.",
    ]
    for mode in ("sample", "argmax"):
        if all(window["by_element"].get("FIRE", {}).get("fire.zero_before_attack/converted", 0) == 0 for window in core[mode][-2:]):
            unresolved.append(f"{mode}: no observed late-window Zero self-damage-to-attack conversion; inspect other coherent self-damage lines before calling Fire complete.")
    qualified = all(checks.values())

    packet = {
        "schema_id": "azuki.terminal_safe_reward_final_decision",
        "schema_version": 2,
        "registration": str(REGISTRATION.relative_to(ROOT)),
        "registration_sha256": sha256(REGISTRATION),
        "evaluation_index": str(EVALUATION_INDEX.relative_to(ROOT)),
        "evaluation_index_sha256": sha256(EVALUATION_INDEX),
        "contract": {
            "arm": "R14",
            "sampled_rows": registration["sampled_rows_per_arm"],
            "seed": registration["seed"],
            "evaluation_updates": evaluation["updates"],
            "trace_games_per_mode_window": evaluation["trace_games_per_mode_window"],
            "qualification_rules": registration["qualification_rules"],
        },
        "arm": arm,
        "prior_references": references,
        "selection_history": {
            "horizon_decision": str(HORIZON_DECISION.relative_to(ROOT)),
            "horizon_decision_sha256": sha256(HORIZON_DECISION),
            "R13_qualifier_decision": str(R13_DECISION.relative_to(ROOT)),
            "R13_qualifier_decision_sha256": sha256(R13_DECISION),
            "R13_disposition": r13["decision"]["disposition"],
            "R14_15M_anchor_mean": horizon["arms"]["R14"]["windows"][-1]["anchor_scores"]["mean"],
        },
        "derived": {
            "anchor_mean_trajectory": anchors,
            "sample_active_sequence_trajectory": sample_breadth,
            "argmax_active_sequence_trajectory": argmax_breadth,
            "qualification_checks": checks,
            "elemental_core_trajectory": core,
            "unresolved_strategy_requirements": unresolved,
        },
        "finalized_recipe": {
            "config": registration["arms"][0]["config"],
            "config_sha256": registration["arms"][0]["config_sha256"],
            "pbrs_mode": "discounted",
            "pbrs_gamma": 0.99,
            "terminal_closure": True,
            "potential_weights": registration["arms"][0]["potential_weights"],
            "potential_tail": registration["arms"][0]["potential_tail"],
            "exploration_tail": registration["arms"][0]["exploration_tail"],
            "anneal_fraction": registration["arms"][0]["anneal_fraction"],
            "direct_leader_weight": 0.0,
            "direct_board_weight": 0.0,
            "early_tempo_bonus": 0.0,
        },
        "decision": {
            "selected_arm": "R14" if qualified else None,
            "production_qualified": False,
            "reward_foundation_qualified": qualified,
            "reward_contract_validated": checks["integrity"] and checks["reward_reconstruction"],
            "disposition": "retain_r14_reward_foundation_with_strategy_caveats" if qualified else "hold_r14_strategy_requalification",
            "reason": [
                "Qualification is computed from current descriptor artifacts; a completed evaluator is not an automatic pass.",
                f"Current checks: {checks}.",
                f"Sampled active breadth: {sample_breadth}; deterministic active breadth: {argmax_breadth}.",
                f"External-anchor trajectory {anchors} is deferred strength-conversion evidence, not the short-run ranker.",
                *unresolved,
                "R3 direct-edge removal is redundant: R14 already has zero leader and board direct deltas.",
                "D2/D3, replay-off, exposure, league runtime/diversity, and S1/S3 are not qualified by this reward-only run.",
            ],
            "next_action": (
                "Review the corrected strategy caveats, then run matched D2/D3 on a healthy qualified runtime. The full recipe remains unqualified; do not launch 1B."
                if qualified else
                "Stop at the registered R14 hold/fallback: resolve and requalify the no-reduction failure. Reward finalization and all downstream recipe comparisons remain blocked; do not launch 1B."
            ),
        },
    }
    output = RESULTS / "final_decision_packet.json"
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
