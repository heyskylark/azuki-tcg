#!/usr/bin/env python3
"""Build the R13 45M qualification decision and R14 fallback selection."""

from __future__ import annotations

import json

from build_shaping_screens import ROOT, sha256
from build_terminal_safe_reward_decision import _arm_summary, _read_json, superseded_decision


RESULTS = ROOT / "train-ablation-1781126582/results/terminal_safe_reward_horizon_followup"
REGISTRATION = RESULTS / "qualifier_registration.json"
EVALUATION_INDEX = RESULTS / "qualifier_evaluation_index.json"
SCREEN_DECISION = RESULTS / "decision_packet.json"


def _active_sequences(window: dict[str, object], mode: str) -> dict[str, int]:
    return {
        element: int(values["active_sequences"])
        for element, values in window[mode]["by_element"].items()
    }


def main() -> None:
    registration = _read_json(REGISTRATION)
    evaluation = _read_json(EVALUATION_INDEX)
    if evaluation.get("registration_sha256") != sha256(REGISTRATION):
        raise ValueError("evaluation index does not match current registration")
    if evaluation.get("completed_at") is None:
        raise ValueError("evaluation index is incomplete")
    if [arm["id"] for arm in evaluation["arms"]] != ["R13"]:
        raise ValueError("qualification decision requires complete R13 artifacts")

    arm = _arm_summary(evaluation["arms"][0], registration["arms"][0])
    screen_decision = _read_json(SCREEN_DECISION)
    screen_r13 = screen_decision["arms"]["R13"]
    screen_r14 = screen_decision["arms"]["R14"]
    windows = arm["windows"]
    anchors = [float(window["anchor_scores"]["mean"]) for window in windows]
    sample_breadth = [_active_sequences(window, "sample") for window in windows]
    argmax_breadth = [_active_sequences(window, "argmax") for window in windows]

    packet = {
        "schema_id": "azuki.terminal_safe_reward_horizon_qualifier_decision",
        "schema_version": 1,
        "registration": str(REGISTRATION.relative_to(ROOT)),
        "registration_sha256": sha256(REGISTRATION),
        "evaluation_index": str(EVALUATION_INDEX.relative_to(ROOT)),
        "evaluation_index_sha256": sha256(EVALUATION_INDEX),
        "contract": {
            "arm": "R13",
            "sampled_rows": registration["sampled_rows_per_arm"],
            "seed": registration["seed"],
            "evaluation_updates": evaluation["updates"],
            "trace_games_per_mode_window": evaluation["trace_games_per_mode_window"],
            "qualification_rules": registration["qualification_rules"],
        },
        "arm": arm,
        "screen_references": {
            "R13": {
                "anchor_scores": screen_r13["windows"][-1]["anchor_scores"],
                "sample_active_sequences": _active_sequences(
                    screen_r13["windows"][-1], "sample"
                ),
            },
            "R14": {
                "anchor_scores": screen_r14["windows"][-1]["anchor_scores"],
                "sample_active_sequences": _active_sequences(
                    screen_r14["windows"][-1], "sample"
                ),
                "curated_score": screen_r14["endpoint_curated"]["score"],
            },
        },
        "derived": {
            "anchor_mean_trajectory": anchors,
            "sample_active_sequence_trajectory": sample_breadth,
            "argmax_active_sequence_trajectory": argmax_breadth,
        },
        "decision": superseded_decision(),
    }
    output = RESULTS / "qualifier_decision_packet.json"
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
