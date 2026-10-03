#!/usr/bin/env python3
"""Build the R7/R8 combat-potential follow-up decision packet."""

from __future__ import annotations

import json
from pathlib import Path

from build_shaping_screens import ROOT, sha256
from build_terminal_safe_reward_decision import (
    _arm_summary,
    _prior_references,
    _read_json,
    superseded_decision,
)


RESULTS = ROOT / "train-ablation-1781126582/results/terminal_safe_reward_followup"
REGISTRATION = RESULTS / "registration.json"
EVALUATION_INDEX = RESULTS / "evaluation_index.json"
PARENT_DECISION = (
    ROOT
    / "train-ablation-1781126582/results/terminal_safe_reward_screens/decision_packet.json"
)
DIRECT_R7_R8 = RESULTS / "h2h_r7_vs_r8.json"


def _direct_summary(path: Path) -> dict[str, object]:
    summary = _read_json(path)["summary"]
    return {
        "artifact": str(path.relative_to(ROOT)),
        "episodes": summary["episodes"],
        "score": summary["score"],
        "paired_lcb_80": summary["paired_lcb_80"],
        "timeout_rate": summary["timeout_rate"],
        "by_candidate_gate": summary["by_candidate_gate"],
    }


def main() -> None:
    registration = _read_json(REGISTRATION)
    evaluation = _read_json(EVALUATION_INDEX)
    if evaluation.get("registration_sha256") != sha256(REGISTRATION):
        raise ValueError("evaluation index does not match current registration")
    if evaluation.get("completed_at") is None:
        raise ValueError("evaluation index is incomplete")

    registered_by_id = {arm["id"]: arm for arm in registration["arms"]}
    evaluated_by_id = {arm["id"]: arm for arm in evaluation["arms"]}
    if set(registered_by_id) != {"R7", "R8"} or set(evaluated_by_id) != {"R7", "R8"}:
        raise ValueError("decision requires complete R7 and R8 artifacts")

    arms = {
        arm_id: _arm_summary(evaluated_by_id[arm_id], registered_by_id[arm_id])
        for arm_id in ("R7", "R8")
    }
    for arm_id in ("R7", "R8"):
        endpoint = evaluated_by_id[arm_id]["windows"][-1]
        arms[arm_id]["direct_vs_R5"] = _direct_summary(
            ROOT / endpoint["h2h_vs_control"]
        )

    direct_r7_r8 = _direct_summary(DIRECT_R7_R8)
    references = _prior_references()
    parent = _read_json(PARENT_DECISION)
    r5 = parent["arms"]["R5"]
    references["R5"] = {
        "anchor_scores": r5["windows"][-1]["anchor_scores"],
        "curated_score": r5["endpoint_curated"]["score"],
        "decision_disposition": parent["decision"]["disposition"],
    }


    packet = {
        "schema_id": "azuki.terminal_safe_reward_followup_decision",
        "schema_version": 1,
        "registration": str(REGISTRATION.relative_to(ROOT)),
        "registration_sha256": sha256(REGISTRATION),
        "evaluation_index": str(EVALUATION_INDEX.relative_to(ROOT)),
        "evaluation_index_sha256": sha256(EVALUATION_INDEX),
        "contract": {
            "arms": ["R7", "R8"],
            "sampled_rows_per_arm": registration["sampled_rows_per_arm"],
            "seed": registration["seed"],
            "evaluation_updates": evaluation["updates"],
            "trace_games_per_mode_window": evaluation["trace_games_per_mode_window"],
            "selection_rules": registration["selection_rules"],
        },
        "arms": arms,
        "prior_references": references,
        "direct_comparison": {
            "R7_vs_R8": direct_r7_r8,
            "R7_score_vs_R8": direct_r7_r8["score"],
            "R8_score_vs_R7": 1.0 - float(direct_r7_r8["score"]),
        },
        "decision": superseded_decision(),
    }
    output = RESULTS / "decision_packet.json"
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
