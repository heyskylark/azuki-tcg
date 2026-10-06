#!/usr/bin/env python3
"""Build the R9/R10 sustained-potential-tail decision packet."""

from __future__ import annotations

import json

from build_shaping_screens import ROOT, sha256
from build_terminal_safe_reward_decision import (
    _arm_summary,
    _prior_references,
    _read_json,
    superseded_decision,
)


RESULTS = ROOT / "train-ablation-1781126582/results/terminal_safe_reward_tail_followup"
REGISTRATION = RESULTS / "registration.json"
EVALUATION_INDEX = RESULTS / "evaluation_index.json"
DIRECT_R9_R10 = RESULTS / "h2h_r9_vs_r10.json"
R7_QUALIFIER_DECISION = (
    ROOT
    / "train-ablation-1781126582/results/terminal_safe_reward_followup/qualifier_decision_packet.json"
)


def _direct_summary(path) -> dict[str, object]:
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
    if set(registered_by_id) != {"R9", "R10"} or set(evaluated_by_id) != {"R9", "R10"}:
        raise ValueError("decision requires complete R9 and R10 artifacts")

    arms = {
        arm_id: _arm_summary(evaluated_by_id[arm_id], registered_by_id[arm_id])
        for arm_id in ("R9", "R10")
    }
    for arm_id in ("R9", "R10"):
        endpoint = evaluated_by_id[arm_id]["windows"][-1]
        arms[arm_id]["direct_vs_R7_45M"] = _direct_summary(
            ROOT / endpoint["h2h_vs_control"]
        )

    direct = _direct_summary(DIRECT_R9_R10)
    references = _prior_references()
    r7_decision = _read_json(R7_QUALIFIER_DECISION)
    r7 = r7_decision["arm"]
    references["R7_45M"] = {
        "anchor_scores": r7["windows"][-1]["anchor_scores"],
        "curated_score": r7["endpoint_curated"]["score"],
        "sample": r7["windows"][-1]["sample"],
        "decision_disposition": r7_decision["decision"]["disposition"],
    }


    packet = {
        "schema_id": "azuki.terminal_safe_reward_tail_decision",
        "schema_version": 1,
        "registration": str(REGISTRATION.relative_to(ROOT)),
        "registration_sha256": sha256(REGISTRATION),
        "evaluation_index": str(EVALUATION_INDEX.relative_to(ROOT)),
        "evaluation_index_sha256": sha256(EVALUATION_INDEX),
        "contract": {
            "arms": ["R9", "R10"],
            "sampled_rows_per_arm": registration["sampled_rows_per_arm"],
            "seed": registration["seed"],
            "evaluation_updates": evaluation["updates"],
            "trace_games_per_mode_window": evaluation["trace_games_per_mode_window"],
            "selection_rules": registration["selection_rules"],
        },
        "arms": arms,
        "prior_references": references,
        "direct_comparison": {
            "R9_vs_R10": direct,
            "R9_score_vs_R10": direct["score"],
            "R10_score_vs_R9": 1.0 - float(direct["score"]),
        },
        "decision": superseded_decision(),
    }
    output = RESULTS / "decision_packet.json"
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
