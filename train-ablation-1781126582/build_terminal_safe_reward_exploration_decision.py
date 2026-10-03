#!/usr/bin/env python3
"""Build the R11/R12 exploration-tail decision packet."""

from __future__ import annotations

import json

from build_shaping_screens import ROOT, sha256
from build_terminal_safe_reward_decision import _arm_summary, _read_json, superseded_decision


RESULTS = ROOT / "train-ablation-1781126582/results/terminal_safe_reward_exploration_followup"
REGISTRATION = RESULTS / "registration.json"
EVALUATION_INDEX = RESULTS / "evaluation_index.json"
DIRECT = RESULTS / "h2h_r11_vs_r12.json"
R9_QUALIFIER_DECISION = (
    ROOT
    / "train-ablation-1781126582/results/terminal_safe_reward_tail_followup/qualifier_decision_packet.json"
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


def _sample_active(window: dict[str, object]) -> dict[str, int]:
    return {
        element: int(values["active_sequences"])
        for element, values in window["sample"]["by_element"].items()
    }


def main() -> None:
    registration = _read_json(REGISTRATION)
    evaluation = _read_json(EVALUATION_INDEX)
    if evaluation.get("registration_sha256") != sha256(REGISTRATION):
        raise ValueError("evaluation index does not match current registration")
    if evaluation.get("completed_at") is None:
        raise ValueError("evaluation index is incomplete")
    registered = {arm["id"]: arm for arm in registration["arms"]}
    evaluated = {arm["id"]: arm for arm in evaluation["arms"]}
    if set(registered) != {"R11", "R12"} or set(evaluated) != {"R11", "R12"}:
        raise ValueError("decision requires complete R11 and R12 artifacts")

    arms = {
        arm_id: _arm_summary(evaluated[arm_id], registered[arm_id])
        for arm_id in ("R11", "R12")
    }
    for arm_id in ("R11", "R12"):
        endpoint = evaluated[arm_id]["windows"][-1]
        arms[arm_id]["direct_vs_R9_45M"] = _direct_summary(
            ROOT / endpoint["h2h_vs_control"]
        )
    direct = _direct_summary(DIRECT)
    r9_decision = _read_json(R9_QUALIFIER_DECISION)
    r9_endpoint = r9_decision["arm"]["windows"][-1]

    packet = {
        "schema_id": "azuki.terminal_safe_reward_exploration_decision",
        "schema_version": 1,
        "registration": str(REGISTRATION.relative_to(ROOT)),
        "registration_sha256": sha256(REGISTRATION),
        "evaluation_index": str(EVALUATION_INDEX.relative_to(ROOT)),
        "evaluation_index_sha256": sha256(EVALUATION_INDEX),
        "contract": {
            "arms": ["R11", "R12"],
            "sampled_rows_per_arm": registration["sampled_rows_per_arm"],
            "seed": registration["seed"],
            "evaluation_updates": evaluation["updates"],
            "trace_games_per_mode_window": evaluation["trace_games_per_mode_window"],
            "selection_rules": registration["selection_rules"],
        },
        "arms": arms,
        "R9_45M_reference": {
            "anchor_scores": r9_endpoint["anchor_scores"],
            "sample_active_sequences": _sample_active(r9_endpoint),
            "decision_disposition": r9_decision["decision"]["disposition"],
        },
        "direct_comparison": {
            "R11_vs_R12": direct,
            "R11_score_vs_R12": direct["score"],
            "R12_score_vs_R11": 1.0 - float(direct["score"]),
        },
        "decision": superseded_decision(),
    }
    output = RESULTS / "decision_packet.json"
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
