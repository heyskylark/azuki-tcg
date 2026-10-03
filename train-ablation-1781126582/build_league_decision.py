#!/usr/bin/env python3
"""Compile the completed L0-L5 league screen into a decision packet."""

from __future__ import annotations

import json
from pathlib import Path

from strategy_descriptor import SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "train-ablation-1781126582/results/league_screens"
INDEX = RESULTS / "evaluation_index.json"
REGISTRATION = RESULTS / "registration.json"
ROLE_NAMES = ("anchor", "recent", "hard", "distinct", "history")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def descriptor_summary(path: Path) -> dict:
    descriptor = read_json(path)
    if descriptor.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"stale strategy descriptor: {path}")
    sequences = {
        element: sum(
            int(sequence["completed"])
            for sequence in descriptor["sequences"].values()
            if sequence["element"] == element
        )
        for element in ("EARTH", "FIRE", "LIGHTNING", "WATER")
    }
    sibling_pairs = descriptor["deck"]["relationships"]["sibling_pairs"]
    sibling_distance = sum(
        float(pair["multiset_distance"]) for pair in sibling_pairs
    ) / len(sibling_pairs)
    return {
        "descriptor": str(path.relative_to(ROOT)),
        "descriptor_id": descriptor["descriptor_id"],
        "elemental_strategy": descriptor["elemental_strategy"],
        "completed_sequences_by_element": sequences,
        "face_share_when_both_legal": descriptor["funnels"]
        ["attack.both_legal_face_choice"]["selected_per_opportunity"],
        "sibling_mean_multiset_distance": sibling_distance,
    }


def curated_score(path: Path) -> float:
    panel = read_json(path)
    games = sum(int(cell["games"]) for cell in panel["by_gate"].values())
    return sum(
        float(cell["score"]) * int(cell["games"])
        for cell in panel["by_gate"].values()
    ) / games


def endpoint_summary(arm: dict) -> dict:
    window = arm["windows"][-1]
    anchors = {
        label: {
            "score": read_json(ROOT / path)["summary"]["score"],
            "paired_lcb_80": read_json(ROOT / path)["summary"]["paired_lcb_80"],
        }
        for label, path in window["h2h"].items()
    }
    direct = (
        read_json(ROOT / window["h2h_vs_control"])["summary"]
        if "h2h_vs_control" in window
        else None
    )
    return {
        "checkpoint": window["checkpoint"],
        "checkpoint_sha256": window["checkpoint_sha256"],
        "steady_sps_median": arm["training"]["steady_sps_median"],
        "direct_vs_l0": (
            {
                "score": direct["score"],
                "paired_lcb_80": direct["paired_lcb_80"],
                "timeout_rate": direct["timeout_rate"],
            }
            if direct is not None
            else None
        ),
        "anchors": anchors,
        "curated_strategy_score": curated_score(ROOT / window["curated_strategy_panel"]),
        "sample": descriptor_summary(ROOT / window["sample_descriptor"]),
        "argmax": descriptor_summary(ROOT / window["argmax_descriptor"]),
    }


def final_role_shares(slug: str) -> dict[str, float] | None:
    log_path = RESULTS / slug / "logs/production.jsonl"
    metric_rows = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line.strip() and '"epoch"' in line
    ]
    final = metric_rows[-1]
    keys = {
        role: f"environment/league/sampling/role/{role}/completed_episode_share"
        for role in ROLE_NAMES
    }
    if not any(key in final for key in keys.values()):
        return None
    return {role: float(final.get(key, 0.0)) for role, key in keys.items()}


def main() -> None:
    index = read_json(INDEX)
    registration = read_json(REGISTRATION)
    arm_registration = {arm["id"]: arm for arm in registration["arms"]}
    arms = {}
    for arm in index["arms"]:
        arm_id = arm["id"]
        endpoint = endpoint_summary(arm)
        endpoint["completed_role_share"] = final_role_shares(arm["name"])
        endpoint["sps_ratio_vs_l0"] = (
            endpoint["steady_sps_median"]
            / index["arms"][0]["training"]["steady_sps_median"]
        )
        endpoint["config"] = arm_registration[arm_id]["config"]
        arms[arm_id] = endpoint

    packet = {
        "schema_id": "azuki.league_screen_decision",
        "schema_version": 1,
        "registration": str(REGISTRATION.relative_to(ROOT)),
        "evaluation_index": str(INDEX.relative_to(ROOT)),
        "parent": "C1",
        "arms": arms,
        "decision": {
            "production_qualified": False,
            "status": "requires_r14_matched_runtime_and_strategy_confirmation",
            "league_45m_qualifiers": [],
            "reason": [
                "These are 15M C1-parent screens, not matched R14 recipe confirmations.",
                "Actual completed-role shares and SPS ratios are reported per arm; configured roles do not establish exposure diversity.",
                "Schema-v3 per-context strategy evidence supersedes v2 conclusions. Early direct or anchor win rate does not select the league parent.",
            ],
            "disposition": {
                "L0": "ratio_control",
                "L1": "rejected_throughput",
                "L2": "rejected_role_and_throughput",
                "L3": "rejected_role_and_throughput",
                "L4": "rejected_role_and_throughput",
                "L5": "requires_strategy_requalification",
            },
            "next_stage": "Qualify runtime and realized role/temporal diversity on the selected R14 draft/exposure recipe before shaping interactions.",
        },
    }
    (RESULTS / "decision_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
