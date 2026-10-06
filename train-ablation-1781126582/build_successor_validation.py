#!/usr/bin/env python3
"""Compile retained successor qualification evidence without launching work."""
from __future__ import annotations

import argparse
from collections import Counter
import configparser
import hashlib
from itertools import combinations
import json
from pathlib import Path

from regenerate_strategy_descriptors import FAMILIES, REWARD_FAMILIES, read_json, resolve
from strategy_descriptor import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "train-ablation-1781126582/results"
OUTPUT = RESULTS / "successor_final_validation/decision_packet.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(path: Path) -> dict:
    return {"path": str(path.resolve().relative_to(ROOT)), "sha256": digest(path)}


def verify_content_id(payload: dict, field: str, path: Path) -> None:
    content = {key: value for key, value in payload.items() if key != field}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if payload[field] != "sha256:" + hashlib.sha256(canonical).hexdigest():
        raise ValueError(f"{field} content hash mismatch: {path}")


def descriptor_at(path: Path, checkpoint_sha256: str) -> dict:
    descriptor = read_json(path)
    if descriptor["schema_id"] != "azuki.strategy_descriptor" or descriptor["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"stale descriptor schema: {path}")
    verify_content_id(descriptor, "descriptor_id", path)
    if descriptor["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(f"descriptor checkpoint mismatch: {path}")
    return descriptor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-evidence", required=True, type=Path)
    args = parser.parse_args()
    runtime = read_json(args.runtime_evidence)
    reward_path = RESULTS / "terminal_safe_reward_horizon_followup/final_decision_packet.json"
    reward = read_json(reward_path)
    for field in ("registration", "evaluation_index"):
        if digest(resolve(reward[field])) != reward[f"{field}_sha256"]:
            raise ValueError(f"R14 {field} changed")

    descriptors = {}
    indexes = []
    for family in FAMILIES + REWARD_FAMILIES:
        family_indexes = sorted((RESULTS / family).glob("*evaluation_index.json"))
        if not family_indexes:
            raise ValueError(f"missing evaluation indexes: {family}")
        for index_path in family_indexes:
            index = read_json(index_path)
            registration_path = resolve(index["registration"])
            registration_hash = digest(registration_path)
            indexes.append({
                **evidence(index_path),
                "registration": str(registration_path.relative_to(ROOT)),
                "recorded_registration_sha256": index["registration_sha256"],
                "current_registration_sha256": registration_hash,
                "registration_matches": registration_hash == index["registration_sha256"],
            })
            for arm in index["arms"]:
                for window in arm["windows"]:
                    for mode in ("sample", "argmax"):
                        path = resolve(window[f"{mode}_descriptor"])
                        descriptor = descriptor_at(path, window["checkpoint_sha256"])
                        descriptors[str(path.relative_to(ROOT))] = descriptor["descriptor_id"]
    if len(descriptors) != 218:
        raise ValueError(f"incomplete retained ablation inventory: {len(descriptors)} != 218")

    baseline_path = RESULTS / "strategy_baseline_v1/baseline_packet.json"
    baseline = read_json(baseline_path)
    verify_content_id(baseline, "packet_id", baseline_path)
    if baseline["descriptor_schema"] != {"schema_id": "azuki.strategy_descriptor", "schema_version": SCHEMA_VERSION}:
        raise ValueError("baseline descriptor schema is stale")
    baseline_descriptors = {}
    for checkpoint in baseline["checkpoints"]:
        path = resolve(checkpoint["descriptor"])
        descriptor = descriptor_at(path, checkpoint["checkpoint_sha256"])
        if descriptor["descriptor_id"] != checkpoint["descriptor_id"]:
            raise ValueError(f"baseline descriptor reference is stale: {path}")
        trace_path = resolve(checkpoint["strategy_trace"])
        if digest(trace_path) != checkpoint["strategy_trace_sha256"]:
            raise ValueError(f"baseline trace changed: {trace_path}")
        baseline_descriptors[str(path.relative_to(ROOT))] = descriptor["descriptor_id"]
    updates = {row["update"] for row in baseline["checkpoints"]}
    pairs = [tuple(sorted((row["left_update"], row["right_update"]))) for row in baseline["pairwise_sensitivity"]]
    if len(baseline["checkpoints"]) != 7 or len(updates) != 7 or len(baseline_descriptors) != 7:
        raise ValueError("baseline must cover all seven retained checkpoints")
    if len(pairs) != 21 or set(pairs) != set(combinations(sorted(updates), 2)):
        raise ValueError("baseline must cover all 21 unique checkpoint pairs")

    manifest_path = RESULTS / "strategy_semantics_v3/prior_decision_manifest.json"
    archives = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(archives) != 17 or len({row["original"] for row in archives}) != 17:
        raise ValueError("prior decision manifest must preserve all 17 originals")
    for archive in archives:
        if digest(resolve(archive["archived"])) != archive["sha256"]:
            raise ValueError(f"prior decision archive changed: {archive['archived']}")

    config_path = resolve(reward["finalized_recipe"]["config"])
    if digest(config_path) != reward["finalized_recipe"]["config_sha256"]:
        raise ValueError("R14 reward config changed")
    config = configparser.ConfigParser()
    config.read(config_path)
    direct = {
        field: config.getfloat("process_env", f"azk_reward_{field}_delta_weight")
        for field in ("leader", "board")
    }
    telemetry = {
        field: reward["arm"]["reward_component_raw_abs_max"][f"direct_{field}_edge"]
        for field in ("leader", "board")
    }
    telemetry_zero = all(value == 0 for value in telemetry.values())
    redundant = all(value == 0 for value in direct.values()) and telemetry_zero
    existing_screens = {}
    for family in ("draft_screens", "curriculum_screens", "league_screens", "shaping_screens"):
        path = RESULTS / family / "registration.json"
        registration = read_json(path)
        existing_screens[family] = {
            **evidence(path),
            "sampled_rows_per_arm": registration["sampled_rows_per_arm"],
            "arms": [arm["id"] for arm in registration["arms"]],
        }

    endpoint = {}
    endpoint_contexts = {}
    for window in reward["arm"]["windows"]:
        for mode in ("sample", "argmax"):
            reference = window[mode]
            path = resolve(reference["descriptor"])
            descriptor = descriptor_at(path, window["checkpoint_sha256"])
            if reference["descriptor_id"] != descriptor["descriptor_id"] or reference["descriptor_schema_version"] != SCHEMA_VERSION:
                raise ValueError(f"R14 descriptor reference is stale: {path}")
            if reference["elemental_strategy"] != descriptor["elemental_strategy"]:
                raise ValueError(f"R14 strategy evidence is stale: {path}")
    for mode in ("sample", "argmax"):
        by_gate = {}
        contexts = reward["arm"]["windows"][-1][mode]["elemental_strategy"]["contexts"]
        for context in contexts.values():
            by_gate.setdefault(context["gate"], Counter()).update(context["counts"])
        endpoint[mode] = {gate: dict(counts) for gate, counts in sorted(by_gate.items())}
        endpoint_contexts[mode] = contexts

    qualified = reward["decision"]["reward_foundation_qualified"]
    checks = reward["derived"]["qualification_checks"]
    if not isinstance(qualified, bool) or qualified != all(checks.values()):
        raise ValueError("R14 qualification disagrees with current checks")
    sources = (
        "strategy_descriptor.py", "analyze_selfplay_games.py", "regenerate_strategy_descriptors.py",
        "build_strategy_baseline_packet.py", "build_terminal_safe_reward_final_decision.py",
        "build_successor_validation.py",
    )
    verification_path = RESULTS / "successor_final_validation/verification.json"
    packet = {
        "schema_id": "azuki.successor_final_validation", "schema_version": 1,
        "decision": "NO_GO", "launch_authorized": False, "frozen_candidate": None,
        "reward_decision": reward["decision"],
        "reward_qualification_checks": checks,
        "reward_unresolved_strategy_requirements": reward["derived"]["unresolved_strategy_requirements"],
        "reward_evidence": evidence(reward_path),
        "runtime_evidence": {**evidence(args.runtime_evidence), "observations": runtime},
        "verification": {**evidence(verification_path), "results": read_json(verification_path)},
        "source_hashes": {name: evidence(ROOT / "train-ablation-1781126582" / name) for name in sources},
        "prior_decisions": {**evidence(manifest_path), "verified_archive_count": len(archives), "archives": archives},
        "evaluator": {
            "descriptor_schema_version": SCHEMA_VERSION,
            "verified_ablation_descriptors": len(descriptors),
            "descriptor_ids": descriptors, "evaluation_indexes": indexes,
            "baseline_checkpoints": len(baseline["checkpoints"]),
            "baseline_pairs": len(pairs), "baseline": evidence(baseline_path),
            "baseline_descriptor_ids": baseline_descriptors,
        },
        "endpoint_by_gate": endpoint, "endpoint_contexts": endpoint_contexts,
        "existing_screens_not_long_confirmations": existing_screens,
        "remaining_boundaries": {
            "reward_foundation": {"status": "qualified" if qualified else "blocked", "failed_checks": [name for name, passed in checks.items() if not passed]},
            "runtime": "blocked: observed GPU access stalls; healthy runtime has not been established",
            "draft_D2_D3": "blocked: no longer matched D2/D3 confirmation on a qualified reward parent",
            "cross_gate_replay_off": "blocked: missing matched replay-off control after credit exclusion",
            "strategic_exposure": "blocked: selected draft parent absent; no-exposure/entity-only/strategic comparison required",
            "league": "blocked: selected parent absent; runtime and realized role/temporal diversity not qualified",
            "R3_direct_edge_removal": {"status": "redundant" if redundant else "unresolved", "config": evidence(config_path), "configured_weights": direct, "raw_abs_max": telemetry, "telemetry_zero": telemetry_zero},
            "S1_S3": "blocked: separate final interactions require otherwise frozen recipe",
            "candidate_freeze": "blocked: prerequisites incomplete; no frozen candidate and no 1B launch",
        },
        "prerequisites": [
            "Resolve failed reward foundation checks under current descriptor semantics before successor-parent qualification.",
            "Establish a healthy runtime before any GPU-dependent confirmation; this report performs no recovery or probes.",
            "Complete matched draft, replay-off, strategic-exposure and league boundaries, then separate S1/S3 final interactions.",
            "Freeze a candidate only after all boundaries qualify; retained short screens do not substitute for confirmations.",
        ],
        "interpretation": [
            "Sparse observed conversion changes are not proof of permanent forgetting; qualification follows the current registered checks.",
            "Water, Earth, Fire and Lightning require separate gate/leader-conditioned deck and effect evidence. No pooled strategy score or early-win-rate ranker is used.",
            "Unregistered coherent lines remain admissible; missing physical-copy IDs, deferred-effect attribution and counterfactual values remain explicitly unmeasured.",
            "Descriptor content hashes verify retained artifacts, not an independent rerun of the evaluator.",
            "No training run, driver reset or host reboot is authorized by this report.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{OUTPUT.relative_to(ROOT)}: NO_GO; {len(descriptors)} verified descriptor IDs")


if __name__ == "__main__":
    main()
