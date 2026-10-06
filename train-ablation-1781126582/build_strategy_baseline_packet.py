#!/usr/bin/env python3
"""Recompute the seven-checkpoint strategy baseline packet."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import time

from analyze_selfplay_games import load_games
from strategy_descriptor import SCHEMA_VERSION, SEQUENCE_REGISTRY, build_strategy_descriptor, payoff_vector_distance


ROOT = Path(__file__).resolve().parent
GATE_ROOT = ROOT / "results" / "corrected_production_1b_lr1500_fresh" / "gates"
EXPERIMENT_ROOT = ROOT.parent / "experiments"


@dataclass(frozen=True)
class CheckpointSpec:
    update: int
    trace: Path
    manifest: Path
    h2h_pattern: str
    holdout: Path
    context_decks: Path | None = None


SPECS = (
    CheckpointSpec(
        21_000,
        GATE_ROOT / "mechanics_800m_p021000.jsonl",
        EXPERIMENT_ROOT / "azuki_local_corrected_production_1b_lr1500_fresh_resume_p19500_178757505266" / "checkpoint_021000.manifest.json",
        "h2h_450m_p021000_vs_*.json",
        GATE_ROOT / "holdout_450m_p021000_formal.json",
    ),
    CheckpointSpec(
        29_300,
        GATE_ROOT / "mechanics_800m_p029300.jsonl",
        EXPERIMENT_ROOT / "azuki_local_corrected_production_1b_lr1500_fresh_resume_p19500_178757505266" / "checkpoint_029300.manifest.json",
        "h2h_450m_p029300_vs_*.json",
        GATE_ROOT / "holdout_450m_p029300_formal.json",
    ),
    CheckpointSpec(
        44_000,
        GATE_ROOT / "mechanics_800m_p044000.jsonl",
        EXPERIMENT_ROOT / "azuki_local_corrected_production_1b_lr1500_fresh_resume_p29300_178765385220" / "checkpoint_044000.manifest.json",
        "h2h_800m_p044000_vs_*.json",
        GATE_ROOT / "holdout_800m_p044000_formal.json",
    ),
    CheckpointSpec(
        52_000,
        GATE_ROOT / "mechanics_800m_p052000.jsonl",
        EXPERIMENT_ROOT / "azuki_local_corrected_production_1b_lr1500_fresh_resume_p29300_178765385220" / "checkpoint_052000.manifest.json",
        "h2h_800m_p052000_vs_*.json",
        GATE_ROOT / "holdout_800m_p052000_formal.json",
    ),
    CheckpointSpec(
        56_000,
        GATE_ROOT / "mechanics_1b_p056000.jsonl",
        EXPERIMENT_ROOT / "azuki_local_corrected_production_1b_lr1500_fresh_resume_p52750_178783658456" / "checkpoint_056000.manifest.json",
        "h2h_1b_p056000_vs_*.json",
        GATE_ROOT / "holdout_1b_p056000_formal.json",
        GATE_ROOT / "context_decks_1b_p056000.json",
    ),
    CheckpointSpec(
        60_000,
        GATE_ROOT / "mechanics_1b_p060000.jsonl",
        EXPERIMENT_ROOT / "azuki_local_corrected_production_1b_lr1500_fresh_resume_p52750_178783658456" / "checkpoint_060000.manifest.json",
        "h2h_1b_p060000_vs_*.json",
        GATE_ROOT / "holdout_1b_p060000_formal.json",
        GATE_ROOT / "context_decks_1b_p060000.json",
    ),
    CheckpointSpec(
        65_105,
        GATE_ROOT / "mechanics_1b_p065105.jsonl",
        EXPERIMENT_ROOT / "azuki_local_corrected_production_1b_lr1500_fresh_resume_p52750_178783658456" / "checkpoint_065105.manifest.json",
        "h2h_1b_p065105_vs_*.json",
        GATE_ROOT / "holdout_1b_p065105_formal.json",
        GATE_ROOT / "context_decks_1b_p065105.json",
    ),
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_hash(manifest: dict) -> str:
    matches = [
        str(artifact["sha256"])
        for artifact in manifest["artifacts"]
        if str(artifact["path"]).startswith("model_")
        and str(artifact["path"]).endswith(".pt")
    ]
    if len(matches) != 1:
        raise ValueError(f"Manifest must contain one model artifact, found {matches}")
    return matches[0]


def _context_key(source: str, opponent: str, game: dict) -> str:
    values = (
        ("source", source),
        ("opponent", opponent),
        ("candidate_gate", game.get("candidate_gate_code", game.get("candidate_gate", "?"))),
        ("candidate_leader", game.get("candidate_leader", "?")),
        ("opponent_gate", game.get("opponent_gate", "?")),
        ("opponent_leader", game.get("opponent_leader", "?")),
        ("candidate_seat", game.get("candidate_seat", "?")),
        ("candidate_started", bool(game.get("candidate_started", False))),
        ("reference_deck", game.get("reference_deck_index", -1)),
    )
    return "|".join(f"{key}={value}" for key, value in values)


def _payoff_cells(spec: CheckpointSpec, checkpoint_hash: str) -> tuple[list[dict], list[str]]:
    paths = sorted(GATE_ROOT.glob(spec.h2h_pattern)) + [spec.holdout]
    grouped: dict[str, list[float]] = defaultdict(list)
    sources = []
    for path in paths:
        payload = _read(path)
        candidate = payload.get("candidate", {})
        observed_hash = str(candidate.get("checkpoint_sha256", ""))
        if observed_hash != checkpoint_hash:
            raise ValueError(
                f"Payoff artifact {path} has checkpoint hash {observed_hash}, expected {checkpoint_hash}"
            )
        mode = str(payload.get("policy_action_mode", "unknown"))
        if mode != "legal_argmax_stable_first":
            raise ValueError(f"Payoff artifact {path} is not deterministic: {mode}")
        source = "holdout" if path == spec.holdout else "h2h"
        opponent = str(
            payload.get("opponent", {}).get(
                "label", payload.get("reference_pilot", {}).get("label", "reference_pool")
            )
        )
        games = payload.get("games", [])
        if not isinstance(games, list) or not games:
            raise ValueError(f"Payoff artifact {path} has no games")
        for game in games:
            score = game.get("candidate_score", game.get("score"))
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ValueError(f"Payoff game in {path} has no numeric score")
            grouped[_context_key(source, opponent, game)].append(float(score))
        sources.append(str(path.relative_to(ROOT)))
    cells = [
        {
            "context_key": key,
            "score": sum(scores) / len(scores),
            "games": len(scores),
        }
        for key, scores in sorted(grouped.items())
    ]
    return cells, sources


def _strategy_vector(descriptor: dict) -> dict[str, float]:
    vector = {}
    for name, funnel in descriptor["funnels"].items():
        for field in (
            "selected_per_opportunity",
            "resolved_per_observed_selection",
            "converted_per_observed_resolution",
        ):
            value = funnel[field]
            if value is not None:
                vector[f"funnel/{name}/{field}"] = float(value)
    for name, sequence in descriptor["sequences"].items():
        for field in ("completion_per_eligible", "conversion_per_completed"):
            value = sequence[field]
            if value is not None:
                vector[f"sequence/{name}/{field}"] = float(value)
    return vector


def _vector_distance(left: dict[str, float], right: dict[str, float]) -> tuple[int, float]:
    common = sorted(left.keys() & right.keys())
    if not common:
        return 0, 0.0
    squared = sum((left[key] - right[key]) ** 2 for key in common)
    return len(common), math.sqrt(squared / len(common))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "strategy_baseline_v1",
    )
    parser.add_argument(
        "--reward-telemetry",
        type=Path,
        default=ROOT / "results" / "reward_component_baseline_v1.json",
    )
    parser.add_argument(
        "--curated-panel",
        type=Path,
        default=(
            ROOT
            / "results"
            / "strategy_baseline_v1"
            / "curated_strategy_panel_p065105.json"
        ),
    )
    args = parser.parse_args()
    descriptor_dir = args.out_dir / "descriptors"
    descriptor_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_rows = []
    descriptors = {}
    total_started = time.perf_counter()
    for spec in SPECS:
        manifest = _read(spec.manifest)
        if int(manifest["update"]) != spec.update:
            raise ValueError(f"Manifest update mismatch for {spec.manifest}")
        checkpoint_hash = _model_hash(manifest)
        payoff_cells, payoff_sources = _payoff_cells(spec, checkpoint_hash)
        load_started = time.perf_counter()
        games = load_games([spec.trace])
        load_seconds = time.perf_counter() - load_started
        evaluate_started = time.perf_counter()
        descriptor = build_strategy_descriptor(
            games,
            label=f"p{spec.update:06d}",
            checkpoint_sha256=checkpoint_hash,
            payoff_cells=payoff_cells,
        )
        evaluate_seconds = time.perf_counter() - evaluate_started
        descriptor_path = descriptor_dir / f"p{spec.update:06d}.json"
        descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
        descriptors[spec.update] = descriptor
        relationships = descriptor["deck"]["relationships"]
        checkpoint_rows.append(
            {
                "update": spec.update,
                "checkpoint_sha256": checkpoint_hash,
                "manifest": str(spec.manifest.relative_to(ROOT.parent)),
                "strategy_trace": str(spec.trace.relative_to(ROOT)),
                "strategy_trace_sha256": _file_sha256(spec.trace),
                "strategy_source_action_mode": "sample_temperature_1_no_smoothing",
                "deterministic_payoff_sources": payoff_sources,
                "context_decks_source": (
                    str(spec.context_decks.relative_to(ROOT))
                    if spec.context_decks is not None
                    else None
                ),
                "descriptor": str(descriptor_path.relative_to(ROOT)),
                "descriptor_id": descriptor["descriptor_id"],
                "games": descriptor["games"],
                "payoff_cells": len(payoff_cells),
                "sibling_exact_collision_pairs": relationships[
                    "sibling_exact_collision_pairs"
                ],
                "runtime": {
                    "load_seconds": load_seconds,
                    "evaluate_seconds": evaluate_seconds,
                    "games_per_evaluator_second": len(games) / evaluate_seconds,
                },
            }
        )

    distances = []
    for left_update, right_update in combinations(sorted(descriptors), 2):
        left = descriptors[left_update]
        right = descriptors[right_update]
        payoff_common, payoff_distance = payoff_vector_distance(
            left["payoff_vector"], right["payoff_vector"]
        )
        strategy_common, strategy_distance = _vector_distance(
            _strategy_vector(left), _strategy_vector(right)
        )
        distances.append(
            {
                "left_update": left_update,
                "right_update": right_update,
                "payoff_common_cells": payoff_common,
                "payoff_rms_distance": payoff_distance,
                "strategy_common_metrics": strategy_common,
                "strategy_rms_distance": strategy_distance,
            }
        )

    reward = _read(args.reward_telemetry)
    curated = _read(args.curated_panel)
    if curated.get("schema_id") != "azuki.curated_strategy_panel":
        raise ValueError("Curated strategy panel has the wrong schema")
    if curated.get("policy_action_mode") != "legal_argmax_stable_first":
        raise ValueError("Curated strategy panel must use deterministic legal argmax")
    if len(curated.get("by_gate", {})) != 8:
        raise ValueError("Curated strategy panel must cover all eight gates")
    curated_records = curated.get("records", [])
    if len(curated_records) != curated.get("games"):
        raise ValueError("Curated strategy panel game count does not match its records")
    if any(not record.get("completed", False) for record in curated_records):
        raise ValueError("Curated strategy panel contains an incomplete game")
    for gate, gate_row in curated["by_gate"].items():
        if gate_row.get("games") != 4 or gate_row.get("completed") != 4:
            raise ValueError(f"Curated strategy panel gate {gate} is incomplete")
    raw_reconstruction_error = reward["metrics"][
        "reward_telemetry/raw_reconstruction_max_abs_error"
    ]
    scaled_reconstruction_error = reward["metrics"][
        "reward_telemetry/scaled_reconstruction_max_abs_error"
    ]
    if raw_reconstruction_error > 2e-6 or scaled_reconstruction_error > 2e-6:
        raise ValueError("Reward telemetry reconstruction exceeds tolerance")
    if len(distances) != 21 or any(
        row["payoff_common_cells"] <= 0 or row["strategy_common_metrics"] <= 0
        for row in distances
    ):
        raise ValueError("Pairwise sensitivity matrix is incomplete")
    if len({row["descriptor_id"] for row in checkpoint_rows}) != 7:
        raise ValueError("Strategy descriptor IDs are not unique")
    packet = {
        "schema_id": "azuki.strategy_baseline_packet",
        "schema_version": 1,
        "descriptor_schema": {
            "schema_id": "azuki.strategy_descriptor",
            "schema_version": SCHEMA_VERSION,
        },
        "checkpoints": checkpoint_rows,
        "pairwise_sensitivity": distances,
        "reward_component_distribution": {
            "path": str(args.reward_telemetry.relative_to(ROOT)),
            "schema_id": reward["schema_id"],
            "schema_version": reward["schema_version"],
            "games": reward["games"],
            "wall_time_seconds": reward["wall_time_seconds"],
            "raw_reconstruction_max_abs_error": raw_reconstruction_error,
            "scaled_reconstruction_max_abs_error": scaled_reconstruction_error,
        },
        "curated_strategy_panel": {
            "path": str(args.curated_panel.relative_to(ROOT)),
            "schema_id": curated["schema_id"],
            "schema_version": curated["schema_version"],
            "games": curated["games"],
            "gates": len(curated["by_gate"]),
            "wall_time_seconds": curated["wall_time_seconds"],
            "control": curated["contract"]["control"],
            "completed_games": len(curated_records),
            "mean_score": sum(record["score"] for record in curated_records)
            / len(curated_records),
        },
        "sensitivity_contract": {
            "stochastic_policy_updates": [spec.update for spec in SPECS],
            "deterministic_payoff_updates": [spec.update for spec in SPECS],
            "gate_contexts": 8,
            "leader_contexts_per_gate": 2,
            "scripted_sequence_positive_negative_fixtures": len(SEQUENCE_REGISTRY),
            "irrelevant_noop_invariance": True,
            "illegal_action_rejection": True,
            "entity_only_face_pressure_probe": {
                "status": "complete",
                "gates": len(curated["by_gate"]),
                "games": curated["games"],
            },
        },
        "runtime": {
            "total_packet_seconds": time.perf_counter() - total_started,
            "reward_collector_seconds": reward["wall_time_seconds"],
            "curated_panel_seconds": curated["wall_time_seconds"],
        },
    }
    canonical = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode("utf-8")
    packet["packet_id"] = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    packet_path = args.out_dir / "baseline_packet.json"
    packet_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {packet_path} ({len(checkpoint_rows)} checkpoints)")


if __name__ == "__main__":
    main()
