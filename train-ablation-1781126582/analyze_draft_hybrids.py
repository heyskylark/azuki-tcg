#!/usr/bin/env python3
"""Analyze matched-versus-sibling gate/leader/main fixed-deck outcomes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


ARMS = ("matched", "sibling_main", "sibling_leader", "sibling_both")
ELEMENT_BY_GATE = {
    "STT01-002": "LIGHTNING",
    "AZK01-120": "LIGHTNING",
    "STT02-002": "WATER",
    "AZK01-126": "WATER",
    "AZK01-122": "FIRE",
    "STT04-002": "FIRE",
    "AZK01-124": "EARTH",
    "STT03-002": "EARTH",
}


def _read_games(paths: list[Path]) -> list[dict]:
    games = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                game = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            games.append(game)
    if not games:
        raise ValueError("No hybrid counterfactual games were provided")
    return games


def _key(game: dict) -> tuple[str, int, int]:
    metadata = game["counterfactual"]
    return (
        str(metadata["target_gate"]),
        int(metadata["reference_index"]),
        int(metadata["candidate_seat"]),
    )


def _score(game: dict) -> float:
    return float(game["counterfactual"]["candidate_score"])


def _summarize_games(games: list[dict]) -> dict[str, object]:
    scores = np.asarray([_score(game) for game in games], dtype=np.float64)
    return {
        "games": len(games),
        "score": float(scores.mean()),
        "wins": int(np.sum(scores == 1.0)),
        "draws": int(np.sum(scores == 0.5)),
        "losses": int(np.sum(scores == 0.0)),
        "truncations": sum(bool(game["outcome"].get("truncated", False)) for game in games),
    }


def _paired_delta(
    by_arm: dict[str, list[dict]],
    candidate_arm: str,
    *,
    gates: frozenset[str] | None = None,
    seed: int,
) -> dict[str, object]:
    baseline = {_key(game): _score(game) for game in by_arm["matched"]}
    candidate = {_key(game): _score(game) for game in by_arm[candidate_arm]}
    if baseline.keys() != candidate.keys():
        raise ValueError(f"{candidate_arm} schedule differs from matched")
    by_block: dict[tuple[str, int], list[float]] = defaultdict(list)
    for key, baseline_score in baseline.items():
        gate, reference_index, _ = key
        if gates is not None and gate not in gates:
            continue
        by_block[(gate, reference_index)].append(candidate[key] - baseline_score)
    malformed = [block for block, values in by_block.items() if len(values) != 2]
    if malformed:
        raise ValueError(f"Blocks do not contain both seats: {malformed[:5]}")
    deltas = np.asarray(
        [float(np.mean(values)) for _, values in sorted(by_block.items())],
        dtype=np.float64,
    )
    if deltas.size == 0:
        raise ValueError("Paired hybrid selector produced no blocks")
    rng = np.random.default_rng(seed)
    sampled = deltas[
        rng.integers(0, len(deltas), size=(30_000, len(deltas)))
    ].mean(axis=1)
    candidate_minus_matched = float(deltas.mean())
    return {
        "paired_blocks": int(deltas.size),
        "seat_games": int(2 * deltas.size),
        "candidate_arm": candidate_arm,
        "candidate_minus_matched": candidate_minus_matched,
        "candidate_minus_matched_ci80": [
            float(np.quantile(sampled, 0.10)),
            float(np.quantile(sampled, 0.90)),
        ],
        "candidate_minus_matched_ci95": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "matched_advantage": -candidate_minus_matched,
        "matched_advantage_ci80": [
            -float(np.quantile(sampled, 0.90)),
            -float(np.quantile(sampled, 0.10)),
        ],
        "matched_better_blocks": int(np.sum(deltas < 0.0)),
        "tied_blocks": int(np.sum(deltas == 0.0)),
        "matched_worse_blocks": int(np.sum(deltas > 0.0)),
    }


def build_report(games: list[dict], label: str) -> dict:
    by_arm: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for game in games:
        arm = str(game["counterfactual"]["arm"])
        if arm not in ARMS:
            raise ValueError(f"Unexpected hybrid arm {arm}")
        unique = (arm, *_key(game))
        if unique in seen:
            raise ValueError(f"Duplicate hybrid task {unique}")
        seen.add(unique)
        by_arm[arm].append(game)
    if set(by_arm) != set(ARMS):
        raise ValueError(f"Hybrid arms differ: {sorted(by_arm)}")
    schedules = [{_key(game) for game in by_arm[arm]} for arm in ARMS]
    if any(schedule != schedules[0] for schedule in schedules[1:]):
        raise ValueError("Hybrid arm schedules do not match")

    arm_summaries = {}
    for arm in ARMS:
        arm_summaries[arm] = {
            "overall": _summarize_games(by_arm[arm]),
            "by_element": {
                element: _summarize_games(
                    [
                        game for game in by_arm[arm]
                        if ELEMENT_BY_GATE[str(game["counterfactual"]["target_gate"])] == element
                    ]
                )
                for element in sorted(set(ELEMENT_BY_GATE.values()))
            },
            "by_gate": {
                gate: _summarize_games(
                    [
                        game for game in by_arm[arm]
                        if str(game["counterfactual"]["target_gate"]) == gate
                    ]
                )
                for gate in ELEMENT_BY_GATE
            },
        }

    comparisons = {}
    for arm_index, arm in enumerate(ARMS[1:], start=1):
        comparisons[arm] = {
            "overall": _paired_delta(
                by_arm,
                arm,
                seed=42_907_000 + arm_index,
            ),
            "by_element": {
                element: _paired_delta(
                    by_arm,
                    arm,
                    gates=frozenset(
                        gate for gate, item_element in ELEMENT_BY_GATE.items()
                        if item_element == element
                    ),
                    seed=42_907_100 + 10 * arm_index + element_index,
                )
                for element_index, element in enumerate(
                    sorted(set(ELEMENT_BY_GATE.values()))
                )
            },
            "by_gate": {
                gate: _paired_delta(
                    by_arm,
                    arm,
                    gates=frozenset((gate,)),
                    seed=42_907_500 + 20 * arm_index + gate_index,
                )
                for gate_index, gate in enumerate(ELEMENT_BY_GATE)
            },
        }
    return {
        "schema_version": 1,
        "label": label,
        "games": len(games),
        "paired_schedule": True,
        "arms": arm_summaries,
        "matched_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--label", required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(_read_games(args.inputs), args.label)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                arm: report["matched_comparisons"][arm]["overall"]["matched_advantage"]
                for arm in ARMS[1:]
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
