#!/usr/bin/env python3
"""Analyze matched gate-leader mains against factorial context mismatches."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


ARMS = ("matched", "sibling_gate_main", "sibling_leader_main", "sibling_both_main")


def _read_games(paths: list[Path]) -> list[dict]:
  games = []
  for path in paths:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
      if not line.strip():
        continue
      try:
        games.append(json.loads(line))
      except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
  if not games:
    raise ValueError("No context-hybrid games were provided")
  return games


def _key(game: dict) -> tuple[str, int, int]:
  metadata = game["counterfactual"]
  return (
    str(metadata["target_gate"]),
    int(metadata["reference_index"]),
    int(metadata["candidate_seat"]),
  )


def _element(context_id: str) -> str:
  pieces = context_id.split(":", 2)
  if len(pieces) != 3 or not all(pieces):
    raise ValueError(f"Invalid context id {context_id!r}")
  return pieces[0]


def _score(game: dict) -> float:
  return float(game["counterfactual"]["candidate_score"])


def _summary(games: list[dict]) -> dict[str, object]:
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
  contexts: frozenset[str] | None,
  seed: int,
) -> dict[str, object]:
  matched = {_key(game): _score(game) for game in by_arm["matched"]}
  candidate = {_key(game): _score(game) for game in by_arm[candidate_arm]}
  if matched.keys() != candidate.keys():
    raise ValueError(f"{candidate_arm} schedule differs from matched")
  blocks: dict[tuple[str, int], list[float]] = defaultdict(list)
  for key, baseline_score in matched.items():
    context_id, reference_index, _ = key
    if contexts is not None and context_id not in contexts:
      continue
    blocks[(context_id, reference_index)].append(candidate[key] - baseline_score)
  malformed = [key for key, values in blocks.items() if len(values) != 2]
  if malformed:
    raise ValueError(f"Context blocks do not contain both seats: {malformed[:5]}")
  deltas = np.asarray(
    [float(np.mean(values)) for _, values in sorted(blocks.items())],
    dtype=np.float64,
  )
  if deltas.size == 0:
    raise ValueError("Context-hybrid selector produced no paired blocks")
  rng = np.random.default_rng(seed)
  sampled = deltas[
    rng.integers(0, deltas.size, size=(30_000, deltas.size))
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
      raise ValueError(f"Unexpected context-hybrid arm {arm}")
    unique = (arm, *_key(game))
    if unique in seen:
      raise ValueError(f"Duplicate context-hybrid task {unique}")
    seen.add(unique)
    by_arm[arm].append(game)
  if set(by_arm) != set(ARMS):
    raise ValueError(f"Context-hybrid arms differ: {sorted(by_arm)}")
  schedules = [{_key(game) for game in by_arm[arm]} for arm in ARMS]
  if any(schedule != schedules[0] for schedule in schedules[1:]):
    raise ValueError("Context-hybrid arm schedules do not match")

  context_ids = sorted({key[0] for key in schedules[0]})
  elements = sorted({_element(context_id) for context_id in context_ids})
  arm_summaries = {}
  for arm in ARMS:
    arm_summaries[arm] = {
      "overall": _summary(by_arm[arm]),
      "by_element": {
        element: _summary([
          game for game in by_arm[arm] if _element(_key(game)[0]) == element
        ])
        for element in elements
      },
      "by_context": {
        context_id: _summary([
          game for game in by_arm[arm] if _key(game)[0] == context_id
        ])
        for context_id in context_ids
      },
    }

  comparisons = {}
  for arm_index, arm in enumerate(ARMS[1:], start=1):
    comparisons[arm] = {
      "overall": _paired_delta(
        by_arm,
        arm,
        contexts=None,
        seed=42_908_000 + arm_index,
      ),
      "by_element": {
        element: _paired_delta(
          by_arm,
          arm,
          contexts=frozenset(
            context_id for context_id in context_ids if _element(context_id) == element
          ),
          seed=42_908_100 + 10 * arm_index + element_index,
        )
        for element_index, element in enumerate(elements)
      },
    }
  return {
    "schema_version": 1,
    "label": label,
    "games": len(games),
    "paired_schedule": True,
    "contexts": context_ids,
    "arms": arm_summaries,
    "matched_comparisons": comparisons,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
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
    ),
    flush=True,
  )


if __name__ == "__main__":
  main()
