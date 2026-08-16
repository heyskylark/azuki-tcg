#!/usr/bin/env python3
"""Merge paired argmax/random-prefix trajectory datasets with unique identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _dataset_spec(value: str) -> tuple[str, Path]:
  name, separator, path = value.partition("=")
  if not separator or not name.strip() or not path.strip():
    raise argparse.ArgumentTypeError("--dataset must use NAME=PATH")
  return name.strip(), Path(path.strip())


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", action="append", type=_dataset_spec, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--summary-json", type=Path, required=True)
  args = parser.parse_args()
  names = [name for name, _ in args.dataset]
  if len(names) < 2 or len(set(names)) != len(names):
    raise ValueError("at least two uniquely named datasets are required")

  merged: list[dict] = []
  by_variant: dict[str, list[dict]] = {}
  seen: set[str] = set()
  for variant, directory in args.dataset:
    shards = sorted(directory.glob("*.jsonl"))
    if not shards:
      raise ValueError(f"dataset {variant} has no JSONL shards: {directory}")
    variant_rows: list[dict] = []
    for shard in shards:
      with shard.open(encoding="utf-8") as handle:
        for line in handle:
          if not line.strip():
            continue
          row = json.loads(line)
          original_id = str(row["trajectory_id"])
          trajectory_id = f"{variant}:{original_id}"
          if trajectory_id in seen:
            raise ValueError(f"duplicate merged trajectory id: {trajectory_id}")
          seen.add(trajectory_id)
          rewritten = {
            **row,
            "trajectory_id": trajectory_id,
            "source_trajectory_id": original_id,
            "dataset_variant": variant,
          }
          variant_rows.append(rewritten)
          merged.append(rewritten)
    by_variant[variant] = variant_rows

  coordinate_maps = {
    variant: {
      (
        str(row["policy_generation"]),
        str(row["opponent_lineage"]),
        str(row["game_id"]),
      ): row
      for row in rows
    }
    for variant, rows in by_variant.items()
  }
  coordinates = [set(mapping) for mapping in coordinate_maps.values()]
  if any(values != coordinates[0] for values in coordinates[1:]):
    raise ValueError("dataset variants do not contain the same paired coordinates")
  paired_coordinates = sorted(coordinates[0])
  baseline_name = names[0]
  pairwise: dict[str, dict[str, float | int]] = {}
  for variant in names[1:]:
    deck_changed = 0
    outcome_changed = 0
    for coordinate in paired_coordinates:
      baseline = coordinate_maps[baseline_name][coordinate]
      candidate = coordinate_maps[variant][coordinate]
      deck_changed += baseline["main_card_ids"] != candidate["main_card_ids"]
      outcome_changed += float(baseline["target"]) != float(candidate["target"])
    pairwise[f"{baseline_name}_vs_{variant}"] = {
      "pairs": len(paired_coordinates),
      "deck_changed": deck_changed,
      "deck_changed_fraction": deck_changed / len(paired_coordinates),
      "outcome_changed": outcome_changed,
      "outcome_changed_fraction": outcome_changed / len(paired_coordinates),
    }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("w", encoding="utf-8") as handle:
    for row in merged:
      handle.write(json.dumps(row, separators=(",", ":")) + "\n")
  summary = {
    "schema_version": 1,
    "variants": {
      variant: {
        "directory": str(directory.resolve()),
        "trajectories": len(by_variant[variant]),
      }
      for variant, directory in args.dataset
    },
    "merged_trajectories": len(merged),
    "paired_coordinates": len(paired_coordinates),
    "pairwise": pairwise,
  }
  args.summary_json.parent.mkdir(parents=True, exist_ok=True)
  args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
  print(
    f"[prefix-outcome-merge] variants={len(names)} trajectories={len(merged)} "
    f"pairs={len(paired_coordinates)}",
    flush=True,
  )


if __name__ == "__main__":
  main()
