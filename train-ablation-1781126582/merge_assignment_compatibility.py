#!/usr/bin/env python3
"""Merge element shards emitted by assignment_compatibility_probe.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from assignment_compatibility_probe import _markdown


def _weighted(contexts: list[dict], key: str) -> float:
    weights = np.asarray([int(item["main_pick_steps"]) for item in contexts], dtype=np.float64)
    values = np.asarray([float(item[key]) for item in contexts], dtype=np.float64)
    return float(np.average(values, weights=weights))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    args = parser.parse_args()
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not shards:
        raise ValueError("At least one shard is required")
    checkpoint = shards[0]["checkpoint"]
    episodes = shards[0]["episodes_per_context"]
    if any(item["checkpoint"] != checkpoint for item in shards):
        raise ValueError("Shard checkpoints differ")
    if any(item["episodes_per_context"] != episodes for item in shards):
        raise ValueError("Shard episode counts differ")
    contexts = [context for shard in shards for context in shard["contexts"]]
    context_ids = {
        (context["element"], context["gate"], context["leader"])
        for context in contexts
    }
    if len(context_ids) != len(contexts):
        raise ValueError("Duplicate context across shards")
    payload = dict(shards[0])
    payload["elements"] = [element for shard in shards for element in shard["elements"]]
    payload["contexts"] = sorted(
        contexts, key=lambda item: (item["element"], item["gate"], item["leader"])
    )
    payload["policy_leader_counts"] = {
        gate: counts
        for shard in shards
        for gate, counts in shard["policy_leader_counts"].items()
    }
    payload["aggregate"] = {
        "contexts": len(contexts),
        "mean_symmetric_kl": _weighted(contexts, "mean_symmetric_kl"),
        "p90_symmetric_kl": _weighted(contexts, "p90_symmetric_kl"),
        "mean_symmetric_tv": _weighted(contexts, "mean_symmetric_tv"),
        "control_mean_kl": _weighted(contexts, "control_mean_kl"),
        "mean_hidden_l2": _weighted(contexts, "mean_hidden_l2"),
        "mean_hidden_cosine": _weighted(contexts, "mean_hidden_cosine"),
    }
    wall = max(float(shard["performance"]["wall_time_seconds"]) for shard in shards)
    decisions = sum(int(shard["performance"]["policy_decisions"]) for shard in shards)
    payload["performance"] = {
        "wall_time_seconds": wall,
        "summed_process_wall_time_seconds": sum(
            float(shard["performance"]["wall_time_seconds"]) for shard in shards
        ),
        "policy_decisions": decisions,
        "policy_decisions_per_second": decisions / max(wall, 1e-9),
        "parallel_shards": len(shards),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(_markdown(payload), encoding="utf-8")
    print(f"wrote {args.json} and {args.md}")


if __name__ == "__main__":
    main()
