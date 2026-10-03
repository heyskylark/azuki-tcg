#!/usr/bin/env python3
"""Compare training runlogs (JSONL) across runs/ablations.

Usage:
  python compare_runs.py LOG1.jsonl LOG2.jsonl ... [--metrics m1,m2] [--points N]

Prints, per run, metric trajectories downsampled to N points aligned by
agent_steps, plus a final-value summary table. Metric names match jsonl keys
with or without the 'environment/' prefix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_METRICS = (
    "SPS",
    "losses/entropy",
    "losses/value_loss",
    "losses/explained_variance",
    "environment/win",
    "environment/azk_episode_length",
    "environment/deckbuild/main_unique",
    "environment/deckbuild/main_avg_cost",
    "environment/deckbuild/main_quad_count",
    "environment/deckbuild/main_copy_entropy_norm",
    "environment/deckbuild_result/gate_match_win_joint",
)


def load_run(path: Path):
    rows = []
    config = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("_event") == "start":
            config = row.get("config", {})
        elif "_step" in row:
            rows.append(row)
    return config, rows


def resolve(row: dict, metric: str):
    if metric in row:
        return row[metric]
    if f"environment/{metric}" in row:
        return row[f"environment/{metric}"]
    return None


def downsample(rows, metric, points):
    series = [(r["_step"], resolve(r, metric)) for r in rows]
    series = [(s, v) for s, v in series if v is not None]
    if not series:
        return []
    if len(series) <= points:
        return series
    stride = len(series) / points
    out = []
    for i in range(points):
        chunk = series[int(i * stride):int((i + 1) * stride)]
        if chunk:
            steps = chunk[-1][0]
            vals = [v for _, v in chunk]
            out.append((steps, sum(vals) / len(vals)))
    return out


def fmt(v):
    if v is None:
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--metrics", type=str, default=None)
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--grep", type=str, default=None,
                        help="Also list final values of all metric keys containing this substring")
    args = parser.parse_args()

    metrics = args.metrics.split(",") if args.metrics else list(DEFAULT_METRICS)
    runs = []
    for path in args.logs:
        config, rows = load_run(path)
        if rows:
            runs.append((path.stem, config, rows))
        else:
            print(f"warning: no data rows in {path}")

    for metric in metrics:
        print(f"\n## {metric}")
        for name, _, rows in runs:
            traj = downsample(rows, metric, args.points)
            if not traj:
                print(f"  {name:<42} (absent)")
                continue
            vals = " ".join(fmt(v) for _, v in traj)
            print(f"  {name:<42} {vals}")

    print("\n## final values")
    header = f"  {'run':<42}" + "".join(f" {m.split('/')[-1][:14]:>14}" for m in metrics)
    print(header)
    for name, _, rows in runs:
        last = rows[-1]
        vals = "".join(f" {fmt(resolve(last, m)):>14}" for m in metrics)
        print(f"  {name:<42}{vals}")

    if args.grep:
        print(f"\n## final keys matching '{args.grep}'")
        for name, _, rows in runs:
            last = rows[-1]
            hits = {k: v for k, v in last.items() if args.grep in k}
            print(f"  {name}:")
            for k in sorted(hits):
                print(f"    {k:<70} {fmt(hits[k])}")


if __name__ == "__main__":
    main()
