#!/usr/bin/env python3
"""Trajectory comparison over runlog JSONL with environment/{0,1}/ agent prefixes.

Averages the two agents' values per row, buckets rows by agent_steps into N
equal-width bins up to --max-steps, and prints aligned trajectories per metric.

Usage:
  traj_compare.py LOG1 LOG2 ... [--max-steps 12000000] [--bins 8] [--metrics ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_METRICS = (
    "win",
    "azk_episode_length",
    "azk_gate_portal_selected_rate",
    "azk_attach_weapon_from_hand_selected_rate",
    "azk_play_spell_from_hand_selected_rate",
    "azk_attack_selected_rate",
    "azk_step_deckbuild/main_avg_cost",
    "azk_step_deckbuild/main_unique",
    "azk_step_deckbuild/main_quad_count",
    "azk_step_deckbuild/main_type_share/WEAPON",
    "azk_step_deckbuild/main_type_share/SPELL",
    "azk_step_deckbuild/main_gate_element_share",
    # decisive same-element pair: weapon share Surge vs Stormchain
    "azk_step_deckbuild_gatecard/STT01-002/type_share/WEAPON",
    "azk_step_deckbuild_gatecard/AZK01-120/type_share/WEAPON",
    # spells: Hydromancy vs EchoedWaves
    "azk_step_deckbuild_gatecard/STT02-002/type_share/SPELL",
    "azk_step_deckbuild_gatecard/AZK01-126/type_share/SPELL",
    # cost: Rushfire (cheap-aggro gate) vs Ragefire
    "azk_step_deckbuild_gatecard/AZK01-122/avg_cost",
    "azk_step_deckbuild_gatecard/STT04-002/avg_cost",
    # earth big-body identity
    "azk_step_deckbuild_gatecard/AZK01-124/avg_cost",
    "azk_step_deckbuild_gatecard/STT03-002/avg_cost",
)

TRAINER_METRICS = ("losses/value_loss", "losses/explained_variance", "losses/entropy", "SPS")


def agent_avg(row: dict, metric: str):
    vals = [row[k] for k in (f"environment/0/{metric}", f"environment/1/{metric}") if k in row]
    if not vals:
        if metric in row:
            return row[metric]
        return None
    return sum(vals) / len(vals)


def load(path: Path, max_steps: int | None):
    rows = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "_step" not in r:
            continue
        steps = r.get("agent_steps", r["_step"])
        if max_steps and steps > max_steps:
            continue
        rows.append((steps, r))
    return rows


def bucket_series(rows, metric: str, bins: int, max_steps: int):
    width = max_steps / bins
    sums = [0.0] * bins
    counts = [0] * bins
    for steps, r in rows:
        v = agent_avg(r, metric)
        if v is None:
            continue
        b = min(bins - 1, int(steps / width))
        sums[b] += v
        counts[b] += 1
    return [sums[i] / counts[i] if counts[i] else None for i in range(bins)]


def fmt(v):
    return "   -  " if v is None else f"{v:6.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--max-steps", type=int, default=12_000_000)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--metrics", type=str, default=None)
    args = ap.parse_args()

    metrics = args.metrics.split(",") if args.metrics else list(DEFAULT_METRICS) + list(TRAINER_METRICS)
    runs = [(p.stem.split("_")[0], load(p, args.max_steps)) for p in args.logs]

    step_hdr = " ".join(f"{int((i + 1) * args.max_steps / args.bins / 1e6):>5}M" for i in range(args.bins))
    print(f"{'metric / run':<46} {step_hdr}")
    for metric in metrics:
        print(f"\n{metric}")
        for name, rows in runs:
            series = bucket_series(rows, metric, args.bins, args.max_steps)
            print(f"  {name:<44} " + " ".join(fmt(v) for v in series))


if __name__ == "__main__":
    main()
