#!/usr/bin/env python3
"""Deeper evidence probes over deck-snapshot JSONLs.

1) Same-element L1 divergence vs a bootstrap noise floor: split each gate's
   decks into random halves; L1 between halves estimates the sampling noise
   floor at that N. Same-element gate-pair L1 is only meaningful above it.
2) Behavior->win correlation per gate: do portal/weapon/spell usage rates
   correlate with winning? Positive r + low usage = exploration failure;
   negative r = the mechanic (as currently played) loses tempo.
3) Leader-conditioning within element: gate pairs share the leader pool, so
   leader-pick divergence between same-element gates is pure gate identity.

Usage: gate_identity_probe.py SNAPSHOT_DIR [--last-frac 0.25] [--boots 40]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

GATE_ELEMENT = {
    "STT01-002": "LIGHTNING", "AZK01-120": "LIGHTNING",
    "STT02-002": "WATER", "AZK01-126": "WATER",
    "AZK01-122": "FIRE", "STT04-002": "FIRE",
    "AZK01-124": "EARTH", "STT03-002": "EARTH",
}
GATE_NAMES = {
    "STT01-002": "Surge", "AZK01-120": "Stormchain",
    "STT02-002": "Hydromancy", "AZK01-126": "EchoedWaves",
    "AZK01-122": "Rushfire", "STT04-002": "Ragefire",
    "AZK01-124": "Devotion", "STT03-002": "Stonehaven",
}
BEHAVIORS = ("portal_rate", "weapon_rate", "spell_rate", "attack_rate", "play_entity_rate")


def load_players(snapshot_dir: Path, last_frac: float):
    records = []
    for path in sorted(snapshot_dir.glob("decks_pid*.jsonl")):
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    records.sort(key=lambda r: r.get("ts", 0.0))
    keep = records[int(len(records) * (1 - last_frac)):]
    by_gate = defaultdict(list)
    for rec in keep:
        for p in rec.get("players", []):
            if p.get("gate"):
                by_gate[p["gate"]].append(p)
    return by_gate, len(keep)


def pick_freq(players):
    counts = Counter()
    for p in players:
        for code, qty in p.get("main", {}).items():
            counts[code] += qty
    total = sum(counts.values()) or 1.0
    return {k: v / total for k, v in counts.items()}


def l1(fa, fb):
    keys = set(fa) | set(fb)
    return sum(abs(fa.get(k, 0.0) - fb.get(k, 0.0)) for k in keys) / 2.0


def bootstrap_floor(players, boots, rng):
    if len(players) < 8:
        return float("nan")
    vals = []
    for _ in range(boots):
        sh = players[:]
        rng.shuffle(sh)
        half = len(sh) // 2
        vals.append(l1(pick_freq(sh[:half]), pick_freq(sh[half:])))
    return sum(vals) / len(vals)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan"), n
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan"), n
    return cov / math.sqrt(vx * vy), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot_dir", type=Path)
    ap.add_argument("--last-frac", type=float, default=0.25)
    ap.add_argument("--boots", type=int, default=40)
    args = ap.parse_args()
    rng = random.Random(0)

    by_gate, n_eps = load_players(args.snapshot_dir, args.last_frac)
    print(f"{args.snapshot_dir} last {args.last_frac:.0%}: {n_eps} episodes")

    # 1) same-element divergence vs noise floor
    print("\n== same-element L1 divergence vs bootstrap noise floor ==")
    elements = defaultdict(list)
    for gate in by_gate:
        elements[GATE_ELEMENT.get(gate, "?")].append(gate)
    for element, gates in sorted(elements.items()):
        if len(gates) != 2:
            continue
        ga, gb = gates
        pa, pb = by_gate[ga], by_gate[gb]
        div = l1(pick_freq(pa), pick_freq(pb))
        floor_a = bootstrap_floor(pa, args.boots, rng)
        floor_b = bootstrap_floor(pb, args.boots, rng)
        floor = (floor_a + floor_b) / 2
        # subsample-matched floor: both halves have ~n/2 decks, pair uses full n;
        # scale floor by sqrt(1/2) to approximate full-N noise
        adj_floor = floor / math.sqrt(2)
        excess = div - adj_floor
        print(
            f"  {element:<9} {GATE_NAMES[ga]:>11}(n={len(pa)}) vs {GATE_NAMES[gb]:<11}(n={len(pb)}) "
            f"L1={div:.4f} floor~{adj_floor:.4f} excess={excess:+.4f}"
        )

    # 2) behavior-win correlation per gate
    print("\n== behavior vs win correlation (per gate, last window) ==")
    hdr = " ".join(f"{b.replace('_rate',''):>12}" for b in BEHAVIORS)
    print(f"  {'gate':<12} n    {hdr}")
    pooled = defaultdict(lambda: ([], []))
    for gate, players in sorted(by_gate.items(), key=lambda kv: GATE_ELEMENT.get(kv[0], "?")):
        cells = []
        for beh in BEHAVIORS:
            xs = [p.get(beh) for p in players if beh in p and "win" in p]
            ys = [p.get("win") for p in players if beh in p and "win" in p]
            r, n = pearson([x for x in xs if x is not None], [y for y in ys if y is not None])
            cells.append(f"{r:+.3f}" if r == r else "   - ")
            px, py = pooled[beh]
            px.extend([x for x in xs if x is not None])
            py.extend([y for y in ys if y is not None])
        print(f"  {GATE_NAMES.get(gate, gate):<12} {len(players):<4} " + " ".join(f"{c:>12}" for c in cells))
    cells = []
    for beh in BEHAVIORS:
        px, py = pooled[beh]
        r, n = pearson(px, py)
        cells.append(f"{r:+.3f}" if r == r else "   - ")
    print(f"  {'POOLED':<12} {len(pooled[BEHAVIORS[0]][0]):<4} " + " ".join(f"{c:>12}" for c in cells))
    print("  (usage means)")
    means = []
    for beh in BEHAVIORS:
        px, _ = pooled[beh]
        means.append(f"{sum(px)/max(len(px),1):.4f}")
    print(f"  {'':<17} " + " ".join(f"{m:>12}" for m in means))

    # 3) leader conditioning within element
    print("\n== leader split per gate (same-element pairs share the pool) ==")
    for element, gates in sorted(elements.items()):
        for gate in gates:
            leaders = Counter(p.get("leader", "?") for p in by_gate[gate])
            total = sum(leaders.values()) or 1
            splits = ", ".join(f"{k}:{v/total:.2f}" for k, v in leaders.most_common())
            print(f"  {element:<9} {GATE_NAMES[gate]:<12} {splits}")


if __name__ == "__main__":
    main()
