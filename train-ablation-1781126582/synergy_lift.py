#!/usr/bin/env python3
"""Card co-occurrence synergy analysis over deck snapshots.

For decks in the analysis window:
  lift(a,b) = P(a & b in same deck) / (P(a) * P(b))
computed within a conditioning scope (global or per gate). Pairs with
lift >> 1 at decent support are drafted together beyond chance = synergy
candidates. Gate-differential lift (lift under gate G vs under all other
gates) is evidence of gate-conditional synergy drafting.

Also reports win-lift: win rate of decks containing the pair minus the
scope's base win rate (does the pair actually help?).

Usage:
  synergy_lift.py SNAPSHOT_DIR [--last-frac 0.25] [--min-support 0.06]
                  [--top 15] [--per-gate]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

CARD_META_PATH = Path(__file__).resolve().parents[1] / "python" / "config" / "policy_card_metadata_v1.json"
GATE_NAMES = {
    "STT01-002": "Surge(L)", "AZK01-120": "Stormchain(L)",
    "STT02-002": "Hydromancy(W)", "AZK01-126": "EchoedWaves(W)",
    "AZK01-122": "Rushfire(F)", "STT04-002": "Ragefire(F)",
    "AZK01-124": "Devotion(E)", "STT03-002": "Stonehaven(E)",
}


def load_meta():
    meta = {}
    if CARD_META_PATH.exists():
        for record in json.loads(CARD_META_PATH.read_text())["records"]:
            meta[record["card_code"]] = record
    return meta


def load_decks(snapshot_dir: Path, last_frac: float):
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
    decks = []
    for rec in keep:
        for p in rec.get("players", []):
            if p.get("gate") and p.get("main"):
                decks.append({
                    "gate": p["gate"],
                    "leader": p.get("leader", "?"),
                    "cards": set(p["main"].keys()),
                    "win": float(p.get("win", 0.0) or 0.0),
                })
    return decks


def pair_stats(decks, min_support: float):
    n = len(decks)
    if n == 0:
        return {}, {}, 0.0
    presence = Counter()
    pair_count = Counter()
    pair_wins = Counter()
    base_win = sum(d["win"] for d in decks) / n
    for d in decks:
        cards = sorted(d["cards"])
        for c in cards:
            presence[c] += 1
        for a, b in combinations(cards, 2):
            pair_count[(a, b)] += 1
            pair_wins[(a, b)] += d["win"]
    out = {}
    for (a, b), cnt in pair_count.items():
        support = cnt / n
        if support < min_support:
            continue
        pa, pb = presence[a] / n, presence[b] / n
        lift = support / (pa * pb) if pa * pb > 0 else 0.0
        win_rate = pair_wins[(a, b)] / cnt
        out[(a, b)] = (lift, support, win_rate - base_win)
    return out, presence, base_win


def label(code, meta):
    m = meta.get(code, {})
    return f"{code}({m.get('name','?')[:18]},{m.get('card_type','?')[:3]},c{m.get('ikz_cost',0)})"


def pair_dlift(decks_g, decks_s, pair):
    """dlift of one pair between two deck groups (presence-based)."""
    def lift(decks):
        n = len(decks)
        if n == 0:
            return 0.0
        a_cnt = b_cnt = ab_cnt = 0
        for d in decks:
            has_a = pair[0] in d["cards"]
            has_b = pair[1] in d["cards"]
            a_cnt += has_a
            b_cnt += has_b
            ab_cnt += has_a and has_b
        pa, pb, pab = a_cnt / n, b_cnt / n, ab_cnt / n
        return pab / (pa * pb) if pa * pb > 0 else 0.0
    return lift(decks_g) - lift(decks_s)


def permutation_pvalue(decks_g, decks_s, pair, observed, permutations, rng):
    pooled = decks_g + decks_s
    n_g = len(decks_g)
    hits = 0
    for _ in range(permutations):
        rng.shuffle(pooled)
        null = pair_dlift(pooled[:n_g], pooled[n_g:], pair)
        if abs(null) >= abs(observed):
            hits += 1
    return (hits + 1) / (permutations + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot_dir", type=Path)
    ap.add_argument("--last-frac", type=float, default=0.25)
    ap.add_argument("--min-support", type=float, default=0.06)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--per-gate", action="store_true")
    ap.add_argument("--permutations", type=int, default=0,
                    help="permutation-null count for sibling-differential p-values")
    args = ap.parse_args()

    meta = load_meta()
    decks = load_decks(args.snapshot_dir, args.last_frac)
    print(f"{len(decks)} decks in window ({args.snapshot_dir}, last {args.last_frac:.0%})")

    print("\n== GLOBAL top-lift pairs ==")
    stats, presence, base_win = pair_stats(decks, args.min_support)
    ranked = sorted(stats.items(), key=lambda kv: -kv[1][0])[: args.top]
    for (a, b), (lift, sup, wl) in ranked:
        print(f"  lift={lift:5.2f} sup={sup:.3f} dwin={wl:+.3f}  {label(a,meta)} + {label(b,meta)}")

    if args.per_gate:
        sibling = {
            "STT01-002": "AZK01-120", "AZK01-120": "STT01-002",
            "STT02-002": "AZK01-126", "AZK01-126": "STT02-002",
            "AZK01-122": "STT04-002", "STT04-002": "AZK01-122",
            "AZK01-124": "STT03-002", "STT03-002": "AZK01-124",
        }
        by_gate = defaultdict(list)
        for d in decks:
            by_gate[d["gate"]].append(d)
        # Differential vs the SAME-ELEMENT sibling gate: identical candidate
        # pool, so any lift gap is gate-conditional strategy, not availability.
        for gate, gdecks in sorted(by_gate.items(), key=lambda kv: -len(kv[1])):
            sib = sibling.get(gate)
            sdecks = by_gate.get(sib, [])
            if len(sdecks) < 20:
                continue
            g_stats, _, g_base = pair_stats(gdecks, args.min_support)
            s_stats, _, _ = pair_stats(sdecks, 0.005)
            rows = []
            for pair, (lift, sup, wl) in g_stats.items():
                s_lift = s_stats.get(pair, (0.0, 0.0, 0.0))[0]
                rows.append((lift - s_lift, lift, s_lift, sup, wl, pair))
            rows.sort(key=lambda r: -r[0])
            print(
                f"\n== {GATE_NAMES.get(gate,gate)} n={len(gdecks)} win={g_base:.3f} "
                f"vs sibling {GATE_NAMES.get(sib,sib)} n={len(sdecks)} — sibling-differential lift =="
            )
            import random as _random
            rng = _random.Random(1234)
            for dlift, lift, s_lift, sup, wl, (a, b) in rows[: args.top]:
                ptxt = ""
                if args.permutations > 0:
                    p = permutation_pvalue(gdecks, sdecks, (a, b), dlift, args.permutations, rng)
                    ptxt = f" p={p:.3f}"
                print(
                    f"  dlift={dlift:+5.2f} (g={lift:4.2f} sib={s_lift:4.2f}) sup={sup:.3f} "
                    f"dwin={wl:+.3f}{ptxt}  {label(a,meta)} + {label(b,meta)}"
                )


if __name__ == "__main__":
    main()
