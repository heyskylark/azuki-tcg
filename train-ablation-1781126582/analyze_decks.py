#!/usr/bin/env python3
"""Aggregate deck-snapshot JSONL files into per-gate archetype reports.

Usage:
  python analyze_decks.py SNAPSHOT_DIR [--buckets N] [--top K] [--csv OUT.csv]

Reads every decks_pid*.jsonl in SNAPSHOT_DIR (written by DeckBuildingParallelEnv
when AZK_DECKBUILD_SNAPSHOT_DIR is set), splits records into N time buckets by
timestamp, and reports per gate:
  - games, win rate, avg cost, type shares, element share
  - top-K most picked cards (pick rate = copies per deck)
  - deck concentration: mean within-gate pairwise Jaccard similarity (sampled)
  - behavioral averages (attack/spell/weapon/portal rates, episode length)
Also reports cross-gate composition divergence (L1 distance of card pick
frequency vectors) to quantify whether gates produce different decks.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

CARD_META_PATH = Path(__file__).resolve().parents[1] / "python" / "config" / "policy_card_metadata_v1.json"

GATE_NAMES = {
    "STT01-002": "Surge(L)",
    "AZK01-120": "Stormchain(L)",
    "STT02-002": "Hydromancy(W)",
    "AZK01-126": "EchoedWaves(W)",
    "AZK01-122": "Rushfire(F)",
    "STT04-002": "Ragefire(F)",
    "AZK01-124": "Devotion(E)",
    "STT03-002": "Stonehaven(E)",
}

BEHAVIOR_KEYS = ("attack_rate", "spell_rate", "weapon_rate", "portal_rate", "play_entity_rate", "noop_rate", "episode_length")


def load_card_meta():
    meta = {}
    if CARD_META_PATH.exists():
        for record in json.loads(CARD_META_PATH.read_text())["records"]:
            meta[record["card_code"]] = {
                "name": record.get("name", "?"),
                "type": record.get("card_type", "?"),
                "cost": record.get("ikz_cost", 0),
                "element": record.get("element", "?"),
            }
    return meta


def load_records(snapshot_dir: Path):
    records = []
    for path in sorted(snapshot_dir.glob("decks_pid*.jsonl")):
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    records.sort(key=lambda r: r.get("ts", 0.0))
    return records


def jaccard(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return inter / union if union else 0.0


def analyze_bucket(records, meta, top_k, sample_pairs=400):
    by_gate = defaultdict(list)
    for record in records:
        ref_seat = int(record.get("ref_seat", -1))
        for idx, player in enumerate(record.get("players", [])):
            if idx == ref_seat:
                continue  # S4 reference seat plays a fixed human deck, not a draft
            gate = player.get("gate")
            if gate:
                by_gate[gate].append(player)

    gate_pick_freq = {}
    report = {}
    for gate, players in sorted(by_gate.items()):
        n = len(players)
        wins = sum(p.get("win", 0.0) for p in players)
        costs = [p.get("avg_cost", 0.0) for p in players]
        type_counts = Counter()
        card_counts = Counter()
        leader_counts = Counter()
        element_match = 0
        total_cards = 0
        behaviors = defaultdict(list)
        for p in players:
            leader_counts[p.get("leader", "?")] += 1
            for code, qty in p.get("main", {}).items():
                card_counts[code] += qty
                total_cards += qty
                m = meta.get(code, {})
                type_counts[m.get("type", "?")] += qty
                if m.get("element") not in ("NORMAL", None):
                    element_match += qty
            for key in BEHAVIOR_KEYS:
                if key in p:
                    behaviors[key].append(p[key])

        decks = [p.get("main", {}) for p in players]
        sims = []
        if len(decks) >= 2:
            rng = random.Random(0)
            for _ in range(min(sample_pairs, len(decks) * (len(decks) - 1) // 2)):
                i, j = rng.sample(range(len(decks)), 2)
                sims.append(jaccard(decks[i], decks[j]))

        pick_freq = {code: card_counts[code] / max(n, 1) for code in card_counts}
        gate_pick_freq[gate] = pick_freq
        report[gate] = {
            "games": n,
            "win_rate": wins / max(n, 1),
            "avg_cost": sum(costs) / max(len(costs), 1),
            "type_share": {t: c / max(total_cards, 1) for t, c in sorted(type_counts.items())},
            "element_share": element_match / max(total_cards, 1),
            "leader_split": dict(leader_counts.most_common()),
            "top_cards": [
                (code, round(cnt / max(n, 1), 2), meta.get(code, {}).get("name", "?")[:24],
                 meta.get(code, {}).get("type", "?")[:6], meta.get(code, {}).get("cost", 0))
                for code, cnt in card_counts.most_common(top_k)
            ],
            "within_gate_jaccard": sum(sims) / max(len(sims), 1),
            "behaviors": {k: sum(v) / max(len(v), 1) for k, v in behaviors.items()},
        }
    return report, gate_pick_freq


def l1_divergence(freq_a: dict, freq_b: dict) -> float:
    keys = set(freq_a) | set(freq_b)
    total_a = sum(freq_a.values()) or 1.0
    total_b = sum(freq_b.values()) or 1.0
    return sum(abs(freq_a.get(k, 0) / total_a - freq_b.get(k, 0) / total_b) for k in keys) / 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_dir", type=Path)
    parser.add_argument("--buckets", type=int, default=4)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--last-only", action="store_true", help="Only print the final bucket")
    args = parser.parse_args()

    meta = load_card_meta()
    records = load_records(args.snapshot_dir)
    if not records:
        print(f"No snapshot records found in {args.snapshot_dir}")
        return

    print(f"{len(records)} episodes ({2 * len(records)} decks) from {args.snapshot_dir}")
    bucket_size = max(1, math.ceil(len(records) / args.buckets))
    csv_rows = []
    for bucket_index in range(args.buckets):
        chunk = records[bucket_index * bucket_size:(bucket_index + 1) * bucket_size]
        if not chunk:
            continue
        report, gate_freq = analyze_bucket(chunk, meta, args.top)
        is_last = bucket_index == args.buckets - 1 or (bucket_index + 2) * bucket_size > len(records)
        if args.last_only and not is_last:
            continue
        print(f"\n=== bucket {bucket_index + 1}/{args.buckets} ({len(chunk)} episodes) ===")
        for gate, stats in report.items():
            name = GATE_NAMES.get(gate, gate)
            behaviors = stats["behaviors"]
            print(
                f"\n[{name}] games={stats['games']} win={stats['win_rate']:.3f} "
                f"avg_cost={stats['avg_cost']:.2f} elem_share={stats['element_share']:.2f} "
                f"jaccard={stats['within_gate_jaccard']:.3f}"
            )
            print(f"  types: " + " ".join(f"{t}={s:.2f}" for t, s in stats["type_share"].items()))
            print(f"  leaders: {stats['leader_split']}")
            if behaviors:
                print("  behavior: " + " ".join(f"{k}={behaviors[k]:.3f}" for k in BEHAVIOR_KEYS if k in behaviors))
            for code, rate, name_, type_, cost in stats["top_cards"]:
                print(f"    {code} x{rate:<5} {type_:<6} c{cost} {name_}")
            csv_rows.append({
                "bucket": bucket_index, "gate": gate, "games": stats["games"],
                "win_rate": round(stats["win_rate"], 4), "avg_cost": round(stats["avg_cost"], 3),
                "element_share": round(stats["element_share"], 4),
                "jaccard": round(stats["within_gate_jaccard"], 4),
                **{f"share_{t}": round(s, 4) for t, s in stats["type_share"].items()},
                **{f"beh_{k}": round(behaviors.get(k, 0.0), 4) for k in BEHAVIOR_KEYS},
            })

        gates = sorted(gate_freq)
        if len(gates) > 1:
            divergences = [
                l1_divergence(gate_freq[a], gate_freq[b])
                for i, a in enumerate(gates) for b in gates[i + 1:]
            ]
            same_element_pairs = []
            for i, a in enumerate(gates):
                for b in gates[i + 1:]:
                    ea = meta.get(a, {}).get("element")
                    eb = meta.get(b, {}).get("element")
                    if ea and ea == eb:
                        same_element_pairs.append(l1_divergence(gate_freq[a], gate_freq[b]))
            print(
                f"\n  cross-gate L1 divergence: mean={sum(divergences)/len(divergences):.3f} "
                f"min={min(divergences):.3f} max={max(divergences):.3f}"
                + (
                    f" | same-element pairs mean={sum(same_element_pairs)/len(same_element_pairs):.3f}"
                    if same_element_pairs else ""
                )
            )

    if args.csv and csv_rows:
        import csv as csv_module
        with args.csv.open("w", newline="") as handle:
            writer = csv_module.DictWriter(handle, fieldnames=sorted({k for row in csv_rows for k in row}))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nwrote {len(csv_rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
