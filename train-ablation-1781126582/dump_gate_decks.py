#!/usr/bin/env python3
"""Dump per-gate drafted decks for a checkpoint to JSON.

For each of the 8 gates: one canonical deck (argmax picks — the deck the
checkpoint builds when playing greedily) plus a sampled aggregate (mean
copies/deck over N stochastic drafts — the deck distribution it actually
plays with in evals). Draft-only episodes on the legacy path with the gate
forced via the sampler hook; the opponent seat is fixed to a constant gate
and ignored.

Usage:
  OMP_NUM_THREADS=6 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/dump_gate_decks.py \
    --checkpoint CKPT --episodes 24 --json OUT.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from analyze_decks import CARD_META_PATH, GATE_NAMES
from probe_gate_kl import EpisodeRunner


def load_names():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("python/config/azuki_deckbuild_3090.ini"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()

    runner = EpisodeRunner(args.config, args.checkpoint, args.device)
    records = runner.catalog.records_by_def_id
    names = load_names()

    def code(def_id):
        rec = records.get(int(def_id))
        return rec.card_code if rec else str(def_id)

    def card_entry(card_code, copies):
        m = names.get(card_code, {})
        return {
            "code": card_code,
            "name": m.get("name", "?"),
            "type": m.get("type", "?"),
            "cost": m.get("cost", 0),
            "element": m.get("element", "?"),
            "copies": copies,
        }

    def drafted_deck():
        st = runner.base_env._states[0]
        leader = code(st.leader_card_def_id)
        mains = Counter(code(st.main_card_def_ids[i]) for i in range(st.main_count))
        return leader, mains

    def deck_summary(mains: Counter, scale: float = 1.0):
        total = sum(mains.values())
        cost_sum = sum(names.get(c, {}).get("cost", 0) * q for c, q in mains.items())
        types = Counter()
        for c, q in mains.items():
            types[names.get(c, {}).get("type", "?")] += q
        return {
            "cards_total": round(total * scale, 2),
            "avg_cost": round(cost_sum / max(total, 1), 3),
            "type_share": {t: round(q / max(total, 1), 3) for t, q in sorted(types.items())},
        }

    def sampled_distribution_summary(decks: list[Counter]) -> dict[str, float]:
        if not decks:
            raise ValueError("sampled deck list must be nonempty")
        unique = []
        singleton_slots = []
        pair_slots = []
        triplet_slots = []
        quad_slots = []
        for deck in decks:
            unique.append(len(deck))
            singleton_slots.append(sum(q for q in deck.values() if q == 1) / 50.0)
            pair_slots.append(sum(q for q in deck.values() if q == 2) / 50.0)
            triplet_slots.append(sum(q for q in deck.values() if q == 3) / 50.0)
            quad_slots.append(sum(q for q in deck.values() if q == 4) / 50.0)
        jaccards = []
        for left, right in combinations(decks, 2):
            keys = set(left).union(right)
            intersection = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
            union = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
            jaccards.append(intersection / union if union else 1.0)
        return {
            "main_unique_mean": round(sum(unique) / len(unique), 6),
            "singleton_slot_share_mean": round(
                sum(singleton_slots) / len(singleton_slots), 6
            ),
            "pair_slot_share_mean": round(sum(pair_slots) / len(pair_slots), 6),
            "triplet_slot_share_mean": round(
                sum(triplet_slots) / len(triplet_slots), 6
            ),
            "quad_slot_share_mean": round(sum(quad_slots) / len(quad_slots), 6),
            "within_gate_pairwise_multiset_jaccard_mean": (
                round(sum(jaccards) / len(jaccards), 6) if jaccards else 1.0
            ),
        }

    def mean_copy_entries(decks: list[Counter]) -> list[dict]:
        aggregate = Counter()
        for deck in decks:
            aggregate.update(deck)
        n = len(decks)
        return [
            {**card_entry(card_code, round(quantity / n, 4))}
            for card_code, quantity in aggregate.most_common()
        ]

    from policy.v2 import tcg_sampler

    out = {"checkpoint": str(args.checkpoint), "sampled_episodes": args.episodes, "gates": {}}
    for gate_code in GATE_NAMES:
        # canonical deck: greedy picks
        tcg_sampler.set_sampling_params(subaction_temperature=1e-6, smoothing_eps=0.0)
        runner.run_episode(90001, gate_code, None)
        leader, argmax_mains = drafted_deck()
        argmax_deck = {
            "leader": card_entry(leader, 1),
            "summary": deck_summary(argmax_mains),
            "cards": [card_entry(c, q) for c, q in sorted(
                argmax_mains.items(), key=lambda t: (names.get(t[0], {}).get("cost", 0), t[0]))],
        }
        # sampled aggregate: the deck distribution under eval sampling
        tcg_sampler.set_sampling_params(subaction_temperature=1.0, smoothing_eps=0.0)
        agg = Counter()
        leaders = Counter()
        sampled_decks = []
        sampled_counters: list[Counter] = []
        counters_by_leader: dict[str, list[Counter]] = {}
        for ep in range(args.episodes):
            seed = 91000 + 37 * ep
            runner.run_episode(seed, gate_code, None)
            lead, mains = drafted_deck()
            leaders[lead] += 1
            agg.update(mains)
            sampled_counters.append(mains)
            counters_by_leader.setdefault(lead, []).append(mains)
            sampled_decks.append(
                {
                    "episode": ep,
                    "seed": seed,
                    "leader": lead,
                    "summary": deck_summary(mains),
                    "cards": [
                        card_entry(card_code, quantity)
                        for card_code, quantity in sorted(mains.items())
                    ],
                }
            )
        n = max(args.episodes, 1)
        by_leader = {}
        for leader_code, leader_decks in sorted(counters_by_leader.items()):
            leader_aggregate = Counter()
            for deck in leader_decks:
                leader_aggregate.update(deck)
            by_leader[leader_code] = {
                "episodes": len(leader_decks),
                "summary": deck_summary(
                    leader_aggregate,
                    scale=1.0 / len(leader_decks),
                ),
                "distribution": sampled_distribution_summary(leader_decks),
                "mean_copies": mean_copy_entries(leader_decks),
            }
        sampled = {
            "leader_split": {l: c for l, c in leaders.most_common()},
            "summary": deck_summary(agg, scale=1.0 / n),
            "distribution": sampled_distribution_summary(sampled_counters),
            "mean_copies": [
                {**card_entry(c, round(q / n, 2))}
                for c, q in agg.most_common(30)
            ],
            "mean_copies_all": mean_copy_entries(sampled_counters),
            "by_leader": by_leader,
            "decks": sampled_decks,
        }
        out["gates"][gate_code] = {
            "gate_name": GATE_NAMES.get(gate_code, gate_code),
            "argmax_deck": argmax_deck,
            "sampled": sampled,
        }
        print(f"[{GATE_NAMES.get(gate_code, gate_code)}] argmax: {argmax_deck['summary']} "
              f"leader={argmax_deck['leader']['name']}", flush=True)

    args.json.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
