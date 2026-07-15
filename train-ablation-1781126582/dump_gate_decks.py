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
        for ep in range(args.episodes):
            runner.run_episode(91000 + 37 * ep, gate_code, None)
            lead, mains = drafted_deck()
            leaders[lead] += 1
            agg.update(mains)
        n = max(args.episodes, 1)
        sampled = {
            "leader_split": {l: c for l, c in leaders.most_common()},
            "summary": deck_summary(agg, scale=1.0 / n),
            "mean_copies": [
                {**card_entry(c, round(q / n, 2))}
                for c, q in agg.most_common(30)
            ],
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
