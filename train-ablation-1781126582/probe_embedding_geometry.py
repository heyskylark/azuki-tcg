#!/usr/bin/env python3
"""Embedding-geometry probe: is the sibling-gate cosine collapse specific to
same-element gate pairs, or is the projected 48-d metadata space globally
compressed?

Reports cosine similarity of the trained card_metadata_projector outputs for:
  - sibling gate pairs (same element)
  - cross-element gate pairs
  - same-element same-type non-gate card pairs (the fair "similar cards" baseline)
  - random card pairs across the pool

Usage: probe_embedding_geometry.py --checkpoint CKPT [--device cpu]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from probe_gate_kl import GATE_CODE_PAIRS, EpisodeRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("python/config/azuki_deckbuild_3090.ini"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    runner = EpisodeRunner(args.config, args.checkpoint, args.device)
    inner = runner.policy
    while not hasattr(inner, "_metadata_embedding_table") and hasattr(inner, "policy"):
        inner = inner.policy
    with torch.no_grad():
        table = inner._metadata_embedding_table().float().cpu().numpy()

    records = runner.catalog.records_by_def_id
    norms = np.linalg.norm(table, axis=1)

    def cos(a: int, b: int) -> float:
        va, vb = table[a], table[b]
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))

    code_to_def = runner.code_to_def
    gates = {el: (code_to_def[a], code_to_def[b]) for el, (a, b) in GATE_CODE_PAIRS.items()}

    print(f"vocab={table.shape[0]} dim={table.shape[1]} mean_norm={norms[norms>0].mean():.3f}")
    print("\n-- sibling gate pairs (same element) --")
    sib = []
    for el, (a, b) in gates.items():
        c = cos(a, b)
        sib.append(c)
        print(f"  {el:<10} {records[a].card_code} vs {records[b].card_code}: cos={c:.4f}")
    print(f"  mean={np.mean(sib):.4f}")

    print("\n-- cross-element gate pairs --")
    all_gates = [g for pair in gates.values() for g in pair]
    cross = [cos(a, b) for a, b in combinations(all_gates, 2)
             if records[a].element != records[b].element]
    print(f"  n={len(cross)} mean={np.mean(cross):.4f} p10={np.quantile(cross,0.1):.4f} p90={np.quantile(cross,0.9):.4f}")

    # group non-gate cards by (element, type)
    by_group: dict[tuple, list[int]] = defaultdict(list)
    present = []
    for d, r in records.items():
        if r.card_type in ("GATE", "LEADER") or d >= table.shape[0] or norms[d] == 0:
            continue
        present.append(d)
        by_group[(r.element, r.card_type)].append(d)

    print("\n-- same-element same-type NON-GATE pairs (similar-cards baseline) --")
    rng = np.random.default_rng(7)
    same_group = []
    for (el, ct), ids in sorted(by_group.items()):
        if len(ids) < 2:
            continue
        pairs = list(combinations(ids, 2))
        take = rng.choice(len(pairs), size=min(60, len(pairs)), replace=False)
        vals = [cos(*pairs[i]) for i in take]
        same_group.extend(vals)
        print(f"  {el:<10} {ct:<7} n={len(ids):>3}: mean={np.mean(vals):.4f} max={np.max(vals):.4f}")
    print(f"  overall mean={np.mean(same_group):.4f} p90={np.quantile(same_group,0.9):.4f} p99={np.quantile(same_group,0.99):.4f}")

    print("\n-- random card pairs (whole pool) --")
    rand_pairs = rng.choice(len(present), size=(3000, 2))
    vals = [cos(present[i], present[j]) for i, j in rand_pairs if i != j]
    print(f"  n={len(vals)} mean={np.mean(vals):.4f} p10={np.quantile(vals,0.1):.4f} p90={np.quantile(vals,0.9):.4f} p99={np.quantile(vals,0.99):.4f}")

    frac_above = float(np.mean([v >= min(sib) for v in same_group]))
    print(f"\nfraction of similar-card baseline pairs with cos >= min sibling-gate cos ({min(sib):.3f}): {frac_above:.3f}")


if __name__ == "__main__":
    main()
