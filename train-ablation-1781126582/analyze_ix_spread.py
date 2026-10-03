#!/usr/bin/env python3
"""Interaction-spread readout: does the critic's sibling differential VARY
across decks (gate x composition interaction) or is it a per-gate constant?

|mean|/std >> 1  => main effect only (constant differential; pre-S3 state)
|mean|/std ~ 1   => differential varies deck-to-deck => interaction learned

Usage: analyze_ix_spread.py critic_final.json [labels...]
"""
import json, sys
import numpy as np

for path in sys.argv[1:]:
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f"{path}: {e}"); continue
    print(f"=== {path} ===")
    ratios = []
    for el, r in d.items():
        vals = np.array(r.get("battle_dv_signed_values", []))
        if len(vals) < 3: continue
        m, s = vals.mean(), vals.std()
        ratios.append(abs(m)/max(s,1e-9))
        print(f"  {el:<10} mean={m:+.5f} std={s:.5f} |m|/std={abs(m)/max(s,1e-9):.2f}")
    if ratios:
        print(f"  MEAN |m|/std = {np.mean(ratios):.2f}  (lower = more interaction)")
