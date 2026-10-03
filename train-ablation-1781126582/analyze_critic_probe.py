#!/usr/bin/env python3
"""Cross-arm comparison of critic gate-sensitivity probe results.

Reads results/critic_probe_<arm>.json (probe_critic_gate.py output) and prints
the go/no-go table: battle-start |dV| vs the A-A control floor and vs the
cross-seed V spread, sign consistency, win-prob-head deltas, and draft-phase
|dV| growth (early/mid/late tertiles).

Usage: analyze_critic_probe.py [--dir results] [--arms ctrl2,anneal1,...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("train-ablation-1781126582/results"))
    ap.add_argument("--arms", type=str, default="ctrl2,anneal1,gateid1,combo1")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    header = (
        f"{'arm':<9} {'element':<10} {'battle|dV|':>10} {'ctrl':>8} {'scale':>8} "
        f"{'ratio':>6} {'sign':>5} {'|dWP|':>8} {'wpscale':>8} {'pick e/m/l':>22}"
    )
    print(header)
    print("-" * len(header))
    for arm in arms:
        path = args.dir / f"critic_probe_{arm}.json"
        if not path.exists():
            print(f"{arm:<9} (missing {path})")
            continue
        data = json.loads(path.read_text())
        arm_dvs, arm_scales, arm_signs, arm_n = [], [], [], 0
        for element, r in data.items():
            terts = r.get("pick_dv_tertiles", {})
            tstr = "/".join(f"{terts.get(k, float('nan')):.4f}" for k in ("early", "mid", "late"))
            n_ep = int(r.get("episodes", 0))
            sign = max(r["battle_sign_pos"], n_ep - r["battle_sign_pos"])
            print(
                f"{arm:<9} {element:<10} {r['battle_mean_abs_dv']:>10.5f} "
                f"{r['battle_ctrl_mean_abs_dv']:>8.1e} {r['battle_v_scale_std']:>8.4f} "
                f"{r['battle_sens_ratio']:>6.2f} {sign:>3}/{n_ep:<2} "
                f"{(r.get('battle_mean_abs_dwp') or float('nan')):>8.5f} "
                f"{(r.get('battle_wp_scale_std') or float('nan')):>8.4f} {tstr:>22}"
            )
            arm_dvs.append(r["battle_mean_abs_dv"])
            arm_scales.append(r["battle_v_scale_std"])
            arm_signs.append(sign / max(n_ep, 1))
            arm_n = n_ep
        if arm_dvs:
            print(
                f"{arm:<9} {'MEAN':<10} {np.mean(arm_dvs):>10.5f} {'':>8} "
                f"{np.mean(arm_scales):>8.4f} {np.mean(arm_dvs)/max(np.mean(arm_scales),1e-12):>6.2f} "
                f"{np.mean(arm_signs)*100:>4.0f}% (n={arm_n}/elem)"
            )
        print()


if __name__ == "__main__":
    main()
