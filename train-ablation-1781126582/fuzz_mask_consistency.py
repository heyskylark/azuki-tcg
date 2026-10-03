#!/usr/bin/env python3
"""Mask-consistency fuzzer for the engine invalid-action desync (task #1).

Drives high-volume random LEGAL actions (uniform over the observation mask
rows — the same stale-mask -> submit flow training uses) through native
deck-building envs. Any stale/fresh mask desync triggers the C-side
"Invalid-action truncation: episode_seed=... gates=[...] tick=..." line on
stderr, giving a deterministic repro seed. Random play visits degenerate
states (e.g. 30-card hands) far more often than trained play.

Usage: fuzz_mask_consistency.py --steps 200000 --envs 16 --seed 1
Exit code 0 = clean; prints any desync lines found (also check stderr log).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sibling-prob", type=float, default=0.35)
    args = ap.parse_args()

    os.environ.pop("AZK_DEBUG_FORCE_GATE_DEF_IDS", None)
    sys.path.insert(0, "build/python/src")
    sys.path.insert(0, "python/src")
    from azk_native import AzukiNativeEnv, NATIVE_DECKBUILD_OBS_DTYPE
    from training_deck_pool import load_training_deck_pool

    env = AzukiNativeEnv(
        num_envs=args.envs,
        deck_pool=load_training_deck_pool(),
        seed=args.seed,
        deck_building=True,
        draft_same_element_matchup_prob=args.sibling_prob,
    )
    env.reset(seed=args.seed)
    view = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(env.num_agents)
    rng = np.random.default_rng(args.seed * 7919 + 13)

    episodes = 0
    for step in range(args.steps):
        acts = np.zeros((env.num_agents, 4), dtype=np.int32)
        for row in range(env.num_agents):
            am = view[row]["action_mask"]
            count = int(am["legal_action_count"])
            if count > 0:
                r = int(rng.integers(0, count))
                acts[row] = (
                    am["legal_primary"][r], am["legal_sub1"][r],
                    am["legal_sub2"][r], am["legal_sub3"][r],
                )
        env.actions[:] = acts.reshape(env.actions.shape)
        env.step()
        episodes += int(env.terminals.any() or env.truncations.any())
        if step % 20_000 == 0:
            print(f"[fuzz seed={args.seed}] step {step} episodes~{episodes}", flush=True)
    print(f"[fuzz seed={args.seed}] DONE steps={args.steps} envs={args.envs} episodes~{episodes}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
