#!/usr/bin/env python3
"""Sibling-gate achievable-gap probe (portal-EV analysis).

Mirror-deck interventional battles: both players get the IDENTICAL leader and
50-card main deck; the only difference in the whole game is the gate card
(P0 gate A vs P1 sibling gate B, seat-swapped across the grid). Any deviation
of gate-A's win rate from 50% is causally attributable to the gate-ability
difference under the probed play mode:

  policy  — trained policy, argmax over legal rows (its actual valuation)
  forced  — whenever a GATE_PORTAL row is legal for the acting player, play
            the best-scoring portal row (bounds what a maximally
            portal-exploiting policy could extract from the gate difference)
  blocked — portal rows masked out (null reference; should sit at ~50%)

Decks are harvested from late-training deck snapshots (the policy's own
drafted decks for that element), used symmetrically on both sides.

Usage:
  probe_gate_gap.py --checkpoint CKPT --element LIGHTNING --mode forced \
      [--decks 6] [--seeds 85] [--seed-offset 0] [--device cpu] [--json OUT]

Shard across processes by splitting --seeds/--seed-offset.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from probe_gate_kl import GATE_CODE_PAIRS, EpisodeRunner

GATE_PORTAL_PRIMARY = 10
MAX_EPISODE_STEPS = 500


def harvest_decks(snapshot_glob: str, element: str, catalog, k: int) -> list[list]:
    """Latest-k distinct decks drafted under either sibling gate of `element`."""
    gate_codes = set(GATE_CODE_PAIRS[element])
    records_by_code = catalog.records_by_code
    seen: dict[str, list] = {}
    files = sorted(glob.glob(snapshot_glob))
    for path in files:
        for line in Path(path).read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for player in rec.get("players", []):
                if player.get("gate") not in gate_codes:
                    continue
                main = player.get("main", {})
                if sum(main.values()) != 50:
                    continue
                if any(code not in records_by_code for code in main):
                    continue
                leader = player.get("leader")
                key = leader + "|" + "|".join(f"{c}:{q}" for c, q in sorted(main.items()))
                deck = [(player["gate"], 1), (leader, 1)] + sorted(main.items())
                seen[key] = deck  # later files overwrite -> keeps latest occurrence
    decks = list(seen.values())[-k:]
    return decks


def build_neutral_decks(catalog, k: int, leader_code: str) -> list[list]:
    """K distinct all-NORMAL 50-card decks (<=4 copies/card), same fixed leader."""
    records = catalog.records_by_def_id
    normal_codes = sorted({
        records[d].card_code
        for defs in catalog.main_def_ids_by_element.values()
        for d in defs
        if records[d].element == "NORMAL"
    })
    decks = []
    for i in range(k):
        rng = np.random.default_rng(4242 + i)
        order = rng.permutation(len(normal_codes))
        main: list[tuple[str, int]] = []
        remaining = 50
        for idx in order:
            if remaining <= 0:
                break
            qty = int(min(remaining, rng.integers(1, 5)))
            main.append((normal_codes[idx], qty))
            remaining -= qty
        if remaining > 0:
            raise SystemExit("NORMAL pool too small for a 50-card deck")
        decks.append([(leader_code, 1)] + sorted(main))
    return decks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("python/config/azuki_deckbuild_3090.ini"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--element", type=str, default=None,
                    help="sibling-pair mode: probe this element's two gates on harvested decks")
    ap.add_argument("--pair", type=str, default=None,
                    help="cross-pair mode: 'GATE_A,GATE_B' card codes, any elements; "
                         "requires --neutral-decks (all-NORMAL mirror decks, same leader both sides)")
    ap.add_argument("--neutral-decks", type=int, default=0,
                    help="construct this many all-NORMAL 50-card decks instead of harvesting")
    ap.add_argument("--mode", type=str, choices=("policy", "forced", "blocked"), required=True)
    ap.add_argument("--snapshots", type=str, default="experiments/abl_snapshots/combo45b/*.jsonl")
    ap.add_argument("--decks", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=85)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    runner = EpisodeRunner(args.config, args.checkpoint, args.device)
    base = runner.base_env
    vecenv = runner.vecenv
    policy = runner.policy
    device = runner.device

    if args.pair:
        gate_a, gate_b = (c.strip() for c in args.pair.split(","))
        element = f"{gate_a}_vs_{gate_b}"
        if args.neutral_decks <= 0:
            raise SystemExit("--pair requires --neutral-decks (cross-element mirrors need neutral decks)")
        # Same leader on BOTH sides (gate A's element, first leader) — removes
        # the leader confound; leader/gate element match is a draft-time rule
        # the battle engine does not enforce.
        gate_a_element = runner.catalog.records_by_code[gate_a].element
        leader_def = runner.catalog.leader_def_ids_by_element[gate_a_element][0]
        leader_code = runner.catalog.records_by_def_id[leader_def].card_code
        decks = build_neutral_decks(runner.catalog, args.neutral_decks, leader_code)
        print(f"[gap] neutral-deck cross pair: fixed leader {leader_code} on both sides")
    else:
        element = args.element.upper()
        gate_a, gate_b = GATE_CODE_PAIRS[element]
        decks = harvest_decks(args.snapshots, element, runner.catalog, args.decks)
    if not decks:
        raise SystemExit(f"no decks available for {element}")
    print(f"[gap] {element} {gate_a} vs {gate_b} mode={args.mode}: {len(decks)} decks, "
          f"{args.seeds} seeds x 2 seat orders")

    def with_gate(deck: list, gate_code: str) -> list:
        out = [(gate_code, 1)] + [entry for entry in deck if entry[0] not in (gate_a, gate_b)]
        return out

    def pick_action(dist, row: int, mode: str) -> np.ndarray:
        count = int(dist.legal_action_count[row])
        if count <= 0:
            return np.zeros(4, dtype=np.int32)
        rows4 = dist.legal_actions[row, :count].detach().cpu().numpy()
        scores = dist.legal_action_logits[row, :count].detach().float().cpu().numpy().copy()
        portal = rows4[:, 0] == GATE_PORTAL_PRIMARY
        if mode == "forced" and portal.any():
            scores[~portal] = -np.inf
        elif mode == "blocked" and (~portal).any():
            scores[portal] = -np.inf
        r = int(np.argmax(scores))
        return rows4[r].astype(np.int32)

    results = defaultdict(lambda: {"wins_a": 0.0, "n": 0, "trunc": 0})
    total_portal_steps = {"a": 0, "b": 0}
    for deck_idx, deck in enumerate(decks):
        for order in (0, 1):  # 0: A on seat0 ; 1: B on seat0
            state_a = base._fixed_state_from_deck(with_gate(deck, gate_a))
            state_b = base._fixed_state_from_deck(with_gate(deck, gate_b))
            seat_of_a = order
            for s in range(args.seeds):
                seed = 90000 + args.seed_offset + s

                def forced_states():
                    pair = [copy.deepcopy(state_a), copy.deepcopy(state_b)]
                    if seat_of_a == 1:
                        pair.reverse()
                    return pair

                base._initial_states = forced_states
                torch.manual_seed(seed)
                vecenv.async_reset(seed=seed)
                obs, _, _, _, _, _, masks = vecenv.recv()
                st = {
                    "lstm_h": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
                    "lstm_c": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
                } if runner.use_rnn else {}
                steps = 0
                done = False
                while not done and steps < MAX_EPISODE_STEPS:
                    obs_t = torch.as_tensor(obs, device=device)
                    step_state = {"mask": torch.as_tensor(masks, device=device)}
                    if runner.use_rnn:
                        step_state["lstm_h"] = st["lstm_h"]
                        step_state["lstm_c"] = st["lstm_c"]
                    with torch.no_grad():
                        logits, _ = policy.forward_eval(obs_t, step_state)
                    if runner.use_rnn:
                        st["lstm_h"] = step_state["lstm_h"]
                        st["lstm_c"] = step_state["lstm_c"]
                    acts = np.zeros((vecenv.num_agents, 4), dtype=np.int32)
                    for row in range(vecenv.num_agents):
                        acts[row] = pick_action(logits, row, args.mode)
                        if acts[row][0] == GATE_PORTAL_PRIMARY:
                            key = "a" if row == seat_of_a else "b"
                            total_portal_steps[key] += 1
                    vecenv.send(acts)
                    obs, _, _, _, _, _, masks = vecenv.recv()
                    steps += 1
                    done = vecenv.envs[0].done
                bucket = results[deck_idx]
                if not done:
                    bucket["trunc"] += 1
                    continue
                info_a = base.infos.get(seat_of_a, {}) or {}
                bucket["wins_a"] += float(info_a.get("win", 0.0) or 0.0)
                bucket["n"] += 1

    tot_w = sum(b["wins_a"] for b in results.values())
    tot_n = sum(b["n"] for b in results.values())
    tot_t = sum(b["trunc"] for b in results.values())
    wr = tot_w / max(tot_n, 1)
    se = (wr * (1 - wr) / max(tot_n, 1)) ** 0.5
    out = {
        "element": element,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "mode": args.mode,
        "n": tot_n,
        "truncated": tot_t,
        "winrate_a": wr,
        "se": se,
        "portal_steps_a": total_portal_steps["a"],
        "portal_steps_b": total_portal_steps["b"],
        "per_deck": {
            str(i): {"winrate_a": b["wins_a"] / max(b["n"], 1), "n": b["n"]}
            for i, b in results.items()
        },
    }
    print(f"[gap] {element} mode={args.mode}: winrate({gate_a})={wr:.4f} +/- {2*se:.4f} "
          f"(n={tot_n}, trunc={tot_t}) portal_steps a={total_portal_steps['a']} b={total_portal_steps['b']}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
