#!/usr/bin/env python3
"""Counterfactual block probe.

At logged response-window decision points (face attack incoming, defender has
an untapped garden entity), replay the episode deterministically to that
point, then branch: force DECLARE_DEFENDER in one arm and NOOP (pass) in the
other, re-seed sampling, and roll out to terminal K times per arm. The
defender's win-rate delta between arms measures the EV of blocking under the
current policy's own continuation.

Two-phase usage:
  # 1) extract + sample branch points from logged games
  probe_block_counterfactual.py extract GAMES.jsonl... --points points.json
  # 2) run rollouts for a slice of points (shardable)
  PYTHONPATH=... probe_block_counterfactual.py run --points points.json \
      --checkpoint CKPT --slice 0/6 --rollouts 10 --out arm0.jsonl
  # 3) summarize
  probe_block_counterfactual.py report arm*.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

DECLARE_DEFENDER = 9
NOOP = 0


# --------------------------- phase 1: extract -------------------------------

def _flags(entry):
    """Trailing flag letters after the hp number (T=tapped, D=defender)."""
    import re

    m = re.search(r"/(-?\d+)([TD]*)$", entry.split("+")[0])
    return m.group(2) if m else ""


def untapped_count(board):
    return sum(1 for s in board if "T" not in _flags(s))


def untapped_defender_count(board):
    return sum(1 for s in board if "T" not in _flags(s) and "D" in _flags(s))


def extract_points(paths, n_accepted, n_declined, rng_seed=1234):
    from analyze_selfplay_games import ELEMENT_OF_GATE, load_games

    games = load_games(paths)
    candidates = []
    for gi, g in enumerate(games):
        if g["outcome"]["winner"] < 0:
            continue
        steps = g["steps"]
        picked_for_game = []
        for i, s in enumerate(steps):
            if s["d"]["t"] != "ATTACK" or s["d"].get("target") != "OPP_LEADER":
                continue
            attacker_p = s["p"]
            j = i + 1
            first_resp = None
            blocked = False
            while j < len(steps) and steps[j]["ph"] == "RESPONSE":
                r = steps[j]
                if r["p"] != attacker_p:
                    if first_resp is None:
                        first_resp = r
                    if r["d"]["t"] == "DECLARE_DEFENDER":
                        blocked = True
                j += 1
            if first_resp is None or untapped_count(first_resp["my_garden"]) == 0:
                continue
            combat = first_resp.get("combat") or {}
            picked_for_game.append(
                {
                    "game_index": gi,
                    "seed": g["seed"],
                    "branch_step": first_resp["i"],
                    "defender": first_resp["p"],
                    "decision": "blocked" if blocked else "declined",
                    "turn": None,
                    "hp_at_branch": first_resp["hp"],
                    "expect_attacker": combat.get("attacker"),
                    "expect_target": combat.get("target"),
                    "logged_action": first_resp["a"],
                    "defender_element": ELEMENT_OF_GATE[g["decks"][first_resp["p"]]["gate"]],
                    "orig_winner": g["outcome"]["winner"],
                    # exact-replay payload: drafted decks in pick order + battle
                    # action prefix (env transitions are deterministic given
                    # seed + applied actions; re-sampling is NOT reproducible
                    # across processes due to CPU float nondeterminism)
                    "decks": [
                        {"leader": g["decks"][p]["leader"], "main": g["decks"][p]["main"]}
                        for p in range(2)
                    ],
                    "prefix": [[st["p"]] + st["a"] for st in steps[: first_resp["i"]]],
                }
            )
        if picked_for_game:
            candidates.append(picked_for_game)

    rng = random.Random(rng_seed)
    # at most one point per game, balanced across decisions
    per_game = [rng.choice(pts) for pts in candidates]
    accepted = [p for p in per_game if p["decision"] == "blocked"]
    declined = [p for p in per_game if p["decision"] == "declined"]
    rng.shuffle(accepted)
    rng.shuffle(declined)
    points = accepted[:n_accepted] + declined[:n_declined]
    print(
        f"candidates: games_with_points={len(candidates)} accepted_pool={len(accepted)} "
        f"declined_pool={len(declined)} -> sampled {len(points)}"
    )
    return points


# --------------------------- phase 2: rollouts ------------------------------

class BranchRunner:
    def __init__(self, config, checkpoint, device):
        import torch  # noqa: F401
        from probe_gate_kl import EpisodeRunner

        self.runner = EpisodeRunner(Path(config), Path(checkpoint), device)
        self.runner.base_env._same_element_matchup_prob = 0.0
        from policy.v2 import tcg_sampler

        tcg_sampler.set_sampling_params(subaction_temperature=1.0, smoothing_eps=0.0)
        self.inner = self.runner.base_env.env
        self.records = self.runner.catalog.records_by_def_id

    def code(self, def_id):
        rec = self.records.get(int(def_id))
        return rec.card_code if rec else f"?{int(def_id)}"

    def legal_actions_for(self, actor):
        raw = self.inner._raw_observation(actor)
        m = raw.action_mask
        n = int(m.legal_action_count)
        return [
            (int(m.legal_primary[k]), int(m.legal_sub1[k]), int(m.legal_sub2[k]), int(m.legal_sub3[k]))
            for k in range(n)
        ]

    def _draft_pick_action(self, cursors):
        """Force the draft pick that reconstructs the logged deck (pick order)."""
        base = self.runner.base_env
        p = int(base._active_player_index)
        state = base._states[p]
        deck = self._point_decks[p]
        if state.leader_card_def_id < 0:
            desired = self.runner.code_to_def[deck["leader"]]
        else:
            desired = self.runner.code_to_def[deck["main"][cursors[p]]]
            cursors[p] += 1
        candidates, _ = base._candidate_def_ids(p)
        try:
            idx = list(candidates).index(int(desired))
        except ValueError:
            return None, f"draft candidate missing for player {p}"
        return np.asarray([3, idx, 0, 0], dtype=np.int32), None  # DECK_PICK_CARD

    def rollout(self, point, arm, rollout_seed, step_cap=4000):
        """Exact-replay episode to branch_step via forced actions, then branch.

        Draft picks are reconstructed from the logged decks (pick order);
        battle prefix actions are the logged ones. Env transitions are
        deterministic given seed + applied actions, so the branch state is
        exact. Only the post-branch suffix is sampled (seeded per rollout).
        """
        import torch
        import azk_puffer.pytorch as azk_pytorch

        runner = self.runner
        torch.manual_seed(rollout_seed)
        runner.vecenv.async_reset(seed=point["seed"])
        obs, rew, term, trunc, info, env_id, masks = runner.vecenv.recv()
        state = {}
        if runner.use_rnn:
            state = {
                "lstm_h": torch.zeros(runner.vecenv.num_agents, runner.policy.hidden_size, device=runner.device),
                "lstm_c": torch.zeros(runner.vecenv.num_agents, runner.policy.hidden_size, device=runner.device),
            }
        self._point_decks = point["decks"]
        cursors = [0, 0]
        prefix = point["prefix"]
        battle_step = -1
        branched = False
        forced_kind = None
        for _ in range(step_cap):
            building = runner.base_env._building
            active = int(runner.base_env._active_player_index)
            if not building:
                battle_step += 1
            obs_tensor = torch.as_tensor(obs, device=runner.device)
            step_state = {"mask": torch.as_tensor(masks, device=runner.device)}
            if runner.use_rnn:
                step_state["lstm_h"] = state["lstm_h"]
                step_state["lstm_c"] = state["lstm_c"]
            with torch.no_grad():
                logits, _ = runner.policy.forward_eval(obs_tensor, step_state)
            if runner.use_rnn:
                state["lstm_h"] = step_state["lstm_h"]
                state["lstm_c"] = step_state["lstm_c"]

            acts = np.zeros((runner.vecenv.num_agents, 4), dtype=np.int32)
            if building:
                forced, err = self._draft_pick_action(cursors)
                if err:
                    return {"error": err}
                acts[active] = forced
            elif battle_step < point["branch_step"]:
                lp = prefix[battle_step]
                if lp[0] != active:
                    return {"error": f"prefix actor mismatch at battle_step {battle_step}: {active} != {lp[0]}"}
                acts[active] = np.asarray(lp[1:], dtype=np.int32)
            elif battle_step == point["branch_step"]:
                if active != point["defender"]:
                    return {"error": f"actor mismatch: {active} != {point['defender']}"}
                raw = self.inner._raw_observation(active)
                cc = raw.combat_context
                if point.get("expect_attacker") and bool(cc.combat_active):
                    att = "LEADER" if cc.attacker_is_leader else self.code(cc.attacker_card_def_id)
                    if att != point["expect_attacker"]:
                        return {"error": f"attacker mismatch: {att} != {point['expect_attacker']}"}
                legal = self.legal_actions_for(active)
                if arm == "block":
                    logged = tuple(point["logged_action"])
                    if logged[0] == DECLARE_DEFENDER and logged in legal:
                        choice = logged
                    else:
                        blocks = [a for a in legal if a[0] == DECLARE_DEFENDER]
                        if not blocks:
                            return {"error": "no legal DECLARE_DEFENDER at branch"}
                        choice = blocks[0]
                    forced_kind = "block"
                else:
                    noops = [a for a in legal if a[0] == NOOP]
                    if not noops:
                        return {"error": "no legal NOOP at branch"}
                    choice = noops[0]
                    forced_kind = "pass"
                acts[active] = np.asarray(choice, dtype=np.int32)
                branched = True
            else:
                sampled, _, _ = azk_pytorch.sample_logits(logits)
                acts = sampled.cpu().numpy().astype(np.int32, copy=True)

            runner.vecenv.send(acts)
            obs, rew, term, trunc, info, env_id, masks = runner.vecenv.recv()
            term = np.asarray(term).reshape(-1)
            trunc = np.asarray(trunc).reshape(-1)
            if bool(term.any()) or bool(trunc.any()):
                if not branched:
                    return {"error": f"episode ended at battle_step {battle_step} before branch {point['branch_step']}"}
                rew = np.asarray(rew, dtype=np.float64).reshape(-1)
                winner = -1
                if rew[0] > rew[1]:
                    winner = 0
                elif rew[1] > rew[0]:
                    winner = 1
                return {
                    "arm": forced_kind,
                    "winner": winner,
                    "defender_won": winner == point["defender"],
                    "rollout_seed": rollout_seed,
                }
        return {"error": "step cap hit"}


# --------------------------- phase 3: report --------------------------------

def report(paths):
    rows = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    print(f"rollouts: {len(rows)} ok={len(ok)} errors={len(errs)}")
    if errs:
        from collections import Counter

        print("  error kinds:", Counter(e["error"].split(":")[0] for e in errs).most_common())

    by_point = defaultdict(lambda: {"block": [], "pass": []})
    for r in ok:
        by_point[(r["seed"], r["branch_step"])][r["arm"]].append(r["defender_won"])
    deltas = []
    for key, arms in by_point.items():
        if arms["block"] and arms["pass"]:
            wb = sum(arms["block"]) / len(arms["block"])
            wp = sum(arms["pass"]) / len(arms["pass"])
            meta = next(r for r in ok if (r["seed"], r["branch_step"]) == key)
            deltas.append({"key": key, "block_wr": wb, "pass_wr": wp, "delta": wb - wp,
                           "decision": meta["decision"], "turn_frac": meta.get("turn_frac"),
                           "element": meta.get("defender_element")})
    print(f"points with both arms: {len(deltas)}")
    for split in ("blocked", "declined"):
        sub = [d for d in deltas if d["decision"] == split]
        if not sub:
            continue
        mb = sum(d["block_wr"] for d in sub) / len(sub)
        mp = sum(d["pass_wr"] for d in sub) / len(sub)
        agree = sum(1 for d in sub if (d["delta"] > 0) == (split == "blocked"))
        print(
            f"  model originally {split:8s}: n={len(sub):3d} "
            f"E[win|block]={mb:.3f} E[win|pass]={mp:.3f} delta={mb - mp:+.3f} "
            f"(policy choice matches better arm at {agree}/{len(sub)} points)"
        )
    pos = [d for d in deltas if d["delta"] > 0.05]
    neg = [d for d in deltas if d["delta"] < -0.05]
    print(f"  points where blocking clearly better: {len(pos)}; clearly worse: {len(neg)}; ~neutral: {len(deltas) - len(pos) - len(neg)}")
    return deltas


# ------------------------------ main ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract")
    ex.add_argument("inputs", nargs="+", type=Path)
    ex.add_argument("--points", type=Path, required=True)
    ex.add_argument("--accepted", type=int, default=18)
    ex.add_argument("--declined", type=int, default=18)

    rn = sub.add_parser("run")
    rn.add_argument("--points", type=Path, required=True)
    rn.add_argument("--checkpoint", type=Path, required=True)
    rn.add_argument("--config", type=Path, default=Path("python/config/azuki_deckbuild_3090.ini"))
    rn.add_argument("--device", type=str, default="cpu")
    rn.add_argument("--rollouts", type=int, default=10)
    rn.add_argument("--slice", type=str, default="0/1")
    rn.add_argument("--out", type=Path, required=True)

    rp = sub.add_parser("report")
    rp.add_argument("inputs", nargs="+", type=Path)

    args = ap.parse_args()

    if args.cmd == "extract":
        points = extract_points(args.inputs, args.accepted, args.declined)
        args.points.write_text(json.dumps(points, indent=2))
        print(f"wrote {args.points} ({len(points)} points)")
        return

    if args.cmd == "report":
        report(args.inputs)
        return

    points = json.loads(args.points.read_text())
    k, n = (int(x) for x in args.slice.split("/"))
    my_points = [p for idx, p in enumerate(points) if idx % n == k]
    runner = BranchRunner(args.config, args.checkpoint, args.device)
    import time

    t0 = time.time()
    done = 0
    total = len(my_points) * 2 * args.rollouts
    with args.out.open("w") as fh:
        for p in my_points:
            for arm in ("block", "pass"):
                for r in range(args.rollouts):
                    # stable across processes (str hash is randomized per process)
                    rollout_seed = (
                        900_000_000
                        + (p["seed"] * 131 + p["branch_step"] * 17 + (0 if arm == "block" else 1) * 7 + r)
                        % 90_000_000
                    )
                    res = runner.rollout(p, arm, rollout_seed)
                    res = res or {"error": "none"}
                    res.update(
                        {
                            "seed": p["seed"],
                            "branch_step": p["branch_step"],
                            "decision": p["decision"],
                            "defender_element": p["defender_element"],
                            "hp_at_branch": p["hp_at_branch"],
                        }
                    )
                    fh.write(json.dumps(res, separators=(",", ":")) + "\n")
                    fh.flush()
                    done += 1
                    if done % 10 == 0:
                        dt = time.time() - t0
                        print(f"[{done}/{total}] {dt:.0f}s ({dt / done:.1f}s/rollout)", flush=True)
                    if "error" in res:
                        break  # don't waste K rollouts on a broken point/arm
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
