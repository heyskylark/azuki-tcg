#!/usr/bin/env python3
"""Per-gate strategy profiles + gate-vs-gate matchup matrix (eval-grade).

For every unordered pair of the 8 gates, run N policy-vs-policy episodes:
BOTH seats draft with the same checkpoint (sampled, not argmax — argmax would
collapse each gate to one deterministic deck per seed), gates forced via the
wrapper's gate sampler hook, seat order alternated for fairness. Each episode
runs draft + battle to completion.

Collected per episode:
  - winner per seat (-> matchup matrix + per-gate win-vs-field, mirrors excluded)
  - battle-phase action histogram per seat, attributed to the ACTIVE player
    (gives declare_defender / attack / portal / ability / confirm rates the
    env info keys do not fully expose)
  - drafted deck composition + env behavior rates via the deck wrapper's
    snapshot records (AZK_DECKBUILD_SNAPSHOT_DIR, every=1), aggregated with
    analyze_decks.analyze_bucket so the report matches the training-meta one.

Outputs: <out-prefix>_matchups.json and <out-prefix>_matchups.txt.

Usage:
  OMP_NUM_THREADS=6 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/profile_gate_matchups.py \
    --checkpoint CKPT [--episodes 16] [--device cpu] [--pairs all|siblings] \
    [--out-prefix train-ablation-1781126582/results/s14_gate]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

GATE_ORDER = [
    "STT01-002", "AZK01-120",  # LIGHTNING: Surge, Stormchain
    "STT02-002", "AZK01-126",  # WATER: Hydromancy, EchoedWaves
    "AZK01-122", "STT04-002",  # FIRE: Rushfire, Ragefire
    "AZK01-124", "STT03-002",  # EARTH: Devotion, Stonehaven
]
SIBLING_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
INFO_KEYS = {
    "attack": "azk_attack_selected_rate",
    "spell": "azk_play_spell_from_hand_selected_rate",
    "weapon": "azk_attach_weapon_from_hand_selected_rate",
    "portal": "azk_gate_portal_selected_rate",
    "play": "azk_play_selected_rate",
    "ability": "azk_ability_selected_rate",
    "noop": "azk_noop_selected_rate",
    "eplen": "azk_episode_length",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("python/config/azuki_deckbuild_3090.ini"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=16, help="episodes per unordered gate pair")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--pairs", choices=("all", "siblings"), default="all")
    ap.add_argument("--out-prefix", type=Path, default=Path("train-ablation-1781126582/results/gate_profile"))
    args = ap.parse_args()

    # Snapshot dir must be armed BEFORE the wrapper is constructed.
    snap_dir = Path(f"{args.out_prefix}_snapshots")
    if snap_dir.exists():
        shutil.rmtree(snap_dir)
    os.environ["AZK_DECKBUILD_SNAPSHOT_DIR"] = str(snap_dir)
    os.environ["AZK_DECKBUILD_SNAPSHOT_EVERY"] = "1"

    import azk_puffer.pytorch as azk_pytorch
    import azk_puffer.vector as azk_vector
    from action import ActionType
    from analyze_decks import GATE_NAMES, analyze_bucket, load_card_meta, load_records
    from evaluate_checkpoint import _apply_checkpoint_resume_policy_config, _unwrap_base_env
    from train import _load_model_weights
    from training_utils import build_policy, build_vecenv, install_tcg_sampler, load_training_config

    action_names = {int(v): v.name for v in ActionType}
    declare_id = int(ActionType.DECLARE_DEFENDER)

    trainer_args = load_training_config(args.config, [])
    trainer_args["train"]["device"] = args.device
    trainer_args.setdefault("env", {})["deck_building_enabled"] = True
    _apply_checkpoint_resume_policy_config(trainer_args, args.checkpoint)
    if trainer_args.get("policy", {}).get("privileged_critic_enabled"):
        trainer_args["env"]["deck_building_privileged_decks"] = True
    install_tcg_sampler()
    device = args.device

    vecenv = build_vecenv(trainer_args, backend=azk_vector.Serial, num_envs=1, seed=17)
    base = _unwrap_base_env(vecenv.envs[0])
    while not getattr(type(base), "is_deck_building_wrapper", False) and hasattr(base, "env"):
        base = base.env
    policy = build_policy(vecenv, trainer_args)
    use_rnn = bool(trainer_args["train"].get("use_rnn", True))

    vecenv.async_reset(seed=17)
    obs, _, _, _, _, _, masks = vecenv.recv()
    warm = {"mask": torch.as_tensor(masks, device=device)}
    if use_rnn:
        warm["lstm_h"] = torch.zeros(vecenv.num_agents, policy.hidden_size, device=device)
        warm["lstm_c"] = torch.zeros(vecenv.num_agents, policy.hidden_size, device=device)
    with torch.no_grad():
        policy.forward_eval(torch.as_tensor(obs, device=device), warm)
    _load_model_weights(policy, args.checkpoint, device=device, strict=False)
    policy.eval()

    code_to_def = {r.card_code: d for d, r in base._catalog.records_by_def_id.items()}

    def force_gates(p0_code: str, p1_code: str):
        forced = [code_to_def[p0_code], code_to_def[p1_code]]
        calls = {"n": 0}

        def fake_sample():
            value = forced[calls["n"] % 2]
            calls["n"] += 1
            return int(value)

        base._sample_gate_def_id = fake_sample

    def run_episode(seed: int, g0: str, g1: str):
        """Returns None on timeout, else dict with wins + per-seat battle stats."""
        force_gates(g0, g1)
        torch.manual_seed(seed)
        vecenv.async_reset(seed=seed)
        obs, _, _, _, _, _, masks = vecenv.recv()
        state = {}
        if use_rnn:
            state = {
                "lstm_h": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
                "lstm_c": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
            }
        acts_hist = (Counter(), Counter())
        battle_steps = 0
        done = False
        for _ in range(args.max_steps):
            obs_t = torch.as_tensor(obs, device=device)
            step_state = {"mask": torch.as_tensor(masks, device=device)}
            if use_rnn:
                step_state["lstm_h"] = state["lstm_h"]
                step_state["lstm_c"] = state["lstm_c"]
            with torch.no_grad():
                logits, _ = policy.forward_eval(obs_t, step_state)
                acts, _, _ = azk_pytorch.sample_logits(logits)
            if use_rnn:
                state["lstm_h"] = step_state["lstm_h"]
                state["lstm_c"] = step_state["lstm_c"]
            acts_np = acts.cpu().numpy().astype(np.int32, copy=True)
            if not base._building:
                active = int(base._active_player_index)
                acts_hist[active][int(acts_np[active][0])] += 1
                battle_steps += 1
            vecenv.send(acts_np)
            obs, _, _, _, _, _, masks = vecenv.recv()
            if vecenv.envs[0].done:
                done = True
                break
        if not done:
            return None
        out = {"battle_steps": battle_steps, "seats": []}
        for seat in (0, 1):
            info = base.infos.get(seat, {}) or {}
            out["seats"].append({
                "win": float(info.get("win", 0.0) or 0.0),
                "hist": dict(acts_hist[seat]),
                "info": {k: float(info.get(v, 0.0) or 0.0) for k, v in INFO_KEYS.items()},
            })
        return out

    if args.pairs == "siblings":
        pair_indices = SIBLING_PAIRS
    else:
        pair_indices = [(i, j) for i in range(len(GATE_ORDER)) for j in range(i, len(GATE_ORDER))]

    matrix = defaultdict(lambda: defaultdict(lambda: {"wins": 0.0, "games": 0}))
    per_gate = defaultdict(lambda: {
        "games": 0, "field_wins": 0.0, "field_games": 0, "hist": Counter(),
        "info_sums": defaultdict(float), "battle_steps": 0.0,
    })
    timeouts = 0
    for pidx, (i, j) in enumerate(pair_indices):
        ga, gb = GATE_ORDER[i], GATE_ORDER[j]
        completed = 0
        for ep in range(args.episodes):
            g0, g1 = (ga, gb) if ep % 2 == 0 else (gb, ga)
            seed = 40000 + pidx * 1000 + ep * 17
            result = run_episode(seed, g0, g1)
            if result is None:
                timeouts += 1
                continue
            completed += 1
            for seat, gate in ((0, g0), (1, g1)):
                opp = g1 if seat == 0 else g0
                srec = result["seats"][seat]
                matrix[gate][opp]["wins"] += srec["win"]
                matrix[gate][opp]["games"] += 1
                pg = per_gate[gate]
                pg["games"] += 1
                pg["battle_steps"] += result["battle_steps"]
                if gate != opp:
                    pg["field_wins"] += srec["win"]
                    pg["field_games"] += 1
                for a, c in srec["hist"].items():
                    pg["hist"][a] += c
                for k, v in srec["info"].items():
                    pg["info_sums"][k] += v
        print(f"[pair {pidx + 1}/{len(pair_indices)}] {GATE_NAMES.get(ga, ga)} vs "
              f"{GATE_NAMES.get(gb, gb)}: {completed}/{args.episodes} completed", flush=True)

    # ---- aggregate ----
    matrix_out = {
        ga: {
            gb: {"games": cell["games"], "win_rate": round(cell["wins"] / max(cell["games"], 1), 4)}
            for gb, cell in sorted(row.items())
        }
        for ga, row in sorted(matrix.items())
    }
    per_gate_out = {}
    for gate, pg in sorted(per_gate.items()):
        total_acts = sum(pg["hist"].values()) or 1
        hist_rates = {
            action_names.get(a, str(a)): round(c / total_acts, 4)
            for a, c in sorted(pg["hist"].items(), key=lambda t: -t[1])
        }
        per_gate_out[gate] = {
            "name": GATE_NAMES.get(gate, gate),
            "games": pg["games"],
            "win_vs_field": round(pg["field_wins"] / max(pg["field_games"], 1), 4),
            "field_games": pg["field_games"],
            "mean_battle_steps": round(pg["battle_steps"] / max(pg["games"], 1), 1),
            "declare_defender_rate": round(pg["hist"].get(declare_id, 0) / total_acts, 4),
            "action_rates": hist_rates,
            "env_info_means": {k: round(v / max(pg["games"], 1), 4) for k, v in sorted(pg["info_sums"].items())},
        }

    # ---- deck composition from wrapper snapshots ----
    meta = load_card_meta()
    records = load_records(snap_dir)
    deck_report, _ = analyze_bucket(records, meta, top_k=10) if records else ({}, {})

    out = {
        "checkpoint": str(args.checkpoint),
        "episodes_per_pair": args.episodes,
        "pairs": args.pairs,
        "timeouts": timeouts,
        "matrix": matrix_out,
        "per_gate": per_gate_out,
        "deck_profiles": deck_report,
    }
    json_path = Path(f"{args.out_prefix}_matchups.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2, default=str))

    # ---- human-readable report ----
    lines = []
    gates_present = [g for g in GATE_ORDER if g in matrix_out]
    short = {g: GATE_NAMES.get(g, g).split("(")[0][:9] for g in GATE_ORDER}
    lines.append(f"gate matchup matrix (row winrate vs col, {args.episodes} eps/pair, "
                 f"seat-fair, sampled; timeouts={timeouts})")
    lines.append("            " + " ".join(f"{short[g]:>9}" for g in gates_present))
    for ga in gates_present:
        row = [f"{matrix_out[ga][gb]['win_rate']:>9.3f}" if gb in matrix_out[ga] else f"{'-':>9}"
               for gb in gates_present]
        lines.append(f"{short[ga]:>11}" + " " + " ".join(row))
    lines.append("")
    for gate in gates_present:
        p = per_gate_out[gate]
        lines.append(f"[{p['name']}] games={p['games']} win_vs_field={p['win_vs_field']:.3f} "
                     f"(n={p['field_games']}) battle_steps={p['mean_battle_steps']}")
        top_acts = ", ".join(f"{k}={v}" for k, v in list(p["action_rates"].items())[:7])
        lines.append(f"  actions: {top_acts}")
        lines.append(f"  declare_defender_rate={p['declare_defender_rate']} "
                     f"portal={p['env_info_means'].get('portal', 0)} "
                     f"attack={p['env_info_means'].get('attack', 0)} "
                     f"noop={p['env_info_means'].get('noop', 0)}")
        d = deck_report.get(gate)
        if d:
            types = " ".join(f"{t}={s:.2f}" for t, s in d["type_share"].items())
            lines.append(f"  deck: avg_cost={d['avg_cost']:.2f} elem_share={d['element_share']:.2f} "
                         f"jaccard={d['within_gate_jaccard']:.3f} types: {types}")
            for code, rate, name, type_, cost in d["top_cards"][:6]:
                lines.append(f"    {code} x{rate:<5} {type_:<6} c{cost} {name}")
        lines.append("")
    text = "\n".join(lines)
    Path(f"{args.out_prefix}_matchups.txt").write_text(text)
    print(text)
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
