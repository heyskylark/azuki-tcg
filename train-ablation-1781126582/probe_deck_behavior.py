#!/usr/bin/env python3
"""Deck->behavior coupling probe (does the policy play what it drafts?).

Fix seat 1 to a constant reference deck. Give seat 0 controlled decks that
differ only in composition (weapon-heavy / spell-heavy / entity-only), same
gate+leader. Run N battles per deck with the SAME policy and measure seat-0
behavior rates (weapon attach, spell cast, attack, portal). A uniform-legal
baseline (logits zeroed) with the same decks calibrates pure availability:
  conditioning evidence = (policy rate gap between decks) >> (uniform gap).

Usage:
  probe_deck_behavior.py --checkpoint CKPT [--episodes 24] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

import azk_puffer.pytorch as azk_pytorch
import azk_puffer.vector as azk_vector
from evaluate_checkpoint import _apply_checkpoint_resume_policy_config, _unwrap_base_env
from train import _load_model_weights
from training_utils import build_policy, build_vecenv, install_tcg_sampler, load_training_config

PROBE_GATE = "STT01-002"    # Surge (L): weapon-replay gate
PROBE_LEADER = "STT01-001"  # Raizan: weapon-synergy leader
BEHAVIOR_KEYS = {
    "weapon": "azk_attach_weapon_from_hand_selected_rate",
    "spell": "azk_play_spell_from_hand_selected_rate",
    "attack": "azk_attack_selected_rate",
    "portal": "azk_gate_portal_selected_rate",
    "play": "azk_play_selected_rate",
    "eplen": "azk_episode_length",
}


def build_probe_decks(catalog):
    """Three 50-card mains over the LIGHTNING+NORMAL pool, same gate/leader."""
    recs = catalog.records_by_def_id
    pool = [recs[d] for d in catalog.main_def_ids_by_element["LIGHTNING"]]

    def take(cards, want):
        out = []
        for r in cards:
            if len(out) >= want:
                break
            qty = min(4, want - len(out))
            out.extend([r.card_code] * qty)
        return out

    weapons = sorted((r for r in pool if r.card_type == "WEAPON"), key=lambda r: r.ikz_cost)
    spells = sorted((r for r in pool if r.card_type == "SPELL"), key=lambda r: r.ikz_cost)
    entities = sorted((r for r in pool if r.card_type == "ENTITY"), key=lambda r: r.ikz_cost)

    def deck(main_codes):
        counts = Counter(main_codes)
        cards = [(PROBE_GATE, 1), (PROBE_LEADER, 1)] + sorted(counts.items())
        return tuple(cards)

    weapon_main = take(weapons, 24) + take(entities, 26)
    spell_main = take(spells, 24) + take(entities, 26)
    entity_main = take(entities, 50)
    for name, m in (("weapon", weapon_main), ("spell", spell_main), ("entity", entity_main)):
        if len(m) != 50:
            raise RuntimeError(f"{name} deck has {len(m)} cards")
    return {"weapon_heavy": deck(weapon_main), "spell_heavy": deck(spell_main), "entity_only": deck(entity_main)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("python/config/azuki_deckbuild_3090.ini"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    trainer_args = load_training_config(args.config, [])
    trainer_args["train"]["device"] = args.device
    env_cfg = trainer_args.setdefault("env", {})
    env_cfg["deck_building_enabled"] = True
    env_cfg["deck_building_fixed_seats"] = "0,1"
    _apply_checkpoint_resume_policy_config(trainer_args, args.checkpoint)
    env_cfg["deck_building_fixed_seats"] = "0,1"
    install_tcg_sampler()

    vecenv = build_vecenv(trainer_args, backend=azk_vector.Serial, num_envs=1, seed=11)
    base = _unwrap_base_env(vecenv.envs[0])
    while not getattr(type(base), "is_deck_building_wrapper", False) and hasattr(base, "env"):
        base = base.env
    policy = build_policy(vecenv, trainer_args)
    use_rnn = bool(trainer_args["train"].get("use_rnn", True))
    device = args.device

    vecenv.async_reset(seed=11)
    obs, _, _, _, _, _, masks = vecenv.recv()
    warm = {"mask": torch.as_tensor(masks, device=device)}
    if use_rnn:
        warm["lstm_h"] = torch.zeros(vecenv.num_agents, policy.hidden_size, device=device)
        warm["lstm_c"] = torch.zeros(vecenv.num_agents, policy.hidden_size, device=device)
    with torch.no_grad():
        policy.forward_eval(torch.as_tensor(obs, device=device), warm)
    if args.checkpoint is not None:
        _load_model_weights(policy, args.checkpoint, device=device, strict=False)
    policy.eval()

    decks = build_probe_decks(base._catalog)
    reference_deck = base._deck_pool[0]

    def run_block(probe_deck, uniform: bool):
        state0 = base._fixed_state_from_deck(probe_deck)
        stateref = base._fixed_state_from_deck(reference_deck)

        def forced_states():
            import copy
            return [copy.deepcopy(state0), copy.deepcopy(stateref)]

        base._initial_states = forced_states
        sums = defaultdict(float)
        wins = 0.0
        n = 0
        for ep in range(args.episodes):
            seed = 5000 + 271 * ep
            torch.manual_seed(seed)
            vecenv.async_reset(seed=seed)
            obs, _, _, _, _, _, masks = vecenv.recv()
            st = {}
            if use_rnn:
                st = {
                    "lstm_h": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
                    "lstm_c": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
                }
            steps = 0
            done = False
            while not done and steps < args.max_steps:
                obs_t = torch.as_tensor(obs, device=device)
                step_state = {"mask": torch.as_tensor(masks, device=device)}
                if use_rnn:
                    step_state["lstm_h"] = st["lstm_h"]
                    step_state["lstm_c"] = st["lstm_c"]
                with torch.no_grad():
                    logits, _ = policy.forward_eval(obs_t, step_state)
                    if uniform and hasattr(logits, "legal_action_logits"):
                        logits.legal_action_logits.zero_()
                    acts, _, _ = azk_pytorch.sample_logits(logits)
                if use_rnn:
                    st["lstm_h"] = step_state["lstm_h"]
                    st["lstm_c"] = step_state["lstm_c"]
                vecenv.send(acts.cpu().numpy().astype(np.int32, copy=True))
                obs, _, _, _, _, _, masks = vecenv.recv()
                steps += 1
                done = vecenv.envs[0].done
            info0 = base.infos.get(0, {}) or {}
            if not done:
                continue
            n += 1
            wins += float(info0.get("win", 0.0) or 0.0)
            for label, key in BEHAVIOR_KEYS.items():
                value = info0.get(key)
                if value is not None:
                    sums[label] += float(value)
        out = {label: sums[label] / max(n, 1) for label in BEHAVIOR_KEYS}
        out["win"] = wins / max(n, 1)
        out["episodes"] = n
        return out

    results = {}
    for mode, uniform in (("policy", False), ("uniform", True)):
        results[mode] = {}
        for deck_name, deck in decks.items():
            r = run_block(deck, uniform)
            results[mode][deck_name] = r
            print(
                f"[{mode:>7}] {deck_name:<13} n={r['episodes']:<3} win={r['win']:.2f} "
                + " ".join(f"{k}={r[k]:.4f}" for k in ("weapon", "spell", "attack", "portal"))
            )

    # conditioning score: policy's weapon-rate gap (weapon vs spell deck)
    # normalized by the uniform-legal gap (availability effect)
    for beh, deck_hi, deck_lo in (("weapon", "weapon_heavy", "spell_heavy"), ("spell", "spell_heavy", "weapon_heavy")):
        pol_gap = results["policy"][deck_hi][beh] - results["policy"][deck_lo][beh]
        uni_gap = results["uniform"][deck_hi][beh] - results["uniform"][deck_lo][beh]
        print(f"conditioning[{beh}]: policy_gap={pol_gap:+.4f} uniform_gap={uni_gap:+.4f} "
              f"ratio={pol_gap / uni_gap if uni_gap else float('nan'):.2f}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
