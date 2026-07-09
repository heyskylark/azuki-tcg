#!/usr/bin/env python3
"""Deck-composition x gate-identity interaction probe.

Question: does the OPTIMAL deck differ between sibling gates? This is the
justification bar for gate-conditional DRAFTING (the sibling gap probe showed
play-level ability differences with decks held fixed; here decks differ and
the gate is held fixed).

Design: mirror-GATE battles — both players get the SAME gate G; P0 plays the
archetype deck (e.g. weapon-heavy), P1 the contrast deck (entity-only), same
leader, seat-swapped, argmax play. Run once with G = sibling A and once with
G = sibling B:

    interaction = WR(archetype | G=A) - WR(archetype | G=B)

|interaction| >> 0  => composition value is gate-dependent => a maximizing
drafter SHOULD draft differently under A vs B. interaction ~ 0 => the
drafter's sibling-indifference is optimal (game-design finding).

Usage:
  probe_deck_gate_interaction.py --checkpoint CKPT --element LIGHTNING \
      --gate STT01-002 [--archetype weapon_heavy] [--contrast entity_only] \
      [--seeds 100] [--seed-offset 0] [--json OUT]
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from probe_gate_kl import GATE_CODE_PAIRS, EpisodeRunner

GATE_PORTAL_PRIMARY = 10
MAX_EPISODE_STEPS = 500

ARCHETYPES = {
    "LIGHTNING": ("weapon_heavy", "entity_only"),
    "WATER": ("spell_heavy", "entity_only"),
    "FIRE": ("cheap_aggro", "entity_only"),
    "EARTH": ("spell_heavy", "entity_only"),
}


def build_archetype_decks(catalog, element: str) -> dict[str, list]:
    """Archetype 50-card mains over the element+NORMAL pool (leader added later)."""
    recs = catalog.records_by_def_id
    pool = [recs[d] for d in catalog.main_def_ids_by_element[element]]

    weapons = sorted((r for r in pool if r.card_type == "WEAPON"), key=lambda r: r.ikz_cost)
    spells = sorted((r for r in pool if r.card_type == "SPELL"), key=lambda r: r.ikz_cost)
    entities = sorted((r for r in pool if r.card_type == "ENTITY"), key=lambda r: r.ikz_cost)
    cheap_entities = [r for r in entities if r.ikz_cost <= 2]

    def deck(*specs: tuple) -> list:
        # copy counts tracked ACROSS takes: overlapping pools (e.g. cheap
        # entities ⊂ entities) must not exceed 4 copies of a card.
        used: Counter = Counter()
        for cards, want in specs:
            got = 0
            for r in cards:
                if got >= want:
                    break
                can = 4 - used[r.card_code]
                if can <= 0:
                    continue
                qty = min(can, want - got)
                used[r.card_code] += qty
                got += qty
        return sorted(used.items())

    decks = {
        "entity_only": deck((entities, 50)),
        "weapon_heavy": deck((weapons, 24), (entities, 26)),
        "spell_heavy": deck((spells, 24), (entities, 26)),
        "cheap_aggro": deck((cheap_entities, 34), (entities, 16)),
    }
    return {k: v for k, v in decks.items() if sum(q for _, q in v) == 50}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("python/config/azuki_deckbuild_3090.ini"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--element", type=str, required=True)
    ap.add_argument("--gate", type=str, required=True, help="gate code BOTH players use")
    ap.add_argument("--archetype", type=str, default=None)
    ap.add_argument("--contrast", type=str, default="entity_only")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    element = args.element.upper()
    archetype = args.archetype or ARCHETYPES[element][0]
    runner = EpisodeRunner(args.config, args.checkpoint, args.device)
    base = runner.base_env
    vecenv = runner.vecenv
    policy = runner.policy
    device = runner.device

    decks = build_archetype_decks(runner.catalog, element)
    if archetype not in decks or args.contrast not in decks:
        raise SystemExit(f"unknown archetype for {element}: {archetype}/{args.contrast} (have {list(decks)})")
    leader_def = runner.catalog.leader_def_ids_by_element[element][0]
    leader_code = runner.catalog.records_by_def_id[leader_def].card_code

    def full_deck(main_entries: list) -> list:
        return [(args.gate, 1), (leader_code, 1)] + list(main_entries)

    state_arch = base._fixed_state_from_deck(full_deck(decks[archetype]))
    state_cont = base._fixed_state_from_deck(full_deck(decks[args.contrast]))

    def pick_action(dist, row: int) -> np.ndarray:
        count = int(dist.legal_action_count[row])
        if count <= 0:
            return np.zeros(4, dtype=np.int32)
        rows4 = dist.legal_actions[row, :count].detach().cpu().numpy()
        scores = dist.legal_action_logits[row, :count].detach().float().cpu().numpy()
        return rows4[int(np.argmax(scores))].astype(np.int32)

    wins_arch = 0.0
    n = trunc = 0
    for order in (0, 1):
        seat_of_arch = order
        for s in range(args.seeds):
            seed = 70000 + args.seed_offset + s

            def forced_states():
                pair = [copy.deepcopy(state_arch), copy.deepcopy(state_cont)]
                if seat_of_arch == 1:
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
                    acts[row] = pick_action(logits, row)
                vecenv.send(acts)
                obs, _, _, _, _, _, masks = vecenv.recv()
                steps += 1
                done = vecenv.envs[0].done
            if not done:
                trunc += 1
                continue
            info = base.infos.get(seat_of_arch, {}) or {}
            wins_arch += float(info.get("win", 0.0) or 0.0)
            n += 1

    wr = wins_arch / max(n, 1)
    se = (wr * (1 - wr) / max(n, 1)) ** 0.5
    out = {
        "element": element,
        "gate": args.gate,
        "archetype": archetype,
        "contrast": args.contrast,
        "winrate_archetype": wr,
        "se": se,
        "n": n,
        "truncated": trunc,
    }
    print(f"[ix] {element} gate={args.gate} {archetype} vs {args.contrast}: "
          f"WR={wr:.4f} +/- {2*se:.4f} (n={n}, trunc={trunc})")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
