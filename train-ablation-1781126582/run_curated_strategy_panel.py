#!/usr/bin/env python3
"""Run deterministic strategic-archetype versus entity-only games for all gates."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from probe_deck_behavior import BEHAVIOR_KEYS
from probe_deck_gate_interaction import ARCHETYPES, build_archetype_decks
from probe_gate_kl import EpisodeRunner, GATE_CODE_PAIRS


MAX_EPISODE_STEPS = 500


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed0", type=int, default=88_301)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")

    runner = EpisodeRunner(args.config, args.checkpoint, args.device)
    base = runner.base_env
    rows = []
    started = time.perf_counter()
    try:
        for element, gates in GATE_CODE_PAIRS.items():
            decks = build_archetype_decks(runner.catalog, element)
            archetype = ARCHETYPES[element][0]
            leader_id = runner.catalog.leader_def_ids_by_element[element][0]
            leader = runner.catalog.records_by_def_id[leader_id].card_code
            for gate_index, gate in enumerate(gates):
                def full_deck(main_entries: list) -> list:
                    return [(gate, 1), (leader, 1), *main_entries]

                strategic_state = base._fixed_state_from_deck(
                    full_deck(decks[archetype])
                )
                entity_state = base._fixed_state_from_deck(
                    full_deck(decks["entity_only"])
                )
                for strategic_seat in (0, 1):
                    for seed_index in range(args.seeds):
                        seed = (
                            args.seed0
                            + 100_003 * gate_index
                            + 10_007 * list(GATE_CODE_PAIRS).index(element)
                            + seed_index
                        )

                        def forced_states():
                            states = [
                                copy.deepcopy(strategic_state),
                                copy.deepcopy(entity_state),
                            ]
                            if strategic_seat == 1:
                                states.reverse()
                            return states

                        base._initial_states = forced_states
                        torch.manual_seed(seed)
                        runner.vecenv.async_reset(seed=seed)
                        obs, _, _, _, _, _, masks = runner.vecenv.recv()
                        state = (
                            {
                                "lstm_h": torch.zeros(
                                    runner.vecenv.num_agents,
                                    runner.policy.hidden_size,
                                    device=runner.device,
                                ),
                                "lstm_c": torch.zeros(
                                    runner.vecenv.num_agents,
                                    runner.policy.hidden_size,
                                    device=runner.device,
                                ),
                            }
                            if runner.use_rnn
                            else {}
                        )
                        steps = 0
                        while not runner.vecenv.envs[0].done and steps < MAX_EPISODE_STEPS:
                            step_state = {
                                "mask": torch.as_tensor(masks, device=runner.device)
                            }
                            if runner.use_rnn:
                                step_state["lstm_h"] = state["lstm_h"]
                                step_state["lstm_c"] = state["lstm_c"]
                            with torch.no_grad():
                                distribution, _ = runner.policy.forward_eval(
                                    torch.as_tensor(obs, device=runner.device),
                                    step_state,
                                )
                            if runner.use_rnn:
                                state["lstm_h"] = step_state["lstm_h"]
                                state["lstm_c"] = step_state["lstm_c"]
                            actions = np.zeros((2, 4), dtype=np.int32)
                            for seat in (0, 1):
                                count = int(distribution.legal_action_count[seat])
                                if count <= 0:
                                    continue
                                legal = distribution.legal_actions[seat, :count]
                                logits = distribution.legal_action_logits[seat, :count]
                                index = int(torch.argmax(logits).item())
                                actions[seat] = legal[index].detach().cpu().numpy()
                            runner.vecenv.send(actions)
                            obs, _, _, _, _, _, masks = runner.vecenv.recv()
                            steps += 1
                        info = base.infos.get(strategic_seat, {}) or {}
                        rows.append(
                            {
                                "element": element,
                                "gate": gate,
                                "leader": leader,
                                "archetype": archetype,
                                "opponent": "entity_only",
                                "strategic_seat": strategic_seat,
                                "seed": seed,
                                "completed": bool(runner.vecenv.envs[0].done),
                                "steps": steps,
                                "score": float(info.get("win", 0.0) or 0.0),
                                "behavior": {
                                    label: float(info.get(key, 0.0) or 0.0)
                                    for label, key in BEHAVIOR_KEYS.items()
                                },
                            }
                        )
        by_gate = {}
        for gate in sorted({str(row["gate"]) for row in rows}):
            selected = [row for row in rows if row["gate"] == gate]
            by_gate[gate] = {
                "games": len(selected),
                "score": sum(float(row["score"]) for row in selected)
                / len(selected),
                "completed": sum(bool(row["completed"]) for row in selected),
                "attack_rate": sum(
                    float(row["behavior"]["attack"]) for row in selected
                )
                / len(selected),
            }
        payload = {
            "schema_id": "azuki.curated_strategy_panel",
            "schema_version": 1,
            "checkpoint": str(args.checkpoint.resolve()),
            "policy_action_mode": "legal_argmax_stable_first",
            "contract": {
                "gates": 8,
                "leaders_per_gate": 1,
                "seat_swap": True,
                "strategic_archetypes": dict(ARCHETYPES),
                "control": "element-matched_entity_only_face_pressure",
                "seeds_per_seat_gate": args.seeds,
            },
            "games": len(rows),
            "wall_time_seconds": time.perf_counter() - started,
            "by_gate": by_gate,
            "records": rows,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json} ({len(rows)} games)")
    finally:
        runner.vecenv.close()


if __name__ == "__main__":
    main()
