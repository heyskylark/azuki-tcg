#!/usr/bin/env python3
"""Collect exact reward-component distributions across all eight gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from azk_native import NATIVE_DECKBUILD_OBS_DTYPE
from deck_building import build_deck_build_catalog
from training_deck_pool import load_training_deck_pool
from training_utils import make_azuki_env


def _first_legal_action(env, structured) -> tuple[int, np.ndarray]:
    player = int(env.active_players()[0])
    if player < 0:
        raise RuntimeError("Environment has no active player")
    row = player
    mask = structured[row]["action_mask"]
    if int(mask["legal_action_count"]) <= 0:
        raise RuntimeError("Active player has no legal action")
    return player, np.asarray(
        [
            mask["legal_primary"][0],
            mask["legal_sub1"][0],
            mask["legal_sub2"][0],
            mask["legal_sub3"][0],
        ],
        dtype=np.int32,
    )


def _play(env, structured, battle_decisions: int) -> None:
    for _ in range(180):
        if int(structured[0]["deck_context"]["mode"]) == 0:
            break
        player, action = _first_legal_action(env, structured)
        env.actions.fill(0)
        env.actions[player] = action
        env.step()
    else:
        raise RuntimeError("Draft did not complete")
    for _ in range(battle_decisions):
        if int(env.active_players()[0]) < 0:
            return
        player, action = _first_legal_action(env, structured)
        env.actions.fill(0)
        env.actions[player] = action
        env.step()
    if int(env.active_players()[0]) >= 0:
        env.force_evaluation_truncations([0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battle-decisions", type=int, default=80)
    parser.add_argument("--seed0", type=int, default=31_771)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.battle_decisions < 1:
        raise ValueError("--battle-decisions must be positive")

    catalog = build_deck_build_catalog(load_training_deck_pool())
    gates = sorted(set(int(value) for value in catalog.gate_def_id_population))
    env = make_azuki_env(
        seed=args.seed0,
        native=True,
        native_envs_per_instance=1,
        deck_building_enabled=True,
        draft_uniform_assignment=True,
        evaluation_mode=True,
        reward_telemetry=True,
    )
    structured = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)
    records = []
    started = time.perf_counter()
    try:
        for index, gate in enumerate(gates):
            element = catalog.records_by_def_id[gate].element
            leaders = catalog.leader_def_ids_by_element[element]
            env.reset_evaluation_games(
                [
                    {
                        "env_index": 0,
                        "seed": args.seed0 + 1009 * index,
                        "gate0": gate,
                        "gate1": gate,
                        "leader0": leaders[0],
                        "leader1": leaders[1],
                    }
                ]
            )
            _play(env, structured, args.battle_decisions)
            drained = env.drain_evaluation_records()
            if len(drained) != 1:
                raise RuntimeError(f"Expected one record for gate {gate}, got {len(drained)}")
            records.append(drained[0])
        metrics = env._deckbuild_helper.process_records(records)
    finally:
        env.close()

    reward_metrics = {
        key: value
        for key, value in sorted(metrics.items())
        if key.startswith("reward_")
    }
    compact_records = []
    for record in records:
        players = []
        for player in record["players"]:
            players.append(
                {
                    "gate": int(player["gate"]),
                    "leader": int(player["leader"]),
                    "reward_telemetry": player["reward_telemetry"],
                }
            )
        telemetry = record["reward_telemetry"]
        compact_records.append(
            {
                "seed": int(record["seed"]),
                "episode_length": float(record["episode_length"]),
                "players": players,
                "reward_telemetry": telemetry,
            }
        )
    payload = {
        "schema_id": "azuki.reward_component_baseline",
        "schema_version": 1,
        "method": {
            "policy": "first_legal_deterministic",
            "gate_coverage": "all_eight_same_gate_matchups",
            "battle_decisions_before_forced_truncation": args.battle_decisions,
            "gamma": 0.99,
            "evaluator_only": True,
        },
        "games": len(records),
        "wall_time_seconds": time.perf_counter() - started,
        "metrics": reward_metrics,
        "records": compact_records,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json} ({len(records)} games)")


if __name__ == "__main__":
    main()
