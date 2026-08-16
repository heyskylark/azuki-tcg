#!/usr/bin/env python3
"""Run paired fixed-deck battles with one frozen policy piloting both seats."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time

from play_selfplay_games import GameLogger
from probe_gate_kl import EpisodeRunner
from training_deck_pool import load_training_deck_labels, load_training_deck_pool


DEFAULT_REFERENCE_INDICES = "1,3,5,7,9,11,13,15,17"
DEFAULT_BASE_SEED = 8_100_001


def _parse_indices(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("indices must be distinct non-negative integers")
    return values


def _native_deck(payload: dict) -> tuple[tuple[str, int], ...]:
    raw = payload.get("native_deck")
    if not isinstance(raw, list):
        raise ValueError("Deck arm entry is missing native_deck")
    deck = tuple((str(entry[0]), int(entry[1])) for entry in raw)
    if sum(quantity for _, quantity in deck) != 62:
        raise ValueError("Fixed deck must contain 62 total cards")
    return deck


def _main_counter(deck_record: dict) -> Counter[str]:
    return Counter(str(code) for code in deck_record["main"])


def _task_schedule(payload: dict, reference_indices: tuple[int, ...]) -> list[dict]:
    arms = payload.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise ValueError("Deck-arm file has no arms")
    arm_names = tuple(arms)
    first_arm = arms[arm_names[0]]
    gate_names = tuple(first_arm)
    for arm in arm_names:
        if tuple(arms[arm]) != gate_names:
            raise ValueError(f"Arm {arm} gate order differs from {arm_names[0]}")

    tasks = []
    for gate_index, gate in enumerate(gate_names):
        for reference_position, reference_index in enumerate(reference_indices):
            block_index = gate_index * len(reference_indices) + reference_position
            seed = DEFAULT_BASE_SEED + 7_919 * block_index
            block_id = f"{gate}:ref{reference_index:02d}"
            for candidate_seat in (0, 1):
                for arm in arm_names:
                    tasks.append(
                        {
                            "arm": arm,
                            "target_gate": gate,
                            "reference_index": reference_index,
                            "candidate_seat": candidate_seat,
                            "seed": seed,
                            "block_id": block_id,
                        }
                    )
    return tasks


def _balanced_shard_tasks(
    all_tasks: list[dict], arm_count: int, shards: int, shard_index: int
) -> list[tuple[int, dict]]:
    """Keep paired arms together while spreading contexts across CPU shards."""
    if arm_count < 1 or len(all_tasks) % arm_count != 0:
        raise ValueError("Task schedule does not contain complete arm groups")

    indexed_tasks = list(enumerate(all_tasks))
    groups: list[list[tuple[int, dict]]] = []
    for offset in range(0, len(indexed_tasks), arm_count):
        group = indexed_tasks[offset : offset + arm_count]
        first = group[0][1]
        identity = (
            first["block_id"],
            first["target_gate"],
            first["reference_index"],
            first["candidate_seat"],
            first["seed"],
        )
        if any(
            (
                task["block_id"],
                task["target_gate"],
                task["reference_index"],
                task["candidate_seat"],
                task["seed"],
            )
            != identity
            for _, task in group
        ):
            raise ValueError("Task schedule split a paired arm group")
        groups.append(group)

    def group_key(group: list[tuple[int, dict]]) -> tuple[bytes, int]:
        global_index, task = group[0]
        identity = (
            f"{DEFAULT_BASE_SEED}:{task['block_id']}:"
            f"seat{task['candidate_seat']}"
        )
        return hashlib.sha256(identity.encode("utf-8")).digest(), global_index

    ordered_groups = sorted(groups, key=group_key)
    return [
        indexed_task
        for position, group in enumerate(ordered_groups)
        if position % shards == shard_index
        for indexed_task in group
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("python/config/azuki_deckbuild_3090.ini")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck-arms", type=Path, required=True)
    parser.add_argument("--reference-indices", default=DEFAULT_REFERENCE_INDICES)
    parser.add_argument("--opponent-deck-arms", type=Path)
    parser.add_argument("--opponent-arm", default="native_p2930")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--step-cap", type=int, default=1_200)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError("--shard-index must be in [0, --shards)")
    if args.step_cap < 1:
        raise ValueError("--step-cap must be positive")

    arm_payload = json.loads(args.deck_arms.read_text(encoding="utf-8"))
    opponent_payload = None
    opponent_gates: tuple[str, ...] = ()
    if args.opponent_deck_arms is not None:
        opponent_payload = json.loads(
            args.opponent_deck_arms.read_text(encoding="utf-8")
        )
        opponent_arms = opponent_payload.get("arms")
        if not isinstance(opponent_arms, dict) or args.opponent_arm not in opponent_arms:
            raise ValueError(f"Opponent deck arms have no {args.opponent_arm} arm")
        selected_opponents = opponent_arms[args.opponent_arm]
        if not isinstance(selected_opponents, dict) or not selected_opponents:
            raise ValueError("Selected opponent arm has no gate decks")
        opponent_gates = tuple(selected_opponents)
        reference_indices = tuple(range(len(opponent_gates)))
    else:
        reference_indices = _parse_indices(args.reference_indices)
    all_tasks = _task_schedule(arm_payload, reference_indices)
    arm_count = len(arm_payload["arms"])
    tasks = _balanced_shard_tasks(
        all_tasks, arm_count, args.shards, args.shard_index
    )
    if args.task_limit is not None:
        tasks = tasks[: max(0, args.task_limit)]

    pool = load_training_deck_pool()
    labels = load_training_deck_labels()
    if opponent_payload is None and max(reference_indices) >= len(pool):
        raise ValueError(f"Reference index exceeds pool size {len(pool)}")

    runner = EpisodeRunner(args.config, args.checkpoint, args.device)
    logger = GameLogger(runner, log_legal_actions=True, action_mode="argmax")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = 0
    candidate_wins = 0
    draws = 0
    try:
        with args.out.open("w", encoding="utf-8") as handle:
            for global_index, task in tasks:
                arm = str(task["arm"])
                gate = str(task["target_gate"])
                reference_index = int(task["reference_index"])
                candidate_seat = int(task["candidate_seat"])
                candidate_deck = _native_deck(arm_payload["arms"][arm][gate])
                if opponent_payload is None:
                    reference_deck = pool[reference_index]
                    reference_label = labels[reference_index]
                    reference_gate = None
                else:
                    reference_gate = opponent_gates[reference_index]
                    reference_deck = _native_deck(
                        opponent_payload["arms"][args.opponent_arm][reference_gate]
                    )
                    reference_label = f"{args.opponent_arm}/{reference_gate}"
                candidate_state = runner.base_env._fixed_state_from_deck(candidate_deck)
                reference_state = runner.base_env._fixed_state_from_deck(reference_deck)
                ordered_states = [None, None]
                ordered_states[candidate_seat] = candidate_state
                ordered_states[1 - candidate_seat] = reference_state

                def forced_states(states=tuple(ordered_states)):
                    return [copy.deepcopy(state) for state in states]

                runner.base_env._initial_states = forced_states
                game = logger.play_game(global_index, int(task["seed"]), step_cap=args.step_cap)
                if game["decks"] is None:
                    raise RuntimeError(f"Game {global_index} did not enter battle")
                expected_candidate = Counter(
                    {
                        code: quantity
                        for code, quantity in candidate_deck
                        if runner.catalog.records_by_code[code].card_type
                        in ("ENTITY", "SPELL", "WEAPON")
                    }
                )
                expected_reference = Counter(
                    {
                        code: quantity
                        for code, quantity in reference_deck
                        if runner.catalog.records_by_code[code].card_type
                        in ("ENTITY", "SPELL", "WEAPON")
                    }
                )
                if _main_counter(game["decks"][candidate_seat]) != expected_candidate:
                    raise RuntimeError(f"Candidate deck mismatch in game {global_index}")
                if _main_counter(game["decks"][1 - candidate_seat]) != expected_reference:
                    raise RuntimeError(f"Reference deck mismatch in game {global_index}")

                winner = int(game["outcome"]["winner"])
                candidate_score = 0.5 if winner < 0 else float(winner == candidate_seat)
                game["counterfactual"] = {
                    **task,
                    "global_task_index": global_index,
                    "reference_label": reference_label,
                    "reference_gate": reference_gate,
                    "candidate_score": candidate_score,
                    "action_mode": "legal_argmax_stable_first",
                    "recurrent_start": "zero_for_both_fixed_seats",
                }
                handle.write(json.dumps(game, separators=(",", ":")) + "\n")
                handle.flush()
                completed += 1
                candidate_wins += int(candidate_score == 1.0)
                draws += int(candidate_score == 0.5)
                if completed == 1 or completed % 5 == 0:
                    elapsed = time.time() - started
                    print(
                        f"[shard {args.shard_index:02d}] {completed}/{len(tasks)} "
                        f"{elapsed / completed:.1f}s/game arm={arm} gate={gate} "
                        f"ref={reference_index} seat={candidate_seat} "
                        f"score={(candidate_wins + 0.5 * draws) / completed:.3f}",
                        flush=True,
                    )
    finally:
        runner.vecenv.close()
    elapsed = time.time() - started
    print(
        f"wrote {args.out} ({completed} games, {elapsed:.1f}s, "
        f"{elapsed / max(completed, 1):.1f}s/game)",
        flush=True,
    )


if __name__ == "__main__":
    main()
