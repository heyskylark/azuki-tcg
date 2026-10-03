#!/usr/bin/env python3
"""Stage 0 frozen-checkpoint battle panel for leader-assignment lifecycles."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from azk_native import NATIVE_DECKBUILD_OBS_DTYPE
from evaluate_checkpoint import _apply_checkpoint_resume_policy_config
from policy.v2.tcg_sampler import tcg_argmax_logits
from probe_gate_kl import GATE_CODE_PAIRS
from train import _load_model_weights
from training_deck_pool import load_training_deck_pool
from training_utils import build_policy, load_training_config, make_azuki_env


ARM_POLICY = "policy_leader"
ARM_FORCED = "forced_leader_row"
ARM_PREFILLED = "prefilled_no_row"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(raw: dict, candidate_seat: int) -> float:
    players = raw["players"]
    candidate_win = float(players[candidate_seat].get("win", 0.0) or 0.0)
    opponent_win = float(players[1 - candidate_seat].get("win", 0.0) or 0.0)
    if candidate_win > opponent_win:
        return 1.0
    if opponent_win > candidate_win:
        return 0.0
    return 0.5


def _deck_metrics(main: list[int]) -> dict[str, float]:
    counts = Counter(int(card_id) for card_id in main if int(card_id) >= 0)
    return {
        "unique_cards": float(len(counts)),
        "singletons": float(sum(count == 1 for count in counts.values())),
        "pairs": float(sum(count == 2 for count in counts.values())),
        "triplets": float(sum(count == 3 for count in counts.values())),
        "quads": float(sum(count == 4 for count in counts.values())),
        "max_copies": float(max(counts.values(), default=0)),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _arm_summary(records: list[dict]) -> dict[str, object]:
    scores = [float(record["candidate_score"]) for record in records]
    by_context: dict[str, list[float]] = defaultdict(list)
    by_seat: dict[int, list[float]] = defaultdict(list)
    selected_leaders: Counter[str] = Counter()
    behavior: dict[str, list[float]] = defaultdict(list)
    decks: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_context[str(record["context_id"])].append(float(record["candidate_score"]))
        by_seat[int(record["candidate_seat"])].append(float(record["candidate_score"]))
        selected_leaders[str(record["candidate_leader"])] += 1
        player = record["players"][int(record["candidate_seat"])]
        for key in (
            "attack_rate",
            "spell_rate",
            "weapon_rate",
            "portal_rate",
            "play_entity_rate",
            "noop_rate",
            "episode_length",
            "leader_health",
        ):
            value = player.get(key)
            if isinstance(value, (int, float)):
                behavior[key].append(float(value))
        for key, value in _deck_metrics(player["main"]).items():
            decks[key].append(value)
    return {
        "games": len(records),
        "score": _mean(scores),
        "normal_completion_rate": _mean(
            [float(int(record["end_reason"]) == 0) for record in records]
        ),
        "by_context": {key: _mean(values) for key, values in sorted(by_context.items())},
        "by_seat": {str(key): _mean(values) for key, values in sorted(by_seat.items())},
        "selected_leaders": dict(sorted(selected_leaders.items())),
        "behavior": {key: _mean(values) for key, values in sorted(behavior.items())},
        "deck": {key: _mean(values) for key, values in sorted(decks.items())},
    }


def _paired_comparison(left: list[dict], right: list[dict]) -> dict[str, object]:
    left_by_key = {record["pair_key"]: record for record in left}
    right_by_key = {record["pair_key"]: record for record in right}
    keys = sorted(set(left_by_key) & set(right_by_key))
    if len(keys) != len(left) or len(keys) != len(right):
        raise RuntimeError(
            f"Paired panel mismatch: shared={len(keys)} left={len(left)} right={len(right)}"
        )
    deltas = np.asarray(
        [
            float(right_by_key[key]["candidate_score"])
            - float(left_by_key[key]["candidate_score"])
            for key in keys
        ],
        dtype=np.float64,
    )
    by_context: dict[str, list[float]] = defaultdict(list)
    by_seat: dict[int, list[float]] = defaultdict(list)
    for key, delta in zip(keys, deltas.tolist()):
        record = right_by_key[key]
        by_context[str(record["context_id"])].append(float(delta))
        by_seat[int(record["candidate_seat"])].append(float(delta))
    standard_error = float(deltas.std(ddof=1) / np.sqrt(deltas.size)) if deltas.size > 1 else 0.0
    return {
        "games": len(keys),
        "left_arm": ARM_FORCED,
        "right_arm": ARM_PREFILLED,
        "right_minus_left_delta": float(deltas.mean()),
        "standard_error": standard_error,
        "normal_approx_95ci": [
            float(deltas.mean() - 1.96 * standard_error),
            float(deltas.mean() + 1.96 * standard_error),
        ],
        "right_better": int(np.sum(deltas > 0.0)),
        "equal": int(np.sum(deltas == 0.0)),
        "left_better": int(np.sum(deltas < 0.0)),
        "by_context": {key: _mean(values) for key, values in sorted(by_context.items())},
        "by_seat": {str(key): _mean(values) for key, values in sorted(by_seat.items())},
    }


def _build_specs(
    arm: str,
    *,
    contexts: list[tuple[str, str, str]],
    games_per_context: int,
    pool_size: int,
) -> list[dict]:
    if games_per_context % 2 != 0:
        raise ValueError("games_per_context must be even for paired seats")
    specs: list[dict] = []
    for context_index, (element, gate, leader) in enumerate(contexts):
        for block in range(games_per_context // 2):
            seed = 11_000_003 + 100_003 * context_index + 7_919 * block
            reference_deck_index = (context_index + block) % pool_size
            for candidate_seat in (0, 1):
                context_id = f"{element}:{gate}:{leader}"
                pair_key = (
                    f"{context_id}:seed{seed}:ref{reference_deck_index}:seat{candidate_seat}"
                )
                specs.append(
                    {
                        "arm": arm,
                        "context_id": context_id,
                        "element": element,
                        "gate_code": gate,
                        "leader_code": leader,
                        "seed": seed,
                        "candidate_seat": candidate_seat,
                        "reference_seat": 1 - candidate_seat,
                        "reference_deck_index": reference_deck_index,
                        "pair_key": pair_key,
                    }
                )
    return specs


def _run_arm(
    arm: str,
    specs: list[dict],
    *,
    trainer_args: dict,
    checkpoint: Path,
    code_to_id: dict[str, int],
    batch_envs: int,
    max_steps: int,
    device: str,
) -> tuple[list[dict], dict[str, float]]:
    uniform = arm == ARM_PREFILLED
    env_cfg = dict(trainer_args["env"])
    env_cfg.update(
        {
            "native": True,
            "native_envs_per_instance": min(batch_envs, len(specs)),
            "deck_building_enabled": True,
            "draft_uniform_assignment": uniform,
            "evaluation_mode": True,
            "draft_same_element_matchup_prob": 0.0,
            "draft_cross_gate_replay_prob": 0.0,
            "deck_snapshot_dir": None,
        }
    )
    env = make_azuki_env(seed=13_000_003, **env_cfg)
    policy = build_policy(type("VecShape", (), {"driver_env": env})(), trainer_args)
    _load_model_weights(policy, checkpoint, device=device, strict=False)
    policy.eval()
    use_rnn = bool(trainer_args["train"].get("use_rnn", True))
    total_rows = int(env.num_agents)
    state_h = None
    state_c = None
    if use_rnn:
        state_h = torch.zeros(total_rows, policy.hidden_size, device=device)
        state_c = torch.zeros(total_rows, policy.hidden_size, device=device)
    structured = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)
    num_envs = total_rows // 2
    active_specs: list[dict | None] = [None] * num_envs
    decision_steps = np.zeros(num_envs, dtype=np.int32)
    queue_index = 0
    records: list[dict] = []
    forward_rows = 0
    started = time.perf_counter()

    def assign(indices: list[int]) -> None:
        nonlocal queue_index
        resets: list[dict] = []
        for env_index in indices:
            if queue_index >= len(specs):
                active_specs[env_index] = None
                continue
            spec = specs[queue_index]
            queue_index += 1
            active_specs[env_index] = spec
            decision_steps[env_index] = 0
            rows = slice(2 * env_index, 2 * env_index + 2)
            if state_h is not None and state_c is not None:
                state_h[rows].zero_()
                state_c[rows].zero_()
            gate = code_to_id[str(spec["gate_code"])]
            leader = code_to_id[str(spec["leader_code"])]
            resets.append(
                {
                    "env_index": env_index,
                    "seed": int(spec["seed"]),
                    "gate0": gate,
                    "gate1": gate,
                    "leader0": leader if uniform else -1,
                    "leader1": leader if uniform else -1,
                    "reference_seat": int(spec["reference_seat"]),
                    "reference_deck_index": int(spec["reference_deck_index"]),
                }
            )
        if resets:
            env.reset_evaluation_games(resets)

    try:
        assign(list(range(num_envs)))
        while len(records) < len(specs):
            active_players = env.active_players()
            running = np.asarray(
                [
                    index
                    for index, player in enumerate(active_players.tolist())
                    if player >= 0 and active_specs[index] is not None
                ],
                dtype=np.int64,
            )
            if running.size == 0:
                raise RuntimeError("Compatibility panel has queued games but no active env")
            seats = active_players[running].astype(np.int64)
            rows = 2 * running + seats
            row_tensor = torch.as_tensor(rows, device=device, dtype=torch.long)
            state = {"mask": torch.ones(rows.size, device=device, dtype=torch.bool)}
            if state_h is not None and state_c is not None:
                state["lstm_h"] = state_h[row_tensor]
                state["lstm_c"] = state_c[row_tensor]
            with torch.inference_mode():
                logits, _ = policy.forward_eval(
                    torch.as_tensor(env.observations[rows], device=device), state
                )
                actions = tcg_argmax_logits(logits).to(dtype=torch.int32).cpu().numpy()
            if state_h is not None and state_c is not None:
                state_h[row_tensor] = state["lstm_h"]
                state_c[row_tensor] = state["lstm_c"]
            forward_rows += int(rows.size)

            if arm == ARM_FORCED:
                for local, (env_index, seat, row) in enumerate(
                    zip(running.tolist(), seats.tolist(), rows.tolist())
                ):
                    spec = active_specs[int(env_index)]
                    if spec is None or int(structured[int(row)]["deck_context"]["mode"]) != 1:
                        continue
                    if int(seat) != int(spec["candidate_seat"]):
                        raise RuntimeError("Reference seat unexpectedly requested a leader action")
                    context = structured[int(row)]["deck_context"]
                    count = int(context["candidate_count"])
                    candidates = np.asarray(
                        context["candidate_card_def_ids"][:count], dtype=np.int64
                    )
                    desired = code_to_id[str(spec["leader_code"])]
                    matches = np.flatnonzero(candidates == desired)
                    if matches.size != 1:
                        raise RuntimeError(
                            f"Leader {spec['leader_code']} unavailable for {spec['gate_code']}"
                        )
                    actions[local] = np.asarray([3, int(matches[0]), 0, 0], dtype=np.int32)

            env.actions.fill(0)
            env.actions[rows] = actions
            env.step()
            decision_steps[running] += 1
            completed: list[int] = []
            for raw in env.drain_evaluation_records():
                env_index = int(raw["env_index"])
                spec = active_specs[env_index]
                if spec is None:
                    raise RuntimeError(f"Record arrived for idle env {env_index}")
                candidate_seat = int(spec["candidate_seat"])
                candidate = raw["players"][candidate_seat]
                record = dict(spec)
                record.update(raw)
                record["candidate_score"] = _score(raw, candidate_seat)
                record["candidate_gate"] = int(candidate["gate"])
                record["candidate_leader"] = int(candidate["leader"])
                records.append(record)
                completed.append(env_index)
                active_specs[env_index] = None
            if completed:
                assign(completed)

            timed_out = [
                int(index)
                for index in running.tolist()
                if active_specs[int(index)] is not None
                and decision_steps[int(index)] >= max_steps
            ]
            if timed_out:
                env.force_evaluation_truncations(timed_out)
                forced_completed: list[int] = []
                for raw in env.drain_evaluation_records():
                    env_index = int(raw["env_index"])
                    spec = active_specs[env_index]
                    if spec is None:
                        raise RuntimeError(f"Timeout record arrived for idle env {env_index}")
                    candidate_seat = int(spec["candidate_seat"])
                    candidate = raw["players"][candidate_seat]
                    record = dict(spec)
                    record.update(raw)
                    record["candidate_score"] = _score(raw, candidate_seat)
                    record["candidate_gate"] = int(candidate["gate"])
                    record["candidate_leader"] = int(candidate["leader"])
                    records.append(record)
                    forced_completed.append(env_index)
                    active_specs[env_index] = None
                if forced_completed:
                    assign(forced_completed)
    finally:
        env.close()
    wall = time.perf_counter() - started
    return records, {
        "wall_time_seconds": wall,
        "policy_forward_rows": float(forward_rows),
        "policy_forward_rows_per_second": float(forward_rows / max(wall, 1e-9)),
        "games_per_second": float(len(records) / max(wall, 1e-9)),
    }


def _markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Stage 0 Frozen Assignment Panel",
        "",
        f"Checkpoint: `{payload['checkpoint']['path']}`",
        "",
        "| Arm | Games | Score vs reference | Normal completion | Unique cards | Rows/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in (ARM_POLICY, ARM_FORCED, ARM_PREFILLED):
        summary = payload["arms"][arm]["summary"]
        perf = payload["arms"][arm]["performance"]
        lines.append(
            f"| {arm} | {summary['games']} | {summary['score']:.4f} | "
            f"{summary['normal_completion_rate']:.4f} | "
            f"{summary['deck']['unique_cards']:.2f} | "
            f"{perf['policy_forward_rows_per_second']:.1f} |"
        )
    comparison = payload["forced_vs_prefilled"]
    lines.extend(
        [
            "",
            "## Forced Row vs Prefilled",
            "",
            f"Paired prefilled-minus-forced score delta: "
            f"**{comparison['right_minus_left_delta']:+.4f}** "
            f"(normal 95% CI {comparison['normal_approx_95ci'][0]:+.4f} to "
            f"{comparison['normal_approx_95ci'][1]:+.4f}).",
            "",
            "| Candidate seat | Delta |",
            "| --- | ---: |",
        ]
    )
    for seat, delta in comparison["by_seat"].items():
        lines.append(f"| {seat} | {delta:+.4f} |")
    lines.extend(["", "## Context Deltas", "", "| Context | Delta |", "| --- | ---: |"])
    for context, delta in comparison["by_context"].items():
        lines.append(f"| {context} | {delta:+.4f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("python/config/azuki_deckbuild_native_3090.ini"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games-per-context", type=int, default=24)
    parser.add_argument("--policy-games-per-gate", type=int, default=24)
    parser.add_argument("--batch-envs", type=int, default=48)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    args = parser.parse_args()

    trainer_args = load_training_config(args.config, [])
    _apply_checkpoint_resume_policy_config(trainer_args, args.checkpoint)
    trainer_args["train"]["device"] = args.device
    deck_pool = load_training_deck_pool(trainer_args["env"].get("deck_pool_path"))
    from deck_building import build_deck_build_catalog

    catalog = build_deck_build_catalog(deck_pool)
    code_to_id = {
        record.card_code: int(def_id)
        for def_id, record in catalog.records_by_def_id.items()
    }
    id_to_code = {
        int(def_id): record.card_code
        for def_id, record in catalog.records_by_def_id.items()
    }
    leaders = {
        element: tuple(id_to_code[int(def_id)] for def_id in def_ids)
        for element, def_ids in catalog.leader_def_ids_by_element.items()
    }
    contexts = [
        (element, gate, leader)
        for element, gate_pair in GATE_CODE_PAIRS.items()
        for gate in gate_pair
        for leader in leaders[element]
    ]
    policy_contexts = [
        (element, gate, "policy_selected")
        for element, gate_pair in GATE_CODE_PAIRS.items()
        for gate in gate_pair
    ]
    arms: dict[str, dict[str, object]] = {}
    raw_by_arm: dict[str, list[dict]] = {}
    for arm, arm_contexts, games_per_context in (
        (ARM_POLICY, policy_contexts, args.policy_games_per_gate),
        (ARM_FORCED, contexts, args.games_per_context),
        (ARM_PREFILLED, contexts, args.games_per_context),
    ):
        if arm == ARM_POLICY:
            specs = _build_specs(
                arm,
                contexts=[
                    (element, gate, leaders[element][0])
                    for element, gate, _ in arm_contexts
                ],
                games_per_context=games_per_context,
                pool_size=len(deck_pool),
            )
            for spec in specs:
                spec["context_id"] = f"{spec['element']}:{spec['gate_code']}:policy_selected"
                spec["pair_key"] = (
                    f"{spec['context_id']}:seed{spec['seed']}:"
                    f"ref{spec['reference_deck_index']}:seat{spec['candidate_seat']}"
                )
        else:
            specs = _build_specs(
                arm,
                contexts=arm_contexts,
                games_per_context=games_per_context,
                pool_size=len(deck_pool),
            )
        print(f"[{arm}] running {len(specs)} games", flush=True)
        records, performance = _run_arm(
            arm,
            specs,
            trainer_args=trainer_args,
            checkpoint=args.checkpoint,
            code_to_id=code_to_id,
            batch_envs=args.batch_envs,
            max_steps=args.max_steps,
            device=args.device,
        )
        for record in records:
            record["candidate_leader_code"] = id_to_code.get(
                int(record["candidate_leader"]), str(record["candidate_leader"])
            )
            record["candidate_gate_code"] = id_to_code.get(
                int(record["candidate_gate"]), str(record["candidate_gate"])
            )
        raw_by_arm[arm] = records
        arms[arm] = {
            "summary": _arm_summary(records),
            "performance": performance,
            "records": records,
        }
        print(
            f"[{arm}] score={arms[arm]['summary']['score']:.4f} "
            f"rows/s={performance['policy_forward_rows_per_second']:.1f}",
            flush=True,
        )

    payload: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": _sha256(args.checkpoint),
        },
        "config": str(args.config.resolve()),
        "games_per_context": args.games_per_context,
        "policy_games_per_gate": args.policy_games_per_gate,
        "contexts": len(contexts),
        "arms": arms,
        "forced_vs_prefilled": _paired_comparison(
            raw_by_arm[ARM_FORCED], raw_by_arm[ARM_PREFILLED]
        ),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(_markdown(payload), encoding="utf-8")
    print(f"wrote {args.json} and {args.md}")


if __name__ == "__main__":
    main()
