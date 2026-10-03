#!/usr/bin/env python3
"""Immutable registered, free-draft, two-checkpoint evaluation (CPU only).

Run from the repository root with PYTHONPATH=build/python/src:python/src.
The world seed and each seat's sampling stream are paired across checkpoints
and modes, not the resulting trajectory. Different choices change subsequent
observations, legal sets and engine RNG consumption; opponent actions are not
replayed. Its private sampling stream is independent of candidate RNG use.

Only the acting seat advances its own model's recurrent state, as in native
head_to_head_eval. State survives the draft/battle boundary. Singleton forwards
are padded to two identical rows using that evaluator's compatibility pattern;
the second row is discarded. GameLogger receives a one-action distribution so
it logs/sends precisely the action already selected by the owning checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from evaluate_checkpoint import _apply_checkpoint_resume_policy_config
from play_selfplay_games import GameLogger
from policy.tcg_distribution import TCGLegalActionDistribution
from policy.v2 import tcg_sampler
from probe_gate_kl import EpisodeRunner
from train import _load_model_weights
from training_utils import build_policy, load_training_config
from deck_building import MAX_DECK_SIZE


DISABLED_PROCESS_ENV = {
    "AZK_DRAFT_REF_SEAT_PROB": "0",
    "AZK_DRAFT_REF_LEARNER_FIXED": "0",
    "AZK_DRAFT_REF_OPPONENT_ONLY": "0",
    "AZK_DRAFT_REF_DECK_INDICES": "",
    "AZK_DRAFT_PREFIX_PROBS": "",
    "AZK_DRAFT_PREFIX_POOL_PATH": "",
    "AZK_DRAFT_NORMAL_PENALTY_CONFIG": "",
    "AZK_DRAFT_NORMAL_PENALTY_COEF_INITIAL": "0",
    "AZK_DRAFT_NORMAL_PENALTY_COEF_FINAL": "0",
    "AZK_FIXED_SEAT_DECK_INDICES": "",
    "AZK_DECKBUILD_SNAPSHOT_DIR": "",
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f"SHA256 drift or invalid hash: {path}")


def validate_config(path: Path) -> dict:
    config = load_training_config(path, [])
    env = config.get("env", {})
    if not config.get("train", {}).get("use_rnn", True):
        raise ValueError(f"Recurrent policy required: {path}")
    for key in ("draft_cross_gate_replay_prob", "draft_same_element_matchup_prob"):
        if float(env.get(key, 0) or 0) != 0:
            raise ValueError(f"{path}: env.{key} must be zero")
    fixed = env.get("deck_building_fixed_seats")
    if fixed is not None and fixed != "" and fixed != [] and fixed != ():
        raise ValueError(f"{path}: fixed decks are forbidden")
    if env.get("deck_pool") is not None:
        raise ValueError(f"{path}: inline supplied decks are forbidden")
    if env.get("deck_snapshot_dir"):
        raise ValueError(f"{path}: deck snapshots must be disabled")
    process = {key.upper(): value for key, value in config.get("process_env", {}).items()}
    for key, disabled in DISABLED_PROCESS_ENV.items():
        value = process.get(key, disabled)
        if disabled == "0":
            valid = float(value or 0) == 0
        else:
            valid = value is None or value == ""
        if not valid:
            raise ValueError(f"{path}: {key} must be disabled, got {value!r}")
    return config


def validate_registration(registration: dict) -> None:
    if registration.get("schema_id") != "azuki.prefix_paired_eval_registration" or registration.get("schema_version") != 1:
        raise ValueError("Unsupported paired evaluation registration")
    if registration.get("step_cap") != 1200:
        raise ValueError("Registered step_cap must be 1200")
    sources = registration["source_sha256"]
    if not sources:
        raise ValueError("source_sha256 cannot be empty")
    registered_paths = {Path(path).resolve() for path in sources}
    for path, expected in sources.items():
        verify_hash(Path(path), expected)
    for collection in ("checkpoints", "opponents"):
        if not registration[collection]:
            raise ValueError(f"Empty {collection}")
        for spec in registration[collection].values():
            checkpoint = Path(spec["checkpoint"])
            verify_hash(checkpoint, spec["checkpoint_sha256"])
            config = Path(spec["config"])
            metadata = checkpoint.with_suffix(checkpoint.suffix + ".meta.json")
            required = [config, metadata] if metadata.exists() else [config]
            for path in required:
                if path.resolve() not in registered_paths:
                    raise ValueError(f"Consumed input missing from source_sha256: {path}")
            validate_config(config)
    ids = set()
    if not registration["tasks"]:
        raise ValueError("No registered tasks")
    for task in registration["tasks"]:
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id or task_id in ids:
            raise ValueError(f"Invalid/duplicate task_id: {task_id!r}")
        ids.add(task_id)
        if type(task["seed"]) is not int or not 0 <= task["seed"] < 2**31:
            raise ValueError(f"Invalid world seed: {task_id}")
        if type(task["candidate_seat"]) is not int or task["candidate_seat"] not in (0, 1):
            raise ValueError(f"Invalid candidate seat: {task_id}")
        if task["opponent_id"] not in registration["opponents"]:
            raise ValueError(f"Unknown opponent: {task_id}")
        for key in ("candidate_gate", "candidate_leader", "opponent_gate", "opponent_leader"):
            if not isinstance(task[key], str) or not task[key]:
                raise ValueError(f"Missing assignment {key}: {task_id}")
        if set(task) != {"task_id", "seed", "candidate_gate", "candidate_leader", "opponent_id", "opponent_gate", "opponent_leader", "candidate_seat"}:
            raise ValueError(f"Unexpected task fields (decks/prefixes forbidden): {task_id}")


def seat_seed(world_seed: int, seat: int) -> int:
    payload = f"azuki.prefix_paired_eval.v1:{world_seed}:seat:{seat}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


class PairedPolicy:
    """GameLogger-compatible active-seat dispatcher; owns all recurrent state."""

    def __init__(self, runner: EpisodeRunner, policies: list, seed: int, mode: str, expected: list):
        self.runner = runner
        self.policies = policies
        self.mode = mode
        self.expected = expected
        self.calls = [0, 0]
        self.states = [
            {name: torch.zeros(1, policy.hidden_size) for name in ("lstm_h", "lstm_c")}
            for policy in policies
        ]
        self.seeds = [seat_seed(seed, seat) for seat in range(2)]
        self.rng_states = [torch.Generator(device="cpu").manual_seed(value).get_state() for value in self.seeds]

    def forward_eval(self, observations, state):
        if not any(self.calls):
            for seat, (gate, leader) in enumerate(self.expected):
                draft = self.runner.base_env._states[seat]
                if (draft.gate_card_def_id, draft.leader_card_def_id) != (gate, leader) or draft.main_count != 0:
                    raise ValueError(f"Seat {seat}: reset did not produce assigned, empty free draft")
        active = int(self.runner.base_env._active_player_index)
        # Native evaluator advances only the active owner's history, not the
        # inactive observation row. Padding is not a second seat or history.
        row = observations[active:active + 1]
        sub_obs = row.expand(2, *row.shape[1:])
        sub_state = {name: value.expand(2, -1).contiguous() for name, value in self.states[active].items()}
        sub_state["mask"] = state["mask"][active:active + 1].repeat(2)
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.rng_states[active])
            logits, _ = self.policies[active].forward_eval(sub_obs, sub_state)
            if self.mode == "sample":
                actions, _, _ = tcg_sampler.tcg_sample_logits(logits)
            else:
                actions = tcg_sampler.tcg_argmax_logits(logits)
            self.rng_states[active] = torch.get_rng_state()
        self.states[active] = {name: sub_state[name][:1].clone() for name in ("lstm_h", "lstm_c")}
        self.calls[active] += 1
        # Only the active row is consumed by the environment. No second policy
        # forward and no candidate-generated action can control the other seat.
        chosen = torch.zeros(2, 1, 4, dtype=torch.long)
        chosen[active, 0] = actions[0]
        return TCGLegalActionDistribution(torch.zeros(2, 1), chosen, torch.ones(2, dtype=torch.long)), torch.zeros(2)


def assignments(runner: EpisodeRunner, task: dict) -> tuple[list, list]:
    by_seat = [None, None]
    by_seat[task["candidate_seat"]] = (task["candidate_gate"], task["candidate_leader"])
    by_seat[1 - task["candidate_seat"]] = (task["opponent_gate"], task["opponent_leader"])
    ids = []
    for gate, leader in by_seat:
        gate_id, leader_id = runner.code_to_def[gate], runner.code_to_def[leader]
        if gate_id not in runner.catalog.gate_def_id_population:
            raise ValueError(f"Not a registered gate: {gate}")
        element = runner.catalog.records_by_def_id[gate_id].element
        if leader_id not in runner.catalog.leader_def_ids_by_element[element]:
            raise ValueError(f"Leader {leader} incompatible with gate {gate}")
        ids.append((gate_id, leader_id))
    return by_seat, ids


def load_opponent(runner: EpisodeRunner, spec: dict):
    config = validate_config(Path(spec["config"]))
    config["train"]["device"] = "cpu"
    _apply_checkpoint_resume_policy_config(config, Path(spec["checkpoint"]))
    policy = build_policy(runner.vecenv, config)
    runner.vecenv.async_reset(seed=7)
    obs, _, _, _, _, _, masks = runner.vecenv.recv()
    state = {name: torch.zeros(2, policy.hidden_size) for name in ("lstm_h", "lstm_c")}
    state["mask"] = torch.as_tensor(masks)
    with torch.no_grad():
        policy.forward_eval(torch.as_tensor(obs), state)
    _load_model_weights(policy, Path(spec["checkpoint"]), device="cpu", strict=True)
    policy.eval()
    policy.requires_grad_(False)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--checkpoint-key", required=True)
    parser.add_argument("--mode", choices=("sample", "argmax"), required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Exclusive-create JSONL; never appended or overwritten")
    parser.add_argument("--task-limit", type=int, help="Smoke only: first N tasks AFTER striding")
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error("Require shards > 0 and 0 <= shard-index < shards")
    if args.task_limit is not None and args.task_limit < 1:
        parser.error("task-limit must be positive")
    if args.out.exists():
        raise FileExistsError(args.out)
    registration_bytes = args.registration.read_bytes()
    registration = json.loads(registration_bytes)
    registration_hash = hashlib.sha256(registration_bytes).hexdigest()
    validate_registration(registration)
    spec = registration["checkpoints"][args.checkpoint_key]
    tasks = list(enumerate(registration["tasks"]))
    tasks = tasks[args.shard_index::args.shards]
    if args.task_limit is not None:
        tasks = tasks[:args.task_limit]
    if not tasks:
        raise ValueError("Shard selects no tasks")
    # Do not apply training process_env: only explicit evaluation exclusions.
    os.environ.update(DISABLED_PROCESS_ENV)
    runner = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        try:
            runner = EpisodeRunner.__new__(EpisodeRunner)
            runner.__init__(Path(spec["config"]), Path(spec["checkpoint"]), "cpu", uniform_assignment=True)
            if runner.vecenv.num_agents != 2 or runner.base_env._fixed_deck_seats or not runner.base_env._uniform_assignment:
                raise ValueError("Expected one two-seat uniform free-draft environment")
            candidate = runner.policy
            # EpisodeRunner's probe loader is permissive; require a complete
            # exact learned state before permitting this registered evaluation.
            _load_model_weights(candidate, Path(spec["checkpoint"]), device="cpu", strict=True)
            candidate.requires_grad_(False)
            opponents = {
                opponent_id: load_opponent(runner, registration["opponents"][opponent_id])
                for opponent_id in sorted({task["opponent_id"] for _, task in tasks})
            }
            tcg_sampler.set_sampling_params(primary_temperature=1.0, subaction_temperature=1.0,
                                            smoothing_eps=0.0, legal_row_temperature=1.0, deck_pick_smoothing_eps=0.0)
            # Dispatcher owns history; prevent GameLogger allocating its own.
            runner.use_rnn = False
            logger = GameLogger(runner, log_legal_actions=True, action_mode=args.mode)
            started = time.monotonic()
            for completed, (task_index, task) in enumerate(tasks, 1):
                verify_hash(args.registration, registration_hash)
                verify_hash(Path(spec["checkpoint"]), spec["checkpoint_sha256"])
                opponent_spec = registration["opponents"][task["opponent_id"]]
                verify_hash(Path(opponent_spec["checkpoint"]), opponent_spec["checkpoint_sha256"])
                for path, expected_hash in registration["source_sha256"].items():
                    verify_hash(Path(path), expected_hash)
                codes, ids = assignments(runner, task)
                runner.force_gates(codes[0][0], codes[1][0])
                runner.force_assigned_leaders(codes[0][1], codes[1][1])
                policies = [opponents[task["opponent_id"]], opponents[task["opponent_id"]]]
                policies[task["candidate_seat"]] = candidate
                dispatcher = PairedPolicy(runner, policies, task["seed"], args.mode, ids)
                runner.policy = dispatcher
                record = logger.play_game(task_index, task["seed"], registration["step_cap"])
                outcome = record["outcome"]
                errors = []
                complete = outcome["terminated"] and not outcome["truncated"]
                if not complete:
                    errors.append("incomplete_game")
                decks = record["decks"]
                if decks is None or any((deck["gate"], deck["leader"]) != codes[seat] or len(deck["main"]) != MAX_DECK_SIZE for seat, deck in enumerate(decks)):
                    errors.append("deck_assignment_or_size_mismatch")
                if record["draft_steps"] != 2 * MAX_DECK_SIZE or any(sum(pick["p"] == seat for pick in record["draft"]) != MAX_DECK_SIZE for seat in range(2)):
                    errors.append("not_full_free_draft")
                if not all(dispatcher.calls):
                    errors.append("missing_policy_actions")
                winner = outcome["winner"]
                score = None if errors else (0.5 if winner == -1 else float(winner == task["candidate_seat"]))
                record["paired_eval"] = {
                    **task, "checkpoint_key": args.checkpoint_key, "candidate_score": score,
                    "candidate": spec, "opponent": opponent_spec,
                    "registration": str(args.registration), "registration_sha256": registration_hash,
                    "task_index": task_index, "mode": args.mode, "shards": args.shards,
                    "shard_index": args.shard_index, "smoke_task_limit": args.task_limit,
                    "world_seed": task["seed"], "seat_sampling_seeds": dispatcher.seeds,
                    "pairing": "same world seed and private per-seat RNG; trajectories may diverge",
                    "recurrent_progression": "active-seat-only; preserved draft through battle; isolated per policy",
                    "policy_forward_counts": dispatcher.calls, "sampling": tcg_sampler.get_sampling_params(),
                    "validation_errors": errors, "complete": not errors,
                }
                output.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                print(json.dumps({"completed": completed, "selected_tasks": len(tasks), "task_id": task["task_id"],
                                  "candidate_score": score, "errors": errors, "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
                if errors:
                    raise RuntimeError(f"Retained invalid trace; failing closed: {task['task_id']}: {errors}")
        finally:
            if runner is not None and hasattr(runner, "vecenv"):
                runner.vecenv.close()


if __name__ == "__main__":
    main()
