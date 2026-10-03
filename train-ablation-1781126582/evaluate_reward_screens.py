#!/usr/bin/env python3
"""Evaluate completed reward screens under their registered contract."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from strategy_descriptor import SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "train-ablation-1781126582/results/reward_screens"
DEFAULT_REGISTRATION = RESULT_ROOT / "registration.json"
DEFAULT_UPDATES = (325, 650, 975)
TRACE_GAMES = 200
ANCHORS = {
    "p021000": (
        ROOT
        / "experiments/azuki_local_corrected_production_1b_lr1500_fresh_resume_p19500_178757505266"
        / "model_azuki_local_021000.pt"
    ),
    "p044000": (
        ROOT
        / "experiments/azuki_local_corrected_production_1b_lr1500_fresh_resume_p29300_178765385220"
        / "model_azuki_local_044000.pt"
    ),
    "p060000": (
        ROOT
        / "experiments/azuki_local_corrected_production_1b_lr1500_fresh_resume_p52750_178783658456"
        / "model_azuki_local_060000.pt"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _checkpoint_from_manifest(experiment_dir: Path, update: int) -> tuple[Path, str]:
    manifest_path = experiment_dir / f"checkpoint_{update:06d}.manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("update") != update:
        raise ValueError(f"checkpoint update mismatch: {manifest_path}")
    model_paths = []
    for artifact in manifest.get("artifacts", []):
        if not isinstance(artifact, dict):
            raise ValueError(f"invalid artifact in {manifest_path}")
        path = experiment_dir / str(artifact.get("path", ""))
        expected = artifact.get("sha256")
        if not path.is_file() or not isinstance(expected, str) or _sha256(path) != expected:
            raise ValueError(f"checkpoint artifact mismatch: {path}")
        if path.name.startswith("model_") and path.suffix == ".pt":
            model_paths.append((path, expected))
    if len(model_paths) != 1:
        raise ValueError(f"manifest must contain one model: {manifest_path}")
    return model_paths[0]


def _experiment_dir(arm: dict, run_status: dict) -> Path:
    recorded = run_status.get("latest_experiment_dir")
    if isinstance(recorded, str):
        path = ROOT / recorded
        if path.is_dir():
            return path
    candidates = sorted(
        [path for path in ROOT.glob(f"experiments/azuki_local_{arm['tag']}_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise FileNotFoundError(f"no experiment directory for {arm['id']}")
    return candidates[-1]


def _valid_trace(path: Path, games: int) -> bool:
    if not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != games:
        return False
    return all(isinstance(json.loads(line), dict) for line in lines)


def _valid_json(path: Path, *, schema_id: str | None = None, schema_version: int | None = None) -> bool:
    if not path.is_file():
        return False
    payload = _read_json(path)
    return (schema_id is None or payload.get("schema_id") == schema_id) and (
        schema_version is None or payload.get("schema_version") == schema_version
    )


def _run(command: list[str], log_path: Path, output_path: Path, validator) -> None:
    if validator(output_path):
        print(f"[reward-eval] reuse {output_path.relative_to(ROOT)}", flush=True)
        return
    if output_path.exists():
        raise RuntimeError(f"refusing invalid partial output: {output_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[reward-eval] start {output_path.relative_to(ROOT)}", flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=_subprocess_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"evaluation failed with {completed.returncode}; inspect {log_path.relative_to(ROOT)}"
        )
    if not validator(output_path):
        raise RuntimeError(f"evaluation produced invalid output: {output_path}")
    print(
        f"[reward-eval] done {output_path.relative_to(ROOT)} "
        f"wall={time.time() - started:.1f}s",
        flush=True,
    )


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [
        str(ROOT / "build/python/src"),
        str(ROOT / "python/src"),
        str(ROOT / "train-ablation-1781126582"),
    ]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/azuki-pyc")
    env.setdefault("OMP_NUM_THREADS", "6")
    return env


def _context_key(opponent: str, game: dict) -> str:
    values = (
        ("source", "h2h"),
        ("opponent", opponent),
        ("candidate_gate", game.get("candidate_gate_code", game.get("candidate_gate", "?"))),
        ("candidate_leader", game.get("candidate_leader", "?")),
        ("opponent_gate", game.get("opponent_gate", "?")),
        ("opponent_leader", game.get("opponent_leader", "?")),
        ("candidate_seat", game.get("candidate_seat", "?")),
        ("candidate_started", bool(game.get("candidate_started", False))),
        ("reference_deck", game.get("reference_deck_index", -1)),
    )
    return "|".join(f"{key}={value}" for key, value in values)


def _write_payoff_cells(paths: list[Path], output: Path) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        payload = _read_json(path)
        if payload.get("policy_action_mode") != "legal_argmax_stable_first":
            raise ValueError(f"non-deterministic payoff artifact: {path}")
        opponent = str(payload["opponent"]["label"])
        for game in payload["games"]:
            grouped[_context_key(opponent, game)].append(float(game["candidate_score"]))
    cells = [
        {"context_key": key, "score": sum(scores) / len(scores), "games": len(scores)}
        for key, scores in sorted(grouped.items())
    ]
    _write_json(output, cells)


def _training_summary(log_path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metrics = [row for row in rows if "_step" in row]
    if not metrics:
        raise ValueError(f"no metric windows: {log_path}")
    sps = [float(row["SPS"]) for row in metrics[1:] or metrics]
    integrity_keys = (
        "environment/timeout_truncation_rate",
        "environment/auto_tick_truncation_rate",
        "environment/zero_legal_action_truncation_rate",
        "environment/draft_episode_credit/incomplete_episodes",
    )
    integrity_maxima = {
        key: max(float(row.get(key, 0.0)) for row in metrics)
        for key in integrity_keys
    }
    invalid_values = [
        float(value)
        for row in metrics
        for key, value in row.items()
        if "invalid" in key and isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    reward_raw_reconstruction = max(
        float(row.get("environment/reward_telemetry/raw_reconstruction_max_abs_error", 0.0))
        for row in metrics
    )
    reward_scaled_reconstruction = max(
        float(row.get("environment/reward_telemetry/scaled_reconstruction_max_abs_error", 0.0))
        for row in metrics
    )
    ppo_component_reconstruction = max(
        float(row.get("losses/ppo_diag_component_reconstruction_error_max", 0.0))
        for row in metrics
    )
    return {
        "metric_windows": len(metrics),
        "last_step": int(metrics[-1]["_step"]),
        "last_epoch": int(metrics[-1]["epoch"]),
        "steady_sps_median": statistics.median(sps),
        "integrity_maxima": integrity_maxima,
        "max_invalid_metric": max(invalid_values, default=0.0),
        "reward_raw_reconstruction_max_abs_error": reward_raw_reconstruction,
        "reward_scaled_reconstruction_max_abs_error": reward_scaled_reconstruction,
        "ppo_component_reconstruction_max_abs_error": ppo_component_reconstruction,
    }


def _evaluate_window(
    *,
    arm: dict,
    config: Path,
    checkpoint: Path,
    checkpoint_hash: str,
    update: int,
    games: int,
    last_update: int,
    device: str,
) -> dict:
    arm_root = (ROOT / arm["result_root"]).resolve()
    output_dir = arm_root / "eval" / f"p{update:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = {}
    descriptors = {}
    for mode in ("sample", "argmax"):
        trace = output_dir / f"trace_{mode}.jsonl"
        command = [
            str(ROOT / ".venv/bin/python"),
            "train-ablation-1781126582/play_selfplay_games.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--games",
            str(games),
            "--seed0",
            str(71_000_000 + update * 10_007),
            "--device",
            device,
            "--action-mode",
            mode,
            "--log-legal-actions",
            "--uniform-assignment",
            "--out",
            str(trace),
        ]
        _run(
            command,
            output_dir / f"trace_{mode}.log",
            trace,
            lambda path, expected=games: _valid_trace(path, expected),
        )
        traces[mode] = trace

    payoff_paths = []
    for label, anchor in ANCHORS.items():
        output = output_dir / f"h2h_vs_{label}.json"
        command = [
            str(ROOT / ".venv/bin/python"),
            "python/src/native_policy_eval.py",
            "--config",
            str(config),
            "--checkpoint-a",
            str(checkpoint),
            "--checkpoint-b",
            str(anchor),
            "--label-a",
            f"{arm['id']}_p{update:06d}",
            "--label-b",
            label,
            "--batch-envs",
            "12",
            "--device",
            device,
            "--json",
            str(output),
        ]
        _run(
            command,
            output_dir / f"h2h_vs_{label}.log",
            output,
            lambda path: _valid_json(path),
        )
        payoff_paths.append(output)

    payoff_cells = output_dir / "payoff_cells.json"
    _write_payoff_cells(payoff_paths, payoff_cells)
    for mode, trace in traces.items():
        descriptor = output_dir / f"strategy_descriptor_{mode}.json"
        command = [
            str(ROOT / ".venv/bin/python"),
            "train-ablation-1781126582/strategy_descriptor.py",
            str(trace),
            "--label",
            f"{arm['id']}_p{update:06d}_{mode}",
            "--checkpoint-sha256",
            checkpoint_hash,
            "--payoff-cells",
            str(payoff_cells),
            "--json",
            str(descriptor),
        ]
        _run(
            command,
            output_dir / f"strategy_descriptor_{mode}.log",
            descriptor,
            lambda path: _valid_json(path, schema_id="azuki.strategy_descriptor", schema_version=SCHEMA_VERSION),
        )
        descriptors[mode] = descriptor

    if update == last_update:
        curated = output_dir / "curated_strategy_panel.json"
        command = [
            str(ROOT / ".venv/bin/python"),
            "train-ablation-1781126582/run_curated_strategy_panel.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--seeds",
            "2",
            "--device",
            device,
            "--json",
            str(curated),
        ]
        _run(
            command,
            output_dir / "curated_strategy_panel.log",
            curated,
            lambda path: _valid_json(path, schema_id="azuki.curated_strategy_panel"),
        )

    return {
        "update": update,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "sample_trace": str(traces["sample"].relative_to(ROOT)),
        "argmax_trace": str(traces["argmax"].relative_to(ROOT)),
        "sample_descriptor": str(descriptors["sample"].relative_to(ROOT)),
        "argmax_descriptor": str(descriptors["argmax"].relative_to(ROOT)),
        "payoff_cells": str(payoff_cells.relative_to(ROOT)),
        "h2h": {
            label: str((output_dir / f"h2h_vs_{label}.json").relative_to(ROOT))
            for label in ANCHORS
        },
        "curated_strategy_panel": (
            str((output_dir / "curated_strategy_panel.json").relative_to(ROOT))
            if update == last_update
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument(
        "--output",
        type=Path,
        help="Evaluation index path; defaults to evaluation_index.json beside registration.",
    )
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--games", type=int, default=TRACE_GAMES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.games < 1:
        raise ValueError("--games must be positive")

    registration = _read_json(args.registration.resolve())
    if registration.get("schema_id") != "azuki.ablation_registration":
        raise ValueError("invalid reward-screen registration schema")
    configured_updates = registration.get("shared_contract", {}).get(
        "evaluation_updates",
        DEFAULT_UPDATES,
    )
    if (
        not isinstance(configured_updates, list | tuple)
        or not configured_updates
        or any(not isinstance(update, int) or update <= 0 for update in configured_updates)
    ):
        raise ValueError("registration evaluation_updates must be positive integers")
    updates = tuple(configured_updates)
    if tuple(sorted(set(updates))) != updates:
        raise ValueError("registration evaluation_updates must be unique and increasing")
    evaluation_root = args.registration.resolve().parent
    index_path = (
        args.output.resolve()
        if args.output is not None
        else evaluation_root / "evaluation_index.json"
    )
    reused_control = registration.get("reused_control")
    if isinstance(reused_control, dict):
        control_id = str(reused_control["id"])
        control_checkpoint: Path | None = ROOT / str(reused_control["checkpoint"])
        expected_control_hash = str(reused_control["checkpoint_sha256"])
        if _sha256(control_checkpoint) != expected_control_hash:
            raise ValueError(f"reused control checkpoint changed: {control_checkpoint}")
    else:
        control_id = str(registration.get("control_arm_id", "R0"))
        control_checkpoint = None
    selected = set(args.arm)
    arms = [arm for arm in registration["arms"] if not selected or arm["name"] in selected]
    names = {arm["name"] for arm in arms}
    if selected and selected != names:
        raise ValueError(f"unknown reward-screen arms: {sorted(selected - names)}")
    for anchor in ANCHORS.values():
        if not anchor.is_file():
            raise FileNotFoundError(anchor)
    if args.plan:
        print(
            json.dumps(
                {
                    "arms": [arm["id"] for arm in arms],
                    "updates": list(updates),
                    "trace_games_per_mode_window": args.games,
                    "anchors": {label: str(path.relative_to(ROOT)) for label, path in ANCHORS.items()},
                    "control": control_id,
                },
                indent=2,
            )
        )
        return

    index = {
        "schema_id": "azuki.ablation_evaluation_index",
        "schema_version": 1,
        "registration": str(args.registration.resolve().relative_to(ROOT)),
        "registration_sha256": _sha256(args.registration.resolve()),
        "updates": list(updates),
        "trace_games_per_mode_window": args.games,
        "policy_modes": ["sample_temperature_1_no_smoothing", "legal_argmax_stable_first"],
        "control": {
            "id": control_id,
            "checkpoint": (
                str(control_checkpoint.relative_to(ROOT))
                if control_checkpoint is not None
                else None
            ),
        },
        "arms": [],
    }
    active_control_checkpoint = control_checkpoint
    for arm in arms:
        result_root = (ROOT / arm["result_root"]).resolve()
        run_status = _read_json(result_root / "run_status.json")
        if run_status.get("state") != "completed" or run_status.get("returncode") != 0:
            raise RuntimeError(f"reward arm is incomplete: {arm['id']}")
        experiment_dir = _experiment_dir(arm, run_status)
        training_config = ROOT / arm["config"]
        if _sha256(training_config) != arm["config_sha256"]:
            raise ValueError(f"registered config changed: {training_config}")
        config = ROOT / arm.get("evaluation_config", arm["config"])
        expected_evaluation_hash = arm.get(
            "evaluation_config_sha256",
            arm["config_sha256"],
        )
        if _sha256(config) != expected_evaluation_hash:
            raise ValueError(f"registered evaluation config changed: {config}")
        status_path = result_root / "evaluation_status.json"
        status = {
            "schema_id": "azuki.reward_screen_evaluation_status",
            "schema_version": 1,
            "arm": arm["id"],
            "state": "running",
            "started_at": time.time(),
        }
        _write_json(status_path, status)
        windows = []
        try:
            for update in updates:
                checkpoint, checkpoint_hash = _checkpoint_from_manifest(experiment_dir, update)
                windows.append(
                    _evaluate_window(
                        arm=arm,
                        config=config,
                        checkpoint=checkpoint,
                        checkpoint_hash=checkpoint_hash,
                        update=update,
                        games=args.games,
                        last_update=updates[-1],
                        device=args.device,
                    )
                )
                if arm["id"] == control_id and update == updates[-1]:
                    active_control_checkpoint = checkpoint
            if arm["id"] != control_id:
                if active_control_checkpoint is None:
                    raise RuntimeError(f"{control_id} must be available before treatment arms")
                final_checkpoint = ROOT / windows[-1]["checkpoint"]
                direct = result_root / "eval" / f"p{updates[-1]:06d}" / "h2h_vs_control.json"
                command = [
                    str(ROOT / ".venv/bin/python"),
                    "python/src/native_policy_eval.py",
                    "--config",
                    str(config),
                    "--checkpoint-a",
                    str(final_checkpoint),
                    "--checkpoint-b",
                    str(active_control_checkpoint),
                    "--label-a",
                    arm["id"],
                    "--label-b",
                    control_id,
                    "--batch-envs",
                    "12",
                    "--device",
                    args.device,
                    "--json",
                    str(direct),
                ]
                _run(
                    command,
                    direct.with_suffix(".log"),
                    direct,
                    lambda path: _valid_json(path),
                )
                windows[-1]["h2h_vs_control"] = str(direct.relative_to(ROOT))
            arm_index = {
                "id": arm["id"],
                "name": arm["name"],
                "training": _training_summary(result_root / "logs/production.jsonl"),
                "windows": windows,
            }
            index["arms"].append(arm_index)
            status.update(
                state="completed",
                finished_at=time.time(),
                windows=len(windows),
            )
            _write_json(status_path, status)
            _write_json(index_path, index)
        except BaseException:
            status.update(state="failed", finished_at=time.time())
            _write_json(status_path, status)
            raise

    index["completed_at"] = time.time()
    _write_json(index_path, index)
    print(f"[reward-eval] wrote {index_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
