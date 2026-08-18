#!/usr/bin/env python3
"""Run the immutable eight-epoch Azuki SPS benchmark and enforce guardrails."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "python" / "config" / "azuki_speed_3090_parallel.ini"
BUILD_PYTHON_DIR = REPO_ROOT / "build" / "python" / "src"
OUTPUT_DIR = Path("/tmp/azuki-autoresearch-fixed-sps")
EPOCH_PATTERN = re.compile(r"^\[epoch\s+(?P<epoch>\d+)\]\s+(?P<payload>\{.*\})\s*$")
EPOCHS = 8
TAIL_EPOCHS = 5

# These controls are deliberately not configurable from argv or the environment.
# Optimizations must improve this workload rather than search configuration space.
FIXED_CONTROLS: tuple[tuple[str, object], ...] = (
    ("vec.num_envs", 120),
    ("vec.num_workers", 4),
    ("vec.batch_size", 120),
    ("vec.zero_copy", True),
    ("vec.seed", 42),
    ("env.direct_parallel", True),
    ("league.enable", False),
    ("train.total_timesteps", 30_720),
    ("train.batch_size", 3_840),
    ("train.minibatch_size", 960),
    ("train.max_minibatch_size", 960),
    ("train.precision", "bfloat16"),
    ("train.compile", False),
    ("train.seed_process_rngs", True),
    ("train.seed", 42),
    ("wandb", False),
    ("neptune", False),
)


def _cli_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _forwarded_cli(data_dir: Path) -> list[str]:
    controls = (*FIXED_CONTROLS, ("train.data_dir", str(data_dir)))
    return [item for key, value in controls for item in (f"--{key}", _cli_value(value))]


def _resolved_value(config: dict, dotted_key: str) -> object:
    current: object = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise RuntimeError(f"Resolved training config is missing fixed control {dotted_key}")
        current = current[part]
    return current


def _validate_resolved_controls(forwarded_cli: list[str], data_dir: Path) -> None:
    sys.path.insert(0, str(BUILD_PYTHON_DIR))
    sys.path.insert(0, str(REPO_ROOT / "python" / "src"))
    from training_utils import load_training_config

    config = load_training_config(CONFIG_PATH, forwarded_cli)
    expected = (*FIXED_CONTROLS, ("train.data_dir", str(data_dir)))
    mismatches = []
    for key, value in expected:
        actual = _resolved_value(config, key)
        if actual != value or type(actual) is not type(value):
            mismatches.append(f"{key}: expected {value!r}, resolved {actual!r}")
    if mismatches:
        raise RuntimeError("Fixed benchmark controls were not preserved:\n  " + "\n  ".join(mismatches))

    train = config.get("train")
    if not isinstance(train, dict):
        raise RuntimeError("Resolved training config has no train section")
    total_timesteps = int(train["total_timesteps"])
    batch_size = int(train["batch_size"])
    if total_timesteps != EPOCHS * batch_size:
        raise RuntimeError(
            f"Benchmark must resolve to exactly {EPOCHS} epochs; "
            f"got total_timesteps={total_timesteps}, batch_size={batch_size}"
        )


def _subprocess_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("HOME", "PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    env.update(
        {
            "AZK_BUILD_PYTHON_DIR": str(BUILD_PYTHON_DIR),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "MKL_NUM_THREADS": "1",
            "NEPTUNE_MODE": "offline",
            "OMP_NUM_THREADS": "1",
            "PYTHONHASHSEED": "42",
            "PYTHONPATH": os.pathsep.join(
                (str(REPO_ROOT / "python" / "src"), str(BUILD_PYTHON_DIR))
            ),
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "RICH_NO_COLOR": "1",
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "disabled",
        }
    )
    return env


def _parse_epoch(line: str) -> dict[str, object] | None:
    match = EPOCH_PATTERN.match(line.strip())
    if match is None:
        return None
    payload = json.loads(match.group("payload"))
    if not isinstance(payload, dict):
        raise RuntimeError("Epoch payload was not a JSON object")
    epoch = int(match.group("epoch"))
    if payload.get("epoch") != epoch:
        raise RuntimeError(
            f"Epoch prefix/payload mismatch: prefix={epoch}, payload={payload.get('epoch')!r}"
        )
    payload["epoch_index"] = epoch
    return payload


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _mean_metric(records: list[dict[str, object]], key: str) -> float:
    values = [record.get(key) for record in records]
    if not values or not all(_finite_number(value) for value in values):
        raise RuntimeError(f"Metric {key} was missing or non-finite in benchmark tail: {values!r}")
    return statistics.fmean(float(value) for value in values)


def _validate_losses(records: list[dict[str, object]]) -> int:
    observed = 0
    for record in records:
        for key, value in record.items():
            if not key.startswith("losses/"):
                continue
            observed += 1
            if not _finite_number(value):
                raise RuntimeError(
                    f"Non-finite loss at epoch {record['epoch_index']}: {key}={value!r}"
                )
    if observed == 0:
        raise RuntimeError("Benchmark emitted no loss metrics; finite-loss guardrail could not run")
    return observed


def _validate_action_sanity(records: list[dict[str, object]]) -> None:
    reasons: list[str] = []
    for seat in (0, 1):
        prefix = f"environment/{seat}/"
        truncation = _mean_metric(records, prefix + "azk_zero_legal_action_truncation")
        noop = _mean_metric(records, prefix + "azk_noop_selected_rate")
        attack = _mean_metric(records, prefix + "azk_attack_selected_rate")
        play = _mean_metric(records, prefix + "azk_play_selected_rate")
        ability = _mean_metric(records, prefix + "azk_ability_selected_rate")
        target = _mean_metric(records, prefix + "azk_target_selected_rate")
        active_mass = attack + play + ability + target
        if truncation > 1e-6:
            reasons.append(f"seat {seat} truncation {truncation:.6f} exceeded 0.000001")
        if not 0.15 <= noop <= 0.55:
            reasons.append(f"seat {seat} noop {noop:.4f} fell outside [0.15, 0.55]")
        if active_mass < 0.35:
            reasons.append(f"seat {seat} non-noop mass {active_mass:.4f} fell below 0.35")
    if reasons:
        raise RuntimeError("Action-sanity guardrail failed:\n  " + "\n  ".join(reasons))


def _run_training(forwarded_cli: list[str]) -> tuple[list[dict[str, object]], float]:
    command = [
        sys.executable,
        str(REPO_ROOT / "python" / "src" / "train.py"),
        "--config",
        str(CONFIG_PATH),
        *forwarded_cli,
    ]
    lines: list[str] = []
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        if process.stdout is None:
            raise RuntimeError("Training subprocess stdout pipe was not created")
        for line in process.stdout:
            lines.append(line)
            record = _parse_epoch(line)
            if record is not None:
                records.append(record)
    finally:
        if process.stdout is not None:
            process.stdout.close()
    exit_code = process.wait()
    runtime = time.perf_counter() - started
    if exit_code != 0:
        sys.stderr.write("".join(lines[-200:]))
        raise RuntimeError(f"Training subprocess exited with status {exit_code}")
    return records, runtime


def main() -> None:
    if not CONFIG_PATH.is_file():
        raise RuntimeError(f"Benchmark config is missing: {CONFIG_PATH}")
    if not any(BUILD_PYTHON_DIR.glob("binding*.so")):
        raise RuntimeError(f"Native Python binding is missing from {BUILD_PYTHON_DIR}")

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True)
    try:
        forwarded_cli = _forwarded_cli(OUTPUT_DIR)
        _validate_resolved_controls(forwarded_cli, OUTPUT_DIR)
        records, runtime = _run_training(forwarded_cli)

        epoch_indices = [record.get("epoch_index") for record in records]
        expected_indices = list(range(1, EPOCHS + 1))
        if epoch_indices != expected_indices:
            raise RuntimeError(
                f"Expected exactly eight epoch records {expected_indices}, got {epoch_indices}"
            )

        _validate_losses(records)
        tail = records[-TAIL_EPOCHS:]
        _validate_action_sanity(tail)
        tail_sps_values = [float(record["SPS"]) for record in tail if _finite_number(record.get("SPS"))]
        if len(tail_sps_values) != TAIL_EPOCHS or any(value <= 0.0 for value in tail_sps_values):
            raise RuntimeError(f"Invalid SPS values in benchmark tail: {tail_sps_values!r}")

        training_sps = statistics.fmean(tail_sps_values)
        tail_sps_stddev = statistics.pstdev(tail_sps_values)
        print(f"METRIC training_sps={training_sps:.6f}")
        print(f"METRIC tail_sps_stddev={tail_sps_stddev:.6f}")
        print(f"METRIC benchmark_runtime_seconds={runtime:.6f}")
        print(f"METRIC benchmark_epochs={EPOCHS}")
    finally:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
