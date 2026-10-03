#!/usr/bin/env python3
"""Run the immutable diagnostic campaign serially; never launch production."""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from evaluate_reward_screens import (
    _checkpoint_from_manifest,
    _evaluate_window,
    _training_summary,
    _write_json,
)

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
MAX_ARM_ROWS = 45_004_800
INTEGRITY_KEYS = (
    "environment/timeout_truncation_rate",
    "environment/auto_tick_truncation_rate",
    "environment/zero_legal_action_truncation_rate",
    "environment/draft_episode_credit/incomplete_episodes",
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_files(hashes: dict) -> None:
    if not hashes:
        raise ValueError("Missing immutable file hashes")
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Registered file changed or missing: {name}")


def verify_registration_files(registration: dict) -> None:
    verify_files(registration["source_sha256"])
    verify_files(registration["artifact_sha256"])
    recorded = set(registration["source_sha256"])
    for pattern in registration["expected_source_globs"] + registration["expected_binary_globs"]:
        current = {str(path.relative_to(ROOT)) for path in ROOT.glob(pattern) if path.is_file()}
        if not current or not current <= recorded:
            raise ValueError(f"Registered file membership changed: {pattern}")


def metric_failure(rows: list[dict]) -> str | None:
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value):
                    return f"non-finite metric {key}={value}"
                if key in INTEGRITY_KEYS and value > 0:
                    return f"integrity failure {key}={value}"
        for key, tolerance in (
            ("environment/reward_telemetry/raw_reconstruction_max_abs_error", 1e-5),
            ("environment/reward_telemetry/scaled_reconstruction_max_abs_error", 1e-5),
            ("losses/ppo_diag_component_reconstruction_error_max", 1e-4),
        ):
            if float(row.get(key, 0)) > tolerance:
                return f"reconstruction failure {key}={row[key]}"
    return None


def read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # The writer may be in the middle of its final line.
    lines = path.read_bytes().split(b"\n")[:-1]
    return [row for line in lines if line.strip()
            if "_step" in (row := json.loads(line))]


def run_training(arm: dict, registration: dict, registration_hash: str) -> dict:
    result_root = ROOT / arm["result_root"]
    status_path = result_root / "run_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("registration_sha256") != registration_hash:
            raise ValueError("Run status belongs to a different registration")
        if status.get("state") == "completed":
            return status
        raise ValueError(f"Refusing to overwrite incomplete run: {status_path}")
    config = ROOT / arm["config"]
    verify_files({arm["config"]: arm["config_sha256"]})
    parsed = configparser.ConfigParser()
    parsed.read(config)
    rows = parsed.getint("train", "total_timesteps")
    if rows != arm["total_timesteps"] or not 0 < rows <= MAX_ARM_ROWS:
        raise ValueError("Arm row count exceeds its diagnostic authorization")
    if parsed.getint("train", "seed") != 42:
        raise ValueError("Matched diagnostic seed must be 42")
    data_dir = ROOT / parsed.get("train", "data_dir", fallback="experiments")
    existing_experiments = set(data_dir.glob(f"azuki_local_{arm['tag']}_*"))
    if existing_experiments:
        raise ValueError(f"Fresh arm already has checkpoint directories: {arm['name']}")
    status = {
        "schema_id": "azuki.ablation_run_status", "schema_version": 1,
        "family": "strategy_discovery_v1", "arm": arm["id"], "name": arm["name"],
        "state": "running", "started_at": time.time(),
        "registration_sha256": registration_hash, "production_qualified": False,
        "config": arm["config"], "config_sha256": arm["config_sha256"],
        "source_sha256": registration["source_sha256"], "sampled_rows": rows, "seed": 42,
    }
    _write_json(status_path, status)
    env = os.environ.copy()
    # Never inherit an old experiment's process knobs into a registered arm.
    env = {key: value for key, value in env.items() if not key.startswith("AZK_")}
    env.update(PYTHONPATH=f"{ROOT / 'build/python/src'}:{ROOT / 'python/src'}", OMP_NUM_THREADS="2")
    log = result_root / "training_console.log"
    print(f"[discovery] starting {arm['id']} {arm['name']} rows={rows}", flush=True)
    failure = None
    with log.open("w") as console:
        process = subprocess.Popen([str(PYTHON), "python/src/train.py", "--config", str(config)],
                                   cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT)
        try:
            last_epoch = -1
            while process.poll() is None:
                time.sleep(10)
                metrics = read_metrics(result_root / "logs/production.jsonl")
                failure = metric_failure(metrics)
                if metrics and int(metrics[-1]["epoch"]) != last_epoch:
                    last_epoch = int(metrics[-1]["epoch"])
                    latest = metrics[-1]
                    status["progress"] = {key: latest[key] for key in (
                        "epoch", "agent_steps", "SPS", "losses/approx_kl",
                        "losses/ppo_diag_component_reconstruction_error_max",
                        "environment/league/strategic_exposure/supplied_learner_battle_row_fraction",
                    ) if key in latest}
                    _write_json(status_path, status)
                    print(f"[discovery] {arm['id']} progress {json.dumps(status['progress'])}", flush=True)
                if len(metrics) >= 12:
                    steady = [float(row["SPS"]) for row in metrics[2:] if "SPS" in row]
                    if steady and statistics.median(steady) < 1235:
                        failure = f"steady median SPS below registered floor: {statistics.median(steady)}"
                if failure:
                    process.terminate()
                    break
            try:
                returncode = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            status.update(state="interrupted", finished_at=time.time())
            _write_json(status_path, status)
            raise
    metrics = read_metrics(result_root / "logs/production.jsonl")
    failure = failure or metric_failure(metrics)
    directories = sorted(
        path for path in data_dir.glob(f"azuki_local_{arm['tag']}_*") if path.is_dir()
    )
    if len(directories) != 1:
        failure = failure or "fresh arm did not produce exactly one checkpoint directory"
    if not metrics:
        failure = failure or "no training metric windows"
    if directories:
        status["latest_experiment_dir"] = str(directories[0].relative_to(ROOT))
    status.update(returncode=returncode, finished_at=time.time(), failure=failure,
                  state="completed" if returncode == 0 and failure is None else "failed")
    if status["state"] == "completed":
        try:
            status["training"] = _training_summary(result_root / "logs/production.jsonl")
            final_update = rows // 15_360
            _checkpoint_from_manifest(directories[0], final_update)
            status["verified_final_update"] = final_update
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            status.update(state="failed", failure=f"final checkpoint verification: {exc}")
    _write_json(status_path, status)
    if status["state"] != "completed":
        raise RuntimeError(f"{arm['name']} failed: {failure}; console={log}")
    print(f"[discovery] completed training {arm['id']}", flush=True)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--deck-diagnostic", type=Path)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    registration_hash = sha256(args.registration)
    if (registration.get("schema_id") != "azuki.ablation_registration"
            or registration.get("family") != "strategy_discovery_v1"
            or registration.get("production_qualified") is not False
            or registration.get("status") != "registered"):
        raise ValueError("Only the registered, non-production strategy discovery campaign is authorized")
    verify_registration_files(registration)
    selected = set(args.arm)
    arms = [arm for arm in registration["arms"] if not selected or arm["name"] in selected]
    if selected and selected != {arm["name"] for arm in arms}:
        raise ValueError("Unknown requested arm")
    for arm in arms:
        verify_files({arm["config"]: arm["config_sha256"],
                      arm["evaluation_config"]: arm["evaluation_config_sha256"]})
        if not 0 < arm["total_timesteps"] <= MAX_ARM_ROWS:
            raise ValueError("Diagnostic row limit exceeded")
    print(f"[discovery] registration verified: {len(arms)} arms; production disabled", flush=True)
    if args.validate_only:
        return
    campaign_path = args.registration.parent / "campaign_status.json"
    campaign = {"schema_id": "azuki.strategy_discovery_status", "schema_version": 1,
                "registration_sha256": registration_hash, "production_qualified": False,
                "state": "running", "started_at": time.time(), "arms": []}
    _write_json(campaign_path, campaign)
    try:
        if args.deck_diagnostic is not None:
            campaign["phase"] = "same_policy_deck_diagnostic"
            _write_json(campaign_path, campaign)
            subprocess.run(
                [str(PYTHON), "train-ablation-1781126582/run_deck_diagnostic.py",
                 "--registration", str(args.deck_diagnostic), "--workers", "6"],
                cwd=ROOT, check=True,
            )
            diagnostic = json.loads(
                (args.deck_diagnostic.parent / "decision_packet.json").read_text()
            )
            if diagnostic["incomplete_games"] != 0:
                raise RuntimeError("Deck diagnostic has incomplete games; campaign held for review")
            campaign["deck_diagnostic"] = {
                "registration": str(args.deck_diagnostic),
                "registration_sha256": sha256(args.deck_diagnostic),
                "result": str(args.deck_diagnostic.parent / "decision_packet.json"),
            }
            campaign["phase"] = "fresh_matched_training"
            _write_json(campaign_path, campaign)
        for arm in arms:
            verify_registration_files(registration)
            status = run_training(arm, registration, registration_hash)
            experiment_dir = ROOT / status["latest_experiment_dir"]
            windows = []
            for update in arm["evaluation_updates"]:
                checkpoint, checkpoint_hash = _checkpoint_from_manifest(experiment_dir, update)
                print(f"[discovery] evaluating {arm['id']} p{update:06d}", flush=True)
                windows.append(_evaluate_window(
                    arm=arm, config=ROOT / arm["evaluation_config"], checkpoint=checkpoint,
                    checkpoint_hash=checkpoint_hash, update=update, games=200,
                    last_update=arm["evaluation_updates"][-1], device="cuda"))
            index = {"schema_id": "azuki.ablation_evaluation_index", "schema_version": 1,
                     "registration": str(args.registration), "registration_sha256": registration_hash,
                     "production_qualified": False, "arms": [{"id": arm["id"], "name": arm["name"],
                     "training": status["training"], "windows": windows}]}
            output = ROOT / arm["result_root"] / "evaluation_index.json"
            _write_json(output, index)
            campaign["arms"].append({"id": arm["id"], "state": "evaluated", "index": str(output.relative_to(ROOT))})
            _write_json(campaign_path, campaign)
        campaign.update(state="completed_observations_require_review", finished_at=time.time())
    except BaseException as exc:
        campaign.update(state="stopped", error=str(exc), finished_at=time.time())
        _write_json(campaign_path, campaign)
        raise
    finally:
        _write_json(campaign_path, campaign)
    print("[discovery] all registered observations complete; no production promotion", flush=True)


if __name__ == "__main__":
    main()
