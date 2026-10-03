#!/usr/bin/env python3
"""Run the matched p11030 control/ref05 regional-deck continuation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
CONFIG = REPO_ROOT / "python" / "config" / "azuki_deckbuild_production_3090.ini"
PARENT_DIR = (
  REPO_ROOT
  / "experiments"
  / "azuki_local_production_soak_20260817T232829Z_phase2_retry_178701029287"
)
PARENT_MODEL = PARENT_DIR / "model_azuki_local_011030.pt"
PARENT_TRAINER = PARENT_DIR / "trainer_state_011030.pt"
PARENT_LEAGUE = PARENT_DIR / "league_state_011030.json"
PARENT_PROMOTION = PARENT_DIR / "promotion_state_011030.json"
SOURCE_CORPUS = REPO_ROOT / ".codex" / "docs" / "azuki_garden_arena_2026-08-15_decks.json"
TRAINING_CORPUS = (
  REPO_ROOT / ".codex" / "docs" / "azuki_garden_arena_2026-08-15_training_decks.json"
)
SPLIT_MANIFEST = (
  REPO_ROOT / ".codex" / "docs" / "azuki_garden_arena_2026-08-15_reference_split.json"
)
CAMPAIGN = "regional_reference_ladder15_v1"
RESULT_ROOT = REPO_ROOT / "train-ablation-1781126582" / "results" / CAMPAIGN
LEAGUE_ROOT = REPO_ROOT / "experiments" / "league" / CAMPAIGN
PARENT_UPDATE = 11030
TARGET_UPDATE = 12000
FULL_HORIZON_UPDATE = 13960
SAMPLES_PER_UPDATE = 15360
FULL_TOTAL_TIMESTEPS = FULL_HORIZON_UPDATE * SAMPLES_PER_UPDATE
ARMS = (("control", 0.0), ("ref05", 0.05))
PARENT_HASHES = {
  "model_azuki_local_011030.pt": "33438e05e7f8675b965833108405a56a3f949ce9396fd455853cb79693eb99d8",
  "model_azuki_local_011030.pt.meta.json": "daa988ab8d6f97108b6184fef7fc83f25bcb85be19bf3e4d5a17912930580fef",
  "trainer_state_011030.pt": "a1aa9f3b38a647e64aea38b0b237d8f3878adc877ea1d02fb8172b8e9ee13e94",
  "league_state_011030.json": "f6f3f144a4b8ee89d3d2218d3852c2b803a476a1a4eeed3c4cc9bbf14a8bb893",
  "promotion_state_011030.json": "ab261e00ece30b7656363988b99ec0082422e0a40451c1b7fedd0b2c8d9629db",
}
INPUT_HASHES = {
  SOURCE_CORPUS: "e97eaf4cde67d689790c541175c61a8e8d18c12e0952b80b30d0a860106be393",
  TRAINING_CORPUS: "74d311650f8f08bd7f862522e4e45b9060f2b5d5d06ace8c7b6ad2035b69112b",
  SPLIT_MANIFEST: "3b75aa457a9982a9a2f4e3a2345ee9dc743302a0c6d295c0e84f041b30d6f728",
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _relative(path: Path) -> str:
  return str(path.resolve().relative_to(REPO_ROOT))


def _write_json(path: Path, payload: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  temporary.replace(path)


def _atomic_copy(source: Path, destination: Path) -> None:
  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_suffix(destination.suffix + ".tmp")
  shutil.copy2(source, temporary)
  temporary.replace(destination)


def _verify_inputs() -> dict[str, Any]:
  for name, expected in PARENT_HASHES.items():
    path = PARENT_DIR / name
    actual = _sha256(path)
    if actual != expected:
      raise ValueError(f"Parent hash mismatch: {path}: expected={expected} actual={actual}")
  for path, expected in INPUT_HASHES.items():
    actual = _sha256(path)
    if actual != expected:
      raise ValueError(f"Input hash mismatch: {path}: expected={expected} actual={actual}")
  split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
  training = split["training"]
  holdout = split["holdout"]
  if training["rows"] != 222 or holdout["rows"] != 15:
    raise ValueError("Regional split cardinality changed")
  if set(training["content_sha256"]) & set(holdout["content_sha256"]):
    raise ValueError("Regional training/holdout signatures overlap")
  return split


def _write_parent_manifest() -> Path:
  path = RESULT_ROOT / "parent_manifest.json"
  payload = {
    "schema_version": 2,
    "purpose": "trusted p11030 source-drift parent for regional corpus migration",
    "model": {
      "path": _relative(PARENT_MODEL),
      "sha256": PARENT_HASHES[PARENT_MODEL.name],
    },
    "metadata": {
      "path": _relative(Path(f"{PARENT_MODEL}.meta.json")),
      "sha256": PARENT_HASHES[f"{PARENT_MODEL.name}.meta.json"],
    },
    "trainer": {
      "path": _relative(PARENT_TRAINER),
      "sha256": PARENT_HASHES[PARENT_TRAINER.name],
    },
    "league": {
      "path": _relative(PARENT_LEAGUE),
      "sha256": PARENT_HASHES[PARENT_LEAGUE.name],
    },
    "promotion": {
      "path": _relative(PARENT_PROMOTION),
      "sha256": PARENT_HASHES[PARENT_PROMOTION.name],
    },
  }
  if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
    raise ValueError(f"Existing parent manifest differs: {path}")
  _write_json(path, payload)
  return path


def _manifest_update(path: Path) -> int:
  payload = json.loads(path.read_text(encoding="utf-8"))
  update = payload.get("update")
  if not isinstance(update, int):
    raise ValueError(f"Checkpoint manifest has no update: {path}")
  return update


def _verify_checkpoint_manifest(path: Path, expected_update: int | None = None) -> dict:
  payload = json.loads(path.read_text(encoding="utf-8"))
  update = payload.get("update")
  if not isinstance(update, int):
    raise ValueError(f"Checkpoint manifest has no update: {path}")
  if expected_update is not None and update != expected_update:
    raise ValueError(f"Checkpoint update mismatch: expected={expected_update} actual={update}")
  artifacts = payload.get("artifacts")
  if not isinstance(artifacts, list) or len(artifacts) < 3:
    raise ValueError(f"Incomplete checkpoint manifest: {path}")
  for entry in artifacts:
    if not isinstance(entry, dict):
      raise ValueError(f"Invalid checkpoint artifact entry: {path}")
    artifact = path.parent / str(entry.get("path", ""))
    expected = entry.get("sha256")
    if not artifact.is_file() or not isinstance(expected, str) or _sha256(artifact) != expected:
      raise ValueError(f"Checkpoint artifact hash mismatch: {artifact}")
  return payload


def _run_dirs(arm: str) -> list[Path]:
  return sorted(
    (REPO_ROOT / "experiments").glob(f"azuki_local_regional_ref15v1_{arm}_*"),
    key=lambda path: path.stat().st_mtime_ns,
  )


def _latest_arm_checkpoint(arm: str) -> tuple[int, Path, Path | None]:
  candidates: list[tuple[int, Path, Path]] = []
  for run_dir in _run_dirs(arm):
    for manifest in run_dir.glob("checkpoint_*.manifest.json"):
      update = _manifest_update(manifest)
      if PARENT_UPDATE < update <= TARGET_UPDATE:
        candidates.append((update, manifest, run_dir))
  if not candidates:
    return PARENT_UPDATE, PARENT_MODEL, None
  update, manifest, run_dir = max(candidates, key=lambda item: item[0])
  payload = _verify_checkpoint_manifest(manifest, update)
  model_name = next(
    str(entry["path"])
    for entry in payload["artifacts"]
    if str(entry.get("path", "")).startswith("model_")
    and str(entry.get("path", "")).endswith(".pt")
  )
  return update, run_dir / model_name, run_dir


def _initialize_arm_state(arm: str) -> tuple[Path, Path, Path, Path]:
  league_dir = LEAGUE_ROOT / arm
  league_path = league_dir / "league_state.json"
  promotion_path = league_dir / "league_state_promotion.json"
  opponent_dir = league_dir / "opponents"
  snapshots = RESULT_ROOT / arm / "snapshots"
  existing = [league_path.exists(), promotion_path.exists()]
  if any(existing) and not all(existing):
    raise ValueError(f"Partial arm state exists: {league_dir}")
  if not all(existing):
    _atomic_copy(PARENT_LEAGUE, league_path)
    _atomic_copy(PARENT_PROMOTION, promotion_path)
  opponent_dir.mkdir(parents=True, exist_ok=True)
  snapshots.mkdir(parents=True, exist_ok=True)
  return league_path, promotion_path, opponent_dir, snapshots


def _training_command(
  arm: str,
  probability: float,
  resume_checkpoint: Path,
  resume_update: int,
  parent_manifest: Path,
  league_path: Path,
  promotion_path: Path,
  opponent_dir: Path,
  snapshots: Path,
) -> list[str]:
  tag = f"regional_ref15v1_{arm}_from_p{resume_update:05d}"
  command = [
    str(PYTHON),
    "python/src/train.py",
    "--config",
    _relative(CONFIG),
    "--resume-checkpoint",
    _relative(resume_checkpoint),
    "--resume-load-optimizer",
    "--resume-strict",
    "--no-resume-auto-reset-critic",
    "--resume-allow-deck-pool-migration",
    "--jsonl-log",
    _relative(RESULT_ROOT / arm / "runlogs"),
    "--tag",
    tag,
    "--parent_manifest",
    _relative(parent_manifest),
    "--env.deck_pool_path",
    _relative(TRAINING_CORPUS),
    "--env.deck_snapshot_every",
    "25",
    "--env.deck_snapshot_dir",
    _relative(snapshots),
    "--train.seed",
    "42",
    "--train.total_timesteps",
    str(FULL_TOTAL_TIMESTEPS),
    "--train.checkpoint_interval",
    str(TARGET_UPDATE),
    "--league.state_path",
    _relative(league_path),
    "--league.promotion_state_path",
    _relative(promotion_path),
    "--league.opponent_dir",
    _relative(opponent_dir),
    "--league.eval_interval",
    "100000",
    "--league.quick_eval_interval",
    "100000",
    "--league.full_eval_interval",
    "100000",
    "--league.promotion_panel_refresh_epochs",
    "100000",
    "--league.promotion_shadow_mode",
    "true",
    "--league.promotion_archive_affects_training_pool",
    "false",
    "--process_env.azk_draft_ref_seat_prob",
    str(probability),
  ]
  if resume_update == PARENT_UPDATE:
    command.append("--resume-restart-lr-schedule")
  else:
    command.extend(("--resume.restart_lr_schedule", "false"))
  return command


def _training_environment() -> dict[str, str]:
  env = dict(os.environ)
  env.update(
    {
      "AZK_DRAFT_REF_OPPONENT_ONLY": "1",
      "AZK_RESUME_ALLOW_BINDING_MISMATCH": "1",
      "PYTHONPATH": "build/python/src:python/src",
      "PYTHONUNBUFFERED": "1",
      "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
  )
  env.pop("AZK_DRAFT_REF_DECK_INDICES", None)
  env.pop("AZK_RESUME_ALLOW_SOURCE_DRIFT", None)
  env.pop("AZK_RESUME_ALLOW_SCHEDULE_REWIND", None)
  return env


def _find_target(arm: str) -> tuple[Path, Path] | None:
  for run_dir in reversed(_run_dirs(arm)):
    manifest = run_dir / f"checkpoint_{TARGET_UPDATE:06d}.manifest.json"
    if manifest.is_file():
      return run_dir, manifest
  return None


def _run_training_arm(arm: str, probability: float, parent_manifest: Path) -> Path:
  arm_result = RESULT_ROOT / arm
  arm_result.mkdir(parents=True, exist_ok=True)
  league_path, promotion_path, opponent_dir, snapshots = _initialize_arm_state(arm)
  target = _find_target(arm)
  if target is None:
    resume_update, resume_checkpoint, _ = _latest_arm_checkpoint(arm)
    command = _training_command(
      arm,
      probability,
      resume_checkpoint,
      resume_update,
      parent_manifest,
      league_path,
      promotion_path,
      opponent_dir,
      snapshots,
    )
    segment_dir = arm_result / "segments" / f"from_{resume_update:06d}"
    if segment_dir.exists():
      suffix = int(time.time())
      segment_dir = arm_result / "segments" / f"from_{resume_update:06d}_{suffix}"
    segment_dir.mkdir(parents=True)
    _write_json(
      segment_dir / "command.json",
      {
        "arm": arm,
        "reference_probability": probability,
        "resume_update": resume_update,
        "target_update": TARGET_UPDATE,
        "full_lr_horizon_update": FULL_HORIZON_UPDATE,
        "command": command,
      },
    )
    print(f"[{CAMPAIGN}] starting {arm} from p{resume_update}", flush=True)
    with (segment_dir / "train.log").open("wb") as log:
      process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=_training_environment(),
        stdout=log,
        stderr=subprocess.STDOUT,
      )
      last_notice = 0.0
      while process.poll() is None:
        target = _find_target(arm)
        if target is not None:
          process.send_signal(signal.SIGINT)
          break
        now = time.monotonic()
        if now - last_notice >= 60:
          print(f"[{CAMPAIGN}] {arm} training pid={process.pid}", flush=True)
          last_notice = now
        time.sleep(5)
      try:
        process.wait(timeout=300)
      except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=60)
    target = _find_target(arm)
    if target is None:
      raise RuntimeError(
        f"{arm} exited before p{TARGET_UPDATE}; rerun this driver to resume the latest atomic recovery"
      )

  run_dir, manifest = target
  payload = _verify_checkpoint_manifest(manifest, TARGET_UPDATE)
  metadata = json.loads(
    (run_dir / f"model_azuki_local_{TARGET_UPDATE:06d}.pt.meta.json").read_text(
      encoding="utf-8"
    )
  )
  if metadata.get("update") != TARGET_UPDATE:
    raise ValueError(f"Target metadata update mismatch: {run_dir}")
  expected_pool = str(TRAINING_CORPUS.resolve())
  actual_pool = metadata.get("resume_config_fingerprint", {}).get("deck_pool_path")
  if actual_pool != expected_pool:
    raise ValueError(f"Target deck pool mismatch: expected={expected_pool} actual={actual_pool}")
  checkpoint = run_dir / f"model_azuki_local_{TARGET_UPDATE:06d}.pt"
  trainer = run_dir / f"trainer_state_{TARGET_UPDATE:06d}.pt"
  _write_json(
    arm_result / "target.json",
    {
      "update": TARGET_UPDATE,
      "sampled_rows": (TARGET_UPDATE - PARENT_UPDATE) * SAMPLES_PER_UPDATE,
      "run_dir": _relative(run_dir),
      "checkpoint": _relative(checkpoint),
      "checkpoint_sha256": _sha256(checkpoint),
      "trainer": _relative(trainer),
      "trainer_sha256": _sha256(trainer),
      "manifest": _relative(manifest),
      "manifest_config_sha256": payload.get("config_sha256"),
    },
  )
  _combine_training_logs(arm, arm_result)
  (arm_result / "TRAIN_DONE").touch()
  return checkpoint


def _combine_training_logs(arm: str, arm_result: Path) -> None:
  rows: list[dict] = []
  for path in sorted((arm_result / "runlogs").glob("*.jsonl"), key=lambda p: p.stat().st_mtime_ns):
    with path.open(encoding="utf-8") as handle:
      for line in handle:
        try:
          row = json.loads(line)
        except json.JSONDecodeError:
          continue
        if isinstance(row, dict):
          rows.append(row)
  rows = [
    row
    for row in rows
    if not isinstance(row.get("epoch"), (int, float))
    or float(row["epoch"]) <= TARGET_UPDATE
  ]
  metric_rows = [row for row in rows if isinstance(row.get("SPS"), (int, float))]
  if not metric_rows:
    raise ValueError(f"No training metric rows found for {arm}")
  for row in metric_rows:
    for key, value in row.items():
      if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
          raise ValueError(f"Non-finite metric for {arm}: {key}={value}")
        lowered = key.lower()
        if any(token in lowered for token in ("timeout", "truncat", "invalid", "incomplete")):
          if float(value) != 0.0:
            raise ValueError(f"Integrity metric is nonzero for {arm}: {key}={value}")
  with (arm_result / "train.jsonl").open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _evaluation_checkpoint(checkpoint: Path, arm_result: Path) -> Path:
  link = arm_result / "evaluation_checkpoint.pt"
  if link.is_symlink() or link.exists():
    link.unlink()
  link.symlink_to(checkpoint.resolve())
  metadata = json.loads(Path(f"{checkpoint}.meta.json").read_text(encoding="utf-8"))
  fingerprint = metadata.get("resume_config_fingerprint")
  if not isinstance(fingerprint, dict):
    raise ValueError(f"Checkpoint metadata has no resume fingerprint: {checkpoint}")
  fingerprint["deck_pool_path"] = str(SOURCE_CORPUS.resolve())
  _write_json(Path(f"{link}.meta.json"), metadata)
  return link


def _run_logged(command: list[str], log_path: Path) -> None:
  print(f"[{CAMPAIGN}] running {log_path.stem}", flush=True)
  with log_path.open("wb") as log:
    subprocess.run(
      command,
      cwd=REPO_ROOT,
      env=_training_environment(),
      stdout=log,
      stderr=subprocess.STDOUT,
      check=True,
    )


def _require_eval(path: Path, expected_games: int) -> None:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if payload.get("summary", {}).get("episodes") != expected_games:
    raise ValueError(f"Evaluation episode count mismatch: {path}")
  games = payload.get("games")
  if not isinstance(games, list) or len(games) != expected_games:
    raise ValueError(f"Evaluation game count mismatch: {path}")
  if float(payload["summary"].get("timeout_rate", 1.0)) != 0.0:
    raise ValueError(f"Evaluation timeout rate is nonzero: {path}")


def _evaluate_arm(arm: str, checkpoint: Path, split: dict[str, Any]) -> None:
  arm_result = RESULT_ROOT / arm
  evaluation_checkpoint = _evaluation_checkpoint(checkpoint, arm_result)
  training_indices = [int(value) for value in split["training"]["source_indices"]]
  holdout_indices = [int(value) for value in split["holdout"]["source_indices"]]
  training_csv = ",".join(map(str, training_indices))
  holdout_csv = ",".join(map(str, holdout_indices))
  for name, indices, index_csv in (
    ("train_reference", training_indices, training_csv),
    ("holdout_reference", holdout_indices, holdout_csv),
  ):
    output = arm_result / f"{name}.json"
    expected_games = len(indices) * 32
    if not output.exists():
      command = [
        str(PYTHON),
        "python/src/native_reference_eval.py",
        "--config",
        _relative(CONFIG),
        "--checkpoint",
        str(evaluation_checkpoint.relative_to(REPO_ROOT)),
        "--opponent-checkpoint",
        _relative(PARENT_MODEL),
        "--candidate-label",
        arm,
        "--opponent-label",
        "p11030_parent",
        "--split",
        name,
        "--deck-indices",
        index_csv,
        "--training-reference-indices",
        training_csv,
        "--holdout-reference-indices",
        holdout_csv,
        "--seeds",
        "42009919,52009922",
        "--batch-envs",
        "12",
        "--max-steps",
        "1200",
        "--device",
        "cuda",
        "--uniform-assignment",
        "--json",
        _relative(output),
      ]
      _run_logged(command, arm_result / f"{name}.log")
    _require_eval(output, expected_games)

  h2h_output = arm_result / "h2h_vs_parent.json"
  if not h2h_output.exists():
    command = [
      str(PYTHON),
      "python/src/native_policy_eval.py",
      "--config",
      _relative(CONFIG),
      "--checkpoint-a",
      str(evaluation_checkpoint.relative_to(REPO_ROOT)),
      "--checkpoint-b",
      _relative(PARENT_MODEL),
      "--label-a",
      arm,
      "--label-b",
      "p11030_parent",
      "--batch-envs",
      "12",
      "--seeds",
      "42001701,52001704,62001707,72001710,82001713,92001716",
      "--max-steps",
      "1200",
      "--device",
      "cuda",
      "--json",
      _relative(h2h_output),
    ]
    _run_logged(command, arm_result / "h2h_vs_parent.log")
  _require_eval(h2h_output, 192)
  (arm_result / "ARM_DONE").touch()


def main() -> None:
  os.chdir(REPO_ROOT)
  split = _verify_inputs()
  RESULT_ROOT.mkdir(parents=True, exist_ok=True)
  parent_manifest = _write_parent_manifest()
  _write_json(
    RESULT_ROOT / "protocol.json",
    {
      "schema_version": 1,
      "parent_update": PARENT_UPDATE,
      "target_update": TARGET_UPDATE,
      "full_lr_horizon_update": FULL_HORIZON_UPDATE,
      "updates_per_arm": TARGET_UPDATE - PARENT_UPDATE,
      "samples_per_update": SAMPLES_PER_UPDATE,
      "sampled_rows_per_arm": (TARGET_UPDATE - PARENT_UPDATE) * SAMPLES_PER_UPDATE,
      "arms": {arm: probability for arm, probability in ARMS},
      "sole_behavioral_arm_difference": "AZK_DRAFT_REF_SEAT_PROB",
      "sps_is_rejection_criterion": False,
      "source_corpus": _relative(SOURCE_CORPUS),
      "training_corpus": _relative(TRAINING_CORPUS),
      "split_manifest": _relative(SPLIT_MANIFEST),
    },
  )
  for arm, probability in ARMS:
    checkpoint = _run_training_arm(arm, probability, parent_manifest)
    _evaluate_arm(arm, checkpoint, split)
  subprocess.run(
    [
      str(PYTHON),
      "train-ablation-1781126582/regional_reference_ladder_report.py",
      _relative(RESULT_ROOT),
    ],
    cwd=REPO_ROOT,
    env=_training_environment(),
    check=True,
  )
  (RESULT_ROOT / "LADDER_DONE").touch()
  print(f"[{CAMPAIGN}] complete: {RESULT_ROOT}", flush=True)


if __name__ == "__main__":
  main()
