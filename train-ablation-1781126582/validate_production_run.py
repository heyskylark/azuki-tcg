#!/usr/bin/env python3
"""Validate pre-distributed production training gates from compact run artifacts."""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
from pathlib import Path
import statistics


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _parse_phase(raw: str) -> tuple[int, int, Path]:
  parts = raw.split(":", 2)
  if len(parts) != 3:
    raise argparse.ArgumentTypeError("phase must be START:END:LOG_PATH")
  start, end = int(parts[0]), int(parts[1])
  if start < 0 or end <= start:
    raise argparse.ArgumentTypeError("phase update range must be increasing")
  return start, end, Path(parts[2])


def _read_metrics(path: Path) -> tuple[dict, list[dict], dict]:
  rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
  starts = [row for row in rows if row.get("_event") == "start"]
  closes = [row for row in rows if row.get("_event") == "close"]
  metrics = [row for row in rows if "_window_updates" in row]
  if len(starts) != 1 or len(closes) != 1 or not metrics:
    raise ValueError(f"incomplete compact log lifecycle: {path}")
  return starts[0], metrics, closes[0]


def _phase_report(start: int, end: int, path: Path) -> dict:
  start_row, metrics, close_row = _read_metrics(path)
  observed_updates = sum(int(row["_window_updates"]) for row in metrics)
  if observed_updates != end - start:
    raise ValueError(
      f"update accounting mismatch for {path}: expected={end-start} observed={observed_updates}"
    )
  final_epoch = int(metrics[-1].get("epoch", -1))
  if final_epoch != end:
    raise ValueError(f"final epoch mismatch for {path}: expected={end} observed={final_epoch}")

  numeric_values = []
  for row in metrics:
    for value in row.values():
      if isinstance(value, bool) or not isinstance(value, (int, float)):
        continue
      numeric_values.append(float(value))
  if any(not math.isfinite(value) for value in numeric_values):
    raise ValueError(f"non-finite metric in {path}")

  health = {}
  for row in metrics:
    for key, value in row.items():
      lowered = key.lower()
      if not isinstance(value, (int, float)):
        continue
      if any(token in lowered for token in ("timeout", "truncat", "invalid", "incomplete")):
        health[key] = max(float(value), health.get(key, 0.0))
  failed_health = {key: value for key, value in health.items() if value != 0.0}
  if failed_health:
    raise ValueError(f"integrity metrics are nonzero in {path}: {failed_health}")

  sps = [float(row["SPS"]) for row in metrics if isinstance(row.get("SPS"), (int, float))]
  if not sps:
    raise ValueError(f"no SPS metrics in {path}")
  return {
    "path": str(path),
    "start_update": start,
    "end_update": end,
    "updates": observed_updates,
    "metric_windows": len(metrics),
    "median_sps": statistics.median(sps),
    "integrity_maxima": health,
    "run_id": start_row.get("run_id"),
    "model_path": close_row.get("model_path"),
  }


def _verify_checkpoint_manifest(path: Path, config_path: Path) -> dict:
  payload = json.loads(path.read_text(encoding="utf-8"))
  for raw in payload.get("artifacts", []):
    artifact = path.parent / raw["path"]
    if not artifact.is_file():
      raise FileNotFoundError(artifact)
    actual = _sha256(artifact)
    if actual != raw["sha256"]:
      raise ValueError(f"checkpoint hash mismatch: {artifact}")
  config = payload.get("config")
  if not isinstance(config, dict):
    raise ValueError(f"checkpoint manifest has no config: {path}")
  if _sha256(config_path) != config["sha256"]:
    raise ValueError(f"checkpoint config hash mismatch: {config_path}")
  return {
    "path": str(path),
    "update": int(payload["update"]),
    "roles": list(payload.get("roles", [])),
    "artifact_count": len(payload.get("artifacts", [])),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--phase", type=_parse_phase, action="append", required=True)
  parser.add_argument("--checkpoint-manifest", type=Path, action="append", required=True)
  parser.add_argument("--baseline-sps", type=float, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  config = configparser.ConfigParser()
  if not config.read(args.config):
    raise FileNotFoundError(args.config)
  gates = config["launch_gates"]
  absolute_floor = gates.getfloat("minimum_median_sps")
  relative_floor = gates.getfloat("minimum_relative_sps")

  phases = [_phase_report(*phase) for phase in args.phase]
  observed_sps = min(phase["median_sps"] for phase in phases)
  required_sps = max(absolute_floor, relative_floor * args.baseline_sps)
  if observed_sps < required_sps:
    raise ValueError(
      f"throughput gate failed: slowest_phase={observed_sps:.3f} required={required_sps:.3f}"
    )

  checkpoints = [
    _verify_checkpoint_manifest(path, args.config)
    for path in args.checkpoint_manifest
  ]
  checkpoint_updates = {checkpoint["update"] for checkpoint in checkpoints}
  expected_ends = {phase["end_update"] for phase in phases}
  if not expected_ends.issubset(checkpoint_updates):
    raise ValueError(
      f"missing phase-end atomic checkpoints: {sorted(expected_ends - checkpoint_updates)}"
    )

  payload = {
    "schema_version": 1,
    "decision": "pass",
    "baseline_sps": args.baseline_sps,
    "required_sps": required_sps,
    "observed_minimum_phase_median_sps": observed_sps,
    "phases": phases,
    "checkpoints": checkpoints,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
  main()
