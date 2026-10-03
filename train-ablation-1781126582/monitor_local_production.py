#!/usr/bin/env python3
"""Passively monitor the local production run without loading a model or using CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

SAMPLES_PER_UPDATE = 15_360
MINIMUM_MEDIAN_SPS = 1_235.0
INTEGRITY_TOKENS = ("timeout", "truncat", "invalid", "incomplete")
MILESTONE_SAMPLES = (
  50_000_000,
  100_000_000,
  200_000_000,
  300_000_000,
  450_000_000,
  800_000_000,
  1_000_000_000,
)


class MonitorFailure(RuntimeError):
  """A production health invariant failed."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  os.replace(temporary, path)


def _read_rows(path: Path) -> list[dict[str, Any]]:
  if not path.is_file():
    raise MonitorFailure(f"compact log is missing: {path}")
  rows: list[dict[str, Any]] = []
  lines = path.read_text(encoding="utf-8").splitlines()
  for index, line in enumerate(lines):
    if not line.strip():
      continue
    try:
      payload = json.loads(line)
    except json.JSONDecodeError:
      if index == len(lines) - 1:
        continue
      raise MonitorFailure(f"invalid JSON at {path}:{index + 1}")
    if not isinstance(payload, dict):
      raise MonitorFailure(f"non-object JSON at {path}:{index + 1}")
    rows.append(payload)
  return rows


def _metric_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  starts = [row for row in rows if row.get("_event") == "start"]
  if len(starts) != 1:
    raise MonitorFailure(f"expected exactly one run start, found {len(starts)}")
  metrics = [row for row in rows if "_window_updates" in row]
  if not metrics:
    raise MonitorFailure("compact log has no metric windows")
  return starts[0], metrics


def _number(value: object, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise MonitorFailure(f"{name} is missing or non-numeric: {value!r}")
  result = float(value)
  if not math.isfinite(result):
    raise MonitorFailure(f"{name} is non-finite: {result}")
  return result


def _expected_lr(epoch: int, *, start_update: int, end_update: int, peak_lr: float) -> float:
  completed = epoch - start_update
  remaining = end_update - start_update
  if completed <= 0:
    return peak_lr
  if completed >= remaining:
    return 0.0
  return peak_lr * (1.0 + math.cos(math.pi * completed / remaining)) / 2.0


def _rolling_values(metrics: list[dict[str, Any]], key: str, count: int) -> list[float]:
  values = []
  for row in metrics[-count:]:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      continue
    numeric = float(value)
    if math.isfinite(numeric):
      values.append(numeric)
  return values


def _trainer_command(pid: int) -> str:
  path = Path(f"/proc/{pid}/cmdline")
  if not path.is_file():
    return ""
  return path.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def _stop_verified_trainer(pid: int | None, run_token: str | None, reason: str) -> bool:
  if pid is None or not run_token:
    return False
  command = _trainer_command(pid)
  if not command or run_token not in command or "python/src/train.py" not in command:
    return False
  process_group = os.getpgid(pid)
  monitor_group = os.getpgrp()
  if process_group == monitor_group:
    print(
      f"CRITICAL_STOP_REFUSED shared_process_group={process_group} reason={reason}",
      flush=True,
    )
    return False
  os.killpg(process_group, signal.SIGTERM)
  print(f"CRITICAL_STOP pid={pid} process_group={process_group} reason={reason}", flush=True)
  return True


def _health_snapshot(
  log_path: Path,
  args: argparse.Namespace,
  *,
  enforce_health_policy: bool = True,
) -> dict[str, Any]:
  start, metrics = _metric_rows(_read_rows(log_path))
  config = start.get("config")
  if not isinstance(config, dict):
    raise MonitorFailure("run start has no config snapshot")
  if config.get("train.seed_process_rngs") is not True:
    raise MonitorFailure("train.seed_process_rngs is not true")
  expected_total = args.end_update * SAMPLES_PER_UPDATE
  if int(config.get("train.total_timesteps", -1)) != expected_total:
    raise MonitorFailure(
      f"configured training horizon mismatch: actual={config.get('train.total_timesteps')} "
      f"expected={expected_total}"
    )

  previous_epoch = args.start_update
  previous_step = -1.0
  integrity_maxima: dict[str, float] = {}
  for row in metrics:
    epoch = int(_number(row.get("epoch"), "epoch"))
    step = _number(row.get("agent_steps"), "agent_steps")
    if epoch <= previous_epoch:
      raise MonitorFailure(f"epoch is not strictly increasing: {previous_epoch} -> {epoch}")
    if step <= previous_step:
      raise MonitorFailure(f"agent_steps is not strictly increasing: {previous_step} -> {step}")
    previous_epoch = epoch
    previous_step = step

    actual_lr = _number(row.get("learning_rate"), f"learning_rate@{epoch}")
    expected_lr = _expected_lr(
      epoch,
      start_update=args.start_update,
      end_update=args.end_update,
      peak_lr=args.peak_lr,
    )
    tolerance = max(1e-12, args.peak_lr * 2e-6)
    if abs(actual_lr - expected_lr) > tolerance:
      raise MonitorFailure(
        f"LR schedule mismatch at epoch {epoch}: actual={actual_lr:.12g} expected={expected_lr:.12g}"
      )

    for key, value in row.items():
      if isinstance(value, bool) or not isinstance(value, (int, float)):
        continue
      numeric = float(value)
      if not math.isfinite(numeric):
        raise MonitorFailure(f"non-finite metric at epoch {epoch}: {key}={numeric}")
      if any(token in key.lower() for token in INTEGRITY_TOKENS):
        integrity_maxima[key] = max(integrity_maxima.get(key, 0.0), numeric)

  nonzero_integrity = {key: value for key, value in integrity_maxima.items() if value != 0.0}
  if nonzero_integrity:
    raise MonitorFailure(f"integrity metrics are nonzero: {nonzero_integrity}")

  last = metrics[-1]
  last_epoch = int(_number(last.get("epoch"), "last epoch"))
  latest_ts = _number(last.get("ts"), "latest metric timestamp")
  age = time.time() - latest_ts
  if enforce_health_policy and age > args.stale_seconds:
    raise MonitorFailure(
      f"metrics are stale: age={age:.1f}s threshold={args.stale_seconds:.1f}s"
    )

  pool_values = _rolling_values(metrics, "environment/league/pool_size_active", 10)
  pool_median = statistics.median(pool_values) if pool_values else None
  if enforce_health_policy and args.expected_pool_size >= 0 and pool_median is not None:
    if pool_median != args.expected_pool_size:
      raise MonitorFailure(f"active opponent pool changed: median={pool_median}")

  trainable = _rolling_values(metrics, "environment/league/trainable_row_fraction", 10)
  if trainable:
    trainable_median = statistics.median(trainable)
    if (
      enforce_health_policy
      and args.expected_trainable_fraction >= 0
      and abs(trainable_median - args.expected_trainable_fraction) > 0.05
    ):
      raise MonitorFailure(f"trainable row fraction changed: median={trainable_median:.6f}")
  else:
    trainable_median = None

  sps20 = _rolling_values(metrics, "SPS", 20)
  sps_median = statistics.median(sps20) if sps20 else None
  if (
    enforce_health_policy
    and len(sps20) >= 20
    and sps_median is not None
    and sps_median < args.minimum_median_sps
  ):
    raise MonitorFailure(f"20-window median SPS below floor: {sps_median:.3f}")

  entropy = _rolling_values(metrics, "losses/entropy", 10)
  entropy_baseline = _rolling_values(metrics[:10], "losses/entropy", 10)
  entropy_median = statistics.median(entropy) if entropy else None
  baseline_entropy = statistics.median(entropy_baseline) if entropy_baseline else None
  if enforce_health_policy and entropy_median is not None and baseline_entropy is not None:
    collapse_floor = max(0.002, baseline_entropy * 0.02)
    if entropy_median < collapse_floor:
      raise MonitorFailure(
        f"entropy collapse: recent={entropy_median:.6g} baseline={baseline_entropy:.6g} floor={collapse_floor:.6g}"
      )

  approx_kl = _rolling_values(metrics, "losses/approx_kl", 10)
  kl_median = statistics.median(approx_kl) if approx_kl else None
  explained_variance = _rolling_values(metrics, "losses/explained_variance", 10)
  explained_variance_median = (
    statistics.median(explained_variance) if explained_variance else None
  )
  if enforce_health_policy and kl_median is not None and kl_median > 0.20:
    raise MonitorFailure(f"sustained approximate KL is excessive: median={kl_median:.6g}")

  policy_loss = _rolling_values(metrics, "losses/policy_loss", 10)
  if (
    enforce_health_policy
    and policy_loss
    and statistics.median(abs(value) for value in policy_loss) > 1.0
  ):
    raise MonitorFailure("sustained losses/policy_loss magnitude is excessive")

  value_loss = _rolling_values(metrics, "losses/value_loss", len(metrics))
  value_loss_median = statistics.median(value_loss[-10:]) if value_loss else None
  if (
    enforce_health_policy
    and args.max_value_loss > 0
    and value_loss_median is not None
    and value_loss_median > args.max_value_loss
  ):
    raise MonitorFailure(
      f"sustained losses/value_loss magnitude is excessive: median={value_loss_median:.6g}"
    )
  if enforce_health_policy and args.adaptive_value_loss and len(value_loss) >= 12:
    historical_windows = [
      statistics.median(value_loss[index : index + 5])
      for index in range(0, len(value_loss) - 4)
    ]
    historical_floor = min(historical_windows[:-1], default=historical_windows[-1])
    recent_value_loss = statistics.median(value_loss[-3:])
    adaptive_ceiling = max(
      args.value_loss_growth_floor,
      historical_floor * args.value_loss_growth_factor,
    )
    if recent_value_loss > adaptive_ceiling:
      raise MonitorFailure(
        "adaptive value-loss divergence: "
        f"recent={recent_value_loss:.6g} historical_floor={historical_floor:.6g} "
        f"ceiling={adaptive_ceiling:.6g}"
      )

  added_samples = max(0, last_epoch - args.start_update) * SAMPLES_PER_UPDATE
  expected_current_lr = _expected_lr(
    last_epoch,
    start_update=args.start_update,
    end_update=args.end_update,
    peak_lr=args.peak_lr,
  )
  return {
    "status": "healthy" if enforce_health_policy else "observed",
    "checked_at": time.time(),
    "run_id": start.get("run_id"),
    "start_update": args.start_update,
    "end_update": args.end_update,
    "peak_lr": args.peak_lr,
    "last_epoch": last_epoch,
    "added_sampled_rows": added_samples,
    "metric_windows": len(metrics),
    "latest_metric_age_seconds": age,
    "current_lr": _number(last.get("learning_rate"), "current learning rate"),
    "expected_lr": expected_current_lr,
    "sps_rolling_median": sps_median,
    "trainable_row_fraction_median": trainable_median,
    "entropy_rolling_median": entropy_median,
    "entropy_baseline_median": baseline_entropy,
    "explained_variance_rolling_median": explained_variance_median,
    "pool_size_rolling_median": pool_median,
    "approx_kl_rolling_median": kl_median,
    "value_loss_rolling_median": value_loss_median,
    "integrity_maxima": integrity_maxima,
    "health_policy_enforced": enforce_health_policy,
  }

def _latest_metric_epoch(log_path: Path) -> int:
  _, metrics = _metric_rows(_read_rows(log_path))
  return int(_number(metrics[-1].get("epoch"), "last epoch"))


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _latest_manifest(checkpoint_dir: Path, maximum_update: int) -> Path | None:
  candidates: list[tuple[int, Path]] = []
  for path in checkpoint_dir.glob("checkpoint_*.manifest.json"):
    try:
      update = int(path.name.removeprefix("checkpoint_").removesuffix(".manifest.json"))
    except ValueError:
      continue
    if update <= maximum_update:
      candidates.append((update, path))
  return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _verify_manifest(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  failures = []
  for item in payload.get("artifacts", []):
    artifact = path.parent / str(item["path"])
    if not artifact.is_file():
      failures.append(f"missing:{artifact}")
      continue
    if artifact.stat().st_size != int(item["bytes"]):
      failures.append(f"size:{artifact}")
      continue
    if _sha256(artifact) != str(item["sha256"]):
      failures.append(f"sha256:{artifact}")
  if failures:
    raise MonitorFailure(f"checkpoint manifest verification failed: {failures}")
  return {
    "path": str(path),
    "update": int(payload["update"]),
    "roles": list(payload.get("roles", [])),
    "artifact_count": len(payload.get("artifacts", [])),
  }


def _target_update(samples: int, start_update: int) -> int:
  return start_update + math.ceil(samples / SAMPLES_PER_UPDATE)


def _write_status_best_effort(path: Path, payload: dict[str, Any]) -> None:
  try:
    _atomic_json(path, payload)
  except Exception as exc:
    print(f"MONITOR_STATUS_WRITE_FAILURE error={type(exc).__name__}: {exc}", flush=True)


def _discord_post(args: argparse.Namespace, content: str) -> bool:
  webhook_path = getattr(args, "discord_webhook_file", None)
  if webhook_path is None:
    return False
  path = Path(webhook_path).expanduser()
  if not path.is_file():
    return False
  if path.stat().st_mode & 0o077:
    print(f"DISCORD_SKIPPED insecure_permissions={path}", flush=True)
    return False
  url = path.read_text(encoding="utf-8").strip()
  if not url:
    return False
  payload = json.dumps({"content": content[:1990]}).encode("utf-8")
  request = urllib_request.Request(
    url,
    data=payload,
    headers={"Content-Type": "application/json", "User-Agent": "azuki-training-monitor/1"},
    method="POST",
  )
  try:
    with urllib_request.urlopen(request, timeout=10) as response:
      response.read()
    return True
  except (OSError, urllib_error.URLError, urllib_error.HTTPError) as exc:
    print(f"DISCORD_POST_FAILURE error={type(exc).__name__}: {exc}", flush=True)
    return False


def _health_discord_message(snapshot: dict[str, Any]) -> str:
  progress = float(snapshot["added_sampled_rows"]) / 1_000_000_000.0
  return (
    "**Azuki 1B training: healthy**\\n"
    f"Run: `{snapshot['run_id']}`\\n"
    f"Progress: {snapshot['added_sampled_rows']:,} / 1,000,000,000 ({progress:.2%})\\n"
    f"Epoch: {snapshot['last_epoch']:,} | LR: {snapshot['current_lr']:.8g}\\n"
    f"SPS median: {snapshot['sps_rolling_median']:.1f} | "
    f"Pool: {snapshot['pool_size_rolling_median']}\\n"
    f"Value loss: {snapshot['value_loss_rolling_median']:.6g} | "
    f"Explained variance: {snapshot['explained_variance_rolling_median']:.4f}\\n"
    f"Entropy: {snapshot['entropy_rolling_median']:.4f} | "
    f"Approx KL: {snapshot['approx_kl_rolling_median']:.4f}"
  )


def _run_health(args: argparse.Namespace) -> int:
  consecutive_health_failures = 0
  last_failure_epoch: int | None = None
  last_discord_post = 0.0
  while True:
    try:
      snapshot = _health_snapshot(args.log, args)
      consecutive_health_failures = 0
      last_failure_epoch = None
      _write_status_best_effort(args.status, snapshot)
      print(
        "HEALTHY "
        f"epoch={snapshot['last_epoch']} added_samples={snapshot['added_sampled_rows']} "
        f"lr={snapshot['current_lr']:.10g} median_sps={snapshot['sps_rolling_median']}",
        flush=True,
      )
      now = time.time()
      if now - last_discord_post >= args.discord_interval_seconds:
        if _discord_post(args, _health_discord_message(snapshot)):
          last_discord_post = now
      if args.once or (args.until_update is not None and snapshot["last_epoch"] >= args.until_update):
        return 0
    except MonitorFailure as exc:
      try:
        observed_epoch = _latest_metric_epoch(args.log)
      except Exception as identity_exc:
        failure = {
          "status": "monitor_error",
          "checked_at": time.time(),
          "error": f"{type(identity_exc).__name__}: {identity_exc}",
        }
        _write_status_best_effort(args.status, failure)
        print(
          f"MONITOR_ERROR no_trainer_action=true error={failure['error']}",
          flush=True,
        )
        if args.once:
          return 3
        time.sleep(args.poll_seconds)
        continue

      distinct_failure = observed_epoch != last_failure_epoch
      if distinct_failure:
        consecutive_health_failures += 1
        last_failure_epoch = observed_epoch
      failure = {
        "status": "health_violation",
        "checked_at": time.time(),
        "observed_epoch": observed_epoch,
        "distinct_failure": distinct_failure,
        "consecutive_failures": consecutive_health_failures,
        "required_failures": args.critical_repeats,
        "error": f"{type(exc).__name__}: {exc}",
      }
      _write_status_best_effort(args.status, failure)
      confirmed = consecutive_health_failures >= args.critical_repeats
      stopped = False
      if confirmed and args.stop_on_critical:
        stopped = _stop_verified_trainer(args.trainer_pid, args.run_token, str(exc))
      print(
        f"HEALTH_VIOLATION epoch={observed_epoch} distinct={distinct_failure} "
        f"confirmed={confirmed} stopped={stopped} error={failure['error']}",
        flush=True,
      )
      if confirmed:
        _discord_post(
          args,
          "**Azuki training health violation**\\n"
          f"Run token: `{args.run_token}`\\n"
          f"Stopped: `{stopped}`\\n"
          f"Reason: `{failure['error']}`",
        )
      if args.once or confirmed:
        return 2
    except Exception as exc:
      failure = {
        "status": "monitor_error",
        "checked_at": time.time(),
        "error": f"{type(exc).__name__}: {exc}",
      }
      _write_status_best_effort(args.status, failure)
      print(f"MONITOR_ERROR no_trainer_action=true error={failure['error']}", flush=True)
      _discord_post(
        args,
        "**Azuki monitor error (trainer not stopped)**\\n"
        f"Error: `{failure['error']}`",
      )
      if args.once:
        return 3
    time.sleep(args.poll_seconds)


def _run_milestones(args: argparse.Namespace) -> int:
  state = {"emitted_samples": []}
  if args.state.is_file():
    loaded = json.loads(args.state.read_text(encoding="utf-8"))
    if isinstance(loaded, dict) and isinstance(loaded.get("emitted_samples"), list):
      state = loaded
  emitted = {int(value) for value in state["emitted_samples"]}

  while True:
    try:
      snapshot = _health_snapshot(args.log, args, enforce_health_policy=False)
      epoch = int(snapshot["last_epoch"])
      for samples in MILESTONE_SAMPLES:
        target = _target_update(samples, args.start_update)
        if samples in emitted or epoch < target:
          continue
        manifest = _latest_manifest(args.checkpoint_dir, epoch)
        if manifest is None:
          raise MonitorFailure(f"no checkpoint manifest exists at milestone {samples}")
        verified = _verify_manifest(manifest)
        milestone = {
          **snapshot,
          "milestone_sampled_rows": samples,
          "target_update": target,
          "checkpoint": verified,
          "human_validation_required": samples >= 1_000_000_000,
          "decision_gate": samples in (200_000_000, 450_000_000, 1_000_000_000),
        }
        output = args.output_dir / f"milestone_{samples}.json"
        _atomic_json(output, milestone)
        emitted.add(samples)
        state["emitted_samples"] = sorted(emitted)
        _atomic_json(args.state, state)
        print(
          f"MILESTONE samples={samples} epoch={epoch} checkpoint={verified['update']} "
          f"human_validation_required={milestone['human_validation_required']}",
          flush=True,
        )
        _discord_post(
          args,
          "**Azuki training milestone reached**\\n"
          f"Samples: {samples:,}\\n"
          f"Epoch: {epoch:,}\\n"
          f"Checkpoint update: {verified['update']:,}\\n"
          f"SPS median: {snapshot['sps_rolling_median']:.1f}\\n"
          f"Value loss: {snapshot['value_loss_rolling_median']:.6g}\\n"
          f"Explained variance: {snapshot['explained_variance_rolling_median']:.4f}\\n"
          f"Decision gate: `{milestone['decision_gate']}`\\n"
          "Reply in the training thread to schedule the checkpoint evaluation.",
        )
      if 1_000_000_000 in emitted:
        return 0
    except Exception as exc:
      failure = {
        "status": "retrying",
        "checked_at": time.time(),
        "error": f"{type(exc).__name__}: {exc}",
      }
      _atomic_json(args.output_dir / "milestone_monitor_failure.json", failure)
      print(
        f"MILESTONE_MONITOR_RETRY no_trainer_action=true error={failure['error']}",
        flush=True,
      )
    time.sleep(args.poll_seconds)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="mode", required=True)

  health = subparsers.add_parser("health")
  health.add_argument("--log", type=Path, required=True)
  health.add_argument("--status", type=Path, required=True)
  health.add_argument("--poll-seconds", type=float, default=30.0)
  health.add_argument("--stale-seconds", type=float, default=900.0)
  health.add_argument("--until-update", type=int)
  health.add_argument("--once", action="store_true")
  health.add_argument("--trainer-pid", type=int)
  health.add_argument("--run-token")
  health.add_argument("--critical-repeats", type=int, default=3)
  health.add_argument("--stop-on-critical", action="store_true")
  health.add_argument("--start-update", type=int, required=True)
  health.add_argument("--end-update", type=int, required=True)
  health.add_argument("--peak-lr", type=float, required=True)
  health.add_argument("--expected-pool-size", type=float, default=-1.0)
  health.add_argument("--expected-trainable-fraction", type=float, default=-1.0)
  health.add_argument("--minimum-median-sps", type=float, default=MINIMUM_MEDIAN_SPS)
  health.add_argument("--max-value-loss", type=float, default=1.0)
  health.add_argument("--adaptive-value-loss", action="store_true")
  health.add_argument("--value-loss-growth-factor", type=float, default=100.0)
  health.add_argument("--value-loss-growth-floor", type=float, default=1_000_000.0)
  health.add_argument("--discord-webhook-file", type=Path)
  health.add_argument("--discord-interval-seconds", type=float, default=3600.0)
  health.set_defaults(run=_run_health)

  milestones = subparsers.add_parser("milestones")
  milestones.add_argument("--log", type=Path, required=True)
  milestones.add_argument("--checkpoint-dir", type=Path, required=True)
  milestones.add_argument("--output-dir", type=Path, required=True)
  milestones.add_argument("--state", type=Path, required=True)
  milestones.add_argument("--poll-seconds", type=float, default=60.0)
  milestones.add_argument("--stale-seconds", type=float, default=900.0)
  milestones.add_argument("--start-update", type=int, required=True)
  milestones.add_argument("--end-update", type=int, required=True)
  milestones.add_argument("--peak-lr", type=float, required=True)
  milestones.add_argument("--expected-pool-size", type=float, default=-1.0)
  milestones.add_argument("--expected-trainable-fraction", type=float, default=-1.0)
  milestones.add_argument("--minimum-median-sps", type=float, default=MINIMUM_MEDIAN_SPS)
  milestones.add_argument("--max-value-loss", type=float, default=0.0)
  milestones.add_argument("--adaptive-value-loss", action="store_true")
  milestones.add_argument("--value-loss-growth-factor", type=float, default=100.0)
  milestones.add_argument("--value-loss-growth-floor", type=float, default=1_000_000.0)
  milestones.add_argument("--discord-webhook-file", type=Path)
  milestones.set_defaults(run=_run_milestones)
  return parser


def main() -> int:
  args = _parser().parse_args()
  return int(args.run(args))


if __name__ == "__main__":
  raise SystemExit(main())
