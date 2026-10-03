#!/usr/bin/env python3
"""Run one registered element-specialist stage with strict resume/manifest checks.

Modeled on control_strategy_scale_30m_v1/run_stage.py. The run root is the
directory written by build_specialist_inputs.py (experiment.json,
source_parent/, league/). The first stage must resume the registered u8223
parent and restart the cosine LR schedule once from its current LR; that one
resume is an explicit native-binding + deck-pool migration. Later stages
resume their own atomic checkpoints with no migration and no restart.
"""
from __future__ import annotations

import argparse
import ast
import configparser
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROWS_PER_UPDATE = 15360


def digest(path):
  with Path(path).open("rb") as handle:
    return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
  temporary.replace(path)


def read_metrics(path):
  if not path.exists():
    return []
  lines = path.read_bytes().splitlines(keepends=True)
  return [json.loads(line) for line in lines if line.endswith(b"\n")]


def check_health(rows, gates):
  metrics = [row for row in rows if "epoch" in row]
  sustained_kl = sustained_clip = 0
  for row in metrics:
    for key, value in row.items():
      if isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError(f"Nonfinite metric: {key}={value}")
    if not math.isclose(row["environment/prebuilt/configured_probability"], gates["configured_probability"], rel_tol=0, abs_tol=1e-12):
      raise ValueError("Configured prebuilt probability changed")
    if row["environment/normalization/update_rows"] != gates["sampled_rows_per_update"]:
      raise ValueError("Production update geometry changed")
    for key in ("environment/timeout_truncation_rate", "environment/zero_legal_action_truncation_rate", "environment/draft_episode_credit/incomplete_episodes"):
      if row.get(key, 0) != 0:
        raise ValueError(f"Training health gate failed: {key}={row[key]}")
    learner_rows = row.get("environment/specialist/learner_battle_rows")
    element_rows = row.get("environment/specialist/learner_battle_element_rows")
    if learner_rows is None or element_rows is None:
      raise ValueError("Missing specialist element telemetry")
    if element_rows != learner_rows * gates["learner_element_fraction"]:
      raise ValueError(f"Learner battle rows off-element: {element_rows}/{learner_rows} at update {row['epoch']}")
    zero_kl = row["losses/ppo_diag_mb_exact_kl_index_00"]
    if abs(zero_kl) > gates["zero_update_exact_kl_abs_max"]:
      raise ValueError(f"Zero-update likelihood mismatch: {zero_kl}")
    replay_kl = row["losses/ppo_diag_rollout_reference_kl_mean"]
    if replay_kl > gates["rollout_reference_kl_window_mean_max"]:
      raise ValueError(f"Rollout/reference mismatch: {replay_kl}")
    sustained_kl = sustained_kl + 1 if row.get("losses/ppo_diag_exact_kl_mean", 0) > gates["ppo_exact_kl_window_mean_limit"] else 0
    sustained_clip = sustained_clip + 1 if row.get("losses/clipfrac", 0) > gates["ppo_clip_fraction_window_limit"] else 0
    if max(sustained_kl, sustained_clip) >= gates["sustained_windows"]:
      raise ValueError("Sustained PPO KL/clipping instability")
  return metrics


def gpu_sample():
  result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
  values = [float(value.strip()) for value in result.stdout.strip().splitlines()[0].split(",")]
  return {"ts": time.time(), "used_mib": values[0], "total_mib": values[1], "utilization_percent": values[2]}


def saved_resume_schedule(checkpoint, plan):
  import torch

  metadata = json.loads(Path(str(checkpoint) + ".meta.json").read_text())
  state_path = checkpoint.parent / f"trainer_state_{metadata['update']:06d}.pt"
  state = torch.load(state_path, map_location="cpu", weights_only=False, mmap=True)
  if state["model_name"] != checkpoint.name:
    raise ValueError("Trainer state belongs to a different model")
  for key in ("global_step", "update", "prebuilt_battle_decisions"):
    if state[key] != metadata[key]:
      raise ValueError(f"Trainer/metadata counter mismatch: {key}")
  if not state["optimizer_state_dict"]["state"] or not state["coordinator_rng_state"]:
    raise ValueError("Missing optimizer history or coordinator RNG state")
  scheduler = state["scheduler_state_dict"]
  parent_update = plan["parent"]["update"]
  is_parent = state["update"] == parent_update
  if is_parent and digest(checkpoint) != plan["parent"]["checkpoint_sha256"]:
    raise ValueError("Initial schedule restart requires the registered parent")
  registered = plan["lr_schedule"]["parent_saved_scheduler" if is_parent else "saved_scheduler"]
  expected_phase = registered["last_epoch"] + (0 if is_parent else state["update"] - parent_update)
  for key in ("T_max", "eta_min", "base_lrs"):
    if scheduler[key] != registered[key]:
      raise ValueError(f"Saved scheduler changed: {key}")
  if scheduler["last_epoch"] != expected_phase or not 0 <= expected_phase < scheduler["T_max"]:
    raise ValueError("Saved scheduler phase changed or exhausted")
  lrs = [float(group["lr"]) for group in state["optimizer_state_dict"]["param_groups"]]
  if lrs != scheduler["_last_lr"]:
    raise ValueError("Optimizer and scheduler learning rates disagree")
  expected_parent_lr = plan["lr_schedule"].get("parent_lr", plan["lr_schedule"]["peak_lr"])
  if is_parent and lrs != [expected_parent_lr]:
    raise ValueError("Registered parent LR is not the parent's current LR")
  return {"global_step": state["global_step"], "update": state["update"], "is_parent": is_parent,
          "optimizer_lrs": lrs, "scheduler": scheduler}


def verify_resume_console(console, saved, plan, restart):
  lines = re.findall(r"\[resume\] restored trainer state: [^\n]+", console)
  if len(lines) != 1:
    raise ValueError("Missing unique full-state restoration evidence")
  for marker in (
    f"global_step={saved['global_step']}, epoch={saved['update']},",
    "optimizer_restored=True", "scheduler_restored=True", "coordinator_rng_restored=True",
  ):
    if marker not in lines[0]:
      raise ValueError(f"Resume did not preserve state: {marker}")
  if "strict=True, missing_keys=0, unexpected_keys=0" not in console:
    raise ValueError("Strict model restoration was not verified")
  if "[specialist] frozen opponent pool verified recent-lineage" not in console:
    raise ValueError("Missing frozen-pool lineage verification")
  match = re.search(r"optimizer_lrs=(\[[^\]]*\])", lines[0])
  if match is None or ast.literal_eval(match.group(1)) != saved["optimizer_lrs"]:
    raise ValueError("Learning rate changed at resume")
  restarts = re.findall(r"\[resume\] restarted LR schedule: [^\n]+", console)
  if restart:
    expected = (
      f"[resume] restarted LR schedule: remaining_epochs={plan['lr_schedule']['additional_updates']}, "
      f"peak_lrs={[plan['lr_schedule']['peak_lr']] if plan['lr_schedule'].get('rewarm') else saved['optimizer_lrs']}, final_lr=0.0"
    )
    if restarts != [expected]:
      raise ValueError("Missing unique current-LR schedule restart evidence")
  elif restarts:
    raise ValueError("Unexpected repeated schedule restart")
  if "[resume] applying anneal step offset" in console:
    raise ValueError("Unexpected progress offset")


def verify_logged_schedule(metrics, plan):
  registered = plan["lr_schedule"]["saved_scheduler"]
  for row in metrics:
    phase = registered["last_epoch"] + int(row["epoch"]) - plan["parent"]["update"]
    if not 0 <= phase < registered["T_max"]:
      raise ValueError("Continuation reached the cosine schedule boundary")
    expected = registered["eta_min"] + (registered["base_lrs"][0] - registered["eta_min"]) * (
      1 + math.cos(math.pi * phase / registered["T_max"])
    ) / 2
    if not math.isclose(row["learning_rate"], expected, rel_tol=1e-9, abs_tol=1e-12):
      raise ValueError(f"Learning-rate discontinuity at update {row['epoch']}")


def verify_config(cfg, plan, config_path):
  if cfg["env"].get("learner_element") != plan["element"]:
    raise ValueError("Config learner_element differs from the registered element")
  if int(cfg["league"]["checkpoint_add_interval"]) != 0:
    raise ValueError("Specialist league must not ingest its own snapshots")
  if float(cfg["train"]["learning_rate"]) != plan["lr_schedule"]["peak_lr"]:
    raise ValueError("Configured learning rate differs from the registered parent LR")
  if int(cfg["train"]["total_timesteps"]) != plan["lr_schedule"]["train_total_timesteps"]:
    raise ValueError("Configured schedule horizon changed")
  if digest(cfg["env"]["deck_pool_path"]) != plan["fixed_recipe"]["deck_pool_sha256"]:
    raise ValueError("Deck pool changed since registration")
  if cfg.getboolean("resume", "restart_lr_schedule", fallback=False):
    raise ValueError("Schedule restart must be explicit on the first-stage command")
  if not math.isclose(float(cfg["env"]["prebuilt_probability"]), 0.8) or float(cfg["league"]["frozen_ratio"]) != 0.4:
    raise ValueError("Recipe drift: prebuilt probability / frozen ratio")
  if float(cfg["train"]["gamma"]) != 1.0:
    raise ValueError("Recipe drift: gamma")


def summarize_specialist(metrics):
  learner = sum(row["environment/specialist/learner_battle_rows"] for row in metrics)
  element = sum(row["environment/specialist/learner_battle_element_rows"] for row in metrics)
  prebuilt = sum(row.get("environment/specialist/prebuilt_episodes", 0) for row in metrics)
  draft = sum(row.get("environment/specialist/draft_episodes", 0) for row in metrics)
  return {
    "learner_battle_rows": int(learner),
    "learner_battle_element_rows": int(element),
    "learner_battle_element_fraction": element / learner if learner else None,
    "prebuilt_episodes": int(prebuilt),
    "draft_episodes": int(draft),
    "prebuilt_episode_fraction": prebuilt / (prebuilt + draft) if prebuilt + draft else None,
  }


def training_result(root, plan, config_path, parent, requested, metrics_path, console_path, memory, runtime):
  rows = read_metrics(metrics_path)
  metrics = check_health(rows, plan["canary_gates"])
  verify_logged_schedule(metrics, plan)
  if not metrics or rows[-1].get("_event") != "close":
    raise ValueError("Missing completed training log/close event")
  directory = Path(rows[-1]["model_path"]).with_suffix("")
  manifests = sorted(directory.glob("checkpoint_*.manifest.json"))
  if not manifests:
    raise ValueError("No endpoint atomic checkpoint manifest")
  manifest_path = manifests[-1]
  manifest = json.loads(manifest_path.read_text())
  if Path(manifest["config"]["path"]).resolve() != config_path.resolve() or digest(config_path) != manifest["config"]["sha256"]:
    raise ValueError("Atomic checkpoint training-config mismatch")
  for item in manifest["artifacts"]:
    if digest(directory / item["path"]) != item["sha256"]:
      raise ValueError(f"Atomic checkpoint artifact mismatch: {item['path']}")
  update = int(manifest["update"])
  checkpoint = directory / f"model_azuki_local_{update:06d}.pt"
  metadata = json.loads(Path(str(checkpoint) + ".meta.json").read_text())
  actual = int(metadata["global_step"]) - int(parent["global_step"])
  if not requested <= actual < requested + ROWS_PER_UPDATE:
    raise ValueError(f"Learner-step stop mismatch: requested={requested}, actual={actual}")
  parity = metadata["checkpoint_parity"]
  if parity["max_abs_diff"] != 0 or any(parity.get(key, 0) for key in ("missing_key_count", "unexpected_key_count", "shape_mismatch_key_count")):
    raise ValueError(f"Checkpoint parity failure: {parity}")
  console = console_path.read_text(errors="replace")
  for required in ("optimizer_restored=True", "scheduler_restored=True", "coordinator_rng_restored=True", "missing_keys=0, unexpected_keys=0", "target_reached=True"):
    if required not in console:
      raise ValueError(f"Missing continuation evidence: {required}")
  binding_path = runtime / "build/python/src/binding.so"
  if Path(metadata["runtime_fingerprint"]["binding_path"]).resolve() != binding_path.resolve():
    raise ValueError("Endpoint used a different native binding")
  elapsed = float(metrics[-1]["uptime"])
  updates = update - int(parent["update"])
  battle_decisions = int(metadata["prebuilt_battle_decisions"]) - int(parent["prebuilt_battle_decisions"])
  return {
    "element": plan["element"], "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
    "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
    "trainer_state": str(directory / f"trainer_state_{update:06d}.pt"), "training_config": str(config_path),
    "start_global_step": int(parent["global_step"]), "learner_global_step": int(metadata["global_step"]),
    "requested_additional_learner_steps": requested, "actual_additional_learner_steps": actual,
    "update": update, "additional_updates": updates, "additional_sampled_rows": updates * ROWS_PER_UPDATE,
    "additional_learner_battle_decisions": battle_decisions, "uptime_seconds": elapsed,
    "raw_rows_per_second_including_startup": updates * ROWS_PER_UPDATE / elapsed,
    "learner_steps_per_second_including_startup": actual / elapsed,
    "median_logged_sps": sorted(row["SPS"] for row in metrics)[len(metrics) // 2],
    "endpoint_learning_rate": metrics[-1]["learning_rate"],
    "specialist": summarize_specialist(metrics),
    "peak_observed_gpu_used_mib": max(sample["used_mib"] for sample in memory) if memory else None,
    "checkpoint_parity": parity, "atomic_hashes_verified": True, "strict_full_state_reload_verified": True,
    "health_gates_passed": True, "metrics": str(metrics_path), "console": str(console_path),
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-root", type=Path, required=True)
  parser.add_argument("--runtime", type=Path, required=True)
  parser.add_argument("--stage", required=True)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--resume-checkpoint", type=Path, required=True)
  parser.add_argument("--additional-learner-steps", type=int, required=True)
  parser.add_argument("--restart-lr-schedule", action="store_true")
  parser.add_argument("--allow-binding-change", action="store_true",
                      help="Resume a specialist checkpoint after an engine-only rebuild (binding_size check only)")
  args = parser.parse_args()
  if args.additional_learner_steps <= 0:
    parser.error("--additional-learner-steps must be positive")
  root, runtime = args.run_root.resolve(), args.runtime.resolve()
  plan = json.loads((root / "experiment.json").read_text())
  stage_dir = root / "stages" / args.stage
  cfg = configparser.ConfigParser(interpolation=None)
  cfg.read(args.config)
  metrics_path = Path(cfg["base"]["jsonl_log"])
  if metrics_path.exists():
    raise ValueError(f"Refusing to overwrite existing stage log: {metrics_path}")
  verify_config(cfg, plan, args.config)
  resuming_parent = args.resume_checkpoint.resolve() == Path(plan["parent"]["checkpoint"]).resolve()
  if args.restart_lr_schedule != resuming_parent:
    raise ValueError("Restart the LR schedule exactly once, when resuming the registered parent")
  inputs = json.loads((root / "source_parent/input_manifest.json").read_text())
  for path, expected in inputs["sha256"].items():
    if digest(path) != expected:
      raise ValueError(f"Frozen parent/league/deck input drift: {path}")
  parent = json.loads(Path(str(args.resume_checkpoint) + ".meta.json").read_text())
  saved_schedule = saved_resume_schedule(args.resume_checkpoint, plan)
  stage_dir.mkdir(parents=True, exist_ok=False)
  console_path = stage_dir / "console.log"
  save(stage_dir / "resume_state_audit.json", saved_schedule)
  environment = {key: value for key, value in os.environ.items() if not key.startswith("AZK_")}
  environment.update(
    PYTHONPATH=f"{runtime}/build/python/src:{runtime}/python/src",
    LD_LIBRARY_PATH=str(runtime / "build/_deps/flecs_src-build"),
    PYTHONUNBUFFERED="1", PYTHONHASHSEED="43", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="1", PYTORCH_ALLOC_CONF="expandable_segments:True",
  )
  command = [str(runtime / ".venv/bin/python"), "python/src/train.py", "--config", str(args.config.resolve()),
             "--resume-checkpoint", str(args.resume_checkpoint.resolve()), "--resume-load-optimizer",
             "--resume-strict", "--stop-after-learner-steps", str(args.additional_learner_steps)]
  if resuming_parent:
    # One explicit migration from the u8223 lineage runtime: new native binding
    # (learner_element + 5 new cards) and the curated deck pool. Observation
    # dtype/size, strict weights, optimizer, scheduler and RNG stay enforced.
    environment["AZK_RESUME_ALLOW_BINDING_MISMATCH"] = "1"
    command += ["--resume-restart-lr-schedule", "--resume-allow-deck-pool-migration"]
  elif args.allow_binding_change:
    environment["AZK_RESUME_ALLOW_BINDING_MISMATCH"] = "1"
  save(stage_dir / "launch.json", {
    "command": command, "cwd": str(runtime), "parent_checkpoint_sha256": digest(args.resume_checkpoint),
    "parent_global_step": parent["global_step"], "parent_update": parent["update"],
    "target_global_step": parent["global_step"] + args.additional_learner_steps,
    "lr_restart": args.restart_lr_schedule, "migration_applied": resuming_parent,
    "binding_change_allowed": resuming_parent or args.allow_binding_change, "started_at_unix": time.time(),
  })
  memory = []
  process = None
  training_ready = False
  try:
    with console_path.open("x") as output, (stage_dir / "gpu_memory.jsonl").open("x") as gpu_log:
      process = subprocess.Popen(command, cwd=runtime, env=environment, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
      print(f"[stage] starting {plan['element']}/{args.stage} pid={process.pid} target={parent['global_step'] + args.additional_learner_steps}", flush=True)
      while process.poll() is None:
        console = console_path.read_text(errors="replace")
        if not training_ready and "[learner-step-budget] start=" in console:
          verify_resume_console(console, saved_schedule, plan, args.restart_lr_schedule)
          save(stage_dir / "resume_verified.json", {
            "verified": True, "global_step": saved_schedule["global_step"], "update": saved_schedule["update"],
            "optimizer_lrs": saved_schedule["optimizer_lrs"], "schedule_restarted": args.restart_lr_schedule,
          })
          training_ready = True
          print(f"[stage] training-ready {plan['element']}/{args.stage}", flush=True)
        sample = gpu_sample()
        memory.append(sample)
        gpu_log.write(json.dumps(sample) + "\n")
        gpu_log.flush()
        metrics = check_health(read_metrics(metrics_path), plan["canary_gates"])
        verify_logged_schedule(metrics, plan)
        time.sleep(5)
      if process.returncode != 0:
        raise RuntimeError(f"Training failed with exit {process.returncode}; console={console_path}")
    report = training_result(root, plan, args.config.resolve(), parent, args.additional_learner_steps,
                             metrics_path, console_path, memory, runtime)
    if not training_ready:
      raise ValueError("Stage ended without verified full-state restoration")
    saved_resume_schedule(Path(report["checkpoint"]), plan)
    save(stage_dir / "result.json", report)
    print("[stage] complete " + json.dumps(report, sort_keys=True), flush=True)
  except BaseException as exc:
    if process is not None and process.poll() is None:
      os.killpg(process.pid, signal.SIGTERM)
      try:
        process.wait(timeout=30)
      except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    save(stage_dir / "failure.json", {"error": str(exc), "console": str(console_path), "ts": time.time()})
    raise


if __name__ == "__main__":
  main()
