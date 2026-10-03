#!/usr/bin/env python3
"""Run the mandatory 100+200 update single-node production resume soak."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "python" / "config" / "azuki_deckbuild_production_3090.ini"
DEFAULT_PARENT = REPO_ROOT / "train-ablation-1781126582" / "results" / "production_launch_v1" / "parent_p10730"
DEFAULT_OUTPUT = REPO_ROOT / "train-ablation-1781126582" / "results" / "production_launch_v1" / "soak"
SAMPLES_PER_UPDATE = 15_360
BASELINE_SPS = 1337.7437704904828


def _closed_model_path(path: Path) -> Path:
  rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
  closes = [row for row in rows if row.get("_event") == "close"]
  if len(closes) != 1 or not isinstance(closes[0].get("model_path"), str):
    raise ValueError(f"compact log has no completed model path: {path}")
  model = Path(closes[0]["model_path"])
  if not model.is_absolute():
    model = REPO_ROOT / model
  if not model.is_file():
    raise FileNotFoundError(model)
  run_dir = model.with_suffix("")
  checkpoints = sorted(run_dir.glob("model_azuki_local_*.pt"))
  if checkpoints:
    return checkpoints[-1].resolve()
  return model.resolve()


def _run_phase(
  *,
  config: Path,
  resume: Path,
  end_update: int,
  tag: str,
  runtime: Path,
  compact_log: Path,
  stdout_log: Path,
) -> Path:
  command = [
    sys.executable,
    str(REPO_ROOT / "python" / "src" / "train.py"),
    "--config",
    str(config),
    "--resume-checkpoint",
    str(resume),
    "--jsonl-log",
    str(compact_log),
    "--tag",
    tag,
    "--train.total_timesteps",
    str(end_update * SAMPLES_PER_UPDATE),
    "--league.state_path",
    str(runtime / "league_state.json"),
    "--league.promotion_state_path",
    str(runtime / "league_state_promotion.json"),
  ]
  env = os.environ.copy()
  env["PYTHONPATH"] = os.pathsep.join(
    [str(REPO_ROOT / "build" / "python" / "src"), str(REPO_ROOT / "python" / "src")]
  )
  stdout_log.parent.mkdir(parents=True, exist_ok=True)
  print(f"[soak] starting {tag}: resume={resume} end_update={end_update}", flush=True)
  with stdout_log.open("w", encoding="utf-8") as handle:
    subprocess.run(
      command,
      cwd=REPO_ROOT,
      env=env,
      stdout=handle,
      stderr=subprocess.STDOUT,
      check=True,
    )
  model = _closed_model_path(compact_log)
  print(f"[soak] completed {tag}: model={model}", flush=True)
  return model


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
  parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
  parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--start-update", type=int, default=10_730)
  parser.add_argument("--phase-one-updates", type=int, default=100)
  parser.add_argument("--phase-two-updates", type=int, default=200)
  args = parser.parse_args()
  if args.phase_one_updates < 1 or args.phase_two_updates < 1:
    raise ValueError("soak phases must both contain at least one update")

  run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  output = args.output_root / run_id
  runtime = output / "runtime"
  runtime.mkdir(parents=True)
  shutil.copy2(args.parent / "state" / "league_state.json", runtime / "league_state.json")
  shutil.copy2(
    args.parent / "state" / "league_state_promotion.json",
    runtime / "league_state_promotion.json",
  )

  start = args.start_update
  phase_one_end = start + args.phase_one_updates
  phase_two_end = phase_one_end + args.phase_two_updates
  phase_one_log = output / "phase1.jsonl"
  phase_two_log = output / "phase2.jsonl"
  parent_model = args.parent / "resume" / f"model_azuki_local_{start:06d}.pt"
  phase_one_model = _run_phase(
    config=args.config,
    resume=parent_model,
    end_update=phase_one_end,
    tag=f"production_soak_{run_id}_phase1",
    runtime=runtime,
    compact_log=phase_one_log,
    stdout_log=output / "phase1.stdout.log",
  )
  phase_two_model = _run_phase(
    config=args.config,
    resume=phase_one_model,
    end_update=phase_two_end,
    tag=f"production_soak_{run_id}_phase2",
    runtime=runtime,
    compact_log=phase_two_log,
    stdout_log=output / "phase2.stdout.log",
  )

  checkpoint_dir = phase_two_model.parent
  manifests = [
    phase_one_model.parent / f"checkpoint_{phase_one_end:06d}.manifest.json",
    checkpoint_dir / f"checkpoint_{((phase_one_end // 1000) + 1) * 1000:06d}.manifest.json",
    checkpoint_dir / f"checkpoint_{phase_two_end:06d}.manifest.json",
  ]
  manifests = list(dict.fromkeys(manifests))
  validator = REPO_ROOT / "train-ablation-1781126582" / "validate_production_run.py"
  command = [
    sys.executable,
    str(validator),
    "--config",
    str(args.config),
    "--phase",
    f"{start}:{phase_one_end}:{phase_one_log}",
    "--phase",
    f"{phase_one_end}:{phase_two_end}:{phase_two_log}",
    "--baseline-sps",
    str(BASELINE_SPS),
    "--output",
    str(output / "soak_report.json"),
  ]
  for manifest in manifests:
    command.extend(["--checkpoint-manifest", str(manifest)])
  subprocess.run(command, cwd=REPO_ROOT, check=True)
  print(f"[soak] PASS report={output / 'soak_report.json'}", flush=True)


if __name__ == "__main__":
  main()
