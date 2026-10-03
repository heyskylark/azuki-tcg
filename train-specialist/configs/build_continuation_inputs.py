#!/usr/bin/env python3
"""Freeze the inputs of a specialist continuation stage resumed from a finished specialist.

Reuses the source run's frozen generalist pool, deck pool and recipe. Changes, all explicit:
  - parent = the source run's final checkpoint;
  - LR re-warmed to the source run's starting peak with a fresh cosine horizon;
  - entropy / subaction-temperature / smoothing anneals pushed past the horizon so the
    recipe stays identical to the source stage (it never reached them).
Writes <run-root>/{source_parent,league,experiment.json} and <run-root>/specialist_<element>_<stage>.ini.
Run with PYTHONPATH=<runtime>/build/python/src:<runtime>/python/src.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

RUNTIME = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RUNTIME / "build/python/src"), str(RUNTIME / "python/src"), str(Path(__file__).resolve().parent)]
from build_specialist_inputs import SAMPLED_ROWS_PER_UPDATE, digest, file_entry, save, saved_schedule  # noqa: E402
from specialist import assert_recent_lineage  # noqa: E402

ANNEAL_OFF_START = 10_000_000_000
ANNEAL_OFF_END = 11_000_000_000


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--source-run", type=Path, required=True, help="finished specialist run root")
  parser.add_argument("--run-root", type=Path, required=True)
  parser.add_argument("--stage", required=True)
  parser.add_argument("--additional-learner-steps", type=int, required=True)
  parser.add_argument("--schedule-updates", type=int, required=True,
                      help="cosine horizon in updates; must exceed the updates the step budget will use")
  args = parser.parse_args()
  source, run_root = args.source_run.resolve(), args.run_root.resolve()
  plan = json.loads((source / "experiment.json").read_text())
  element = plan["element"]
  final = json.loads((source / "final_result.json").read_text())
  model = Path(final["checkpoint"])
  metadata = Path(str(model) + ".meta.json")
  update = json.loads(metadata.read_text())["update"]
  trainer = model.parent / f"trainer_state_{update:06d}.pt"
  atomic = model.parent / f"checkpoint_{update:06d}.manifest.json"
  for item in json.loads(atomic.read_text())["artifacts"]:
    if digest(model.parent / item["path"]) != item["sha256"]:
      raise ValueError(f"Source artifact hash mismatch: {item['path']}")
  assert_recent_lineage([model])
  schedule = saved_schedule(trainer)
  if schedule["update"] != update:
    raise ValueError("Source trainer state update mismatch")
  peak_lr = float(plan["lr_schedule"]["peak_lr"])
  total_timesteps = (update + args.schedule_updates) * SAMPLED_ROWS_PER_UPDATE

  league_dir = run_root / "league"
  league_dir.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(source / "league/league_state.json", league_dir / "league_state.json")
  shutil.copyfile(source / "league/league_state_promotion.json", league_dir / "league_state_promotion.json")
  pool = json.loads((league_dir / "league_state.json").read_text())
  pool_checkpoints = [Path(entry["checkpoint_path"]) for entry in pool["policies"].values()]
  assert_recent_lineage(pool_checkpoints)

  parent_manifest = {
    "schema_version": 2, "epoch": update, "model": file_entry(model), "metadata": file_entry(metadata),
    "trainer": file_entry(trainer), "source_atomic_manifest": file_entry(atomic), "source_experiment": str(source),
  }
  save(run_root / "source_parent/parent_manifest.json", parent_manifest)
  deck_pool = Path(plan["fixed_recipe"]["deck_pool"])
  inputs = [model, metadata, trainer, league_dir / "league_state.json", *pool_checkpoints, deck_pool]
  save(run_root / "source_parent/input_manifest.json",
       {"schema_version": 1, "sha256": {str(path): digest(path) for path in dict.fromkeys(inputs)}})

  text = Path(plan["config"]).read_text()
  text = text.replace(str(source), str(run_root))
  substitutions = {
    "jsonl_log": str(run_root / "logs" / f"specialist_{element}_{args.stage}.jsonl"),
    "total_timesteps": str(total_timesteps),
    "learning_rate": repr(peak_lr),
    "ent_coef_anneal_start_step": str(ANNEAL_OFF_START), "ent_coef_anneal_end_step": str(ANNEAL_OFF_END),
    "subaction_temperature_anneal_start_step": str(ANNEAL_OFF_START),
    "subaction_temperature_anneal_end_step": str(ANNEAL_OFF_END),
    "smoothing_eps_anneal_start_step": str(ANNEAL_OFF_START), "smoothing_eps_anneal_end_step": str(ANNEAL_OFF_END),
  }
  for key, value in substitutions.items():
    text, count = re.subn(rf"(?m)^{re.escape(key)} = .*$", f"{key} = {value}", text)
    if count != 1:
      raise ValueError(f"Expected exactly one '{key}' line in the source config, found {count}")
  if str(source) in text.replace(str(run_root), ""):
    raise ValueError("Source run paths remain in the continuation config")
  config_out = run_root / f"specialist_{element}_{args.stage}.ini"
  config_out.write_text(text)

  save(run_root / "experiment.json", {
    **plan,
    "config": str(config_out),
    "source_run": str(source),
    "parent": {
      "checkpoint": str(model), "checkpoint_sha256": parent_manifest["model"]["sha256"],
      "global_step": schedule["global_step"], "update": update,
      "prebuilt_battle_decisions": schedule["prebuilt_battle_decisions"],
    },
    "continuation": {"stage": args.stage, "additional_learner_steps": args.additional_learner_steps},
    "lr_schedule": {
      "type": "CosineAnnealingLR re-warmed to the source stage's starting peak",
      "peak_lr": peak_lr,
      "rewarm": True,
      "parent_lr": schedule["optimizer_lrs"][0],
      "additional_updates": args.schedule_updates,
      "train_total_timesteps": total_timesteps,
      "parent_saved_scheduler": schedule["scheduler"],
      "saved_scheduler": {"T_max": args.schedule_updates, "eta_min": 0.0, "base_lrs": [peak_lr], "last_epoch": 0},
      "entropy": f"ent_coef 0.01; entropy/temperature/smoothing anneals moved to {ANNEAL_OFF_START} global steps",
    },
  })
  print(json.dumps({"element": element, "parent_update": update, "parent_lr": schedule["optimizer_lrs"][0],
                    "peak_lr": peak_lr, "total_timesteps": total_timesteps, "config": str(config_out)}))


if __name__ == "__main__":
  main()
