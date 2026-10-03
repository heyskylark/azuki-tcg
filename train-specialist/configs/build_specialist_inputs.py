#!/usr/bin/env python3
"""Freeze the inputs of one element-specialist fine-tune and write its config.

Writes under train-specialist/runs/<element>/ (or --run-root):
  source_parent/parent_manifest.json  trusted u8223 parent (schema 2, hashes)
  source_parent/input_manifest.json   sha256 of every frozen input
  league/league_state.json            frozen recent-lineage generalist pool
  league/league_state_promotion.json  empty promotion archive (shadow only)
  experiment.json                     registered plan read by run_specialist_stage.py
and the training config train-specialist/configs/specialist_<element>.ini
(or --config-out). Run with PYTHONPATH=<runtime>/build/python/src:<runtime>/python/src.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

RUNTIME = Path(__file__).resolve().parents[2]
SPECIALIST_ROOT = RUNTIME / "train-specialist"
RESULTS = Path("/home/skylark/git/azuki-tcg/train-ablation-1781126582/results/control_strategy_scale_30m_v1")
PARENT_DIR = RESULTS / "runs/control/artifacts/azuki_local_control_strategy_scale_30m_s43_179038406361"
PARENT_UPDATE = 8223
U9305 = (
  RESULTS / "runs/control/artifacts/azuki_local_control_strategy_scale_30m_s43_179039793722"
  / "model_azuki_local_009305.pt"
)
CURATED_POOL = SPECIALIST_ROOT / "decks/curated_deck_pool.json"
SAMPLED_ROWS_PER_UPDATE = 15360
ADDITIONAL_LEARNER_STEPS = 30_000_000
# Same margin rule as the parent campaign: register a cosine horizon beyond the
# learner-step endpoint (~3245 updates at the parent's 9246 learner steps per
# update) and stop at the step budget rather than cross the boundary.
SCHEDULE_UPDATES = 4500
FROZEN_POOL_KEEP = {"keep_recent": 8, "keep_mid": 4, "keep_old": 3}

sys.path[:0] = [str(RUNTIME / "build/python/src"), str(RUNTIME / "python/src")]
from prebuilt_deck_pool import load_specialist_deck_groups, specialist_learner_deck_indices  # noqa: E402
from specialist import SPECIALIST_ELEMENTS, assert_recent_lineage  # noqa: E402


def digest(path: Path) -> str:
  with path.open("rb") as handle:
    return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path: Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n")
  temporary.replace(path)


def file_entry(path: Path) -> dict:
  return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def parent_paths() -> dict[str, Path]:
  suffix = f"{PARENT_UPDATE:06d}"
  return {
    "model": PARENT_DIR / f"model_azuki_local_{suffix}.pt",
    "metadata": PARENT_DIR / f"model_azuki_local_{suffix}.pt.meta.json",
    "trainer": PARENT_DIR / f"trainer_state_{suffix}.pt",
    "league": PARENT_DIR / f"league_state_{suffix}.json",
    "atomic_manifest": PARENT_DIR / f"checkpoint_{suffix}.manifest.json",
  }


def verify_parent_atomic_manifest(paths: dict[str, Path]) -> None:
  manifest = json.loads(paths["atomic_manifest"].read_text())
  if int(manifest["update"]) != PARENT_UPDATE:
    raise ValueError("Parent atomic manifest update mismatch")
  for item in manifest["artifacts"]:
    if digest(PARENT_DIR / item["path"]) != item["sha256"]:
      raise ValueError(f"Parent artifact hash mismatch: {item['path']}")


def frozen_pool_state(paths: dict[str, Path]) -> dict:
  """Active parent league members + u8223 + u9305; nothing else, never legacy."""
  parent = json.loads(paths["league"].read_text())
  policies = {pid: entry for pid, entry in parent["policies"].items() if entry["active"]}
  next_index = int(parent["next_policy_index"])
  for checkpoint, epoch in ((paths["model"], PARENT_UPDATE), (U9305, 9305)):
    policy_id = f"p{next_index:06d}"
    next_index += 1
    policies[policy_id] = {
      "active": True,
      "bucket": "recent",
      "checkpoint_path": str(checkpoint),
      "created_by_learner_id": None,
      "created_epoch": epoch,
      "created_ts": 0.0,
      "policy_id": policy_id,
      "rating": {"draws": 0, "elo": 1000.0, "games": 0, "losses": 0, "policy_id": policy_id, "wins": 0},
      "source": "generalist_anchor",
    }
  checkpoints = [Path(entry["checkpoint_path"]) for entry in policies.values()]
  missing = [str(path) for path in checkpoints if not path.is_file()]
  if missing:
    raise FileNotFoundError(f"Frozen pool checkpoints missing: {missing}")
  assert_recent_lineage(checkpoints)
  if len(policies) > sum(FROZEN_POOL_KEEP.values()):
    raise ValueError("Frozen pool exceeds keep budget; pruning would drop generalists")
  return {
    "champion_policy_id": None,
    "current_candidate_policy_id": None,
    "history": [{"event": "specialist_frozen_pool", "source_league_state": str(paths["league"])}],
    "learner_policy_id": None,
    "next_policy_index": next_index,
    "policies": policies,
    "version": 1,
  }


def saved_schedule(trainer_path: Path) -> dict:
  state = torch.load(trainer_path, map_location="cpu", weights_only=False, mmap=True)
  scheduler = state["scheduler_state_dict"]
  lrs = [float(group["lr"]) for group in state["optimizer_state_dict"]["param_groups"]]
  if lrs != scheduler["_last_lr"]:
    raise ValueError("Parent optimizer and scheduler learning rates disagree")
  return {
    "global_step": int(state["global_step"]),
    "update": int(state["update"]),
    "prebuilt_battle_decisions": int(state["prebuilt_battle_decisions"]),
    "optimizer_lrs": lrs,
    "scheduler": {key: scheduler[key] for key in ("T_max", "eta_min", "base_lrs", "last_epoch", "_last_lr")},
  }


def render_config(*, element: str, run_root: Path, deck_pool: Path, peak_lr: float, total_timesteps: int,
                  stage: str) -> str:
  template = (SPECIALIST_ROOT / "configs/specialist_template.ini").read_text()
  return template.format(
    element=element,
    run_root=run_root,
    deck_pool=deck_pool,
    peak_lr=repr(peak_lr),
    total_timesteps=total_timesteps,
    stage=stage,
    **FROZEN_POOL_KEEP,
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--element", choices=SPECIALIST_ELEMENTS, required=True)
  parser.add_argument("--run-root", type=Path)
  parser.add_argument("--deck-pool", type=Path, default=CURATED_POOL)
  parser.add_argument("--config-out", type=Path)
  parser.add_argument("--stage", default="s30m")
  args = parser.parse_args()
  run_root = (args.run_root or SPECIALIST_ROOT / "runs" / args.element).resolve()
  config_out = (args.config_out or SPECIALIST_ROOT / "configs" / f"specialist_{args.element}.ini").resolve()
  deck_pool = args.deck_pool.resolve()

  groups = load_specialist_deck_groups(deck_pool)
  learner_decks = specialist_learner_deck_indices(deck_pool, args.element)
  paths = parent_paths()
  verify_parent_atomic_manifest(paths)
  assert_recent_lineage([paths["model"]])
  schedule = saved_schedule(paths["trainer"])
  if schedule["update"] != PARENT_UPDATE:
    raise ValueError("Parent trainer state update mismatch")
  peak_lr = schedule["optimizer_lrs"][0]
  total_timesteps = (PARENT_UPDATE + SCHEDULE_UPDATES) * SAMPLED_ROWS_PER_UPDATE

  pool_state = frozen_pool_state(paths)
  league_dir = run_root / "league"
  save(league_dir / "league_state.json", pool_state)
  shutil.copyfile(PARENT_DIR / f"promotion_state_{PARENT_UPDATE:06d}.json", league_dir / "league_state_promotion.json")

  parent_manifest = {
    "schema_version": 2,
    "epoch": PARENT_UPDATE,
    "model": file_entry(paths["model"]),
    "metadata": file_entry(paths["metadata"]),
    "trainer": file_entry(paths["trainer"]),
    "source_atomic_manifest": file_entry(paths["atomic_manifest"]),
    "source_experiment": str(RESULTS),
  }
  save(run_root / "source_parent/parent_manifest.json", parent_manifest)
  inputs = [paths[key] for key in ("model", "metadata", "trainer", "league")]
  inputs += [Path(entry["checkpoint_path"]) for entry in pool_state["policies"].values()]
  inputs.append(deck_pool)
  save(run_root / "source_parent/input_manifest.json",
       {"schema_version": 1, "sha256": {str(path): digest(path) for path in dict.fromkeys(inputs)}})

  config_out.parent.mkdir(parents=True, exist_ok=True)
  config_out.write_text(render_config(
    element=args.element, run_root=run_root, deck_pool=deck_pool, peak_lr=peak_lr,
    total_timesteps=total_timesteps, stage=args.stage,
  ))
  save(run_root / "experiment.json", {
    "schema_id": "azuki.element_specialist",
    "schema_version": 1,
    "element": args.element,
    "runtime": str(RUNTIME),
    "config": str(config_out),
    "parent": {
      "checkpoint": str(paths["model"]),
      "checkpoint_sha256": parent_manifest["model"]["sha256"],
      "global_step": schedule["global_step"],
      "update": PARENT_UPDATE,
      "prebuilt_battle_decisions": schedule["prebuilt_battle_decisions"],
    },
    "fixed_recipe": {
      "prebuilt_probability": 0.8,
      "ordinary_draft_probability": 0.2,
      "terminal_rewards": [5.0, -5.0, 0.0],
      "reward_shaping": False,
      "gamma": 1.0,
      "frozen_ratio": 0.4,
      "sampled_rows_per_update": SAMPLED_ROWS_PER_UPDATE,
      "learner_element": args.element,
      "learner_seats": "every learner-controlled seat (both seats in current-vs-current) plays the element",
      "frozen_pool": "recent-lineage generalists only (parent active league + u8223 + u9305); no self-snapshots",
      "deck_pool": str(deck_pool),
      "deck_pool_sha256": digest(deck_pool),
      "prebuilt_groups": len(groups),
      "learner_prebuilt_decks_weighted": list(learner_decks),
    },
    "continuation": {"stage": args.stage, "additional_learner_steps": ADDITIONAL_LEARNER_STEPS},
    "lr_schedule": {
      "type": "CosineAnnealingLR restarted once from the parent's current optimizer LR",
      "peak_lr": peak_lr,
      "additional_updates": SCHEDULE_UPDATES,
      "train_total_timesteps": total_timesteps,
      "parent_saved_scheduler": schedule["scheduler"],
      "saved_scheduler": {"T_max": SCHEDULE_UPDATES, "eta_min": 0.0, "base_lrs": [peak_lr], "last_epoch": 0},
      "entropy": "ent_coef 0.01 (parent anneal starts at 150M global steps; not reached)",
    },
    "canary_gates": {
      "configured_probability": 0.8,
      "sampled_rows_per_update": SAMPLED_ROWS_PER_UPDATE,
      "learner_element_fraction": 1.0,
      # Pre-step old/new likelihood agreement. Bitwise 0 for water/earth/lightning;
      # fire reproducibly shows ~2.4e-5 (bf16 path noise), ~10x below one
      # optimizer step's KL. 1e-4 still catches real old/new policy mismatches.
      "zero_update_exact_kl_abs_max": 1e-04,
      "rollout_reference_kl_window_mean_max": 0.001,
      "sustained_windows": 3,
      "ppo_exact_kl_window_mean_limit": 0.02,
      "ppo_clip_fraction_window_limit": 0.2,
    },
  })
  print(json.dumps({"element": args.element, "config": str(config_out), "run_root": str(run_root),
                    "frozen_pool": len(pool_state["policies"]), "learner_decks": learner_decks,
                    "peak_lr": peak_lr, "total_timesteps": total_timesteps}))


if __name__ == "__main__":
  main()
