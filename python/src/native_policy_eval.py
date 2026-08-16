from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from dataclasses import replace
import json
from pathlib import Path

import azk_puffer.vector as azk_vector

from deck_building import build_deck_build_catalog
from evaluate_checkpoint import _apply_checkpoint_resume_policy_config
from league_eval import MatchRequest, NativeLeagueEvaluator
from league_promotion import (
  PromotionThresholds,
  build_panel_schedule,
  summarize_panel,
)
from league_promotion_store import checkpoint_sha256
from train import _load_model_weights
from training_deck_pool import load_training_deck_pool
from training_utils import DEFAULT_CONFIG_PATH, build_policy, build_vecenv, load_training_config


DEFAULT_SEEDS = "42001701,52001704,62001707,72001710,82001713,92001716"
BEHAVIOR_KEYS = (
  "attack_rate",
  "spell_rate",
  "weapon_rate",
  "portal_rate",
  "play_entity_rate",
  "noop_rate",
  "ability_rate",
  "leader_health",
)


def _parse_seeds(value: str) -> tuple[int, ...]:
  try:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
  except ValueError as exc:
    raise argparse.ArgumentTypeError("--seeds must be a comma-separated integer list") from exc
  if not seeds or len(set(seeds)) != len(seeds):
    raise argparse.ArgumentTypeError("--seeds must be non-empty and unique")
  if any(seed < 0 or seed > 0xFFFFFFFF for seed in seeds):
    raise argparse.ArgumentTypeError("--seeds must contain uint32 values")
  return seeds


def _mean(records: list[dict], key: str) -> float:
  values = [
    float(record[key])
    for record in records
    if isinstance(record.get(key), (int, float)) and not isinstance(record.get(key), bool)
  ]
  return float(sum(values) / len(values)) if values else 0.0


def _group_score(records: list[dict], key: str) -> dict[str, dict]:
  grouped: dict[str, list[dict]] = defaultdict(list)
  for record in records:
    grouped[str(record[key])].append(record)
  return {
    value: {
      "games": len(selected),
      "score": _mean(selected, "candidate_score"),
      "timeout_rate": sum(not bool(record["completed_normally"]) for record in selected)
      / len(selected),
    }
    for value, selected in sorted(grouped.items())
  }


def _build_policy(checkpoint: Path, trainer_args: dict, vecenv, *, device: str):
  policy_args = copy.deepcopy(trainer_args)
  _apply_checkpoint_resume_policy_config(policy_args, checkpoint)
  policy_args["train"]["device"] = device
  policy = build_policy(vecenv, policy_args)
  _load_model_weights(policy, checkpoint, device=device, strict=False)
  policy.eval()
  for parameter in policy.parameters():
    parameter.requires_grad_(False)
  return policy


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Deterministic native seat-and-gate-balanced policy head-to-head evaluation."
  )
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
  parser.add_argument("--checkpoint-a", type=Path, required=True)
  parser.add_argument("--checkpoint-b", type=Path, required=True)
  parser.add_argument("--label-a", default="candidate")
  parser.add_argument("--label-b", default="opponent")
  parser.add_argument("--seeds", default=DEFAULT_SEEDS)
  parser.add_argument("--batch-envs", type=int, default=12)
  parser.add_argument("--max-steps", type=int, default=600)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--json", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  seeds = _parse_seeds(args.seeds)
  if args.batch_envs < 1 or args.max_steps < 1:
    raise ValueError("--batch-envs and --max-steps must be positive")

  trainer_args = load_training_config(args.config, [])
  trainer_args["train"]["device"] = args.device
  _apply_checkpoint_resume_policy_config(trainer_args, args.checkpoint_a)
  trainer_args["env"]["native_envs_per_instance"] = 1
  pool_path = trainer_args.get("env", {}).get("deck_pool_path")
  pool = load_training_deck_pool(pool_path)
  catalog = build_deck_build_catalog(pool)

  vecenv = build_vecenv(
    trainer_args,
    backend=azk_vector.Serial,
    num_envs=1,
    seed=seeds[0],
  )
  try:
    policy_a = _build_policy(
      args.checkpoint_a, trainer_args, vecenv, device=args.device
    )
    if args.checkpoint_a.resolve() == args.checkpoint_b.resolve():
      policy_b = policy_a
    else:
      policy_b = _build_policy(
        args.checkpoint_b, trainer_args, vecenv, device=args.device
      )
  finally:
    vecenv.close()

  schedule = []
  for seed_index, seed in enumerate(seeds):
    seed_prefix = f"seed{seed_index:02d}"
    seed_schedule = build_panel_schedule(
      [args.label_b],
      catalog.records_by_code,
      base_seed=seed,
      include_confirmation=True,
    )
    schedule.extend(
      replace(
        game,
        game_id=f"{seed_prefix}:{game.game_id}",
        block_id=f"{seed_prefix}:{game.block_id}",
        schedule_version=f"{game.schedule_version}:{seed_prefix}",
      )
      for game in seed_schedule
    )

  evaluator = NativeLeagueEvaluator()
  result = evaluator.evaluate_schedule(
    trainer_args,
    policy_a=policy_a,
    policy_b=policy_b,
    request=MatchRequest(
      episodes=len(schedule),
      max_steps=args.max_steps,
      seed=seeds[0],
      device=args.device,
      batch_envs=args.batch_envs,
    ),
    games=schedule,
  )
  panel = summarize_panel(
    result.records,
    thresholds=PromotionThresholds(),
    bootstrap_seed=seeds[0] + 101,
  )
  raw_by_id = {str(record["game_id"]): record for record in result.raw_records}
  games = []
  for record in result.records:
    raw = raw_by_id[record.game_id]
    candidate = raw["players"][record.candidate_seat]
    gate = catalog.records_by_def_id[record.candidate_gate].card_code
    game = {
      **record.to_dict(),
      "candidate_gate_code": gate,
      "candidate_started": record.starting_player == record.candidate_seat,
      "completed_normally": record.completed_normally,
      "episode_length": float(raw["episode_length"]),
    }
    for key in BEHAVIOR_KEYS:
      value = candidate.get(key)
      if isinstance(value, (int, float)) and not isinstance(value, bool):
        game[key] = float(value)
    games.append(game)

  summary = {
    "episodes": len(games),
    "score": float(panel.pooled_score),
    "paired_lcb_80": float(panel.paired_lcb),
    "timeout_rate": float(panel.timeout_rate),
    "wins": int(result.wins_a),
    "losses": int(result.wins_b),
    "draws": int(result.draws),
    "by_candidate_gate": _group_score(games, "candidate_gate_code"),
    "by_candidate_seat": _group_score(games, "candidate_seat"),
    "by_candidate_started": _group_score(games, "candidate_started"),
    "battle_metrics": {
      **{f"{key}_mean": _mean(games, key) for key in BEHAVIOR_KEYS},
      "episode_length_mean": _mean(games, "episode_length"),
    },
  }
  payload = {
    "schema_version": 1,
    "evaluator_version": evaluator.evaluator_version,
    "policy_action_mode": "legal_argmax_stable_first",
    "candidate": {
      "label": args.label_a,
      "checkpoint": str(args.checkpoint_a.resolve()),
      "checkpoint_sha256": checkpoint_sha256(args.checkpoint_a),
    },
    "opponent": {
      "label": args.label_b,
      "checkpoint": str(args.checkpoint_b.resolve()),
      "checkpoint_sha256": checkpoint_sha256(args.checkpoint_b),
    },
    "seeds": list(seeds),
    "batch_envs": args.batch_envs,
    "max_steps": args.max_steps,
    "wall_time_seconds": result.wall_time_seconds,
    "timings": result.timings,
    "summary": summary,
    "games": games,
  }
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  print(
    f"[native-policy-eval] {args.label_a} vs {args.label_b} games={len(games)} "
    f"score={summary['score']:.4f} lcb80={summary['paired_lcb_80']:.4f} "
    f"timeout={summary['timeout_rate']:.4f} wall={result.wall_time_seconds:.2f}s",
    flush=True,
  )


if __name__ == "__main__":
  main()
