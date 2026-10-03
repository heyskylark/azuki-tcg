from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import replace
import json
from pathlib import Path

import azk_puffer.vector as azk_vector

from deck_building import MAIN_CARD_TYPES, build_deck_build_catalog
from evaluate_checkpoint import _apply_checkpoint_resume_policy_config
from league_eval import MatchRequest, NativeLeagueEvaluator
from league_promotion import build_reference_schedule
from league_promotion_store import checkpoint_sha256
from train import _load_model_weights
from training_deck_pool import load_training_deck_labels, load_training_deck_pool
from training_utils import DEFAULT_CONFIG_PATH, build_policy, build_vecenv, load_training_config


DEFAULT_TRAIN_INDICES = "0,2,4,6,8,10,12,14,16"
DEFAULT_HOLDOUT_INDICES = "1,3,5,7,9,11,13,15,17"
DEFAULT_SEEDS = "42009919,52009922"
BEHAVIOR_KEYS = (
  "attack_rate",
  "spell_rate",
  "weapon_rate",
  "portal_rate",
  "play_entity_rate",
  "noop_rate",
  "ability_rate",
  "leader_health",
  "episode_length",
)


def assign_uniform_candidate_leaders(games, catalog, *, seed_index: int):
  """Assign valid leaders to both scheduled seats and pair candidate contexts."""
  leader_offset = int(seed_index) % 2
  assigned = []
  for game in games:
    candidate_gate = game.gate0 if game.candidate_seat == 0 else game.gate1
    gate_record = catalog.records_by_def_id[int(candidate_gate)]
    leaders = tuple(
      sorted(int(value) for value in catalog.leader_def_ids_by_element[gate_record.element])
    )
    if len(leaders) != 2:
      raise ValueError(
        f"Uniform reference evaluation requires two leaders for {gate_record.element}"
      )
    candidate_seat = int(game.candidate_seat)
    candidate_leader = leaders[(candidate_seat + leader_offset) % 2]
    reference_leader = leaders[((1 - candidate_seat) + leader_offset) % 2]
    assigned.append(
      replace(
        game,
        leader0=candidate_leader if candidate_seat == 0 else reference_leader,
        leader1=candidate_leader if candidate_seat == 1 else reference_leader,
        schedule_version=f"{game.schedule_version}:uniform-context-v1",
      )
    )
  return assigned


def _parse_int_tuple(value: str, *, label: str) -> tuple[int, ...]:
  try:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
  except ValueError as exc:
    raise argparse.ArgumentTypeError(f"{label} must be a comma-separated integer list") from exc
  if not parsed:
    raise argparse.ArgumentTypeError(f"{label} must not be empty")
  if len(set(parsed)) != len(parsed):
    raise argparse.ArgumentTypeError(f"{label} must not contain duplicates")
  if any(item < 0 for item in parsed):
    raise argparse.ArgumentTypeError(f"{label} must contain non-negative integers")
  return parsed


def multiset_jaccard(left: Counter[str], right: Counter[str]) -> float:
  keys = set(left).union(right)
  intersection = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
  union = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
  return float(intersection / union) if union else 0.0


def _mean(records: list[dict], key: str) -> float:
  values = [
    float(record[key])
    for record in records
    if isinstance(record.get(key), (int, float)) and not isinstance(record.get(key), bool)
  ]
  return float(sum(values) / len(values)) if values else 0.0


def _percentile(records: list[dict], key: str, quantile: float) -> float:
  values = sorted(
    float(record[key])
    for record in records
    if isinstance(record.get(key), (int, float)) and not isinstance(record.get(key), bool)
  )
  if not values:
    return 0.0
  index = min(len(values) - 1, max(0, int(round((len(values) - 1) * quantile))))
  return values[index]


def _reference_main_decks(pool, catalog) -> list[Counter[str]]:
  mains = []
  for deck in pool:
    mains.append(
      Counter(
        {
          card_code: int(quantity)
          for card_code, quantity in deck
          if catalog.records_by_code[card_code].card_type in MAIN_CARD_TYPES
        }
      )
    )
  return mains


def _main_from_def_ids(def_ids: list[int], catalog) -> Counter[str]:
  main: Counter[str] = Counter()
  for card_id_raw in def_ids:
    card_id = int(card_id_raw)
    if card_id < 0:
      continue
    record = catalog.records_by_def_id.get(card_id)
    if record is None:
      raise ValueError(f"Evaluation record contains unknown card definition id {card_id}")
    if record.card_type in MAIN_CARD_TYPES:
      main[record.card_code] += 1
  return main


def _nearest_reference(
  main: Counter[str],
  reference_mains: list[Counter[str]],
  reference_labels: tuple[str, ...],
  indices: tuple[int, ...],
) -> tuple[int, str, float]:
  similarities = {
    index: multiset_jaccard(main, reference_mains[index])
    for index in indices
  }
  nearest = max(similarities, key=similarities.get)
  return nearest, reference_labels[nearest], float(similarities[nearest])


def _group_score(records: list[dict], key: str) -> dict[str, dict]:
  groups: dict[str, list[dict]] = defaultdict(list)
  for record in records:
    groups[str(record[key])].append(record)
  out = {}
  for value, selected in sorted(groups.items()):
    games = len(selected)
    out[value] = {
      "games": games,
      "score": _mean(selected, "score"),
      "win_rate": sum(float(record["score"]) == 1.0 for record in selected) / games,
      "draw_rate": sum(float(record["score"]) == 0.5 for record in selected) / games,
      "timeout_rate": sum(not bool(record["completed_normally"]) for record in selected) / games,
    }
  return out


def _deck_summary(records: list[dict]) -> dict:
  if not records:
    return {}
  signatures = Counter(str(record["main_signature"]) for record in records)
  train_labels = Counter(str(record["nearest_train_reference_label"]) for record in records)
  holdout_labels = Counter(str(record["nearest_holdout_reference_label"]) for record in records)
  train_mean = _mean(records, "nearest_train_reference_jaccard")
  holdout_mean = _mean(records, "nearest_holdout_reference_jaccard")
  return {
    "main_unique_mean": _mean(records, "main_unique"),
    "avg_copies_per_unique_mean": _mean(records, "avg_copies_per_unique"),
    "singleton_slot_share_mean": _mean(records, "singleton_slot_share"),
    "quad_slot_share_mean": _mean(records, "quad_slot_share"),
    "distinct_main_decks": len(signatures),
    "dominant_main_deck_share": max(signatures.values()) / len(records),
    "nearest_train_reference_jaccard_mean": train_mean,
    "nearest_train_reference_jaccard_p90": _percentile(
      records, "nearest_train_reference_jaccard", 0.90
    ),
    "nearest_holdout_reference_jaccard_mean": holdout_mean,
    "nearest_holdout_reference_jaccard_p90": _percentile(
      records, "nearest_holdout_reference_jaccard", 0.90
    ),
    "train_holdout_similarity_gap": train_mean - holdout_mean,
    "largest_nearest_train_label_share": max(train_labels.values()) / len(records),
    "largest_nearest_train_label": train_labels.most_common(1)[0][0],
    "largest_nearest_holdout_label_share": max(holdout_labels.values()) / len(records),
    "largest_nearest_holdout_label": holdout_labels.most_common(1)[0][0],
    "by_candidate_gate": {
      gate: {
        "games": len(selected),
        "main_unique_mean": _mean(selected, "main_unique"),
        "nearest_train_reference_jaccard_mean": _mean(
          selected, "nearest_train_reference_jaccard"
        ),
        "nearest_holdout_reference_jaccard_mean": _mean(
          selected, "nearest_holdout_reference_jaccard"
        ),
        "distinct_main_decks": len(
          {str(record["main_signature"]) for record in selected}
        ),
      }
      for gate, selected in sorted(
        (
          (gate, [record for record in records if record["candidate_gate"] == gate])
          for gate in {str(record["candidate_gate"]) for record in records}
        )
      )
    },
  }


def summarize_native_reference_result(
  *,
  result,
  catalog,
  pool,
  reference_labels: tuple[str, ...],
  training_reference_indices: tuple[int, ...],
  holdout_reference_indices: tuple[int, ...],
) -> tuple[list[dict], dict]:
  if len(result.records) != len(result.raw_records):
    raise RuntimeError(
      f"Native evaluator record mismatch: {len(result.records)} promotion records vs "
      f"{len(result.raw_records)} raw records"
    )
  promotion_by_id = {record.game_id: record for record in result.records}
  reference_mains = _reference_main_decks(pool, catalog)
  games: list[dict] = []
  for raw in result.raw_records:
    game_id = str(raw["game_id"])
    promotion = promotion_by_id.get(game_id)
    if promotion is None:
      raise RuntimeError(f"Missing promotion record for native game {game_id}")
    candidate_seat = int(raw["candidate_seat"])
    players = raw.get("players")
    if not isinstance(players, list) or len(players) != 2:
      raise ValueError(f"Native game {game_id} has invalid player telemetry")
    candidate = players[candidate_seat]
    main = _main_from_def_ids(candidate["main"], catalog)
    main_total = sum(main.values())
    if main_total != 50:
      raise ValueError(f"Native game {game_id} candidate main deck has {main_total} cards")
    copy_histogram = Counter(main.values())
    train_index, train_label, train_jaccard = _nearest_reference(
      main,
      reference_mains,
      reference_labels,
      training_reference_indices,
    )
    holdout_index, holdout_label, holdout_jaccard = _nearest_reference(
      main,
      reference_mains,
      reference_labels,
      holdout_reference_indices,
    )
    gate_id = int(candidate["gate"])
    leader_id = int(candidate["leader"])
    gate_record = catalog.records_by_def_id.get(gate_id)
    leader_record = catalog.records_by_def_id.get(leader_id)
    if gate_record is None or leader_record is None:
      raise ValueError(f"Native game {game_id} has unknown gate or leader id")
    reference_index = int(raw["ref_deck_index"])
    game = {
      "game_id": game_id,
      "block_id": promotion.block_id,
      "schedule_version": promotion.schedule_version,
      "seed": int(raw["seed"]),
      "candidate_seat": candidate_seat,
      "starting_player": int(raw["starting_player"]),
      "candidate_started": int(raw["starting_player"]) == candidate_seat,
      "reference_deck_index": reference_index,
      "reference_deck_label": reference_labels[reference_index],
      "candidate_gate": gate_record.card_code,
      "candidate_leader": leader_record.card_code,
      "candidate_context": f"{gate_record.card_code}:{leader_record.card_code}",
      "score": float(promotion.candidate_score),
      "completed_normally": bool(promotion.completed_normally),
      "end_reason": promotion.end_reason,
      "steps": int(promotion.steps),
      "main_unique": len(main),
      "avg_copies_per_unique": main_total / max(len(main), 1),
      "singleton_slot_share": copy_histogram.get(1, 0) / main_total,
      "quad_slot_share": 4 * copy_histogram.get(4, 0) / main_total,
      "main_signature": "|".join(f"{code}:{quantity}" for code, quantity in sorted(main.items())),
      "nearest_train_reference_index": train_index,
      "nearest_train_reference_label": train_label,
      "nearest_train_reference_jaccard": train_jaccard,
      "nearest_holdout_reference_index": holdout_index,
      "nearest_holdout_reference_label": holdout_label,
      "nearest_holdout_reference_jaccard": holdout_jaccard,
      "episode_length": float(raw["episode_length"]),
    }
    for key in BEHAVIOR_KEYS:
      if key == "episode_length":
        continue
      value = candidate.get(key)
      if isinstance(value, (int, float)) and not isinstance(value, bool):
        game[key] = float(value)
    games.append(game)

  summary = {
    "episodes": len(games),
    "score": _mean(games, "score"),
    "win_rate": sum(float(game["score"]) == 1.0 for game in games) / max(len(games), 1),
    "draw_rate": sum(float(game["score"]) == 0.5 for game in games) / max(len(games), 1),
    "timeout_rate": sum(not bool(game["completed_normally"]) for game in games)
    / max(len(games), 1),
    "by_reference_deck": _group_score(games, "reference_deck_index"),
    "by_candidate_gate": _group_score(games, "candidate_gate"),
    "by_candidate_leader": _group_score(games, "candidate_leader"),
    "by_candidate_context": _group_score(games, "candidate_context"),
    "by_candidate_seat": _group_score(games, "candidate_seat"),
    "by_candidate_started": _group_score(games, "candidate_started"),
    "deck_metrics": _deck_summary(games),
    "battle_metrics": {f"{key}_mean": _mean(games, key) for key in BEHAVIOR_KEYS},
  }
  return games, summary


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
    description="Fast paired native evaluation of a drafting policy against fixed reference decks."
  )
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--opponent-checkpoint", type=Path, default=None)
  parser.add_argument("--candidate-label", default="candidate")
  parser.add_argument("--opponent-label", default="reference_pilot")
  parser.add_argument("--split", default="reference")
  parser.add_argument("--deck-indices", default=DEFAULT_TRAIN_INDICES)
  parser.add_argument("--training-reference-indices", default=DEFAULT_TRAIN_INDICES)
  parser.add_argument("--holdout-reference-indices", default=DEFAULT_HOLDOUT_INDICES)
  parser.add_argument("--seeds", default=DEFAULT_SEEDS)
  parser.add_argument("--batch-envs", type=int, default=12)
  parser.add_argument("--max-steps", type=int, default=600)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--uniform-assignment", action="store_true")
  parser.add_argument("--json", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  deck_indices = _parse_int_tuple(args.deck_indices, label="--deck-indices")
  train_indices = _parse_int_tuple(
    args.training_reference_indices, label="--training-reference-indices"
  )
  holdout_indices = _parse_int_tuple(
    args.holdout_reference_indices, label="--holdout-reference-indices"
  )
  seeds = _parse_int_tuple(args.seeds, label="--seeds")
  if args.batch_envs < 1 or args.max_steps < 1:
    raise ValueError("--batch-envs and --max-steps must be positive")

  trainer_args = load_training_config(args.config, [])
  trainer_args["train"]["device"] = args.device
  _apply_checkpoint_resume_policy_config(trainer_args, args.checkpoint)
  trainer_args["env"]["native_envs_per_instance"] = 1
  trainer_args["env"]["draft_uniform_assignment"] = bool(args.uniform_assignment)
  pool_path = trainer_args.get("env", {}).get("deck_pool_path")
  pool = load_training_deck_pool(pool_path)
  reference_labels = load_training_deck_labels(pool_path)
  all_indices = set(deck_indices).union(train_indices).union(holdout_indices)
  if max(all_indices) >= len(pool):
    raise ValueError(f"Reference deck index exceeds pool size {len(pool)}")
  catalog = build_deck_build_catalog(pool)

  vecenv = build_vecenv(
    trainer_args,
    backend=azk_vector.Serial,
    num_envs=1,
    seed=seeds[0],
  )
  try:
    candidate_policy = _build_policy(
      args.checkpoint, trainer_args, vecenv, device=args.device
    )
    opponent_checkpoint = args.opponent_checkpoint or args.checkpoint
    if opponent_checkpoint.resolve() == args.checkpoint.resolve():
      opponent_policy = candidate_policy
    else:
      opponent_policy = _build_policy(
        opponent_checkpoint, trainer_args, vecenv, device=args.device
      )
  finally:
    vecenv.close()

  schedule = []
  for seed_index, seed in enumerate(seeds):
    seed_schedule = build_reference_schedule(
      deck_indices,
      catalog.records_by_code,
      base_seed=seed,
      schedule_id=f"{args.split}-seed{seed_index:02d}",
    )
    if args.uniform_assignment:
      seed_schedule = assign_uniform_candidate_leaders(
        seed_schedule,
        catalog,
        seed_index=seed_index,
      )
    schedule.extend(seed_schedule)
  evaluator = NativeLeagueEvaluator()
  result = evaluator.evaluate_schedule(
    trainer_args,
    policy_a=candidate_policy,
    policy_b=opponent_policy,
    request=MatchRequest(
      episodes=len(schedule),
      max_steps=args.max_steps,
      seed=seeds[0],
      device=args.device,
      batch_envs=args.batch_envs,
    ),
    games=schedule,
  )
  games, summary = summarize_native_reference_result(
    result=result,
    catalog=catalog,
    pool=pool,
    reference_labels=reference_labels,
    training_reference_indices=train_indices,
    holdout_reference_indices=holdout_indices,
  )
  payload = {
    "schema_version": 1,
    "evaluator_version": evaluator.evaluator_version,
    "policy_action_mode": "legal_argmax_stable_first",
    "assignment_contract": (
      "uniform_gate_same_element_leader"
      if args.uniform_assignment
      else "policy_selects_leader"
    ),
    "split": args.split,
    "candidate": {
      "label": args.candidate_label,
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
    },
    "reference_pilot": {
      "label": args.opponent_label,
      "checkpoint": str(opponent_checkpoint.resolve()),
      "checkpoint_sha256": checkpoint_sha256(opponent_checkpoint),
    },
    "deck_indices": list(deck_indices),
    "training_reference_indices": list(train_indices),
    "holdout_reference_indices": list(holdout_indices),
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
    f"[native-reference-eval] split={args.split} games={result.episodes} "
    f"score={summary['score']:.4f} timeout={summary['timeout_rate']:.4f} "
    f"wall={result.wall_time_seconds:.2f}s output={args.json}",
    flush=True,
  )


if __name__ == "__main__":
  main()
