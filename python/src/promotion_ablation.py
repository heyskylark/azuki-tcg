from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import azk_puffer.vector as azk_vector
import numpy as np
import torch

from deck_building import build_deck_build_catalog
from evaluate_checkpoint import _apply_checkpoint_resume_policy_config
from league_eval import MatchRequest, NativeLeagueEvaluator
from league_promotion import (
  CONFIRMATION_PHASE,
  PromotionGameRecord,
  PromotionThresholds,
  build_panel_schedule,
  build_reference_schedule,
  compare_panel_records,
  compare_reference_records,
  decide_promotion,
  paired_bootstrap_lower_bound,
  summarize_panel,
)
from league_promotion_store import (
  checkpoint_sha256,
  deck_sha256,
  promotion_record_from_dict,
  write_immutable_json,
)
from train import _load_model_weights
from training_deck_pool import load_training_deck_pool
from training_utils import build_policy, build_vecenv, load_training_config


DEFAULT_MANIFEST = Path("train-ablation-1781126582/promotion-checkpoints-v1.json")
DEFAULT_OUTPUT = Path("train-ablation-1781126582/results/promotion_ablation_v1")


def _load_manifest(path: Path) -> dict:
  payload = json.loads(path.read_text(encoding="utf-8"))
  policies = payload.get("policies")
  if not isinstance(policies, list) or len(policies) < 5:
    raise ValueError("Promotion manifest must contain at least five policies")
  ids = [str(item.get("id", "")) for item in policies]
  if any(not policy_id for policy_id in ids) or len(set(ids)) != len(ids):
    raise ValueError("Promotion manifest policy ids must be non-empty and unique")
  if payload.get("anchor_id") not in ids:
    raise ValueError("Promotion manifest anchor_id is not in policies")
  for item in policies:
    path_value = Path(str(item["path"]))
    if not path_value.exists():
      raise FileNotFoundError(f"Historical checkpoint is missing: {path_value}")
  return payload


def _build_historical_policies(manifest: dict, *, device: str):
  policy_items = manifest["policies"]
  anchor_item = next(item for item in policy_items if item["id"] == manifest["anchor_id"])
  anchor_path = Path(anchor_item["path"])
  trainer_args = load_training_config(Path(manifest["config_path"]), [])
  trainer_args["train"]["device"] = device
  _apply_checkpoint_resume_policy_config(trainer_args, anchor_path)
  trainer_args["env"]["native_envs_per_instance"] = 1
  vecenv = build_vecenv(
    trainer_args,
    backend=azk_vector.Serial,
    num_envs=1,
    seed=int(manifest["schedule_seeds"][0]),
  )
  policies = {}
  hashes = {}
  try:
    for index, item in enumerate(policy_items, start=1):
      policy_id = str(item["id"])
      checkpoint = Path(item["path"])
      started = time.perf_counter()
      policy_args = copy.deepcopy(trainer_args)
      _apply_checkpoint_resume_policy_config(policy_args, checkpoint)
      policy_args["train"]["device"] = device
      policy = build_policy(vecenv, policy_args)
      _load_model_weights(policy, checkpoint, device=device, strict=False)
      policy.eval()
      for parameter in policy.parameters():
        parameter.requires_grad_(False)
      policies[policy_id] = policy
      hashes[policy_id] = checkpoint_sha256(checkpoint)
      print(
        f"[promotion-ablation] loaded {index}/{len(policy_items)} {policy_id} "
        f"in {time.perf_counter() - started:.2f}s",
        flush=True,
      )
  finally:
    vecenv.close()
  return trainer_args, policies, hashes


def _catalog(trainer_args: dict):
  pool_path = trainer_args.get("env", {}).get("deck_pool_path")
  pool = load_training_deck_pool(pool_path)
  return pool, build_deck_build_catalog(pool)


def _run_artifact(
  path: Path,
  *,
  kind: str,
  policy_a: str,
  policy_b: str,
  checkpoint_hashes: dict[str, str],
  seed: int,
  schedule,
  records,
  wall_seconds: float,
  evaluator_version: str,
  max_steps: int,
  batch_envs: int,
  timings: dict[str, float],
) -> None:
  write_immutable_json(
    path,
    {
      "schema_version": 1,
      "kind": kind,
      "policy_a": policy_a,
      "policy_b": policy_b,
      "policy_a_checkpoint_hash": checkpoint_hashes[policy_a],
      "policy_b_checkpoint_hash": checkpoint_hashes[policy_b],
      "seed": int(seed),
      "schedule": [game.to_dict() for game in schedule],
      "evaluator_version": evaluator_version,
      "policy_action_mode": "legal_argmax_stable_first",
      "max_steps": int(max_steps),
      "batch_envs": int(batch_envs),
      "wall_time_seconds": float(wall_seconds),
      "timings": {key: float(value) for key, value in timings.items()},
      "games": [record.to_dict() for record in records],
    },
  )


def run_replays(args) -> None:
  manifest = _load_manifest(args.manifest)
  output = args.output
  output.mkdir(parents=True, exist_ok=True)
  trainer_args, policies, hashes = _build_historical_policies(manifest, device=args.device)
  pool, catalog = _catalog(trainer_args)
  reference_indices = [int(index) for index in manifest["reference_deck_indices"]]
  holdout_indices = [int(index) for index in manifest.get("reference_holdout_deck_indices", [])]
  if reference_indices + holdout_indices and max(reference_indices + holdout_indices) >= len(pool):
    raise ValueError("Reference manifest contains an out-of-range deck index")
  matrix_max_steps = int(manifest.get("matrix_max_steps", args.max_steps))
  reference_max_steps = int(manifest.get("reference_max_steps", args.max_steps))
  replay_batch_envs = int(manifest.get("batch_envs", args.batch_envs))
  write_immutable_json(
    output / "resolved-manifest.json",
    {
      "schema_version": 1,
      "source_manifest": str(args.manifest.resolve()),
      "anchor_id": manifest["anchor_id"],
      "schedule_seeds": manifest["schedule_seeds"],
      "reference_seeds": manifest["reference_seeds"],
      "policies": [
        {**item, "checkpoint_hash": hashes[str(item["id"])]}
        for item in manifest["policies"]
      ],
      "reference_split": {
        "version": "reference-split-v1",
        "promotion": [
          {"deck_index": index, "deck_hash": deck_sha256(pool[index])}
          for index in reference_indices
        ],
        "holdout": [
          {"deck_index": index, "deck_hash": deck_sha256(pool[index])}
          for index in holdout_indices
        ],
      },
      "evaluator_version": NativeLeagueEvaluator.evaluator_version,
      "policy_action_mode": "legal_argmax_stable_first",
      "evaluation_parameters": {
        "batch_envs": replay_batch_envs,
        "matrix_max_steps": matrix_max_steps,
        "reference_max_steps": reference_max_steps,
      },
    },
  )
  evaluator = NativeLeagueEvaluator()
  policy_items = manifest["policies"]
  policy_ids = [str(item["id"]) for item in policy_items]
  if args.policy_ids:
    requested = [value.strip() for value in args.policy_ids.split(",") if value.strip()]
    unknown = sorted(set(requested) - set(policy_ids))
    if unknown:
      raise ValueError(f"Unknown --policy-ids: {unknown}")
    policy_ids = requested
  seeds = [int(seed) for seed in manifest["schedule_seeds"]]

  if args.stage in {"all", "matrix"}:
    for seed_index, seed in enumerate(seeds):
      pair_dir = output / "pairs" / f"seed{seed_index:02d}"
      for left_index, policy_a in enumerate(policy_ids):
        for policy_b in policy_ids[left_index:]:
          path = pair_dir / f"{policy_a}__vs__{policy_b}.json"
          if path.exists():
            continue
          schedule = build_panel_schedule(
            [policy_b],
            catalog.records_by_code,
            base_seed=seed,
            include_confirmation=True,
          )
          started = time.perf_counter()
          result = evaluator.evaluate_schedule(
            trainer_args,
            policy_a=policies[policy_a],
            policy_b=policies[policy_b],
            request=MatchRequest(
              episodes=len(schedule),
              max_steps=matrix_max_steps,
              seed=seed,
              device=args.device,
              batch_envs=replay_batch_envs,
            ),
            games=schedule,
          )
          wall = time.perf_counter() - started
          _run_artifact(
            path,
            kind="paired_policy_match",
            policy_a=policy_a,
            policy_b=policy_b,
            checkpoint_hashes=hashes,
            seed=seed,
            schedule=schedule,
            records=result.records,
            wall_seconds=wall,
            evaluator_version=evaluator.evaluator_version,
            max_steps=matrix_max_steps,
            batch_envs=replay_batch_envs,
            timings=result.timings,
          )
          print(
            f"[promotion-ablation] matrix seed={seed_index} {policy_a} vs {policy_b}: "
            f"score={result.score_a:.3f} games={result.episodes} wall={wall:.2f}s",
            flush=True,
          )

  if args.stage in {"all", "reference"}:
    anchor_id = str(manifest["anchor_id"])
    for seed_index, seed in enumerate(int(value) for value in manifest["reference_seeds"]):
      reference_dir = output / "reference" / f"seed{seed_index:02d}"
      schedule = build_reference_schedule(
        reference_indices,
        catalog.records_by_code,
        base_seed=seed,
      )
      for policy_id in policy_ids:
        path = reference_dir / f"{policy_id}.json"
        if path.exists():
          continue
        started = time.perf_counter()
        result = evaluator.evaluate_schedule(
          trainer_args,
          policy_a=policies[policy_id],
          policy_b=policies[anchor_id],
          request=MatchRequest(
            episodes=len(schedule),
            max_steps=reference_max_steps,
            seed=seed,
            device=args.device,
            batch_envs=replay_batch_envs,
          ),
          games=schedule,
        )
        wall = time.perf_counter() - started
        _run_artifact(
          path,
          kind="fixed_reference_match",
          policy_a=policy_id,
          policy_b=anchor_id,
          checkpoint_hashes=hashes,
          seed=seed,
          schedule=schedule,
          records=result.records,
          wall_seconds=wall,
          evaluator_version=evaluator.evaluator_version,
          max_steps=reference_max_steps,
          batch_envs=replay_batch_envs,
          timings=result.timings,
        )
        print(
          f"[promotion-ablation] reference seed={seed_index} {policy_id}: "
          f"score={result.score_a:.3f} games={result.episodes} wall={wall:.2f}s",
          flush=True,
        )

  if args.stage in {"all", "self-reference"}:
    for seed_index, seed in enumerate(int(value) for value in manifest["reference_seeds"]):
      reference_dir = output / "self_reference" / f"seed{seed_index:02d}"
      schedule = build_reference_schedule(
        reference_indices,
        catalog.records_by_code,
        base_seed=seed,
      )
      for policy_id in policy_ids:
        path = reference_dir / f"{policy_id}.json"
        if path.exists():
          continue
        started = time.perf_counter()
        result = evaluator.evaluate_schedule(
          trainer_args,
          policy_a=policies[policy_id],
          policy_b=policies[policy_id],
          request=MatchRequest(
            episodes=len(schedule),
            max_steps=reference_max_steps,
            seed=seed,
            device=args.device,
            batch_envs=replay_batch_envs,
          ),
          games=schedule,
        )
        wall = time.perf_counter() - started
        _run_artifact(
          path,
          kind="self_controlled_reference_match",
          policy_a=policy_id,
          policy_b=policy_id,
          checkpoint_hashes=hashes,
          seed=seed,
          schedule=schedule,
          records=result.records,
          wall_seconds=wall,
          evaluator_version=evaluator.evaluator_version,
          max_steps=reference_max_steps,
          batch_envs=replay_batch_envs,
          timings=result.timings,
        )
        print(
          f"[promotion-ablation] self-reference seed={seed_index} {policy_id}: "
          f"score={result.score_a:.3f} games={result.episodes} wall={wall:.2f}s",
          flush=True,
        )

  if args.stage in {"all", "holdout-self-reference"}:
    for seed_index, seed in enumerate(int(value) for value in manifest["reference_seeds"]):
      reference_dir = output / "holdout_self_reference" / f"seed{seed_index:02d}"
      schedule = build_reference_schedule(
        holdout_indices,
        catalog.records_by_code,
        base_seed=seed,
      )
      for policy_id in policy_ids:
        path = reference_dir / f"{policy_id}.json"
        if path.exists():
          continue
        started = time.perf_counter()
        result = evaluator.evaluate_schedule(
          trainer_args,
          policy_a=policies[policy_id],
          policy_b=policies[policy_id],
          request=MatchRequest(
            episodes=len(schedule),
            max_steps=reference_max_steps,
            seed=seed,
            device=args.device,
            batch_envs=replay_batch_envs,
          ),
          games=schedule,
        )
        wall = time.perf_counter() - started
        _run_artifact(
          path,
          kind="self_controlled_holdout_reference_match",
          policy_a=policy_id,
          policy_b=policy_id,
          checkpoint_hashes=hashes,
          seed=seed,
          schedule=schedule,
          records=result.records,
          wall_seconds=wall,
          evaluator_version=evaluator.evaluator_version,
          max_steps=reference_max_steps,
          batch_envs=replay_batch_envs,
          timings=result.timings,
        )
        print(
          f"[promotion-ablation] holdout self-reference seed={seed_index} {policy_id}: "
          f"score={result.score_a:.3f} games={result.episodes} wall={wall:.2f}s",
          flush=True,
        )


def _artifact_records(path: Path) -> tuple[dict, list[PromotionGameRecord]]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  return payload, [promotion_record_from_dict(item) for item in payload["games"]]


def _namespace_records(
  records: list[PromotionGameRecord],
  namespace: str,
) -> list[PromotionGameRecord]:
  return [
    PromotionGameRecord(
      **{
        **asdict(record),
        "game_id": f"{namespace}:{record.game_id}",
        "block_id": f"{namespace}:{record.block_id}",
        "schedule_version": f"{record.schedule_version}:{namespace}",
      }
    )
    for record in records
  ]


def _reorient_records(
  records: list[PromotionGameRecord],
  *,
  source_candidate: str,
  target_candidate: str,
  opponent_id: str,
) -> list[PromotionGameRecord]:
  if source_candidate == target_candidate:
    return [
      PromotionGameRecord(
        **{
          **asdict(record),
          "game_id": f"{target_candidate}:{opponent_id}:{record.game_id}",
          "block_id": f"{target_candidate}:{opponent_id}:{record.block_id}",
          "opponent_id": opponent_id,
        }
      )
      for record in records
    ]
  return [
    PromotionGameRecord(
      **{
        **asdict(record),
        "game_id": f"{target_candidate}:{opponent_id}:{record.game_id}:inverse",
        "block_id": f"{target_candidate}:{opponent_id}:{record.block_id}:inverse",
        "opponent_id": opponent_id,
        "candidate_seat": 1 - record.candidate_seat,
        "candidate_gate": record.opponent_gate,
        "opponent_gate": record.candidate_gate,
      }
    )
    for record in records
  ]


def _pair_records(output: Path, policy_a: str, policy_b: str, seed_index: int):
  left, right = sorted((policy_a, policy_b), key=lambda value: value)
  # Artifacts use manifest order, not lexical order.
  pair_dir = output / "pairs" / f"seed{seed_index:02d}"
  direct = pair_dir / f"{policy_a}__vs__{policy_b}.json"
  reverse = pair_dir / f"{policy_b}__vs__{policy_a}.json"
  path = direct if direct.exists() else reverse
  if not path.exists():
    raise FileNotFoundError(f"Missing pair artifact for {policy_a} vs {policy_b}: {pair_dir}")
  payload, records = _artifact_records(path)
  source = str(payload["policy_a"])
  return _reorient_records(
    records,
    source_candidate=source,
    target_candidate=policy_a,
    opponent_id=policy_b,
  )


def _payoff_matrix(manifest: dict, output: Path, seed_index: int) -> dict[str, dict[str, float]]:
  ids = [str(item["id"]) for item in manifest["policies"]]
  matrix: dict[str, dict[str, float]] = {policy_id: {} for policy_id in ids}
  for policy_a in ids:
    for policy_b in ids:
      records = _pair_records(output, policy_a, policy_b, seed_index)
      matrix[policy_a][policy_b] = float(np.mean([record.candidate_score for record in records]))
  return matrix


def _select_panel(manifest: dict, matrix: dict[str, dict[str, float]]) -> list[dict]:
  anchor = str(manifest["anchor_id"])
  items = {str(item["id"]): item for item in manifest["policies"]}
  eligible = [policy_id for policy_id, item in items.items() if bool(item.get("panel_eligible", True))]
  qualified = [policy_id for policy_id in eligible if bool(items[policy_id].get("qualified_prior", False))]
  selected = [{"policy_id": anchor, "role": "production_anchor", "evidence": {}}]
  selected_ids = {anchor}

  recent = max(
    (policy_id for policy_id in qualified if policy_id not in selected_ids),
    key=lambda policy_id: (int(items[policy_id].get("epoch", 0)), policy_id),
  )
  selected.append(
    {
      "policy_id": recent,
      "role": "recent_quality",
      "evidence": {"epoch": int(items[recent].get("epoch", 0))},
    }
  )
  selected_ids.add(recent)

  hardest = min(
    (policy_id for policy_id in eligible if policy_id not in selected_ids),
    key=lambda policy_id: (matrix[anchor][policy_id], policy_id),
  )
  selected.append(
    {
      "policy_id": hardest,
      "role": "hardest_retained",
      "evidence": {"anchor_score": matrix[anchor][hardest]},
    }
  )
  selected_ids.add(hardest)

  def distance(policy_id: str) -> float:
    distances = []
    for selected_id in selected_ids:
      vector_a = np.asarray([matrix[policy_id][other] for other in eligible], dtype=np.float64)
      vector_b = np.asarray([matrix[selected_id][other] for other in eligible], dtype=np.float64)
      distances.append(float(np.sqrt(np.mean((vector_a - vector_b) ** 2))))
    return min(distances)

  distinct = max(
    (policy_id for policy_id in qualified if policy_id not in selected_ids),
    key=lambda policy_id: (distance(policy_id), policy_id),
  )
  selected.append(
    {
      "policy_id": distinct,
      "role": "historically_distinct",
      "evidence": {"payoff_vector_distance": distance(distinct)},
    }
  )
  return selected


def _legacy_paired_pass(records_by_opponent: dict[str, list[PromotionGameRecord]], anchor: str) -> bool:
  anchor_records = records_by_opponent[anchor]
  wins = sum(record.candidate_score == 1.0 for record in anchor_records)
  games = len(anchor_records)
  if games < 16 or wins / games < 0.55:
    return False
  # Legacy z=1.0 lower bound, applied to paired data to isolate pairing (P1).
  p = wins / games
  z = 1.0
  lower = (p + z * z / (2 * games) - z * ((p * (1 - p) + z * z / (4 * games)) / games) ** 0.5) / (1 + z * z / games)
  if lower < 0.55:
    return False
  baselines = [policy_id for policy_id in records_by_opponent if policy_id != anchor][:2]
  for policy_id in baselines:
    values = records_by_opponent[policy_id]
    if len(values) < 8 or np.mean([record.candidate_score for record in values]) < 0.50:
      return False
  return True


def _p3_pass(summary, thresholds: PromotionThresholds) -> bool:
  return (
    summary.pooled_score >= thresholds.pooled_min
    and sum(score >= thresholds.opponent_quorum_min for score in summary.opponent_scores.values())
    >= thresholds.opponent_quorum_count
    and sum(score >= 0.5 for score in summary.opponent_scores.values())
    >= thresholds.opponent_break_even_count
    and min(summary.opponent_scores.values()) >= thresholds.opponent_floor
    and min(summary.seat_scores.values()) >= thresholds.seat_floor
    and summary.timeout_rate <= thresholds.max_timeout_rate
  )


def select_panel_replays(args) -> None:
  manifest = _load_manifest(args.manifest)
  output = args.output
  policy_ids = [str(item["id"]) for item in manifest["policies"]]
  matrices = [
    _payoff_matrix(manifest, output, index)
    for index in range(len(manifest["schedule_seeds"]))
  ]
  mean_matrix = {
    policy_a: {
      policy_b: float(np.mean([matrix[policy_a][policy_b] for matrix in matrices]))
      for policy_b in policy_ids
    }
    for policy_a in policy_ids
  }
  panel = _select_panel(manifest, mean_matrix)
  manifest_items = {str(item["id"]): item for item in manifest["policies"]}
  payload = {
    "schema_version": 1,
    "panel_version": 1,
    "anchor_id": str(manifest["anchor_id"]),
    "activated_epoch": 0,
    "schedule_seed": int(manifest["schedule_seeds"][0]),
    "selection_source": "paired-retrospective-v1",
    "members": [
      {
        "policy_id": item["policy_id"],
        "role": item["role"],
        "checkpoint_path": str(manifest_items[item["policy_id"]]["path"]),
        "checkpoint_hash": checkpoint_sha256(Path(manifest_items[item["policy_id"]]["path"])),
        "epoch": int(manifest_items[item["policy_id"]].get("epoch", 0)),
        "quality_qualified": bool(
          item["policy_id"] == manifest["anchor_id"]
          or manifest_items[item["policy_id"]].get("qualified_prior", False)
        ),
        "selection_evidence": item["evidence"],
      }
      for item in panel
    ],
  }
  write_immutable_json(output / "panel-v1.json", payload)
  print(json.dumps(payload, indent=2))


def analyze_replays(args) -> None:
  manifest = _load_manifest(args.manifest)
  output = args.output
  policy_ids = [str(item["id"]) for item in manifest["policies"]]
  matrices = [_payoff_matrix(manifest, output, index) for index in range(len(manifest["schedule_seeds"]))]
  mean_matrix = {
    policy_a: {
      policy_b: float(np.mean([matrix[policy_a][policy_b] for matrix in matrices]))
      for policy_b in policy_ids
    }
    for policy_a in policy_ids
  }
  panel = _select_panel(manifest, mean_matrix)
  panel_ids = [item["policy_id"] for item in panel]
  manifest_items = {str(item["id"]): item for item in manifest["policies"]}
  panel_manifest = {
    "schema_version": 1,
    "panel_version": 1,
    "anchor_id": str(manifest["anchor_id"]),
    "activated_epoch": 0,
    "schedule_seed": int(manifest["schedule_seeds"][0]),
    "selection_source": "paired-retrospective-v1",
    "members": [
      {
        "policy_id": item["policy_id"],
        "role": item["role"],
        "checkpoint_path": str(manifest_items[item["policy_id"]]["path"]),
        "checkpoint_hash": checkpoint_sha256(
          Path(manifest_items[item["policy_id"]]["path"])
        ),
        "epoch": int(manifest_items[item["policy_id"]].get("epoch", 0)),
        "quality_qualified": bool(
          item["policy_id"] == manifest["anchor_id"]
          or manifest_items[item["policy_id"]].get("qualified_prior", False)
        ),
        "selection_evidence": item["evidence"],
      }
      for item in panel
    ],
  }
  write_immutable_json(output / "panel-v1.json", panel_manifest)
  anchor_id = str(manifest["anchor_id"])
  thresholds = PromotionThresholds()
  decisions: dict[str, dict[str, dict]] = defaultdict(dict)

  for seed_index in range(len(manifest["schedule_seeds"])):
    anchor_panel_records = [
      record
      for opponent_id in panel_ids
      for record in _pair_records(output, anchor_id, opponent_id, seed_index)
    ]
    anchor_reference_path = (
      output / "self_reference" / f"seed{seed_index:02d}" / f"{anchor_id}.json"
    )
    _, anchor_reference = _artifact_records(anchor_reference_path)
    _, anchor_fixed_reference = _artifact_records(
      output / "reference" / f"seed{seed_index:02d}" / f"{anchor_id}.json"
    )
    for policy_id in policy_ids:
      records_by_opponent = {
        opponent_id: _pair_records(output, policy_id, opponent_id, seed_index)
        for opponent_id in panel_ids
      }
      panel_records = [record for records in records_by_opponent.values() for record in records]
      summary = summarize_panel(
        panel_records,
        thresholds=thresholds,
        bootstrap_seed=int(manifest["schedule_seeds"][seed_index]) + 31,
      )
      panel_comparison = compare_panel_records(
        panel_records,
        anchor_panel_records,
        confidence=thresholds.bootstrap_confidence,
        samples=thresholds.bootstrap_samples,
        seed=int(manifest["schedule_seeds"][seed_index]) + 37,
      )
      _, candidate_reference = _artifact_records(
        output / "self_reference" / f"seed{seed_index:02d}" / f"{policy_id}.json"
      )
      reference = compare_reference_records(
        candidate_reference,
        anchor_reference,
        confidence=thresholds.bootstrap_confidence,
        samples=thresholds.bootstrap_samples,
        seed=int(manifest["reference_seeds"][seed_index]) + 29,
      )
      p4 = decide_promotion(
        panel_records,
        thresholds=thresholds,
        reference=reference,
        require_reference=True,
        bootstrap_seed=int(manifest["schedule_seeds"][seed_index]) + 31,
        panel_comparison=panel_comparison,
      )
      _, candidate_fixed_reference = _artifact_records(
        output / "reference" / f"seed{seed_index:02d}" / f"{policy_id}.json"
      )
      fixed_anchor_reference = compare_reference_records(
        candidate_fixed_reference,
        anchor_fixed_reference,
        confidence=thresholds.bootstrap_confidence,
        samples=thresholds.bootstrap_samples,
        seed=int(manifest["reference_seeds"][seed_index]) + 41,
      )
      holdout_reference = None
      anchor_holdout_path = (
        output
        / "holdout_self_reference"
        / f"seed{seed_index:02d}"
        / f"{anchor_id}.json"
      )
      candidate_holdout_path = (
        output
        / "holdout_self_reference"
        / f"seed{seed_index:02d}"
        / f"{policy_id}.json"
      )
      if anchor_holdout_path.exists() and candidate_holdout_path.exists():
        _, anchor_holdout = _artifact_records(anchor_holdout_path)
        _, candidate_holdout = _artifact_records(candidate_holdout_path)
        holdout_reference = compare_reference_records(
          candidate_holdout,
          anchor_holdout,
          confidence=thresholds.bootstrap_confidence,
          samples=thresholds.bootstrap_samples,
          seed=int(manifest["reference_seeds"][seed_index]) + 43,
        )
      decisions[policy_id][f"seed{seed_index:02d}"] = {
        "p1": _legacy_paired_pass(records_by_opponent, anchor_id),
        "p2": summary.pooled_score >= thresholds.pooled_min,
        "p3": _p3_pass(summary, thresholds),
        "p4": p4.to_dict(),
        "fixed_anchor_reference": fixed_anchor_reference.to_dict(),
        "holdout_self_reference": (
          None if holdout_reference is None else holdout_reference.to_dict()
        ),
      }

  # Live v1 uses the first paired panel schedule and both fixed promotion
  # reference seeds. Keep the per-seed decisions above as the stability audit.
  live_inputs = {}
  for policy_id in policy_ids:
    anchor_reference_all: list[PromotionGameRecord] = []
    candidate_reference_all: list[PromotionGameRecord] = []
    anchor_fixed_reference_all: list[PromotionGameRecord] = []
    candidate_fixed_reference_all: list[PromotionGameRecord] = []
    anchor_holdout_all: list[PromotionGameRecord] = []
    candidate_holdout_all: list[PromotionGameRecord] = []
    holdout_complete = True
    for seed_index in range(len(manifest["reference_seeds"])):
      namespace = f"seed{seed_index:02d}"
      _, anchor_seed_records = _artifact_records(
        output / "self_reference" / namespace / f"{anchor_id}.json"
      )
      _, candidate_seed_records = _artifact_records(
        output / "self_reference" / namespace / f"{policy_id}.json"
      )
      anchor_reference_all.extend(_namespace_records(anchor_seed_records, namespace))
      candidate_reference_all.extend(_namespace_records(candidate_seed_records, namespace))

      _, anchor_fixed_seed_records = _artifact_records(
        output / "reference" / namespace / f"{anchor_id}.json"
      )
      _, candidate_fixed_seed_records = _artifact_records(
        output / "reference" / namespace / f"{policy_id}.json"
      )
      anchor_fixed_reference_all.extend(
        _namespace_records(anchor_fixed_seed_records, namespace)
      )
      candidate_fixed_reference_all.extend(
        _namespace_records(candidate_fixed_seed_records, namespace)
      )

      anchor_holdout_path = (
        output / "holdout_self_reference" / namespace / f"{anchor_id}.json"
      )
      candidate_holdout_path = (
        output / "holdout_self_reference" / namespace / f"{policy_id}.json"
      )
      if anchor_holdout_path.exists() and candidate_holdout_path.exists():
        _, anchor_holdout_seed = _artifact_records(anchor_holdout_path)
        _, candidate_holdout_seed = _artifact_records(candidate_holdout_path)
        anchor_holdout_all.extend(_namespace_records(anchor_holdout_seed, namespace))
        candidate_holdout_all.extend(_namespace_records(candidate_holdout_seed, namespace))
      else:
        holdout_complete = False

    combined_reference = compare_reference_records(
      candidate_reference_all,
      anchor_reference_all,
      confidence=thresholds.bootstrap_confidence,
      samples=thresholds.bootstrap_samples,
      seed=int(manifest["reference_seeds"][0]) + 101,
    )
    live_panel_records = [
      record
      for opponent_id in panel_ids
      for record in _pair_records(output, policy_id, opponent_id, 0)
    ]
    live_anchor_panel_records = [
      record
      for opponent_id in panel_ids
      for record in _pair_records(output, anchor_id, opponent_id, 0)
    ]
    live_panel_comparison = compare_panel_records(
      live_panel_records,
      live_anchor_panel_records,
      confidence=thresholds.bootstrap_confidence,
      samples=thresholds.bootstrap_samples,
      seed=int(manifest["schedule_seeds"][0]) + 37,
    )
    live_decision = decide_promotion(
      live_panel_records,
      thresholds=thresholds,
      reference=combined_reference,
      require_reference=True,
      bootstrap_seed=int(manifest["schedule_seeds"][0]) + 31,
      panel_comparison=live_panel_comparison,
    )
    live_inputs[policy_id] = (
      live_panel_records,
      live_panel_comparison,
      combined_reference,
    )
    combined_fixed_reference = compare_reference_records(
      candidate_fixed_reference_all,
      anchor_fixed_reference_all,
      confidence=thresholds.bootstrap_confidence,
      samples=thresholds.bootstrap_samples,
      seed=int(manifest["reference_seeds"][0]) + 103,
    )
    combined_holdout = None
    if holdout_complete:
      combined_holdout = compare_reference_records(
        candidate_holdout_all,
        anchor_holdout_all,
        confidence=thresholds.bootstrap_confidence,
        samples=thresholds.bootstrap_samples,
        seed=int(manifest["reference_seeds"][0]) + 107,
      )
    decisions[policy_id]["live_v1"] = {
      "p4": live_decision.to_dict(),
      "fixed_anchor_reference": combined_fixed_reference.to_dict(),
      "holdout_self_reference": (
        None if combined_holdout is None else combined_holdout.to_dict()
      ),
      "reference_seed_count": len(manifest["reference_seeds"]),
    }

  sensitivity = []
  for relative_pooled_min in (-0.05, -0.02, 0.0):
    for relative_opponent_floor in (-0.15, -0.10, -0.05):
      for relative_paired_lcb_min in (-0.08, -0.05, -0.02):
        for reference_noninferiority in (-0.20, -0.15, -0.10):
          tuned = PromotionThresholds(
            bootstrap_samples=2_000,
            reference_noninferiority=reference_noninferiority,
            anchored_relative_pooled_min=relative_pooled_min,
            anchored_relative_opponent_floor=relative_opponent_floor,
            anchored_relative_paired_lcb_min=relative_paired_lcb_min,
          )
          passes = {}
          for policy_id in policy_ids:
            per_seed = []
            for seed_index in range(len(manifest["schedule_seeds"])):
              panel_records = [
                record
                for opponent_id in panel_ids
                for record in _pair_records(output, policy_id, opponent_id, seed_index)
              ]
              anchor_panel_records = [
                record
                for opponent_id in panel_ids
                for record in _pair_records(output, anchor_id, opponent_id, seed_index)
              ]
              panel_comparison = compare_panel_records(
                panel_records,
                anchor_panel_records,
                confidence=tuned.bootstrap_confidence,
                samples=tuned.bootstrap_samples,
                seed=int(manifest["schedule_seeds"][seed_index]) + 37,
              )
              _, anchor_reference = _artifact_records(
                output
                / "self_reference"
                / f"seed{seed_index:02d}"
                / f"{anchor_id}.json"
              )
              _, candidate_reference = _artifact_records(
                output
                / "self_reference"
                / f"seed{seed_index:02d}"
                / f"{policy_id}.json"
              )
              reference = compare_reference_records(
                candidate_reference,
                anchor_reference,
                confidence=tuned.bootstrap_confidence,
                samples=tuned.bootstrap_samples,
                seed=int(manifest["reference_seeds"][seed_index]) + 29,
              )
              per_seed.append(
                decide_promotion(
                  panel_records,
                  thresholds=tuned,
                  reference=reference,
                  require_reference=True,
                  bootstrap_seed=int(manifest["schedule_seeds"][seed_index]) + 31,
                  panel_comparison=panel_comparison,
                ).admitted
              )
            passes[policy_id] = per_seed
          sensitivity.append(
            {
              "relative_pooled_min": relative_pooled_min,
              "relative_opponent_floor": relative_opponent_floor,
              "relative_paired_lcb_min": relative_paired_lcb_min,
              "reference_noninferiority": reference_noninferiority,
              "passes": passes,
            }
          )

  reference_floor_sensitivity = []
  for candidate_floor in (0.48, 0.50, 0.52):
    tuned = PromotionThresholds(
      bootstrap_samples=2_000,
      reference_candidate_floor=candidate_floor,
    )
    passes = {}
    for policy_id, (panel_records, panel_comparison, reference) in live_inputs.items():
      passes[policy_id] = decide_promotion(
        panel_records,
        thresholds=tuned,
        reference=reference,
        require_reference=True,
        bootstrap_seed=int(manifest["schedule_seeds"][0]) + 31,
        panel_comparison=panel_comparison,
      ).admitted
    reference_floor_sensitivity.append(
      {
        "reference_candidate_floor": candidate_floor,
        "passes": passes,
      }
    )

  expected = {
    str(item["id"]): bool(item["expected_p4"])
    for item in manifest["policies"]
    if "expected_p4" in item
  }
  sentinel = {
    policy_id: {
      "expected": expected_value,
      "live_v1_result": bool(decisions[policy_id]["live_v1"]["p4"]["admitted"]),
      "seed_results": [
        bool(decisions[policy_id][f"seed{index:02d}"]["p4"]["admitted"])
        for index in range(len(manifest["schedule_seeds"]))
      ],
    }
    for policy_id, expected_value in expected.items()
  }
  payload = {
    "schema_version": 1,
    "manifest": str(args.manifest),
    "panel": panel,
    "payoff_matrix_by_seed": matrices,
    "payoff_matrix_mean": mean_matrix,
    "decisions": dict(decisions),
    "sentinel": sentinel,
    "sensitivity": sensitivity,
    "reference_floor_sensitivity": reference_floor_sensitivity,
  }
  output_path = output / "analysis.json"
  output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
  print(f"[promotion-ablation] analysis written to {output_path}")
  print(json.dumps({"panel": panel, "sentinel": sentinel}, indent=2))


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Deterministic promotion retrospective runner")
  parser.add_argument("command", choices=("run", "select-panel", "analyze"))
  parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--batch-envs", type=int, default=12)
  parser.add_argument("--max-steps", type=int, default=400)
  parser.add_argument(
    "--policy-ids",
    default="",
    help="optional comma-separated policy subset for replay stages",
  )
  parser.add_argument(
    "--stage",
    choices=(
      "all",
      "matrix",
      "reference",
      "self-reference",
      "holdout-self-reference",
    ),
    default="all",
  )
  return parser


def main() -> None:
  args = _parser().parse_args()
  if args.command == "run":
    run_replays(args)
  elif args.command == "select-panel":
    select_panel_replays(args)
  else:
    analyze_replays(args)


if __name__ == "__main__":
  main()
