#!/usr/bin/env python3
"""Paired 16-context evaluation for the uniform gate/leader lifecycle."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

import azk_puffer.vector as azk_vector

from deck_building import MAIN_CARD_TYPES, build_deck_build_catalog
from evaluate_checkpoint import _apply_checkpoint_resume_policy_config
from league_eval import MatchRequest, NativeLeagueEvaluator
from league_promotion import PromotionGameSpec, PromotionThresholds, summarize_panel
from league_promotion_store import checkpoint_sha256
from probe_gate_kl import GATE_CODE_PAIRS
from train import _load_model_weights
from training_deck_pool import load_training_deck_pool
from training_utils import build_policy, build_vecenv, load_training_config


DEFAULT_CONFIG_PATH = Path("python/config/azuki_deckbuild_native_3090.ini")


BEHAVIOR_KEYS = (
  "attack_rate",
  "spell_rate",
  "weapon_rate",
  "portal_rate",
  "play_entity_rate",
  "noop_rate",
  "ability_rate",
  "garden_or_leader_ability_rate",
  "alley_ability_rate",
  "play_entity_to_garden_rate",
  "play_entity_to_alley_rate",
  "target_rate",
  "contextual_response_opportunities",
  "temporary_charge_realized",
  "temporary_attack_damage_realized",
  "generated_ikz_created",
  "generated_ikz_converted",
  "entity_damage_dealt",
  "entity_damage_taken",
  "leader_health",
)


def _mean(records: list[dict], key: str) -> float:
  values = [
    float(record[key])
    for record in records
    if isinstance(record.get(key), (int, float)) and not isinstance(record.get(key), bool)
  ]
  return float(sum(values) / len(values)) if values else 0.0


def _group_summary(records: list[dict], key: str) -> dict[str, dict]:
  groups: dict[str, list[dict]] = defaultdict(list)
  for record in records:
    groups[str(record[key])].append(record)
  return {
    value: {
      "games": len(selected),
      "score": _mean(selected, "candidate_score"),
      "timeout_rate": sum(not bool(record["completed_normally"]) for record in selected)
      / len(selected),
      "main_unique_mean": _mean(selected, "main_unique"),
      "avg_copies_per_unique_mean": _mean(selected, "avg_copies_per_unique"),
      **{f"{metric}_mean": _mean(selected, metric) for metric in BEHAVIOR_KEYS},
    }
    for value, selected in sorted(groups.items())
  }


def _policy_summary(records: list[dict]) -> dict[str, object]:
  return {
    "score": _mean(records, "candidate_score"),
    "by_context": _group_summary(records, "context_id"),
    "by_element": _group_summary(records, "element"),
    "by_gate": _group_summary(records, "gate_code"),
    "by_leader": _group_summary(records, "leader_code"),
    "by_seat": _group_summary(records, "candidate_seat"),
    "by_start": _group_summary(records, "candidate_started"),
    "deck_metrics": {
      key: _mean(records, key)
      for key in (
        "main_unique",
        "avg_copies_per_unique",
        "singleton_slot_share",
        "quad_slot_share",
        "max_copies",
        "entity_slots",
        "spell_slots",
        "weapon_slots",
        "mean_ikz_cost",
      )
    },
    "battle_metrics": {
      **{f"{key}_mean": _mean(records, key) for key in BEHAVIOR_KEYS},
      "episode_length_mean": _mean(records, "episode_length"),
    },
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


def _build_contexts(catalog) -> list[dict]:
  contexts: list[dict] = []
  for element, gate_pair in GATE_CODE_PAIRS.items():
    leader_ids = catalog.leader_def_ids_by_element[element]
    for gate_code in gate_pair:
      gate_id = int(catalog.records_by_code[gate_code].card_def_id)
      for leader_id_raw in leader_ids:
        leader_id = int(leader_id_raw)
        leader_code = catalog.records_by_def_id[leader_id].card_code
        contexts.append(
          {
            "element": element,
            "gate_code": gate_code,
            "gate_id": gate_id,
            "leader_code": leader_code,
            "leader_id": leader_id,
            "context_id": f"{element}:{gate_code}:{leader_code}",
          }
        )
  if len(contexts) != 16:
    raise RuntimeError(f"Expected 16 gate-leader contexts, found {len(contexts)}")
  return contexts


def _build_schedule(contexts: list[dict], games_per_context: int) -> list[PromotionGameSpec]:
  if games_per_context < 2 or games_per_context % 2:
    raise ValueError("--games-per-context must be a positive even integer")
  games: list[PromotionGameSpec] = []
  for context_index, context in enumerate(contexts):
    for block in range(games_per_context // 2):
      seed = 31_000_019 + 100_003 * context_index + 7_919 * block
      block_id = f"ctx{context_index:02d}:block{block:02d}"
      for candidate_seat in (0, 1):
        games.append(
          PromotionGameSpec(
            game_id=f"{block_id}:seat{candidate_seat}",
            block_id=block_id,
            phase="uniform_context",
            opponent_id="matched_control",
            seed=seed,
            candidate_seat=candidate_seat,
            gate0=int(context["gate_id"]),
            gate1=int(context["gate_id"]),
            leader0=int(context["leader_id"]),
            leader1=int(context["leader_id"]),
            schedule_version="uniform-context-v1",
          )
        )
  return games


def _markdown(payload: dict) -> str:
  summary = payload["summary"]
  lines = [
    "# Uniform Gate-Leader Context Panel",
    "",
    f"Candidate: `{payload['candidate']['label']}`",
    "",
    f"Control: `{payload['control']['label']}`",
    "",
    f"Paired direct score: **{summary['score']:.4f}** over "
    f"{summary['episodes']} games; timeout rate `{summary['timeout_rate']:.4f}`.",
    "",
    "| Context | Games | Score | Unique | Copies/unique | Attack | Spell | Portal | Garden play | Ability |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for context, values in summary["by_context"].items():
    lines.append(
      f"| {context} | {values['games']} | {values['score']:.3f} | "
      f"{values['main_unique_mean']:.2f} | {values['avg_copies_per_unique_mean']:.2f} | "
      f"{values['attack_rate_mean']:.3f} | {values['spell_rate_mean']:.3f} | "
      f"{values['portal_rate_mean']:.3f} | "
      f"{values['play_entity_to_garden_rate_mean']:.3f} | "
      f"{values['ability_rate_mean']:.3f} |"
    )
  lines.extend(
    [
      "",
      "## Aggregate Mechanics",
      "",
      "| Metric | Mean |",
      "| --- | ---: |",
    ]
  )
  for key, value in summary["battle_metrics"].items():
    lines.append(f"| {key} | {value:.6f} |")
  lines.append("")
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
  parser.add_argument("--checkpoint-a", type=Path, required=True)
  parser.add_argument("--checkpoint-b", type=Path, required=True)
  parser.add_argument("--label-a", default="candidate")
  parser.add_argument("--label-b", default="control")
  parser.add_argument("--games-per-context", type=int, default=24)
  parser.add_argument("--batch-envs", type=int, default=48)
  parser.add_argument("--max-steps", type=int, default=600)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--json", type=Path, required=True)
  parser.add_argument("--md", type=Path, required=True)
  args = parser.parse_args()

  trainer_args = load_training_config(args.config, [])
  _apply_checkpoint_resume_policy_config(trainer_args, args.checkpoint_a)
  trainer_args["train"]["device"] = args.device
  trainer_args["env"]["draft_uniform_assignment"] = True
  trainer_args["env"]["native_envs_per_instance"] = 1
  pool = load_training_deck_pool(trainer_args["env"].get("deck_pool_path"))
  catalog = build_deck_build_catalog(pool)
  contexts = _build_contexts(catalog)
  schedule = _build_schedule(contexts, args.games_per_context)
  context_by_prefix = {
    f"ctx{index:02d}": context for index, context in enumerate(contexts)
  }

  vecenv = build_vecenv(
    trainer_args,
    backend=azk_vector.Serial,
    num_envs=1,
    seed=int(schedule[0].seed),
  )
  try:
    policy_a = _build_policy(args.checkpoint_a, trainer_args, vecenv, device=args.device)
    if args.checkpoint_a.resolve() == args.checkpoint_b.resolve():
      policy_b = policy_a
    else:
      policy_b = _build_policy(args.checkpoint_b, trainer_args, vecenv, device=args.device)
  finally:
    vecenv.close()

  base_a = getattr(policy_a, "policy", policy_a)
  base_b = getattr(policy_b, "policy", policy_b)
  if not bool(getattr(base_a, "_native_layout", False)) or not bool(
    getattr(base_b, "_native_layout", False)
  ):
    raise RuntimeError("Uniform-context policies must use the packed native layout")
  print(
    f"[uniform-context] packed policy layout verified; games={len(schedule)}",
    flush=True,
  )

  evaluator = NativeLeagueEvaluator()
  result = evaluator.evaluate_schedule(
    trainer_args,
    policy_a=policy_a,
    policy_b=policy_b,
    request=MatchRequest(
      episodes=len(schedule),
      max_steps=args.max_steps,
      seed=int(schedule[0].seed),
      device=args.device,
      batch_envs=args.batch_envs,
    ),
    games=schedule,
  )
  panel = summarize_panel(
    result.records,
    thresholds=PromotionThresholds(),
    bootstrap_seed=int(schedule[0].seed) + 101,
  )
  raw_by_id = {str(record["game_id"]): record for record in result.raw_records}
  games: list[dict] = []
  control_games: list[dict] = []
  for record in result.records:
    raw = raw_by_id[record.game_id]
    context = context_by_prefix[record.block_id.split(":", 1)[0]]
    for policy_name, seat in (
      ("policy_a", record.candidate_seat),
      ("policy_b", 1 - record.candidate_seat),
    ):
      player = raw["players"][seat]
      counts = Counter(
        int(card_id) for card_id in player["main"] if int(card_id) >= 0
      )
      main_total = sum(counts.values())
      type_counts: Counter[str] = Counter()
      total_cost = 0
      for card_id, quantity in counts.items():
        card = catalog.records_by_def_id[card_id]
        if card.card_type in MAIN_CARD_TYPES:
          type_counts[card.card_type] += quantity
          total_cost += int(card.ikz_cost) * quantity
      game = {
        **record.to_dict(),
        "policy": policy_name,
        "candidate_seat": seat,
        "candidate_score": (
          float(record.candidate_score)
          if policy_name == "policy_a"
          else float(1.0 - record.candidate_score)
        ),
        "completed_normally": record.completed_normally,
        "element": context["element"],
        "gate_code": context["gate_code"],
        "leader_code": context["leader_code"],
        "context_id": context["context_id"],
        "candidate_started": record.starting_player == seat,
        "episode_length": float(raw["episode_length"]),
        "main_unique": len(counts),
        "avg_copies_per_unique": main_total / max(len(counts), 1),
        "singleton_slot_share": sum(qty for qty in counts.values() if qty == 1)
        / max(main_total, 1),
        "quad_slot_share": sum(qty for qty in counts.values() if qty == 4)
        / max(main_total, 1),
        "max_copies": max(counts.values(), default=0),
        "entity_slots": int(type_counts["ENTITY"]),
        "spell_slots": int(type_counts["SPELL"]),
        "weapon_slots": int(type_counts["WEAPON"]),
        "mean_ikz_cost": total_cost / max(main_total, 1),
        "main_signature": "|".join(
          f"{card_id}:{quantity}" for card_id, quantity in sorted(counts.items())
        ),
      }
      for key in BEHAVIOR_KEYS:
        value = player.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
          game[key] = float(value)
      if policy_name == "policy_a":
        games.append(game)
      else:
        control_games.append(game)

  policy_a_summary = _policy_summary(games)
  policy_b_summary = _policy_summary(control_games)
  summary = {
    "episodes": len(games),
    "score": float(panel.pooled_score),
    "paired_lcb_80": float(panel.paired_lcb),
    "timeout_rate": float(panel.timeout_rate),
    "wins": int(result.wins_a),
    "losses": int(result.wins_b),
    "draws": int(result.draws),
    **policy_a_summary,
    "policy_a": policy_a_summary,
    "policy_b": policy_b_summary,
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
    "control": {
      "label": args.label_b,
      "checkpoint": str(args.checkpoint_b.resolve()),
      "checkpoint_sha256": checkpoint_sha256(args.checkpoint_b),
    },
    "games_per_context": args.games_per_context,
    "contexts": contexts,
    "wall_time_seconds": result.wall_time_seconds,
    "timings": result.timings,
    "summary": summary,
    "games": games,
    "control_games": control_games,
  }
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  args.md.parent.mkdir(parents=True, exist_ok=True)
  args.md.write_text(_markdown(payload), encoding="utf-8")
  print(
    f"[uniform-context] {args.label_a} vs {args.label_b} games={len(games)} "
    f"score={summary['score']:.4f} timeout={summary['timeout_rate']:.4f} "
    f"wall={result.wall_time_seconds:.2f}s",
    flush=True,
  )


if __name__ == "__main__":
  main()
