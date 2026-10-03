"""Evaluate a per-seat element SpecialistEnsemble against frozen generalists.

Suites (both candidate seats, paired: the two seat assignments of a block
share one world seed):
  fixed       constructed vs constructed: candidate plays each held-out deck,
              opponent plays each held-out deck.
  free_draft  candidate drafts every gate/leader context; opponent plays each
              held-out deck.
Elements without a --specialist fall back to --fallback (u8223 by default).

Run: PYTHONPATH=build/python/src:python/src python python/src/specialist_ensemble_eval.py \
  --config <specialist .ini> --specialist water=<ckpt> --json out.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path

import azk_puffer.vector as azk_vector

from deck_building import build_deck_build_catalog
from league_eval import MatchRequest, NativeLeagueEvaluator
from league_promotion import PromotionGameSpec, _derived_seed, build_reference_schedule
from league_promotion_store import checkpoint_sha256
from native_policy_eval import _build_policy, _parse_seeds
from specialist import SPECIALIST_ELEMENTS, assert_recent_lineage, gate_element_codes, LEARNER_ELEMENT_CODES
from specialist_ensemble import SpecialistEnsemble
from training_deck_pool import load_training_deck_pool
from training_utils import build_vecenv, load_training_config

_ARTIFACTS = Path("/home/skylark/git/azuki-tcg/train-ablation-1781126582/results/control_strategy_scale_30m_v1/runs/control/artifacts")
U8223 = _ARTIFACTS / "azuki_local_control_strategy_scale_30m_s43_179038406361/model_azuki_local_008223.pt"
U9305 = _ARTIFACTS / "azuki_local_control_strategy_scale_30m_s43_179039793722/model_azuki_local_009305.pt"
_ELEMENT_BY_CODE = {code: name for name, code in LEARNER_ELEMENT_CODES.items()}


def _pairs(values: list[str], label: str) -> dict[str, Path]:
  out: dict[str, Path] = {}
  for value in values:
    key, separator, path = value.partition("=")
    if not separator or not key or not path:
      raise ValueError(f"{label} must be KEY=PATH, got {value!r}")
    out[key.strip().lower() if label == "--specialist" else key.strip()] = Path(path).expanduser()
  return out


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--specialist", action="append", default=[], help="ELEMENT=CHECKPOINT")
  parser.add_argument("--fallback", type=Path, default=U8223)
  parser.add_argument("--opponent", action="append", default=[], help="LABEL=CHECKPOINT (default u8223 and u9305)")
  parser.add_argument("--deck-pool", type=Path, help="pool holding the held-out decks (default: config env.deck_pool_path)")
  parser.add_argument("--heldout-deck-indices", default="", help="comma list (default: pool summary holdout_reference_deck_indices)")
  parser.add_argument("--suites", default="fixed,free_draft")
  parser.add_argument("--seeds", default="42001701,52001704")
  parser.add_argument("--max-games-per-suite", type=int, default=0, help="smoke cap per opponent and suite (0 = all)")
  parser.add_argument("--batch-envs", type=int, default=24)
  parser.add_argument("--max-steps", type=int, default=1200)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--json", type=Path, required=True)
  parser.add_argument("--candidate-elements", default="",
                      help="comma list; keep only games whose candidate seat plays these elements (default all)")
  return parser.parse_args()


def _fixed_schedule(opponent: str, decks: list[int], seeds: tuple[int, ...]) -> list[PromotionGameSpec]:
  games = []
  for seed_index, base_seed in enumerate(seeds):
    for candidate_position, candidate_deck in enumerate(decks):
      for opponent_position, opponent_deck in enumerate(decks):
        block_id = f"fixed:{opponent}:s{seed_index}:cand{candidate_deck}:opp{opponent_deck}"
        seed = _derived_seed(base_seed, candidate_position, 3, opponent_position)
        for candidate_seat in (0, 1):
          games.append(PromotionGameSpec(
            game_id=f"{block_id}:seat{candidate_seat}", block_id=block_id, phase="fixed",
            opponent_id=opponent, seed=seed, candidate_seat=candidate_seat, gate0=-1, gate1=-1,
            schedule_version="specialist-fixed-v1", reference_seat=1 - candidate_seat,
            reference_deck_index=opponent_deck, other_deck_index=candidate_deck,
          ))
  return games


def _free_draft_schedule(opponent: str, decks: list[int], seeds: tuple[int, ...], catalog) -> list[PromotionGameSpec]:
  # The native draft catalog only knows gates present in the deck pool; skip
  # promotion-schedule gate contexts the pool cannot draft.
  draftable = set(int(gate) for gate in catalog.gate_def_id_population)
  games = []
  for seed_index, base_seed in enumerate(seeds):
    for game in build_reference_schedule(
      decks, catalog.records_by_code, base_seed=base_seed, schedule_id=f"{opponent}:s{seed_index}",
      leader_ids_by_element=catalog.leader_def_ids_by_element,
    ):
      if game.gate0 in draftable and game.gate1 in draftable:
        games.append(replace(game, opponent_id=opponent, phase="free_draft"))
  return games


def _candidate_element(game: PromotionGameSpec, pool_gate_codes: list[str], catalog) -> str:
  if game.other_deck_index >= 0:
    gate = int(catalog.records_by_code[pool_gate_codes[game.other_deck_index]].card_def_id)
  else:
    gate = game.gate0 if game.candidate_seat == 0 else game.gate1
  return _ELEMENT_BY_CODE.get(int(gate_element_codes([gate])[0]), "unknown")


def _cap(games: list[PromotionGameSpec], limit: int) -> list[PromotionGameSpec]:
  if limit <= 0:
    return games
  blocks: dict[str, list[PromotionGameSpec]] = defaultdict(list)
  for game in games:
    blocks[game.block_id].append(game)
  # Spread the cap across blocks round-robin while keeping both seats of a block.
  ordered = list(blocks.values())
  step = max(1, len(ordered) // max(1, limit // 2))
  kept = [game for block in ordered[::step] for game in block]
  return kept[: limit + (limit % 2)]


def _score(rows: list[dict]) -> dict:
  if not rows:
    return {"games": 0}
  return {
    "games": len(rows),
    "score": sum(row["candidate_score"] for row in rows) / len(rows),
    "wins": sum(row["candidate_score"] == 1.0 for row in rows),
    "losses": sum(row["candidate_score"] == 0.0 for row in rows),
    "timeouts": sum(row["end_reason"] != "gameover" for row in rows),
  }


def main() -> None:
  args = _parse_args()
  seeds = _parse_seeds(args.seeds)
  specialists = _pairs(args.specialist, "--specialist")
  unknown = set(specialists) - set(SPECIALIST_ELEMENTS)
  if unknown:
    raise ValueError(f"Unknown specialist elements {sorted(unknown)}")
  opponents = _pairs(args.opponent, "--opponent") or {"u8223": U8223, "u9305": U9305}
  checkpoints = [args.fallback, *specialists.values(), *opponents.values()]
  assert_recent_lineage(checkpoints)

  trainer_args = load_training_config(args.config, [])
  trainer_args["train"]["device"] = args.device
  env = trainer_args["env"]
  if args.deck_pool is not None:
    env["deck_pool_path"] = str(args.deck_pool.resolve())
  env.update(prebuilt_curriculum=False, prebuilt_probability=0.0, learner_element="none", native_envs_per_instance=1)
  pool_path = Path(env["deck_pool_path"])
  pool = load_training_deck_pool(pool_path)
  catalog = build_deck_build_catalog(pool)
  if args.heldout_deck_indices:
    heldout = [int(part) for part in args.heldout_deck_indices.split(",") if part.strip()]
  else:
    heldout = list(json.loads(pool_path.read_text())["summary"]["holdout_reference_deck_indices"])

  vecenv = build_vecenv(trainer_args, backend=azk_vector.Serial, num_envs=1, seed=seeds[0])
  try:
    cache: dict[Path, object] = {}

    def load(path: Path):
      key = path.resolve()
      if key not in cache:
        cache[key] = _build_policy(key, trainer_args, vecenv, device=args.device)
      return cache[key]

    ensemble = SpecialistEnsemble(
      {element: (f"{element}:{path.name}", load(path)) for element, path in specialists.items()},
      (f"fallback:{args.fallback.name}", load(args.fallback)),
    ).to(args.device)
    opponent_policies = {label: load(path) for label, path in opponents.items()}
  finally:
    vecenv.close()

  suites = [suite.strip() for suite in args.suites.split(",") if suite.strip()]
  keep_elements = {part.strip().lower() for part in args.candidate_elements.split(",") if part.strip()}
  unknown_elements = keep_elements - set(SPECIALIST_ELEMENTS)
  if unknown_elements:
    raise ValueError(f"Unknown --candidate-elements {sorted(unknown_elements)}")
  pool_gate_codes = [deck["gate_card_id"] for deck in json.loads(pool_path.read_text())["decks"]]
  evaluator = NativeLeagueEvaluator()
  games_out: list[dict] = []
  wall = 0.0
  for label, opponent in opponent_policies.items():
    for suite in suites:
      if suite == "fixed":
        schedule = _fixed_schedule(label, heldout, seeds)
      elif suite == "free_draft":
        schedule = _free_draft_schedule(label, heldout, seeds, catalog)
      else:
        raise ValueError(f"Unknown suite {suite!r}")
      if keep_elements:
        schedule = [game for game in schedule if _candidate_element(game, pool_gate_codes, catalog) in keep_elements]
      schedule = _cap(schedule, args.max_games_per_suite)
      before = Counter(ensemble.routed_rows)
      result = evaluator.evaluate_schedule(
        trainer_args, policy_a=ensemble, policy_b=opponent,
        request=MatchRequest(episodes=len(schedule), max_steps=args.max_steps, seed=seeds[0],
                             device=args.device, batch_envs=args.batch_envs),
        games=schedule,
      )
      wall += result.wall_time_seconds
      routed = Counter(ensemble.routed_rows)
      routed.subtract(before)
      print(f"[ensemble-eval] {suite} vs {label}: games={len(result.records)} routed_rows={dict(+routed)}", flush=True)
      for record in result.records:
        element = _ELEMENT_BY_CODE.get(int(gate_element_codes([record.candidate_gate])[0]), "unknown")
        member = specialists.get(element)
        games_out.append({
          **record.to_dict(),
          "candidate_score": float(record.candidate_score),
          "suite": suite,
          "candidate_element": element,
          "candidate_gate_code": catalog.records_by_def_id[record.candidate_gate].card_code,
          "routed_to": f"{element}:{member.name}" if member is not None else f"fallback:{args.fallback.name}",
        })

  summary: dict = {"overall": _score(games_out)}
  for key in ("suite", "opponent_id", "candidate_element", "candidate_seat"):
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in games_out:
      groups[str(row[key])].append(row)
    summary[f"by_{key}"] = {name: _score(rows) for name, rows in sorted(groups.items())}
  payload = {
    "schema_version": 1,
    "evaluator_version": evaluator.evaluator_version,
    "policy_action_mode": "legal_argmax_stable_first",
    "deck_pool": str(pool_path),
    "heldout_deck_indices": heldout,
    "seeds": list(seeds),
    "specialists": {element: {"checkpoint": str(path), "sha256": checkpoint_sha256(path)} for element, path in specialists.items()},
    "fallback": {"checkpoint": str(args.fallback), "sha256": checkpoint_sha256(args.fallback)},
    "opponents": {label: {"checkpoint": str(path), "sha256": checkpoint_sha256(path)} for label, path in opponents.items()},
    "routing_rows": dict(ensemble.routed_rows),
    "wall_time_seconds": wall,
    "summary": summary,
    "games": games_out,
  }
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  print(f"[ensemble-eval] games={len(games_out)} score={summary['overall'].get('score')} wall={wall:.1f}s json={args.json}", flush=True)


if __name__ == "__main__":
  main()
