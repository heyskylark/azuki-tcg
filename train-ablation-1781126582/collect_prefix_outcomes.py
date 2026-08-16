#!/usr/bin/env python3
"""Collect exact terminal outcomes paired with ordered 50-card draft prefixes."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import azk_puffer.vector as azk_vector
import numpy as np

from deck_building import build_deck_build_catalog
from evaluate_checkpoint import _apply_checkpoint_resume_policy_config
from league_eval import MatchRequest, NativeLeagueEvaluator
from league_promotion import PromotionGameSpec
from league_promotion_store import checkpoint_sha256
from observation import DECKBUILD_OBSERVATION_CTYPE
from probe_gate_kl import GATE_CODE_PAIRS
from train import _load_model_weights
from training_deck_pool import load_training_deck_pool
from training_utils import build_policy, build_vecenv, load_training_config


DEFAULT_CONFIG_PATH = Path("python/config/azuki_deckbuild_native_3090.ini")
_UINT64_MASK = (1 << 64) - 1


def _priority(episode_seed: int, seat: int, salt: int) -> int:
  value = (
    (int(salt) & _UINT64_MASK)
    ^ (((int(episode_seed) + 1) * 0xD2B74407B1CE6E93) & _UINT64_MASK)
    ^ (((int(seat) + 1) * 0xCA5A826395121157) & _UINT64_MASK)
  )
  value ^= value >> 30
  value = (value * 0xBF58476D1CE4E5B9) & _UINT64_MASK
  value ^= value >> 27
  value = (value * 0x94D049BB133111EB) & _UINT64_MASK
  value ^= value >> 31
  return int(value)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
  return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
  return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _prefix_length(
  episode_seed: int,
  seat: int,
  salt: int,
  lengths: tuple[int, ...],
  probabilities: tuple[float, ...],
) -> int:
  unit = _priority(episode_seed, seat, salt) / float(1 << 64)
  cumulative = 0.0
  for length, probability in zip(lengths, probabilities):
    cumulative += probability
    if unit < cumulative:
      return int(length)
  return int(lengths[-1])


class PrefixBroadeningEvaluator(NativeLeagueEvaluator):
  def __init__(
    self,
    *,
    lengths: tuple[int, ...],
    probabilities: tuple[float, ...],
    seed: int,
  ):
    self.lengths = lengths
    self.probabilities = probabilities
    self.seed = int(seed)
    self.forced_actions = 0
    self.prefix_lengths: list[int] = []
    self._seen_games: set[str] = set()
    self._dtype = np.dtype(DECKBUILD_OBSERVATION_CTYPE)

  def _override_actions(
    self,
    *,
    env,
    running_envs: np.ndarray,
    seats: np.ndarray,
    row_indices: np.ndarray,
    candidate_role: np.ndarray,
    active_specs: list[PromotionGameSpec | None],
  ) -> None:
    observations = env.observations.view(self._dtype).reshape(-1)
    for local_index in np.nonzero(candidate_role)[0].tolist():
      env_index = int(running_envs[local_index])
      seat = int(seats[local_index])
      row_index = int(row_indices[local_index])
      spec = active_specs[env_index]
      if spec is None:
        continue
      length = _prefix_length(
        int(spec.seed), seat, self.seed, self.lengths, self.probabilities
      )
      if spec.game_id not in self._seen_games:
        self._seen_games.add(spec.game_id)
        self.prefix_lengths.append(length)
      observation = observations[row_index]
      deck_context = observation["deck_context"]
      main_count = int(deck_context["main_count"])
      if int(deck_context["mode"]) != 2 or main_count >= length:
        continue
      action_mask = observation["action_mask"]
      legal_count = int(action_mask["legal_action_count"])
      if legal_count < 1:
        raise RuntimeError("random prefix row has no legal actions")
      candidate = _priority(
        int(spec.seed),
        seat,
        self.seed ^ ((main_count + 1) * 0x9E3779B97F4A7C15),
      ) % legal_count
      env.actions[row_index] = np.asarray(
        [
          action_mask["legal_primary"][candidate],
          action_mask["legal_sub1"][candidate],
          action_mask["legal_sub2"][candidate],
          action_mask["legal_sub3"][candidate],
        ],
        dtype=np.int32,
      )
      self.forced_actions += 1


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


def _contexts(catalog) -> list[tuple[int, int, str]]:
  contexts: list[tuple[int, int, str]] = []
  for element, gate_codes in GATE_CODE_PAIRS.items():
    for gate_code in gate_codes:
      gate_id = int(catalog.records_by_code[gate_code].card_def_id)
      for leader_id_raw in catalog.leader_def_ids_by_element[element]:
        leader_id = int(leader_id_raw)
        leader_code = catalog.records_by_def_id[leader_id].card_code
        contexts.append((gate_id, leader_id, f"{element}:{gate_code}:{leader_code}"))
  if len(contexts) != 16:
    raise RuntimeError(f"expected 16 gate-leader contexts, found {len(contexts)}")
  return contexts


def _schedule(
  contexts: list[tuple[int, int, str]],
  *,
  games_per_context: int,
  base_seed: int,
  opponent_lineage: str,
) -> list[PromotionGameSpec]:
  if games_per_context < 2 or games_per_context % 2:
    raise ValueError("games_per_context must be a positive even integer")
  games: list[PromotionGameSpec] = []
  for context_index, (gate_id, leader_id, context_id) in enumerate(contexts):
    for block in range(games_per_context // 2):
      seed = int(base_seed + 100_003 * context_index + 7_919 * block)
      block_id = f"ctx{context_index:02d}:block{block:03d}"
      for candidate_seat in (0, 1):
        games.append(
          PromotionGameSpec(
            game_id=f"{block_id}:seat{candidate_seat}",
            block_id=block_id,
            phase="prefix_outcome_collection",
            opponent_id=opponent_lineage,
            seed=seed,
            candidate_seat=candidate_seat,
            gate0=gate_id,
            gate1=gate_id,
            leader0=leader_id,
            leader1=leader_id,
            schedule_version=f"prefix-outcome-v1:{context_id}",
          )
        )
  return games


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--opponent-checkpoint", type=Path, required=True)
  parser.add_argument("--policy-generation", required=True)
  parser.add_argument("--opponent-lineage", required=True)
  parser.add_argument("--games-per-context", type=int, default=12)
  parser.add_argument("--base-seed", type=int, required=True)
  parser.add_argument("--batch-envs", type=int, default=48)
  parser.add_argument("--max-steps", type=int, default=1200)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--random-prefix-lengths", default="0")
  parser.add_argument("--random-prefix-probabilities", default="1")
  parser.add_argument("--random-prefix-seed", type=int, default=420053)
  parser.add_argument("--jsonl", type=Path, required=True)
  parser.add_argument("--summary-json", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.batch_envs < 1 or args.max_steps < 1:
    raise ValueError("batch_envs and max_steps must be positive")
  if not 0 <= args.base_seed <= 0xFFFFFFFF:
    raise ValueError("base_seed must be a uint32 value")
  prefix_lengths = _parse_int_tuple(args.random_prefix_lengths)
  prefix_probabilities = _parse_float_tuple(args.random_prefix_probabilities)
  if not prefix_lengths or len(prefix_lengths) != len(prefix_probabilities):
    raise ValueError("random prefix lengths and probabilities must align")
  if any(length < 0 or length > 50 for length in prefix_lengths):
    raise ValueError("random prefix lengths must be in [0, 50]")
  if any(probability < 0.0 for probability in prefix_probabilities) or abs(
    sum(prefix_probabilities) - 1.0
  ) > 1e-9:
    raise ValueError("random prefix probabilities must be nonnegative and sum to one")

  trainer_args = load_training_config(args.config, [])
  _apply_checkpoint_resume_policy_config(trainer_args, args.checkpoint)
  trainer_args["train"]["device"] = args.device
  trainer_args["env"]["draft_uniform_assignment"] = True
  trainer_args["env"]["native_envs_per_instance"] = 1
  catalog = build_deck_build_catalog(
    load_training_deck_pool(trainer_args["env"].get("deck_pool_path"))
  )
  contexts = _contexts(catalog)
  schedule = _schedule(
    contexts,
    games_per_context=args.games_per_context,
    base_seed=args.base_seed,
    opponent_lineage=args.opponent_lineage,
  )

  vecenv = build_vecenv(
    trainer_args,
    backend=azk_vector.Serial,
    num_envs=1,
    seed=int(schedule[0].seed),
  )
  try:
    policy = _build_policy(args.checkpoint, trainer_args, vecenv, device=args.device)
    opponent = _build_policy(
      args.opponent_checkpoint, trainer_args, vecenv, device=args.device
    )
  finally:
    vecenv.close()

  evaluator = PrefixBroadeningEvaluator(
    lengths=prefix_lengths,
    probabilities=prefix_probabilities,
    seed=args.random_prefix_seed,
  )
  result = evaluator.evaluate_schedule(
    trainer_args,
    policy_a=policy,
    policy_b=opponent,
    request=MatchRequest(
      episodes=len(schedule),
      max_steps=args.max_steps,
      seed=int(schedule[0].seed),
      device=args.device,
      batch_envs=args.batch_envs,
    ),
    games=schedule,
  )
  raw_by_id = {str(record["game_id"]): record for record in result.raw_records}
  trajectories: list[dict[str, object]] = []
  dropped = 0
  for record in result.records:
    raw = raw_by_id[record.game_id]
    if not record.completed_normally:
      dropped += 1
      continue
    player = raw["players"][record.candidate_seat]
    opponent_player = raw["players"][1 - record.candidate_seat]
    main = [int(card_id) for card_id in player["main"]]
    if len(main) != 50 or any(card_id < 0 for card_id in main):
      raise RuntimeError(f"trajectory {record.game_id} does not contain 50 main cards")
    if record.winner_seat < 0:
      target = 0.5
    else:
      target = 1.0 if record.winner_seat == record.candidate_seat else 0.0
    trajectories.append(
      {
        "schema_version": 1,
        "trajectory_id": (
          f"{args.policy_generation}:{args.opponent_lineage}:{record.game_id}"
        ),
        "game_id": record.game_id,
        "block_id": record.block_id,
        "seed": int(record.seed),
        "world_seed": int(record.world_seed),
        "policy_generation": args.policy_generation,
        "opponent_lineage": args.opponent_lineage,
        "seat": int(record.candidate_seat),
        "starting_player": int(record.starting_player),
        "gate_id": int(player["gate"]),
        "leader_id": int(player["leader"]),
        "opponent_gate_id": int(opponent_player["gate"]),
        "opponent_leader_id": int(opponent_player["leader"]),
        "main_card_ids": main,
        "target": target,
        "end_reason": record.end_reason,
        "steps": int(record.steps),
        "forced_prefix_length": _prefix_length(
          int(record.seed),
          int(record.candidate_seat),
          args.random_prefix_seed,
          prefix_lengths,
          prefix_probabilities,
        ),
      }
    )

  args.jsonl.parent.mkdir(parents=True, exist_ok=True)
  with args.jsonl.open("w", encoding="utf-8") as handle:
    for trajectory in trajectories:
      handle.write(json.dumps(trajectory, separators=(",", ":")) + "\n")
  summary = {
    "schema_version": 1,
    "policy_generation": args.policy_generation,
    "opponent_lineage": args.opponent_lineage,
    "checkpoint": str(args.checkpoint.resolve()),
    "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
    "opponent_checkpoint": str(args.opponent_checkpoint.resolve()),
    "opponent_checkpoint_sha256": checkpoint_sha256(args.opponent_checkpoint),
    "scheduled_games": len(schedule),
    "complete_trajectories": len(trajectories),
    "dropped_incomplete": dropped,
    "wins": sum(float(row["target"]) == 1.0 for row in trajectories),
    "losses": sum(float(row["target"]) == 0.0 for row in trajectories),
    "draws": sum(float(row["target"]) == 0.5 for row in trajectories),
    "wall_time_seconds": result.wall_time_seconds,
    "timings": result.timings,
    "contexts": len(contexts),
    "games_per_context": args.games_per_context,
    "base_seed": args.base_seed,
    "random_prefix": {
      "lengths": list(prefix_lengths),
      "probabilities": list(prefix_probabilities),
      "seed": args.random_prefix_seed,
      "forced_actions": evaluator.forced_actions,
      "sampled_lengths": {
        str(length): evaluator.prefix_lengths.count(length) for length in prefix_lengths
      },
      "sampled_mean": (
        sum(evaluator.prefix_lengths) / len(evaluator.prefix_lengths)
        if evaluator.prefix_lengths
        else 0.0
      ),
    },
  }
  args.summary_json.parent.mkdir(parents=True, exist_ok=True)
  args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
  print(
    f"[prefix-outcome-collect] generation={args.policy_generation} "
    f"opponent={args.opponent_lineage} complete={len(trajectories)}/"
    f"{len(schedule)} wall={result.wall_time_seconds:.2f}s",
    flush=True,
  )


if __name__ == "__main__":
  main()
