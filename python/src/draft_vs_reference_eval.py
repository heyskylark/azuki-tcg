from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
from collections import Counter, defaultdict
from pathlib import Path

import azk_puffer.pytorch as azk_pytorch
import azk_puffer.vector as azk_vector
import numpy as np
import torch

from evaluate_checkpoint import (
  _apply_checkpoint_resume_policy_config,
  _unwrap_base_env,
)
from train import (
  _apply_sampler_anneal,
  _build_sampler_anneal_config,
  _load_model_weights,
  _peek_resume_global_step,
)
from training_utils import (
  DEFAULT_CONFIG_PATH,
  build_policy,
  build_vecenv,
  install_tcg_sampler,
  load_training_config,
)


_BEHAVIOR_INFO_KEYS = {
  "attack_rate": "azk_attack_selected_rate",
  "spell_rate": "azk_play_spell_from_hand_selected_rate",
  "weapon_rate": "azk_attach_weapon_from_hand_selected_rate",
  "portal_rate": "azk_gate_portal_selected_rate",
  "play_entity_rate": "azk_play_selected_rate",
  "noop_rate": "azk_noop_selected_rate",
  "episode_length": "azk_episode_length",
  "leader_health": "leader_health",
}


def multiset_jaccard(left: Counter[str], right: Counter[str]) -> float:
  keys = set(left).union(right)
  intersection = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
  union = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
  return float(intersection / union) if union else 0.0


def _reference_main_decks(base_env) -> tuple[list[Counter[str]], list[str]]:
  from deck_building import MAIN_CARD_TYPES
  from training_deck_pool import load_training_deck_labels

  records_by_code = base_env._catalog.records_by_code
  mains: list[Counter[str]] = []
  for deck in base_env._deck_pool:
    mains.append(
      Counter(
        {
          card_code: int(quantity)
          for card_code, quantity in deck
          if records_by_code[card_code].card_type in MAIN_CARD_TYPES
        }
      )
    )
  return mains, list(load_training_deck_labels())


def _mean(records: list[dict], key: str) -> float:
  values = [float(record[key]) for record in records if isinstance(record.get(key), (int, float))]
  return float(sum(values) / len(values)) if values else 0.0


def _percentile(records: list[dict], key: str, quantile: float) -> float:
  values = sorted(
    float(record[key]) for record in records if isinstance(record.get(key), (int, float))
  )
  if not values:
    return 0.0
  index = min(len(values) - 1, max(0, int(round((len(values) - 1) * quantile))))
  return values[index]


def run_eval(
  *,
  config_path: Path,
  checkpoint: Path | None,
  episodes: int,
  fixed_seat: int,
  device: str,
  seed: int,
  max_steps: int,
  argmax: bool,
):
  """One policy controls both seats; `fixed_seat` plays a sampled reference
  deck from the pool while the other seat drafts. Returns drafter stats."""
  trainer_args = load_training_config(config_path, [])
  trainer_args["train"]["device"] = device
  env_cfg = trainer_args.setdefault("env", {})
  env_cfg["deck_building_enabled"] = True
  # Seat forcing and per-seat info reads need the legacy wrapper chain.
  env_cfg["native"] = False
  env_cfg.pop("native_envs_per_instance", None)
  env_cfg["deck_building_fixed_seats"] = str(fixed_seat)
  _apply_checkpoint_resume_policy_config(trainer_args, checkpoint)
  env_cfg["deck_building_fixed_seats"] = str(fixed_seat)
  install_tcg_sampler()

  sampler_cfg = _build_sampler_anneal_config(
    trainer_args,
    total_timesteps=int(trainer_args.get("train", {}).get("total_timesteps", 0)),
  )
  checkpoint_step = _peek_resume_global_step(checkpoint, None) if checkpoint else None
  _apply_sampler_anneal(sampler_cfg, global_step=0 if checkpoint_step is None else int(checkpoint_step))
  if argmax:
    from policy.v2 import tcg_sampler

    tcg_sampler.set_sampling_params(subaction_temperature=1e-6, smoothing_eps=0.0)

  vecenv = build_vecenv(trainer_args, backend=azk_vector.Serial, num_envs=1, seed=seed)
  base_env = _unwrap_base_env(vecenv.envs[0])
  # Wrapper __getattr__ forwarding can satisfy the marker one level early;
  # walk until the class itself is the deck-building wrapper.
  while not getattr(type(base_env), "is_deck_building_wrapper", False) and hasattr(base_env, "env"):
    base_env = base_env.env
  policy = build_policy(vecenv, trainer_args)
  use_rnn = bool(trainer_args["train"].get("use_rnn", True))
  drafter_seat = 1 - int(fixed_seat)

  vecenv.async_reset(seed=seed)
  warm_obs, _, _, _, _, _, warm_masks = vecenv.recv()
  warm_state = {"mask": torch.as_tensor(warm_masks, device=device)}
  if use_rnn:
    warm_state["lstm_h"] = torch.zeros(vecenv.num_agents, policy.hidden_size, device=device)
    warm_state["lstm_c"] = torch.zeros(vecenv.num_agents, policy.hidden_size, device=device)
  with torch.no_grad():
    policy.forward_eval(torch.as_tensor(warm_obs, device=device), warm_state)

  if checkpoint is not None:
    _load_model_weights(policy, checkpoint, device=device, strict=False)
  policy.eval()

  records = []
  per_gate = defaultdict(lambda: {"games": 0, "wins": 0.0})
  catalog = base_env._catalog
  reference_mains, reference_labels = _reference_main_decks(base_env)
  train_reference_indices = tuple(range(0, len(reference_mains), 2))
  holdout_reference_indices = tuple(range(1, len(reference_mains), 2))

  try:
    for episode_idx in range(episodes):
      state = {}
      if use_rnn:
        state = {
          "lstm_h": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
          "lstm_c": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
        }
      vecenv.async_reset(seed=seed + 1000 * (episode_idx + 1))
      obs, _, _, _, _, _, masks = vecenv.recv()
      done = False
      steps = 0
      while not done and steps < max_steps:
        obs_tensor = torch.as_tensor(obs, device=device)
        step_state = {"mask": torch.as_tensor(masks, device=device)}
        if use_rnn:
          step_state["lstm_h"] = state["lstm_h"]
          step_state["lstm_c"] = state["lstm_c"]
        with torch.no_grad():
          logits, _ = policy.forward_eval(obs_tensor, step_state)
          actions, _, _ = azk_pytorch.sample_logits(logits)
        if use_rnn:
          state["lstm_h"] = step_state["lstm_h"]
          state["lstm_c"] = step_state["lstm_c"]
        vecenv.send(actions.cpu().numpy().astype(np.int32, copy=True))
        obs, _, _, _, _, _, masks = vecenv.recv()
        steps += 1
        done = vecenv.envs[0].done

      drafter_state = base_env._states[drafter_seat]
      gate_record = catalog.records_by_def_id.get(drafter_state.gate_card_def_id)
      gate_code = gate_record.card_code if gate_record else "?"
      info = base_env.infos.get(drafter_seat, {})
      win = float(info.get("win", 0.0) or 0.0)
      timeout = bool(steps >= max_steps and not done)
      main_ids = [
        int(card_id)
        for card_id in drafter_state.main_card_def_ids[: drafter_state.main_count]
        if int(card_id) >= 0
      ]
      main = Counter(catalog.records_by_def_id[card_id].card_code for card_id in main_ids)
      copy_histogram = Counter(main.values())
      train_similarities = {
        index: multiset_jaccard(main, reference_mains[index])
        for index in train_reference_indices
      }
      holdout_similarities = {
        index: multiset_jaccard(main, reference_mains[index])
        for index in holdout_reference_indices
      }
      nearest_train = max(train_similarities, key=train_similarities.get)
      nearest_holdout = max(holdout_similarities, key=holdout_similarities.get)
      record = {
        "gate": gate_code,
        "win": win,
        "steps": steps,
        "timeout": timeout,
        "main_unique": len(main),
        "avg_copies_per_unique": len(main_ids) / max(len(main), 1),
        "singleton_slot_share": copy_histogram.get(1, 0) / max(len(main_ids), 1),
        "quad_slot_share": 4 * copy_histogram.get(4, 0) / max(len(main_ids), 1),
        "nearest_train_reference_index": nearest_train,
        "nearest_train_reference_label": reference_labels[nearest_train],
        "nearest_train_reference_jaccard": train_similarities[nearest_train],
        "nearest_holdout_reference_index": nearest_holdout,
        "nearest_holdout_reference_label": reference_labels[nearest_holdout],
        "nearest_holdout_reference_jaccard": holdout_similarities[nearest_holdout],
      }
      for metric_name, info_key in _BEHAVIOR_INFO_KEYS.items():
        value = info.get(info_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
          record[metric_name] = float(value)
      ability_rate = 0.0
      for info_key in (
        "azk_activate_garden_or_leader_ability_selected_rate",
        "azk_activate_alley_ability_selected_rate",
      ):
        value = info.get(info_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
          ability_rate += float(value)
      record["ability_rate"] = ability_rate
      records.append(record)
      per_gate[gate_code]["games"] += 1
      per_gate[gate_code]["wins"] += win
  finally:
    vecenv.close()

  games = max(len(records), 1)
  nearest_train_labels = Counter(
    str(record["nearest_train_reference_label"])
    for record in records
    if "nearest_train_reference_label" in record
  )
  result = {
    "episodes": len(records),
    "drafter_seat": drafter_seat,
    "fixed_seat": int(fixed_seat),
    "drafter_win_rate": sum(r["win"] for r in records) / games,
    "timeout_rate": sum(1 for r in records if r["timeout"]) / games,
    "avg_steps": sum(r["steps"] for r in records) / games,
    "deck_metrics": {
      "main_unique_mean": _mean(records, "main_unique"),
      "avg_copies_per_unique_mean": _mean(records, "avg_copies_per_unique"),
      "singleton_slot_share_mean": _mean(records, "singleton_slot_share"),
      "quad_slot_share_mean": _mean(records, "quad_slot_share"),
      "nearest_train_reference_jaccard_mean": _mean(
        records, "nearest_train_reference_jaccard"
      ),
      "nearest_train_reference_jaccard_p90": _percentile(
        records, "nearest_train_reference_jaccard", 0.9
      ),
      "nearest_holdout_reference_jaccard_mean": _mean(
        records, "nearest_holdout_reference_jaccard"
      ),
      "nearest_holdout_reference_jaccard_p90": _percentile(
        records, "nearest_holdout_reference_jaccard", 0.9
      ),
      "largest_nearest_train_reference_share": (
        max(nearest_train_labels.values(), default=0) / games
      ),
      "nearest_train_reference_labels": dict(nearest_train_labels.most_common()),
    },
    "battle_metrics": {
      metric_name: _mean(records, metric_name)
      for metric_name in (*_BEHAVIOR_INFO_KEYS, "ability_rate", "steps")
    },
    "per_gate": {
      gate: {"games": stats["games"], "win_rate": stats["wins"] / max(stats["games"], 1)}
      for gate, stats in sorted(per_gate.items())
    },
  }
  return result


def _run_eval_job(kwargs):
  return run_eval(**kwargs)


def main():
  parser = argparse.ArgumentParser(description="Drafted-deck vs reference-deck evaluation.")
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--episodes", type=int, default=100)
  parser.add_argument("--fixed-seat", type=str, choices=["0", "1", "both"], default="both")
  parser.add_argument("--device", type=str, default="cuda")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--max-steps", type=int, default=500)
  parser.add_argument("--argmax", action="store_true")
  parser.add_argument(
    "--parallel-seats",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="evaluate fixed seats 0 and 1 concurrently when both are requested",
  )
  parser.add_argument("--json", type=Path, default=None)
  parser.add_argument(
    "--deck-indices", type=str, default=None,
    help="csv of pool indices the fixed seat may play (S4 holdout evals)")
  args = parser.parse_args()
  if args.deck_indices is not None:
    import os

    os.environ["AZK_FIXED_SEAT_DECK_INDICES"] = args.deck_indices

  seats = [0, 1] if args.fixed_seat == "both" else [int(args.fixed_seat)]
  jobs = []
  for seat in seats:
    jobs.append({
      "config_path": args.config,
      "checkpoint": args.checkpoint,
      "episodes": args.episodes // len(seats),
      "fixed_seat": seat,
      "device": args.device,
      "seed": args.seed + seat * 99991,
      "max_steps": args.max_steps,
      "argmax": args.argmax,
    })

  if len(jobs) == 2 and args.parallel_seats:
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
      max_workers=2,
      mp_context=context,
    ) as executor:
      results = list(executor.map(_run_eval_job, jobs))
  else:
    results = [_run_eval_job(job) for job in jobs]

  for result in results:
    print(json.dumps(result, indent=2))

  if len(results) == 2:
    total_games = sum(r["episodes"] for r in results)
    combined = sum(r["drafter_win_rate"] * r["episodes"] for r in results) / max(total_games, 1)
    print(f"combined drafter win rate (seat-fair): {combined:.4f} over {total_games} episodes")
  if args.json:
    args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
  main()
