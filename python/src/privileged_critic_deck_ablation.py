from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import azk_puffer.trainer as pufferl
import numpy as np
import torch

from critic_eval import collect_dataset, evaluate_dataset
from train import (
  _apply_ent_coef_anneal,
  _apply_sampler_anneal,
  _build_ent_coef_anneal_config,
  _build_sampler_anneal_config,
  _checkpoint_parity_guard,
  _compute_runtime_fingerprint,
  _resume_config_fingerprint,
  _save_checkpoint_metadata,
  _save_per_checkpoint_trainer_state,
)
from training_utils import (
  build_policy,
  build_vecenv,
  install_tcg_sampler,
  load_training_config,
)


PRIMARY_ACTION_NAMES = {
  0: "noop",
  1: "play_entity_to_garden",
  2: "play_entity_to_alley",
  6: "attack",
  7: "attach_weapon_from_hand",
  8: "play_spell_from_hand",
  9: "declare_defender",
  10: "gate_portal",
  11: "activate_garden_or_leader_ability",
  12: "activate_alley_ability",
  13: "select_cost_target",
  14: "select_effect_target",
  16: "confirm_ability",
  18: "select_from_selection",
  19: "bottom_deck_card",
  20: "bottom_deck_all",
  21: "select_to_alley",
  22: "select_to_equip",
  23: "select_to_garden",
  24: "top_deck_card",
  25: "mulligan_shuffle",
}

TRAINING_TAIL_KEYS = (
  "SPS",
  "agent_steps",
  "epoch",
  "learning_rate",
  "losses/value_loss",
  "losses/explained_variance",
  "losses/win_prob_aux_accuracy",
  "losses/win_prob_aux_brier",
  "environment/0/azk_noop_selected_rate",
  "environment/1/azk_noop_selected_rate",
  "environment/0/azk_zero_legal_action_truncation",
  "environment/1/azk_zero_legal_action_truncation",
)


class CaptureLogger:
  def __init__(self, run_id: str):
    self.run_id = run_id
    self.records: list[dict[str, Any]] = []

  def log(self, logs, step):
    record = {"step": int(step)}
    record.update(_json_safe(logs))
    self.records.append(record)

  def close(self, model_path):
    return None


def _json_safe(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(key): _json_safe(inner) for key, inner in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(inner) for inner in value]
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, np.generic):
    return value.item()
  if torch.is_tensor(value):
    if value.numel() == 1:
      return value.detach().cpu().item()
    return value.detach().cpu().tolist()
  return value


def _effective_batch_size(train_cfg: dict[str, Any], vecenv) -> int:
  configured_batch_size = train_cfg.get("batch_size", "auto")
  configured_bptt = train_cfg.get("bptt_horizon", "auto")
  if configured_batch_size == "auto":
    if configured_bptt == "auto":
      raise ValueError("Both train.batch_size and train.bptt_horizon are auto; cannot infer epoch size")
    return int(vecenv.num_agents * int(configured_bptt))
  return int(configured_batch_size)


def _summarize_training_tail(log_records: list[dict[str, Any]]) -> dict[str, Any]:
  if not log_records:
    return {}
  last = log_records[-1]
  return {
    key: last[key]
    for key in TRAINING_TAIL_KEYS
    if key in last
  }


def _summarize_action_ratios(dataset_path: Path) -> dict[str, Any]:
  payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
  if not isinstance(payload, dict):
    raise ValueError(f"Unexpected dataset payload in {dataset_path}")

  counts: Counter[int] = Counter()
  total_actions = 0
  for episode in payload.get("episodes", []):
    for action in episode.get("actions", []):
      primary = int(action[0])
      counts[primary] += 1
      total_actions += 1

  top_actions = []
  for primary, count in counts.most_common():
    top_actions.append(
      {
        "primary_action": int(primary),
        "name": PRIMARY_ACTION_NAMES.get(primary, f"action_{primary}"),
        "count": int(count),
        "ratio": float(count / total_actions) if total_actions > 0 else 0.0,
      }
    )

  return {
    "dataset_path": str(dataset_path),
    "episodes": int(payload.get("episode_count", 0)),
    "timeouts": int(payload.get("timeout_count", 0)),
    "total_actions": int(total_actions),
    "noop_ratio": float(counts.get(0, 0) / total_actions) if total_actions > 0 else 0.0,
    "top_primary_actions": top_actions,
  }


def _find_result_for_checkpoint(payload: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
  checkpoint_resolved = str(checkpoint.resolve())
  for result in payload.get("results", []):
    if result.get("checkpoint") == checkpoint_resolved:
      return result
  raise KeyError(f"Checkpoint result not found for {checkpoint_resolved}")


def _build_variant_overrides(args: argparse.Namespace, encoder_type: str) -> list[str]:
  return [
    "--vec.num_envs", str(args.num_envs),
    "--vec.num_workers", str(args.num_workers),
    "--vec.batch_size", str(args.vec_batch_size),
    "--train.total_timesteps", str(args.total_timesteps),
    "--train.minibatch_size", str(args.minibatch_size),
    "--train.max_minibatch_size", str(args.max_minibatch_size),
    "--train.device", args.device,
    "--policy.privileged_critic_enabled", "true",
    "--policy.privileged_critic_deck_encoder_type", encoder_type,
    "--league.enable", "false",
    "--wandb", "false",
    "--neptune", "false",
  ]


def _train_variant(
  *,
  config_path: Path,
  output_dir: Path,
  label: str,
  encoder_type: str,
  seed: int,
  forwarded_cli: list[str],
) -> dict[str, Any]:
  install_tcg_sampler()

  trainer_args = load_training_config(config_path, forwarded_cli)
  trainer_args["wandb"] = False
  trainer_args["neptune"] = False
  trainer_args.setdefault("league", {})["enable"] = False
  trainer_args.setdefault("policy", {})["privileged_critic_enabled"] = True
  trainer_args["policy"]["privileged_critic_deck_encoder_type"] = encoder_type
  trainer_args.setdefault("train", {})["device"] = str(trainer_args["train"].get("device", "cpu"))
  trainer_args["train"]["seed"] = int(seed)
  trainer_args["train"]["data_dir"] = str(output_dir / "training")
  trainer_args["train"]["env"] = str(trainer_args.get("env_name", "azuki_local"))
  trainer_args.setdefault("vec", {})["seed"] = int(seed)

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  vecenv = build_vecenv(trainer_args, seed=seed)
  runtime_fingerprint = _compute_runtime_fingerprint(vecenv)
  resume_config_fingerprint = _resume_config_fingerprint(trainer_args)

  train_cfg = trainer_args["train"]
  effective_batch_size = _effective_batch_size(train_cfg, vecenv)
  requested_total_timesteps = int(train_cfg.get("total_timesteps", 0))
  remainder = requested_total_timesteps % effective_batch_size
  if remainder != 0:
    aligned_total_timesteps = requested_total_timesteps + (effective_batch_size - remainder)
    train_cfg["total_timesteps"] = aligned_total_timesteps
  else:
    aligned_total_timesteps = requested_total_timesteps

  configured_minibatch = int(train_cfg.get("minibatch_size", effective_batch_size))
  if configured_minibatch > effective_batch_size:
    train_cfg["minibatch_size"] = effective_batch_size

  configured_max_minibatch = int(train_cfg.get("max_minibatch_size", effective_batch_size))
  if configured_max_minibatch < int(train_cfg["minibatch_size"]):
    train_cfg["max_minibatch_size"] = int(train_cfg["minibatch_size"])

  run_id = f"{label}_{int(time.time() * 1000)}"
  logger = CaptureLogger(run_id)
  policy = build_policy(vecenv, trainer_args)
  trainer = pufferl.PuffeRL(train_cfg, vecenv, policy, logger=logger)

  last_checkpoint_path: Path | None = None
  original_save_checkpoint = trainer.save_checkpoint

  def _save_checkpoint_with_metadata():
    nonlocal last_checkpoint_path
    checkpoint_raw = original_save_checkpoint()
    if not checkpoint_raw:
      return checkpoint_raw
    checkpoint_path = Path(checkpoint_raw)
    parity_summary = _checkpoint_parity_guard(trainer, checkpoint_path)
    state_path = _save_per_checkpoint_trainer_state(
      trainer,
      checkpoint_path,
      runtime_fingerprint,
      resume_config_fingerprint,
    )
    _save_checkpoint_metadata(
      checkpoint_path,
      {
        "model_name": checkpoint_path.name,
        "global_step": int(trainer.global_step),
        "update": int(trainer.epoch),
        "trainer_state_path": str(state_path.name),
        "runtime_fingerprint": runtime_fingerprint,
        "resume_config_fingerprint": resume_config_fingerprint,
        "checkpoint_parity": parity_summary,
      },
    )
    last_checkpoint_path = checkpoint_path
    return checkpoint_raw

  trainer.save_checkpoint = _save_checkpoint_with_metadata

  sampler_anneal_config = _build_sampler_anneal_config(
    trainer_args,
    total_timesteps=int(trainer.total_epochs * trainer.config["batch_size"]),
  )
  ent_coef_anneal_config = _build_ent_coef_anneal_config(
    trainer_args,
    total_timesteps=int(trainer.total_epochs * trainer.config["batch_size"]),
  )
  _apply_sampler_anneal(sampler_anneal_config, global_step=0)
  _apply_ent_coef_anneal(trainer, ent_coef_anneal_config, global_step=0)

  started_at = time.time()
  try:
    while trainer.epoch < trainer.total_epochs:
      anneal_step = int(trainer.global_step)
      current_temp, current_smoothing = _apply_sampler_anneal(
        sampler_anneal_config,
        global_step=anneal_step,
      )
      current_ent_coef = _apply_ent_coef_anneal(
        trainer,
        ent_coef_anneal_config,
        global_step=anneal_step,
      )
      trainer.stats["sampler/subaction_temperature"].append(float(current_temp))
      trainer.stats["sampler/smoothing_eps"].append(float(current_smoothing))
      trainer.stats["anneal/ent_coef"].append(float(current_ent_coef))
      trainer.evaluate()
      trainer.train()
  finally:
    trainer.close()

  if last_checkpoint_path is None:
    raise RuntimeError(f"No checkpoint was produced for variant {label}")

  return {
    "label": label,
    "deck_encoder_type": encoder_type,
    "seed": int(seed),
    "requested_total_timesteps": int(requested_total_timesteps),
    "aligned_total_timesteps": int(aligned_total_timesteps),
    "effective_batch_size": int(effective_batch_size),
    "checkpoint": str(last_checkpoint_path.resolve()),
    "checkpoint_dir": str(last_checkpoint_path.parent.resolve()),
    "runtime_seconds": float(time.time() - started_at),
    "training_tail": _summarize_training_tail(logger.records),
    "training_log_count": int(len(logger.records)),
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run a short matched ablation for privileged critic deck encoders."
  )
  parser.add_argument(
    "--config",
    type=Path,
    default=Path("python/config/azuki_speed_3090_parallel.ini"),
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Directory for checkpoints, datasets, and summary JSON.",
  )
  parser.add_argument("--device", type=str, default="cuda")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--total-timesteps", type=int, default=16384)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--num-workers", type=int, default=4)
  parser.add_argument("--vec-batch-size", type=int, default=32)
  parser.add_argument("--minibatch-size", type=int, default=1024)
  parser.add_argument("--max-minibatch-size", type=int, default=1024)
  parser.add_argument("--rollout-episodes", type=int, default=12)
  parser.add_argument("--rollout-max-steps", type=int, default=300)
  parser.add_argument("--critic-eval-episodes", type=int, default=24)
  parser.add_argument("--critic-eval-max-steps", type=int, default=300)
  parser.add_argument(
    "--variants",
    nargs="+",
    default=["transformer", "gru"],
    choices=["transformer", "gru"],
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  output_dir = args.output_dir
  if output_dir is None:
    output_dir = Path("experiments") / f"privileged_deck_encoder_ablation_{int(time.time())}"
  output_dir.mkdir(parents=True, exist_ok=True)

  variant_results: list[dict[str, Any]] = []
  checkpoints: list[Path] = []
  for encoder_type in args.variants:
    print(f"[ablation] training variant={encoder_type}")
    result = _train_variant(
      config_path=args.config,
      output_dir=output_dir,
      label=encoder_type,
      encoder_type=encoder_type,
      seed=args.seed,
      forwarded_cli=_build_variant_overrides(args, encoder_type),
    )
    checkpoint_path = Path(result["checkpoint"])
    checkpoints.append(checkpoint_path)

    rollout_dataset_path = output_dir / f"{encoder_type}_rollout_dataset.pt"
    collect_dataset(
      config_path=args.config,
      output_path=rollout_dataset_path,
      collector_checkpoint=checkpoint_path,
      collector_mode="checkpoint",
      episodes=args.rollout_episodes,
      device=args.device,
      seed=args.seed + 1000,
      max_steps=args.rollout_max_steps,
    )
    result["rollout_action_summary"] = _summarize_action_ratios(rollout_dataset_path)
    result["rollout_dataset_path"] = str(rollout_dataset_path.resolve())
    variant_results.append(result)

  collector_label = "transformer" if "transformer" in args.variants else args.variants[0]
  collector_index = args.variants.index(collector_label)
  collector_checkpoint = checkpoints[collector_index]
  fixed_dataset_path = output_dir / f"critic_eval_{collector_label}_dataset.pt"
  collect_dataset(
    config_path=args.config,
    output_path=fixed_dataset_path,
    collector_checkpoint=collector_checkpoint,
    collector_mode="checkpoint",
    episodes=args.critic_eval_episodes,
    device=args.device,
    seed=args.seed + 2000,
    max_steps=args.critic_eval_max_steps,
  )
  fixed_eval_results_path = output_dir / "critic_eval_results.json"
  fixed_eval_payload = evaluate_dataset(
    dataset_path=fixed_dataset_path,
    config_path=args.config,
    checkpoints=checkpoints,
    device=args.device,
    output_path=fixed_eval_results_path,
  )

  result_by_checkpoint = {
    result["checkpoint"]: result
    for result in fixed_eval_payload["results"]
  }
  for variant in variant_results:
    variant["critic_eval"] = result_by_checkpoint[variant["checkpoint"]]

  summary = {
    "created_at_unix": int(time.time()),
    "config": str(args.config.resolve()),
    "device": args.device,
    "seed": int(args.seed),
    "variants": variant_results,
    "critic_eval_dataset_path": str(fixed_dataset_path.resolve()),
    "critic_eval_collector": collector_label,
    "critic_eval_results_path": str(fixed_eval_results_path.resolve()),
  }

  summary_path = output_dir / "summary.json"
  summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
  print(json.dumps(summary, indent=2, sort_keys=True, default=_json_safe))
  print(f"[ablation] summary={summary_path}")


if __name__ == "__main__":
  main()
