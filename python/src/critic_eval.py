from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import azk_puffer.pytorch as azk_pytorch
import azk_puffer.vector as azk_vector
import numpy as np
import torch

from evaluate_checkpoint import _random_legal_action, _unwrap_base_env
from observation import observation_to_dict
from training_utils import (
  DEFAULT_CONFIG_PATH,
  build_policy,
  build_vecenv,
  install_tcg_sampler,
  load_training_config,
)
from train import (
  _compute_runtime_fingerprint,
  _load_model_weights,
  _resume_config_fingerprint,
  _validate_resume_metadata,
)

DATASET_VERSION = 1
REWARD_TOL = 1e-6
LOG_LOSS_EPS = 1e-6
CALIBRATION_BUCKET_COUNT = 10

PHASE_NAMES = {
  0: "pregame_mulligan",
  1: "start_of_turn",
  2: "main",
  3: "response_window",
  4: "combat_resolve",
  5: "end_turn_action",
  6: "end_turn",
  7: "end_match",
}

ABILITY_PHASE_NAMES = {
  0: "none",
  1: "confirmation",
  2: "cost_selection",
  3: "effect_selection",
  4: "selection_pick",
  5: "bottom_deck",
}


def _checkpoint_metadata_path(model_path: Path) -> Path:
  return model_path.with_suffix(model_path.suffix + ".meta.json")


def _read_checkpoint_metadata(model_path: Path) -> dict[str, Any]:
  metadata_path = _checkpoint_metadata_path(model_path)
  if not metadata_path.exists():
    return {}
  try:
    payload = json.loads(metadata_path.read_text())
  except Exception as exc:
    print(f"[critic-eval] warning: failed to read checkpoint metadata {metadata_path}: {exc}")
    return {}
  return payload if isinstance(payload, dict) else {}


def _build_trainer_args(
  config_path: Path,
  *,
  device: str,
  checkpoint: Path | None = None,
) -> dict[str, Any]:
  trainer_args = load_training_config(config_path, [])
  trainer_args["train"]["device"] = device

  if checkpoint is None:
    return trainer_args

  payload = _read_checkpoint_metadata(checkpoint)
  resume_cfg = payload.get("resume_config_fingerprint")
  if not isinstance(resume_cfg, dict):
    return trainer_args

  env_cfg = trainer_args.setdefault("env", {})
  policy_cfg = trainer_args.setdefault("policy", {})

  deck_pool_path = resume_cfg.get("deck_pool_path")
  if isinstance(deck_pool_path, str) and deck_pool_path:
    env_cfg["deck_pool_path"] = deck_pool_path

  model_version = resume_cfg.get("policy_model_version")
  if isinstance(model_version, str) and model_version:
    policy_cfg["model_version"] = model_version

  actor_head_type = resume_cfg.get("policy_actor_head_type")
  if isinstance(actor_head_type, str) and actor_head_type:
    policy_cfg["actor_head_type"] = actor_head_type

  legal_action_scorer_use_references = resume_cfg.get(
    "policy_legal_action_scorer_use_references"
  )
  if isinstance(legal_action_scorer_use_references, bool):
    policy_cfg["legal_action_scorer_use_references"] = legal_action_scorer_use_references

  critic_head_type = resume_cfg.get("policy_critic_head_type")
  if isinstance(critic_head_type, str) and critic_head_type:
    policy_cfg["critic_head_type"] = critic_head_type

  privileged_critic_enabled = resume_cfg.get("policy_privileged_critic_enabled")
  if isinstance(privileged_critic_enabled, bool):
    policy_cfg["privileged_critic_enabled"] = privileged_critic_enabled
  elif checkpoint is not None:
    policy_cfg["privileged_critic_enabled"] = False

  privileged_critic_embed_dim = resume_cfg.get("policy_privileged_critic_embed_dim")
  if isinstance(privileged_critic_embed_dim, (int, float)):
    policy_cfg["privileged_critic_embed_dim"] = int(privileged_critic_embed_dim)

  privileged_critic_deck_encoder_type = resume_cfg.get("policy_privileged_critic_deck_encoder_type")
  if isinstance(privileged_critic_deck_encoder_type, str) and privileged_critic_deck_encoder_type:
    policy_cfg["privileged_critic_deck_encoder_type"] = privileged_critic_deck_encoder_type

  privileged_critic_deck_heads = resume_cfg.get("policy_privileged_critic_deck_heads")
  if isinstance(privileged_critic_deck_heads, (int, float)):
    policy_cfg["privileged_critic_deck_heads"] = int(privileged_critic_deck_heads)

  privileged_critic_deck_layers = resume_cfg.get("policy_privileged_critic_deck_layers")
  if isinstance(privileged_critic_deck_layers, (int, float)):
    policy_cfg["privileged_critic_deck_layers"] = int(privileged_critic_deck_layers)

  privileged_critic_deck_ff_size = resume_cfg.get("policy_privileged_critic_deck_ff_size")
  if isinstance(privileged_critic_deck_ff_size, (int, float)):
    policy_cfg["privileged_critic_deck_ff_size"] = int(privileged_critic_deck_ff_size)

  privileged_critic_fusion_hidden_size = resume_cfg.get(
    "policy_privileged_critic_fusion_hidden_size"
  )
  if isinstance(privileged_critic_fusion_hidden_size, (int, float)):
    policy_cfg["privileged_critic_fusion_hidden_size"] = int(privileged_critic_fusion_hidden_size)

  privileged_critic_fusion_projection_size = resume_cfg.get(
    "policy_privileged_critic_fusion_projection_size"
  )
  if isinstance(privileged_critic_fusion_projection_size, (int, float)):
    policy_cfg["privileged_critic_fusion_projection_size"] = int(
      privileged_critic_fusion_projection_size
    )

  privileged_critic_feature_scale = resume_cfg.get("policy_privileged_critic_feature_scale")
  if isinstance(privileged_critic_feature_scale, (int, float)):
    policy_cfg["privileged_critic_feature_scale"] = float(privileged_critic_feature_scale)

  win_prob_aux_enabled = resume_cfg.get("policy_win_prob_aux_enabled")
  if isinstance(win_prob_aux_enabled, bool):
    policy_cfg["win_prob_aux_enabled"] = win_prob_aux_enabled
  elif checkpoint is not None:
    policy_cfg["win_prob_aux_enabled"] = False

  win_prob_aux_coef = resume_cfg.get("policy_win_prob_aux_coef")
  if isinstance(win_prob_aux_coef, (int, float)):
    policy_cfg["win_prob_aux_coef"] = float(win_prob_aux_coef)
  elif checkpoint is not None and not bool(policy_cfg.get("win_prob_aux_enabled", False)):
    policy_cfg["win_prob_aux_coef"] = 0.0

  split_value_heads_enabled = resume_cfg.get("policy_split_value_heads_enabled")
  if isinstance(split_value_heads_enabled, bool):
    policy_cfg["split_value_heads_enabled"] = split_value_heads_enabled
  elif checkpoint is not None:
    policy_cfg["split_value_heads_enabled"] = False

  split_value_component_coef = resume_cfg.get("policy_split_value_component_coef")
  if isinstance(split_value_component_coef, (int, float)):
    policy_cfg["split_value_component_coef"] = float(split_value_component_coef)
  elif checkpoint is not None and not bool(policy_cfg.get("split_value_heads_enabled", False)):
    policy_cfg["split_value_component_coef"] = 0.0

  return trainer_args


def _warm_policy(policy, vecenv, *, device: str, use_rnn: bool, seed: int) -> None:
  vecenv.async_reset(seed=seed)
  warm_obs, _, _, _, _, _, warm_masks = vecenv.recv()
  warm_state: dict[str, torch.Tensor] = {
    "mask": torch.as_tensor(warm_masks, device=device),
  }
  if use_rnn:
    warm_state["lstm_h"] = torch.zeros(vecenv.num_agents, policy.hidden_size, device=device)
    warm_state["lstm_c"] = torch.zeros(vecenv.num_agents, policy.hidden_size, device=device)
  with torch.no_grad():
    policy.forward_eval(torch.as_tensor(warm_obs, device=device), warm_state)


def _serialize_float_pair(values: np.ndarray | list[float] | tuple[float, float]) -> list[float]:
  array = np.asarray(values, dtype=np.float64).reshape(-1)
  return [float(array[0]), float(array[1])]


def _build_eval_action_array(active_index: int, action: list[int] | tuple[int, ...] | np.ndarray, num_agents: int) -> np.ndarray:
  action_np = np.zeros((num_agents, 4), dtype=np.int32)
  src = np.asarray(action, dtype=np.int32).reshape(4)
  action_np[active_index] = src
  return action_np


def _phase_name(phase: int) -> str:
  return PHASE_NAMES.get(int(phase), f"unknown_{int(phase)}")


def _ability_phase_name(phase: int) -> str:
  return ABILITY_PHASE_NAMES.get(int(phase), f"unknown_{int(phase)}")


def _stage_bucket(step_index: int, episode_length: int) -> str:
  if episode_length <= 1:
    return "opening"
  progress = float(step_index) / float(max(episode_length - 1, 1))
  if progress < (1.0 / 3.0):
    return "opening"
  if progress < (2.0 / 3.0):
    return "mid"
  return "late"


def _health_bucket(edge: int) -> str:
  if edge <= -3:
    return "trailing"
  if edge >= 3:
    return "ahead"
  return "even"


def _safe_explained_variance(targets: np.ndarray, preds: np.ndarray) -> float:
  if targets.size <= 1:
    return float("nan")
  target_var = float(np.var(targets))
  if target_var <= 1e-12:
    return float("nan")
  residual_var = float(np.var(targets - preds))
  return 1.0 - (residual_var / target_var)


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  if not rows:
    return {
      "count": 0,
      "mse": None,
      "rmse": None,
      "mae": None,
      "bias": None,
      "explained_variance": None,
      "pred_mean": None,
      "target_mean": None,
    }

  preds = np.asarray([float(row["value_prediction"]) for row in rows], dtype=np.float64)
  targets = np.asarray([float(row["return_target"]) for row in rows], dtype=np.float64)
  errors = preds - targets

  mse = float(np.mean(np.square(errors)))
  mae = float(np.mean(np.abs(errors)))
  bias = float(np.mean(errors))

  return {
    "count": int(preds.size),
    "mse": mse,
    "rmse": float(math.sqrt(mse)),
    "mae": mae,
    "bias": bias,
    "explained_variance": _safe_explained_variance(targets, preds),
    "pred_mean": float(np.mean(preds)),
    "target_mean": float(np.mean(targets)),
  }


def _group_metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
  buckets: dict[str, list[dict[str, Any]]] = {}
  for row in rows:
    bucket = str(row[key])
    buckets.setdefault(bucket, []).append(row)
  return {
    bucket: _metric_summary(bucket_rows)
    for bucket, bucket_rows in sorted(
      buckets.items(),
      key=lambda item: (-len(item[1]), item[0]),
    )
  }


def _binary_metric_summary(
  rows: list[dict[str, Any]],
  *,
  pred_key: str,
  target_key: str,
) -> dict[str, Any]:
  valid_rows = [
    row
    for row in rows
    if row.get(pred_key) is not None and row.get(target_key) is not None
  ]
  if not valid_rows:
    return {
      "count": 0,
      "accuracy": None,
      "brier": None,
      "log_loss": None,
      "pred_mean": None,
      "target_mean": None,
      "calibration_ece": None,
      "calibration_mce": None,
      "calibration_buckets": [],
    }

  preds = np.asarray([float(row[pred_key]) for row in valid_rows], dtype=np.float64)
  targets = np.asarray([float(row[target_key]) for row in valid_rows], dtype=np.float64)
  clipped_preds = np.clip(preds, LOG_LOSS_EPS, 1.0 - LOG_LOSS_EPS)
  predicted_labels = preds >= 0.5
  target_labels = targets >= 0.5
  bucket_edges = np.linspace(0.0, 1.0, CALIBRATION_BUCKET_COUNT + 1, dtype=np.float64)

  calibration_buckets: list[dict[str, Any]] = []
  ece = 0.0
  mce = 0.0
  total_count = max(int(preds.size), 1)

  for bucket_idx in range(CALIBRATION_BUCKET_COUNT):
    low = float(bucket_edges[bucket_idx])
    high = float(bucket_edges[bucket_idx + 1])
    if bucket_idx == CALIBRATION_BUCKET_COUNT - 1:
      mask = (preds >= low) & (preds <= high)
    else:
      mask = (preds >= low) & (preds < high)
    count = int(np.count_nonzero(mask))
    if count > 0:
      bucket_pred_mean = float(np.mean(preds[mask]))
      bucket_target_mean = float(np.mean(targets[mask]))
      gap = abs(bucket_pred_mean - bucket_target_mean)
      ece += (count / total_count) * gap
      mce = max(mce, gap)
    else:
      bucket_pred_mean = None
      bucket_target_mean = None
      gap = None
    calibration_buckets.append(
      {
        "bucket": f"{low:.1f}-{high:.1f}",
        "count": count,
        "pred_mean": bucket_pred_mean,
        "target_mean": bucket_target_mean,
        "gap": gap,
      }
    )

  return {
    "count": int(preds.size),
    "accuracy": float(np.mean(predicted_labels == target_labels)),
    "brier": float(np.mean(np.square(preds - targets))),
    "log_loss": float(
      -np.mean(targets * np.log(clipped_preds) + (1.0 - targets) * np.log(1.0 - clipped_preds))
    ),
    "pred_mean": float(np.mean(preds)),
    "target_mean": float(np.mean(targets)),
    "calibration_ece": float(ece),
    "calibration_mce": float(mce),
    "calibration_buckets": calibration_buckets,
  }


def _group_binary_metric_summary(
  rows: list[dict[str, Any]],
  key: str,
  *,
  pred_key: str,
  target_key: str,
) -> dict[str, dict[str, Any]]:
  buckets: dict[str, list[dict[str, Any]]] = {}
  for row in rows:
    bucket = str(row[key])
    buckets.setdefault(bucket, []).append(row)
  return {
    bucket: _binary_metric_summary(bucket_rows, pred_key=pred_key, target_key=target_key)
    for bucket, bucket_rows in sorted(
      buckets.items(),
      key=lambda item: (-len(item[1]), item[0]),
    )
  }


def _dataset_validation_summary(dataset: dict[str, Any], trainer_args: dict[str, Any], runtime_fingerprint: dict[str, Any]) -> None:
  saved_runtime = dataset.get("runtime_fingerprint")
  if isinstance(saved_runtime, dict):
    for key in ("obs_dtype_descr", "obs_dtype_itemsize", "binding_size"):
      saved_value = saved_runtime.get(key)
      current_value = runtime_fingerprint.get(key)
      if saved_value is not None and current_value is not None and saved_value != current_value:
        raise RuntimeError(
          "[critic-eval] dataset/runtime incompatibility detected: "
          f"{key}: dataset={saved_value!r} current={current_value!r}"
        )

  saved_cfg = dataset.get("resume_config_fingerprint")
  if not isinstance(saved_cfg, dict):
    return

  current_cfg = _resume_config_fingerprint(trainer_args)
  keys = (
    "use_rnn",
    "direct_parallel",
    "deck_pool_path",
    "policy_model_version",
  )
  mismatches = []
  for key in keys:
    if key in saved_cfg and key in current_cfg and saved_cfg[key] != current_cfg[key]:
      mismatches.append((key, saved_cfg[key], current_cfg[key]))
  if mismatches:
    details = ", ".join(f"{key}: dataset={saved!r} current={current!r}" for key, saved, current in mismatches)
    raise RuntimeError(f"[critic-eval] dataset/config incompatibility detected. {details}")


def collect_dataset(
  *,
  config_path: Path,
  output_path: Path,
  collector_checkpoint: Path | None,
  collector_mode: str,
  episodes: int,
  device: str,
  seed: int,
  max_steps: int,
) -> dict[str, Any]:
  install_tcg_sampler()

  trainer_args = _build_trainer_args(
    config_path,
    device=device,
    checkpoint=collector_checkpoint,
  )
  use_rnn = bool(trainer_args.get("train", {}).get("use_rnn", True))
  gamma = float(trainer_args.get("train", {}).get("gamma", 0.99))

  vecenv = build_vecenv(
    trainer_args,
    backend=azk_vector.Serial,
    num_envs=1,
    seed=seed,
  )
  runtime_fingerprint = _compute_runtime_fingerprint(vecenv)
  resume_config_fingerprint = _resume_config_fingerprint(trainer_args)
  base_env = _unwrap_base_env(vecenv.envs[0])

  policy = None
  if collector_mode == "checkpoint":
    if collector_checkpoint is None:
      raise ValueError("collector_checkpoint is required for checkpoint collection")
    policy = build_policy(vecenv, trainer_args)
    _warm_policy(policy, vecenv, device=device, use_rnn=use_rnn, seed=seed)
    _validate_resume_metadata(collector_checkpoint, runtime_fingerprint, resume_config_fingerprint)
    _load_model_weights(policy, collector_checkpoint, device=device, strict=False)
    policy.eval()

  rng = np.random.default_rng(seed + 17)
  torch.manual_seed(seed)
  episodes_out: list[dict[str, Any]] = []
  timeout_count = 0

  try:
    for episode_idx in range(episodes):
      episode_seed = int(seed + episode_idx)
      if use_rnn and policy is not None:
        recurrent_state: dict[str, torch.Tensor] = {
          "lstm_h": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
          "lstm_c": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
        }
      else:
        recurrent_state = {}

      vecenv.async_reset(seed=episode_seed)
      obs, _, _, _, _, _, masks = vecenv.recv()
      deck_indices = [int(index) for index in getattr(base_env, "_current_deck_indices", (-1, -1))]

      active_indices: list[int] = []
      actions: list[list[int]] = []
      rewards: list[list[float]] = []
      done = False
      step_count = 0

      while not done and step_count < max_steps:
        active_index = int(getattr(base_env, "_active_player_index", 0))
        active_indices.append(active_index)

        if collector_mode == "checkpoint":
          assert policy is not None
          step_state: dict[str, torch.Tensor] = {
            "mask": torch.as_tensor(masks, device=device),
          }
          if use_rnn:
            step_state["lstm_h"] = recurrent_state["lstm_h"]
            step_state["lstm_c"] = recurrent_state["lstm_c"]
          with torch.no_grad():
            logits, _ = policy.forward_eval(torch.as_tensor(obs, device=device), step_state)
            sampled_actions, _, _ = azk_pytorch.sample_logits(logits)
          active_action = sampled_actions.detach().cpu().numpy().astype(np.int32, copy=False)[active_index].tolist()
          if use_rnn:
            recurrent_state["lstm_h"] = step_state["lstm_h"]
            recurrent_state["lstm_c"] = step_state["lstm_c"]
        else:
          active_action = _random_legal_action(base_env, rng).astype(np.int32, copy=False).tolist()

        action_np = _build_eval_action_array(active_index, active_action, vecenv.num_agents)
        vecenv.send(action_np)
        obs, reward_step, _, _, _, _, masks = vecenv.recv()

        actions.append([int(value) for value in active_action])
        rewards.append(_serialize_float_pair(reward_step))
        step_count += 1
        done = bool(vecenv.envs[0].done)

      if not done and step_count >= max_steps:
        timeout_count += 1

      final_infos = getattr(base_env, "infos", {})
      seat_wins = [float(final_infos.get(seat, {}).get("win", 0.0)) for seat in range(vecenv.num_agents)]
      seat_returns = [float(final_infos.get(seat, {}).get("azk_episode_return", 0.0)) for seat in range(vecenv.num_agents)]
      episodes_out.append(
        {
          "seed": episode_seed,
          "deck_indices": deck_indices,
          "active_indices": active_indices,
          "actions": actions,
          "rewards": rewards,
          "completed": bool(done),
          "timed_out": bool((not done) and step_count >= max_steps),
          "seat_wins": seat_wins,
          "seat_episode_returns": seat_returns,
          "step_count": int(step_count),
        }
      )
  finally:
    vecenv.close()

  output_path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "dataset_version": DATASET_VERSION,
    "created_at_unix": int(time.time()),
    "config_path": str(config_path.resolve()),
    "collector": {
      "mode": collector_mode,
      "checkpoint": str(collector_checkpoint.resolve()) if collector_checkpoint is not None else None,
    },
    "episodes": episodes_out,
    "episode_count": int(len(episodes_out)),
    "timeout_count": int(timeout_count),
    "max_steps": int(max_steps),
    "seed": int(seed),
    "gamma": gamma,
    "runtime_fingerprint": runtime_fingerprint,
    "resume_config_fingerprint": resume_config_fingerprint,
  }
  torch.save(payload, output_path)
  return {
    "dataset_path": str(output_path),
    "episodes": int(len(episodes_out)),
    "timeout_count": int(timeout_count),
    "avg_episode_length": float(np.mean([episode["step_count"] for episode in episodes_out]) if episodes_out else 0.0),
  }


def _evaluate_checkpoint_on_dataset(
  *,
  dataset: dict[str, Any],
  config_path: Path,
  checkpoint: Path,
  device: str,
) -> dict[str, Any]:
  install_tcg_sampler()

  trainer_args = _build_trainer_args(
    config_path,
    device=device,
    checkpoint=checkpoint,
  )
  use_rnn = bool(trainer_args.get("train", {}).get("use_rnn", True))
  gamma = float(dataset.get("gamma", trainer_args.get("train", {}).get("gamma", 0.99)))

  vecenv = build_vecenv(
    trainer_args,
    backend=azk_vector.Serial,
    num_envs=1,
    seed=int(dataset.get("seed", 0)),
  )
  runtime_fingerprint = _compute_runtime_fingerprint(vecenv)
  _dataset_validation_summary(dataset, trainer_args, runtime_fingerprint)
  resume_config_fingerprint = _resume_config_fingerprint(trainer_args)
  _validate_resume_metadata(checkpoint, runtime_fingerprint, resume_config_fingerprint)

  policy = build_policy(vecenv, trainer_args)
  _warm_policy(policy, vecenv, device=device, use_rnn=use_rnn, seed=int(dataset.get("seed", 0)))
  _load_model_weights(policy, checkpoint, device=device, strict=False)
  policy.eval()

  base_env = _unwrap_base_env(vecenv.envs[0])
  rows: list[dict[str, Any]] = []
  replay_warnings = 0
  policy_cfg = trainer_args.get("policy", {})

  try:
    for episode_index, episode in enumerate(dataset.get("episodes", [])):
      episode_seed = int(episode["seed"])
      actions = episode["actions"]
      active_indices = episode["active_indices"]
      rewards_ref = episode["rewards"]
      if not (len(actions) == len(active_indices) == len(rewards_ref)):
        raise RuntimeError(
          f"[critic-eval] malformed dataset episode {episode_index}: "
          f"actions={len(actions)} active_indices={len(active_indices)} rewards={len(rewards_ref)}"
        )

      if use_rnn:
        recurrent_state: dict[str, torch.Tensor] = {
          "lstm_h": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
          "lstm_c": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
        }
      else:
        recurrent_state = {}

      vecenv.async_reset(seed=episode_seed)
      obs, _, _, _, _, _, masks = vecenv.recv()
      current_deck_indices = [int(index) for index in getattr(base_env, "_current_deck_indices", (-1, -1))]
      if current_deck_indices != [int(index) for index in episode["deck_indices"]]:
        raise RuntimeError(
          f"[critic-eval] deck mismatch in episode {episode_index}: "
          f"dataset={episode['deck_indices']} current={current_deck_indices}"
        )

      episode_rows: list[dict[str, Any]] = []
      done = False

      for step_index, (action, expected_active_index, expected_reward_pair) in enumerate(
        zip(actions, active_indices, rewards_ref, strict=True)
      ):
        active_index = int(getattr(base_env, "_active_player_index", 0))
        if active_index != int(expected_active_index):
          raise RuntimeError(
            f"[critic-eval] active player mismatch in episode {episode_index} step {step_index}: "
            f"dataset={expected_active_index} current={active_index}"
          )

        step_state: dict[str, torch.Tensor] = {
          "mask": torch.as_tensor(masks, device=device),
        }
        if use_rnn:
          step_state["lstm_h"] = recurrent_state["lstm_h"]
          step_state["lstm_c"] = recurrent_state["lstm_c"]

        with torch.no_grad():
          _, values = policy.forward_eval(torch.as_tensor(obs, device=device), step_state)

        if use_rnn:
          recurrent_state["lstm_h"] = step_state["lstm_h"]
          recurrent_state["lstm_c"] = step_state["lstm_c"]

        value_tensor = values.reshape(-1)
        active_value = float(value_tensor[active_index].item())
        win_prob_prediction = None
        win_prob_logits = step_state.get("_azk_win_prob_logits")
        if torch.is_tensor(win_prob_logits):
          win_prob_tensor = win_prob_logits.reshape(-1)
          win_prob_prediction = float(torch.sigmoid(win_prob_tensor[active_index]).item())
        active_obs = observation_to_dict(base_env._raw_observation(active_index))
        leader_health_edge = int(active_obs["player"]["leader"]["cur_hp"]) - int(active_obs["opponent"]["leader"]["cur_hp"])
        deck_pair = f"{current_deck_indices[0]}_vs_{current_deck_indices[1]}"
        win_prob_target = None
        seat_wins = np.asarray(episode["seat_wins"], dtype=np.float64)
        if bool(episode.get("completed", False)) and np.isclose(np.sum(seat_wins), 1.0, atol=REWARD_TOL, rtol=0.0):
          win_prob_target = float(seat_wins[active_index])

        action_np = _build_eval_action_array(active_index, action, vecenv.num_agents)
        vecenv.send(action_np)
        obs, reward_step, _, _, _, _, masks = vecenv.recv()
        reward_pair = _serialize_float_pair(reward_step)
        if not np.allclose(
          np.asarray(reward_pair, dtype=np.float64),
          np.asarray(expected_reward_pair, dtype=np.float64),
          atol=REWARD_TOL,
          rtol=0.0,
        ):
          raise RuntimeError(
            f"[critic-eval] reward mismatch in episode {episode_index} step {step_index}: "
            f"dataset={expected_reward_pair} current={reward_pair}"
          )

        episode_rows.append(
          {
            "episode_index": int(episode_index),
            "episode_seed": int(episode_seed),
            "step_index": int(step_index),
            "active_seat": int(active_index),
            "deck_pair": deck_pair,
            "phase": _phase_name(active_obs["phase"]),
            "ability_phase": _ability_phase_name(active_obs["ability_context"]["phase"]),
            "leader_health_bucket": _health_bucket(leader_health_edge),
            "leader_health_edge": int(leader_health_edge),
            "combat_active": bool(active_obs["combat_context"]["combat_active"]),
            "value_prediction": active_value,
            "win_prob_prediction": win_prob_prediction,
            "win_prob_target": win_prob_target,
            "reward_pair": reward_pair,
            "seat_win": float(episode["seat_wins"][active_index]),
          }
        )

        done = bool(vecenv.envs[0].done)

      if bool(episode.get("completed", False)) != done:
        replay_warnings += 1
        print(
          f"[critic-eval] warning: episode completion mismatch for episode {episode_index}: "
          f"dataset={episode.get('completed')} current={done}"
        )

      returns_by_seat = [0.0, 0.0]
      for row in reversed(episode_rows):
        reward_pair = row["reward_pair"]
        returns_by_seat[0] = float(reward_pair[0]) + gamma * returns_by_seat[0]
        returns_by_seat[1] = float(reward_pair[1]) + gamma * returns_by_seat[1]
        row["return_target"] = float(returns_by_seat[row["active_seat"]])

      episode_length = len(episode_rows)
      for row in episode_rows:
        row["stage"] = _stage_bucket(int(row["step_index"]), episode_length)
      rows.extend(episode_rows)
  finally:
    vecenv.close()

  summary = {
    "checkpoint": str(checkpoint.resolve()),
    "policy_actor_head_type": policy_cfg.get("actor_head_type", "legal_action_scorer"),
    "policy_legal_action_scorer_use_references": bool(
      policy_cfg.get("legal_action_scorer_use_references", True)
    ),
    "policy_critic_head_type": policy_cfg.get("critic_head_type", "full_lstm_mlp"),
    "policy_privileged_critic_enabled": bool(policy_cfg.get("privileged_critic_enabled", False)),
    "policy_privileged_critic_embed_dim": int(policy_cfg.get("privileged_critic_embed_dim", 64)),
    "policy_privileged_critic_deck_encoder_type": str(
      policy_cfg.get("privileged_critic_deck_encoder_type", "transformer")
    ),
    "policy_privileged_critic_deck_heads": int(policy_cfg.get("privileged_critic_deck_heads", 4)),
    "policy_privileged_critic_deck_layers": int(policy_cfg.get("privileged_critic_deck_layers", 2)),
    "policy_privileged_critic_deck_ff_size": int(policy_cfg.get("privileged_critic_deck_ff_size", 256)),
    "policy_privileged_critic_fusion_hidden_size": int(
      policy_cfg.get("privileged_critic_fusion_hidden_size", 512)
    ),
    "policy_privileged_critic_fusion_projection_size": int(
      policy_cfg.get("privileged_critic_fusion_projection_size", 128)
    ),
    "policy_privileged_critic_feature_scale": float(
      policy_cfg.get("privileged_critic_feature_scale", 1.0)
    ),
    "policy_win_prob_aux_enabled": bool(policy_cfg.get("win_prob_aux_enabled", False)),
    "gamma": float(gamma),
    "overall": _metric_summary(rows),
    "win_prob_overall": _binary_metric_summary(rows, pred_key="win_prob_prediction", target_key="win_prob_target"),
    "by_phase": _group_metric_summary(rows, "phase"),
    "win_prob_by_phase": _group_binary_metric_summary(
      rows, "phase", pred_key="win_prob_prediction", target_key="win_prob_target"
    ),
    "by_ability_phase": _group_metric_summary(rows, "ability_phase"),
    "win_prob_by_ability_phase": _group_binary_metric_summary(
      rows, "ability_phase", pred_key="win_prob_prediction", target_key="win_prob_target"
    ),
    "by_stage": _group_metric_summary(rows, "stage"),
    "win_prob_by_stage": _group_binary_metric_summary(
      rows, "stage", pred_key="win_prob_prediction", target_key="win_prob_target"
    ),
    "by_deck_pair": _group_metric_summary(rows, "deck_pair"),
    "win_prob_by_deck_pair": _group_binary_metric_summary(
      rows, "deck_pair", pred_key="win_prob_prediction", target_key="win_prob_target"
    ),
    "by_active_seat": _group_metric_summary(rows, "active_seat"),
    "win_prob_by_active_seat": _group_binary_metric_summary(
      rows, "active_seat", pred_key="win_prob_prediction", target_key="win_prob_target"
    ),
    "by_leader_health_bucket": _group_metric_summary(rows, "leader_health_bucket"),
    "win_prob_by_leader_health_bucket": _group_binary_metric_summary(
      rows, "leader_health_bucket", pred_key="win_prob_prediction", target_key="win_prob_target"
    ),
    "by_seat_win": _group_metric_summary(rows, "seat_win"),
    "win_prob_by_seat_win": _group_binary_metric_summary(
      rows, "seat_win", pred_key="win_prob_prediction", target_key="win_prob_target"
    ),
    "row_count": int(len(rows)),
    "replay_warnings": int(replay_warnings),
  }
  return summary


def evaluate_dataset(
  *,
  dataset_path: Path,
  config_path: Path,
  checkpoints: list[Path],
  device: str,
  output_path: Path | None,
) -> dict[str, Any]:
  dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
  if not isinstance(dataset, dict):
    raise ValueError(f"[critic-eval] unsupported dataset format: {dataset_path}")
  if int(dataset.get("dataset_version", -1)) != DATASET_VERSION:
    raise ValueError(
      f"[critic-eval] unsupported dataset version {dataset.get('dataset_version')}; "
      f"expected {DATASET_VERSION}"
    )

  results = []
  for checkpoint in checkpoints:
    results.append(
      _evaluate_checkpoint_on_dataset(
        dataset=dataset,
        config_path=config_path,
        checkpoint=checkpoint,
        device=device,
      )
    )

  payload = {
    "dataset_path": str(dataset_path.resolve()),
    "config_path": str(config_path.resolve()),
    "checkpoint_count": int(len(results)),
    "results": results,
  }

  if output_path is not None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
  return payload


def parse_args():
  parser = argparse.ArgumentParser(
    description="Collect fixed rollout traces and evaluate critic quality offline."
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  collect_parser = subparsers.add_parser(
    "collect",
    help="Collect a fixed rollout dataset using a checkpoint or random-legal policy.",
  )
  collect_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
  collect_parser.add_argument("--output", type=Path, required=True)
  collect_parser.add_argument(
    "--collector-mode",
    choices=("checkpoint", "random_legal"),
    default="checkpoint",
  )
  collect_parser.add_argument("--collector-checkpoint", type=Path, default=None)
  collect_parser.add_argument("--episodes", type=int, default=64)
  collect_parser.add_argument("--device", type=str, default="cuda")
  collect_parser.add_argument("--seed", type=int, default=1234)
  collect_parser.add_argument("--max-steps", type=int, default=400)

  eval_parser = subparsers.add_parser(
    "evaluate",
    help="Replay a fixed rollout dataset and score one or more checkpoints.",
  )
  eval_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
  eval_parser.add_argument("--dataset", type=Path, required=True)
  eval_parser.add_argument("--checkpoint", type=Path, action="append", required=True)
  eval_parser.add_argument("--device", type=str, default="cuda")
  eval_parser.add_argument("--output", type=Path, default=None)

  return parser.parse_args()


def _print_eval_summary(payload: dict[str, Any]) -> None:
  print(f"[critic-eval] dataset={payload['dataset_path']}")
  for result in payload["results"]:
    overall = result["overall"]
    win_prob = result.get("win_prob_overall", {})
    win_prob_suffix = ""
    if int(win_prob.get("count") or 0) > 0:
      win_prob_suffix = (
        f" win_prob_count={win_prob['count']}"
        f" win_prob_brier={win_prob['brier']:.6f}"
        f" win_prob_logloss={win_prob['log_loss']:.6f}"
        f" win_prob_acc={win_prob['accuracy']:.6f}"
        f" win_prob_ece={win_prob['calibration_ece']:.6f}"
      )
    print(
      "[critic-eval] "
      f"critic_head={result['policy_critic_head_type']} "
      f"deck_encoder={result.get('policy_privileged_critic_deck_encoder_type', 'transformer')} "
      f"checkpoint={result['checkpoint']} "
      f"count={overall['count']} "
      f"mse={overall['mse']:.6f} "
      f"mae={overall['mae']:.6f} "
      f"ev={overall['explained_variance']:.6f}"
      f"{win_prob_suffix}"
    )


def main():
  args = parse_args()
  if args.command == "collect":
    result = collect_dataset(
      config_path=args.config,
      output_path=args.output,
      collector_checkpoint=args.collector_checkpoint,
      collector_mode=args.collector_mode,
      episodes=args.episodes,
      device=args.device,
      seed=args.seed,
      max_steps=args.max_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return

  payload = evaluate_dataset(
    dataset_path=args.dataset,
    config_path=args.config,
    checkpoints=[Path(checkpoint) for checkpoint in args.checkpoint],
    device=args.device,
    output_path=args.output,
  )
  _print_eval_summary(payload)


if __name__ == "__main__":
  main()
