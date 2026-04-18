from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import azk_puffer.pytorch as azk_pytorch
import azk_puffer.vector as azk_vector
import numpy as np
import torch

from evaluate_checkpoint import _apply_checkpoint_resume_policy_config
from policy.tcg_distribution import TCGActionDistribution, TCGLegalActionDistribution
from training_utils import (
  DEFAULT_CONFIG_PATH,
  build_policy,
  build_vecenv,
  install_tcg_sampler,
  load_training_config,
)
from train import _apply_sampler_anneal, _build_sampler_anneal_config, _load_model_weights, _peek_resume_global_step

RANDOM_LEGAL_LABEL = "random_legal"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Batched seat-fair evaluation for Azuki checkpoints.",
  )
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
  parser.add_argument("--checkpoint-a", type=Path, required=True)
  parser.add_argument("--checkpoint-b", type=Path, default=None)
  parser.add_argument("--label-a", type=str, default="checkpoint_a")
  parser.add_argument("--label-b", type=str, default="checkpoint_b")
  parser.add_argument("--episodes", type=int, default=48)
  parser.add_argument("--num-envs", type=int, default=0)
  parser.add_argument("--backend", type=str, default="Serial")
  parser.add_argument("--num-workers", type=int, default=None)
  parser.add_argument("--batch-size", type=int, default=None)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--max-steps", type=int, default=400)
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument("--device-staging", type=str, default="auto")
  parser.add_argument("--amp", action="store_true")
  parser.add_argument("--precision", type=str, default="float16")
  parser.add_argument("--compile", action="store_true")
  parser.add_argument("--compile-mode", type=str, default="max-autotune-no-cudagraphs")
  parser.add_argument("--json", type=Path, default=None)
  return parser.parse_args()


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


def _build_policy_spec(
  *,
  checkpoint: Path,
  config_path: Path,
  vecenv,
  device: str,
  warm_seed: int,
  use_amp: bool,
  precision: str,
  compile_eval: bool,
  compile_mode: str,
) -> dict[str, Any]:
  trainer_args = load_training_config(config_path, [])
  trainer_args["train"]["device"] = device
  _apply_checkpoint_resume_policy_config(trainer_args, checkpoint)
  policy = build_policy(vecenv, trainer_args)
  use_rnn = bool(trainer_args["train"].get("use_rnn", True))
  _warm_policy(policy, vecenv, device=device, use_rnn=use_rnn, seed=warm_seed)
  _load_model_weights(policy, checkpoint, device=device, strict=False)
  if compile_eval:
    policy.forward_eval = torch.compile(policy.forward_eval, mode=compile_mode)
  policy.eval()
  amp_context = contextlib.nullcontext()
  if use_amp and str(device).startswith("cuda"):
    amp_context = torch.amp.autocast(device_type="cuda", dtype=getattr(torch, precision))
  return {
    "policy": policy,
    "use_rnn": use_rnn,
    "amp_context": amp_context,
  }


def _target_episodes_per_env(total_episodes: int, num_envs: int) -> np.ndarray:
  if total_episodes < 1:
    raise ValueError("episodes must be >= 1")
  if num_envs < 1:
    raise ValueError("num_envs must be >= 1")
  base = total_episodes // num_envs
  remainder = total_episodes % num_envs
  targets = np.full(num_envs, base, dtype=np.int32)
  if remainder > 0:
    targets[:remainder] += 1
  return targets


def _resolve_backend(name: str):
  backend = getattr(azk_vector, str(name), None)
  if backend is None:
    raise ValueError(f"Unknown vector backend '{name}'")
  return backend


def _resolve_num_envs(
  total_episodes: int,
  requested_num_envs: int,
  backend_name: str,
  device: str,
) -> int:
  if requested_num_envs > 0:
    return int(requested_num_envs)
  if total_episodes < 1:
    raise ValueError("episodes must be >= 1")
  if str(backend_name) == "Serial":
    serial_cap = 20 if str(device).startswith("cuda") else 12
    return int(min(total_episodes, serial_cap))
  return int(min(total_episodes, 8))


def _resolve_device_staging(device_staging: str, device: str) -> str:
  if device_staging == "auto":
    return "full_batch" if str(device).startswith("cuda") else "grouped"
  if device_staging in {"grouped", "full_batch"}:
    return device_staging
  raise ValueError(f"Unsupported device_staging '{device_staging}'")


def _reset_env_policy_state(state: dict[str, torch.Tensor], env_index: int) -> None:
  state["lstm_h"][env_index].zero_()
  state["lstm_c"][env_index].zero_()


def _seat_owner_name(seat_assignment: np.ndarray, env_index: int, seat_index: int, labels: list[str]) -> str:
  owner_index = int(seat_assignment[env_index, seat_index])
  return labels[owner_index]


def _build_info_by_env(info, env_indices: np.ndarray) -> dict[int, object]:
  ordered_envs = np.unique(env_indices)
  info_entries = list(info) if isinstance(info, (list, tuple)) else []
  if len(info_entries) == len(ordered_envs):
    return {
      int(env_idx): info_entries[pos]
      for pos, env_idx in enumerate(ordered_envs)
    }
  if len(info_entries) == len(env_indices):
    info_by_env: dict[int, object] = {}
    for pos, env_idx in enumerate(env_indices):
      info_by_env.setdefault(int(env_idx), info_entries[pos])
    return info_by_env
  return {}


def _extract_terminal_win_labels(env_info, agents_per_env: int) -> dict[int, float]:
  if not isinstance(env_info, dict):
    return {}

  labels: dict[int, float] = {}
  for seat in range(agents_per_env):
    seat_info = env_info.get(seat)
    if seat_info is None:
      seat_info = env_info.get(str(seat))
    if not isinstance(seat_info, dict):
      continue
    win_value = seat_info.get("win")
    if win_value is None:
      continue
    labels[int(seat)] = float(win_value)
  return labels


def _random_legal_action_from_mask(
  legal_action_count: np.ndarray,
  legal_primary: np.ndarray,
  legal_sub1: np.ndarray,
  legal_sub2: np.ndarray,
  legal_sub3: np.ndarray,
  row_index: int,
  rng: np.random.Generator,
) -> np.ndarray:
  legal_count = int(legal_action_count[row_index])
  if legal_count <= 0:
    return np.asarray([0, 0, 0, 0], dtype=np.int32)
  choice = int(rng.integers(0, legal_count))
  return np.asarray(
    [
      int(legal_primary[row_index, choice]),
      int(legal_sub1[row_index, choice]),
      int(legal_sub2[row_index, choice]),
      int(legal_sub3[row_index, choice]),
    ],
    dtype=np.int32,
  )


def _upcast_distribution_for_sampling(logits):
  if isinstance(logits, TCGLegalActionDistribution):
    return TCGLegalActionDistribution(
      legal_action_logits=logits.legal_action_logits.float(),
      legal_actions=logits.legal_actions,
      legal_action_count=logits.legal_action_count,
    )
  if isinstance(logits, TCGActionDistribution):
    updates = {}
    for field in dataclasses.fields(logits):
      value = getattr(logits, field.name)
      if torch.is_tensor(value) and value.is_floating_point():
        updates[field.name] = value.float()
      else:
        updates[field.name] = value
    return TCGActionDistribution(**updates)
  return logits


def evaluate_head_to_head(
  *,
  config_path: Path,
  checkpoint_a: Path,
  checkpoint_b: Path | None,
  label_a: str,
  label_b: str,
  episodes: int,
  num_envs: int,
  backend: str,
  num_workers: int | None,
  batch_size: int | None,
  seed: int,
  max_steps: int,
  device: str,
  device_staging: str,
  use_amp: bool,
  precision: str,
  compile_eval: bool,
  compile_mode: str,
) -> dict[str, Any]:
  install_tcg_sampler()
  device_staging = _resolve_device_staging(device_staging, device)
  base_args = load_training_config(config_path, [])
  base_args["train"]["device"] = device
  if str(device).startswith("cuda"):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
  backend_cls = _resolve_backend(backend)
  num_envs = _resolve_num_envs(episodes, num_envs, backend_cls.__name__, device)
  vec_cfg = base_args.setdefault("vec", {})
  resolved_batch_size = int(num_envs if batch_size is None else batch_size)
  if resolved_batch_size != int(num_envs):
    raise ValueError(
      f"head_to_head_eval requires batch_size == num_envs for stable env-local state tracking; "
      f"got batch_size={resolved_batch_size}, num_envs={num_envs}"
    )
  vec_cfg["batch_size"] = resolved_batch_size
  if num_workers is not None:
    vec_cfg["num_workers"] = int(num_workers)

  anneal_total_timesteps = int(base_args.get("train", {}).get("total_timesteps", 0))
  sampler_cfg = _build_sampler_anneal_config(base_args, total_timesteps=anneal_total_timesteps)

  vecenv = build_vecenv(
    base_args,
    backend=backend_cls,
    num_envs=num_envs,
    seed=seed,
  )
  agents_per_env = int(vecenv.driver_env.num_agents)
  if agents_per_env != 2:
    raise ValueError(f"Expected 2 agents per env, got {agents_per_env}")
  rngs = [np.random.default_rng(seed + 10_000 + env_index) for env_index in range(num_envs)]

  labels = [label_a]
  if checkpoint_b is None:
    labels.append(RANDOM_LEGAL_LABEL)
  else:
    labels.append(label_b)

  policy_specs: list[dict[str, Any] | None] = [
    _build_policy_spec(
      checkpoint=checkpoint_a,
      config_path=config_path,
      vecenv=vecenv,
      device=device,
      warm_seed=seed,
      use_amp=use_amp,
      precision=precision,
      compile_eval=compile_eval,
      compile_mode=compile_mode,
    ),
    None,
  ]
  if checkpoint_b is not None:
    policy_specs[1] = _build_policy_spec(
      checkpoint=checkpoint_b,
      config_path=config_path,
      vecenv=vecenv,
      device=device,
      warm_seed=seed + 1,
      use_amp=use_amp,
      precision=precision,
      compile_eval=compile_eval,
      compile_mode=compile_mode,
    )
  obs_struct_dtype = next(
    getattr(spec["policy"], "policy", spec["policy"])._obs_struct_dtype
    for spec in policy_specs
    if spec is not None
  )

  sampler_steps = []
  for checkpoint in (checkpoint_a, checkpoint_b):
    if checkpoint is None:
      continue
    checkpoint_global_step = _peek_resume_global_step(checkpoint, None)
    if checkpoint_global_step is not None:
      sampler_steps.append(int(checkpoint_global_step))
  _apply_sampler_anneal(
    sampler_cfg,
    global_step=max(sampler_steps) if sampler_steps else 0,
  )

  policy_states: list[dict[str, torch.Tensor] | None] = []
  for spec in policy_specs:
    if spec is None:
      policy_states.append(None)
      continue
    if spec["use_rnn"]:
      hidden_size = int(spec["policy"].hidden_size)
      policy_states.append(
        {
          "lstm_h": torch.zeros(num_envs, hidden_size, device=device),
          "lstm_c": torch.zeros(num_envs, hidden_size, device=device),
        }
      )
    else:
      policy_states.append(None)

  episode_targets = _target_episodes_per_env(episodes, num_envs)
  completed = np.zeros(num_envs, dtype=np.int32)
  draining = np.zeros(num_envs, dtype=bool)
  episode_lengths = np.zeros(num_envs, dtype=np.int32)
  episode_rewards = np.zeros((num_envs, 2), dtype=np.float64)
  seat_assignment = np.zeros((num_envs, 2), dtype=np.int32)
  for env_index in range(num_envs):
    first_owner = env_index % 2
    seat_assignment[env_index, 0] = first_owner
    seat_assignment[env_index, 1] = 1 - first_owner

  scoreboard: dict[str, dict[str, Any]] = {}
  for label in labels:
    scoreboard[label] = {
      "wins": 0,
      "losses": 0,
      "draws": 0,
      "timeouts": 0,
      "episode_lengths": [],
      "episode_rewards": [],
      "seat0_wins": 0,
      "seat1_wins": 0,
    }

  vecenv.async_reset(seed=seed)
  obs, rewards, terminals, truncations, _, _, masks = vecenv.recv()
  started_at = time.perf_counter()
  decode_seconds = 0.0
  device_stage_seconds = 0.0
  forward_seconds = 0.0
  env_seconds = 0.0
  step_iterations = 0

  try:
    while np.any(completed < episode_targets):
      step_iterations += 1
      obs_np = np.asarray(obs)
      mask_np = np.asarray(masks)
      action_np = np.zeros((vecenv.num_agents, 4), dtype=np.int32)
      obs_device = None
      mask_device = None
      decode_started = time.perf_counter()
      structured_obs = azk_pytorch.nativize_tensor(torch.as_tensor(obs_np), obs_struct_dtype)
      action_mask_struct = structured_obs["action_mask"]
      legal_action_count_t = action_mask_struct["legal_action_count"].to(dtype=torch.long).view(
        num_envs,
        agents_per_env,
      )
      legal_row_active = legal_action_count_t > 0
      active_env_mask = legal_row_active.any(dim=1)
      active_seats = torch.argmax(legal_action_count_t, dim=1)
      legal_action_count = legal_action_count_t.reshape(-1).detach().cpu().numpy()
      legal_primary = action_mask_struct["legal_actions"]["legal_primary"].detach().cpu().numpy()
      legal_sub1 = action_mask_struct["legal_actions"]["legal_sub1"].detach().cpu().numpy()
      legal_sub2 = action_mask_struct["legal_actions"]["legal_sub2"].detach().cpu().numpy()
      legal_sub3 = action_mask_struct["legal_actions"]["legal_sub3"].detach().cpu().numpy()
      decode_seconds += time.perf_counter() - decode_started
      if device_staging == "full_batch":
        stage_started = time.perf_counter()
        obs_device = torch.as_tensor(obs_np, device=device)
        mask_device = torch.as_tensor(mask_np, device=device)
        device_stage_seconds += time.perf_counter() - stage_started

      grouped_rows: list[list[int]] = [[], []]
      grouped_envs: list[list[int]] = [[], []]

      for env_index in range(num_envs):
        if not bool(active_env_mask[env_index].item()):
          continue
        active_seat = int(active_seats[env_index].item())
        row_index = env_index * agents_per_env + active_seat
        if draining[env_index]:
          action_np[row_index] = _random_legal_action_from_mask(
            legal_action_count,
            legal_primary,
            legal_sub1,
            legal_sub2,
            legal_sub3,
            row_index,
            rngs[env_index],
          )
          continue
        owner_index = int(seat_assignment[env_index, active_seat])
        if completed[env_index] >= episode_targets[env_index]:
          action_np[row_index] = _random_legal_action_from_mask(
            legal_action_count,
            legal_primary,
            legal_sub1,
            legal_sub2,
            legal_sub3,
            row_index,
            rngs[env_index],
          )
          continue
        if policy_specs[owner_index] is None:
          action_np[row_index] = _random_legal_action_from_mask(
            legal_action_count,
            legal_primary,
            legal_sub1,
            legal_sub2,
            legal_sub3,
            row_index,
            rngs[env_index],
          )
          continue
        grouped_rows[owner_index].append(row_index)
        grouped_envs[owner_index].append(env_index)

      for owner_index, spec in enumerate(policy_specs):
        if spec is None:
          continue
        rows = grouped_rows[owner_index]
        env_indices = grouped_envs[owner_index]
        if not rows:
          continue
        if device_staging == "full_batch":
          row_tensor = torch.as_tensor(rows, device=device, dtype=torch.long)
          sub_obs = obs_device.index_select(0, row_tensor)
          sub_mask = mask_device.index_select(0, row_tensor)
        else:
          sub_obs = torch.as_tensor(obs_np[rows], device=device)
          sub_mask = torch.as_tensor(mask_np[rows], device=device)
        step_state: dict[str, torch.Tensor] = {"mask": sub_mask}
        state_bank = policy_states[owner_index]
        if spec["use_rnn"] and state_bank is not None:
          env_tensor = torch.as_tensor(env_indices, device=device, dtype=torch.long)
          step_state["lstm_h"] = state_bank["lstm_h"].index_select(0, env_tensor)
          step_state["lstm_c"] = state_bank["lstm_c"].index_select(0, env_tensor)
        padded_singleton = len(rows) == 1
        if padded_singleton:
          sub_obs = torch.cat([sub_obs, sub_obs], dim=0)
          step_state["mask"] = torch.cat([step_state["mask"], step_state["mask"]], dim=0)
          if spec["use_rnn"] and state_bank is not None:
            step_state["lstm_h"] = torch.cat([step_state["lstm_h"], step_state["lstm_h"]], dim=0)
            step_state["lstm_c"] = torch.cat([step_state["lstm_c"], step_state["lstm_c"]], dim=0)
        forward_started = time.perf_counter()
        with torch.inference_mode(), spec["amp_context"]:
          logits, _ = spec["policy"].forward_eval(sub_obs, step_state)
        if use_amp and str(device).startswith("cuda"):
          logits = _upcast_distribution_for_sampling(logits)
        with torch.inference_mode():
          sampled_actions, _, _ = azk_pytorch.sample_logits(logits)
        forward_seconds += time.perf_counter() - forward_started
        if padded_singleton:
          sampled_actions = sampled_actions[:1]
          if spec["use_rnn"] and state_bank is not None:
            step_state["lstm_h"] = step_state["lstm_h"][:1]
            step_state["lstm_c"] = step_state["lstm_c"][:1]
        sampled_np = sampled_actions.detach().cpu().numpy().astype(np.int32, copy=False)
        for batch_index, row_index in enumerate(rows):
          action_np[row_index] = sampled_np[batch_index]
        if spec["use_rnn"] and state_bank is not None:
          env_tensor = torch.as_tensor(env_indices, device=device, dtype=torch.long)
          next_h = step_state["lstm_h"].to(dtype=state_bank["lstm_h"].dtype)
          next_c = step_state["lstm_c"].to(dtype=state_bank["lstm_c"].dtype)
          state_bank["lstm_h"].index_copy_(0, env_tensor, next_h)
          state_bank["lstm_c"].index_copy_(0, env_tensor, next_c)

      env_started = time.perf_counter()
      vecenv.send(action_np)
      obs, rewards, terminals, truncations, infos, _, masks = vecenv.recv()
      env_seconds += time.perf_counter() - env_started
      rewards_np = np.asarray(rewards, dtype=np.float32).reshape(num_envs, agents_per_env)
      done_np = np.logical_or(
        np.asarray(terminals, dtype=bool).reshape(num_envs, agents_per_env),
        np.asarray(truncations, dtype=bool).reshape(num_envs, agents_per_env),
      )
      done_env = done_np.all(axis=1)
      info_by_env = _build_info_by_env(
        infos,
        np.repeat(np.arange(num_envs, dtype=np.int32), agents_per_env),
      )

      for env_index in range(num_envs):
        if draining[env_index]:
          if done_env[env_index]:
            draining[env_index] = False
            episode_lengths[env_index] = 0
            episode_rewards[env_index] = 0.0
            for state_bank in policy_states:
              if state_bank is not None:
                _reset_env_policy_state(state_bank, env_index)
          continue

        if completed[env_index] >= episode_targets[env_index]:
          continue

        episode_lengths[env_index] += 1
        for seat_index in range(agents_per_env):
          owner_index = int(seat_assignment[env_index, seat_index])
          episode_rewards[env_index, owner_index] += float(rewards_np[env_index, seat_index])

        if not done_env[env_index] and episode_lengths[env_index] < max_steps:
          continue

        timed_out = not done_env[env_index] and episode_lengths[env_index] >= max_steps
        env_labels = [
          _seat_owner_name(seat_assignment, env_index, 0, labels),
          _seat_owner_name(seat_assignment, env_index, 1, labels),
        ]
        for owner_index, label in enumerate(labels):
          scoreboard[label]["episode_lengths"].append(int(episode_lengths[env_index]))
          scoreboard[label]["episode_rewards"].append(float(episode_rewards[env_index, owner_index]))

        if timed_out:
          for label in labels:
            scoreboard[label]["draws"] += 1
            scoreboard[label]["timeouts"] += 1
          draining[env_index] = True
        else:
          winner_label: str | None = None
          winner_seat: int | None = None
          seat_labels = _extract_terminal_win_labels(info_by_env.get(env_index), agents_per_env)
          for seat_index, win_value in seat_labels.items():
            if float(win_value) >= 0.5:
              winner_label = env_labels[seat_index]
              winner_seat = seat_index
              break
          if winner_label is None:
            for label in labels:
              scoreboard[label]["draws"] += 1
          else:
            loser_label = env_labels[1 - winner_seat]
            scoreboard[winner_label]["wins"] += 1
            scoreboard[loser_label]["losses"] += 1
            seat_key = f"seat{winner_seat}_wins"
            scoreboard[winner_label][seat_key] += 1

        completed[env_index] += 1
        episode_lengths[env_index] = 0
        episode_rewards[env_index] = 0.0
        seat_assignment[env_index, 0], seat_assignment[env_index, 1] = (
          seat_assignment[env_index, 1],
          seat_assignment[env_index, 0],
        )
        for state_bank in policy_states:
          if state_bank is not None:
            _reset_env_policy_state(state_bank, env_index)

    summary = {
      "episodes": int(episodes),
      "num_envs": int(num_envs),
      "backend": str(backend_cls.__name__),
      "num_workers": None if getattr(vecenv, "num_workers", None) is None else int(vecenv.num_workers),
      "batch_size": int(resolved_batch_size),
      "max_steps": int(max_steps),
      "device": device,
      "device_staging": device_staging,
      "amp": bool(use_amp and str(device).startswith("cuda")),
      "precision": precision,
      "compile": bool(compile_eval),
      "compile_mode": compile_mode if compile_eval else None,
      "elapsed_seconds": float(time.perf_counter() - started_at),
      "step_iterations": int(step_iterations),
      "timing": {
        "decode_seconds": float(decode_seconds),
        "device_stage_seconds": float(device_stage_seconds),
        "forward_seconds": float(forward_seconds),
        "env_seconds": float(env_seconds),
      },
      "checkpoint_a": str(checkpoint_a.resolve()),
      "checkpoint_b": None if checkpoint_b is None else str(checkpoint_b.resolve()),
      "results": {},
    }
    for label, metrics in scoreboard.items():
      wins = int(metrics["wins"])
      losses = int(metrics["losses"])
      draws = int(metrics["draws"])
      total = wins + losses + draws
      non_draw_total = wins + losses
      summary["results"][label] = {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": int(metrics["timeouts"]),
        "win_rate": (wins / total) if total > 0 else None,
        "non_draw_win_rate": (wins / non_draw_total) if non_draw_total > 0 else None,
        "mean_episode_length": float(np.mean(metrics["episode_lengths"])) if metrics["episode_lengths"] else None,
        "mean_episode_reward": float(np.mean(metrics["episode_rewards"])) if metrics["episode_rewards"] else None,
        "seat0_wins": int(metrics["seat0_wins"]),
        "seat1_wins": int(metrics["seat1_wins"]),
      }
    return summary
  finally:
    vecenv.close()


def main() -> None:
  args = _parse_args()
  summary = evaluate_head_to_head(
    config_path=args.config,
    checkpoint_a=args.checkpoint_a,
    checkpoint_b=args.checkpoint_b,
    label_a=args.label_a,
    label_b=args.label_b,
    episodes=args.episodes,
    num_envs=args.num_envs,
    backend=args.backend,
    num_workers=args.num_workers,
    batch_size=args.batch_size,
    seed=args.seed,
    max_steps=args.max_steps,
    device=args.device,
    device_staging=args.device_staging,
    use_amp=args.amp,
    precision=args.precision,
    compile_eval=args.compile,
    compile_mode=args.compile_mode,
  )
  rendered = json.dumps(summary, indent=2)
  print(rendered)
  if args.json is not None:
    args.json.write_text(rendered + "\n")


if __name__ == "__main__":
  main()
