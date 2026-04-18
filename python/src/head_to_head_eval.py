from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import azk_puffer.pytorch as azk_pytorch
import azk_puffer.vector as azk_vector
import numpy as np
import torch

from evaluate_checkpoint import _apply_checkpoint_resume_policy_config, _random_legal_action, _unwrap_base_env
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
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--max-steps", type=int, default=400)
  parser.add_argument("--device", type=str, default="cuda:0")
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
) -> dict[str, Any]:
  trainer_args = load_training_config(config_path, [])
  trainer_args["train"]["device"] = device
  _apply_checkpoint_resume_policy_config(trainer_args, checkpoint)
  policy = build_policy(vecenv, trainer_args)
  use_rnn = bool(trainer_args["train"].get("use_rnn", True))
  _warm_policy(policy, vecenv, device=device, use_rnn=use_rnn, seed=warm_seed)
  _load_model_weights(policy, checkpoint, device=device, strict=False)
  policy.eval()
  return {
    "policy": policy,
    "use_rnn": use_rnn,
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


def _reset_env_policy_state(state: dict[str, torch.Tensor], env_index: int) -> None:
  state["lstm_h"][env_index].zero_()
  state["lstm_c"][env_index].zero_()


def _seat_owner_name(seat_assignment: np.ndarray, env_index: int, seat_index: int, labels: list[str]) -> str:
  owner_index = int(seat_assignment[env_index, seat_index])
  return labels[owner_index]


def evaluate_head_to_head(
  *,
  config_path: Path,
  checkpoint_a: Path,
  checkpoint_b: Path | None,
  label_a: str,
  label_b: str,
  episodes: int,
  num_envs: int,
  seed: int,
  max_steps: int,
  device: str,
) -> dict[str, Any]:
  install_tcg_sampler()
  base_args = load_training_config(config_path, [])
  base_args["train"]["device"] = device
  anneal_total_timesteps = int(base_args.get("train", {}).get("total_timesteps", 0))
  sampler_cfg = _build_sampler_anneal_config(base_args, total_timesteps=anneal_total_timesteps)

  vecenv = build_vecenv(
    base_args,
    backend=azk_vector.Serial,
    num_envs=num_envs,
    seed=seed,
  )
  agents_per_env = int(vecenv.driver_env.num_agents)
  if agents_per_env != 2:
    raise ValueError(f"Expected 2 agents per env, got {agents_per_env}")

  wrapped_envs = list(vecenv.envs)
  base_envs = [_unwrap_base_env(env) for env in wrapped_envs]
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

  try:
    while np.any(completed < episode_targets):
      obs_tensor = torch.as_tensor(obs, device=device)
      mask_tensor = torch.as_tensor(masks, device=device)
      action_np = np.zeros((vecenv.num_agents, 4), dtype=np.int32)

      grouped_rows: list[list[int]] = [[], []]
      grouped_envs: list[list[int]] = [[], []]

      for env_index, base_env in enumerate(base_envs):
        if wrapped_envs[env_index].done:
          continue
        active_seat = int(getattr(base_env, "_active_player_index", 0))
        row_index = env_index * agents_per_env + active_seat
        owner_index = int(seat_assignment[env_index, active_seat])
        if completed[env_index] >= episode_targets[env_index]:
          action_np[row_index] = _random_legal_action(base_env, rngs[env_index])
          continue
        if policy_specs[owner_index] is None:
          action_np[row_index] = _random_legal_action(base_env, rngs[env_index])
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
        row_tensor = torch.as_tensor(rows, device=device, dtype=torch.long)
        sub_obs = obs_tensor.index_select(0, row_tensor)
        sub_mask = mask_tensor.index_select(0, row_tensor)
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
        with torch.no_grad():
          logits, _ = spec["policy"].forward_eval(sub_obs, step_state)
          sampled_actions, _, _ = azk_pytorch.sample_logits(logits)
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
          state_bank["lstm_h"].index_copy_(0, env_tensor, step_state["lstm_h"])
          state_bank["lstm_c"].index_copy_(0, env_tensor, step_state["lstm_c"])

      vecenv.send(action_np)
      obs, rewards, terminals, truncations, _, _, masks = vecenv.recv()
      rewards_np = np.asarray(rewards, dtype=np.float32).reshape(num_envs, agents_per_env)

      for env_index in range(num_envs):
        if completed[env_index] >= episode_targets[env_index]:
          continue

        episode_lengths[env_index] += 1
        for seat_index in range(agents_per_env):
          owner_index = int(seat_assignment[env_index, seat_index])
          episode_rewards[env_index, owner_index] += float(rewards_np[env_index, seat_index])

        wrapped_env = wrapped_envs[env_index]
        base_env = base_envs[env_index]
        if not wrapped_env.done and episode_lengths[env_index] < max_steps:
          continue

        timed_out = not wrapped_env.done and episode_lengths[env_index] >= max_steps
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
        else:
          winner_label: str | None = None
          winner_seat: int | None = None
          for seat_index in range(agents_per_env):
            info = base_env.infos.get(seat_index, {})
            if float(info.get("win", 0.0)) >= 0.5:
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
      "max_steps": int(max_steps),
      "device": device,
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
    seed=args.seed,
    max_steps=args.max_steps,
    device=args.device,
  )
  rendered = json.dumps(summary, indent=2)
  print(rendered)
  if args.json is not None:
    args.json.write_text(rendered + "\n")


if __name__ == "__main__":
  main()
