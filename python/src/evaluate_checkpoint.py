from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import pufferlib.pytorch
import pufferlib.vector

from training_utils import (
  DEFAULT_CONFIG_PATH,
  build_policy,
  build_vecenv,
  install_tcg_sampler,
  load_training_config,
)
from train import (
  _apply_sampler_anneal,
  _build_sampler_anneal_config,
  _compute_runtime_fingerprint,
  _load_model_weights,
  _peek_resume_global_step,
  _resume_config_fingerprint,
  _validate_resume_metadata,
)


def _unwrap_base_env(env):
  current = getattr(env, "env", env)
  seen = set()
  while hasattr(current, "env"):
    nxt = getattr(current, "env")
    if nxt is current or nxt in seen:
      break
    seen.add(current)
    current = nxt
  return current


def _random_legal_action(base_env, rng: np.random.Generator):
  active_index = int(getattr(base_env, "_active_player_index", 0))
  raw_obs = base_env._raw_observation(active_index)
  mask = raw_obs.action_mask
  legal_count = int(mask.legal_action_count)
  if legal_count <= 0:
    return np.asarray([0, 0, 0, 0], dtype=np.int32)
  choice = int(rng.integers(0, legal_count))
  return np.asarray(
    [
      int(mask.legal_primary[choice]),
      int(mask.legal_sub1[choice]),
      int(mask.legal_sub2[choice]),
      int(mask.legal_sub3[choice]),
    ],
    dtype=np.int32,
  )


def evaluate(
  *,
  config_path: Path,
  checkpoint: Path | None,
  episodes: int,
  policy_seat: int,
  device: str,
  seed: int,
  max_steps: int,
):
  trainer_args = load_training_config(config_path, [])
  trainer_args["train"]["device"] = device
  install_tcg_sampler()
  anneal_total_timesteps = int(trainer_args.get("train", {}).get("total_timesteps", 0))
  sampler_cfg = _build_sampler_anneal_config(
    trainer_args,
    total_timesteps=anneal_total_timesteps,
  )
  checkpoint_global_step = _peek_resume_global_step(checkpoint, None) if checkpoint is not None else None
  temp_now, smoothing_now = _apply_sampler_anneal(
    sampler_cfg,
    global_step=0 if checkpoint_global_step is None else int(checkpoint_global_step),
  )
  print(
    "[sampler] eval settings: "
    f"global_step={0 if checkpoint_global_step is None else int(checkpoint_global_step)}, "
    f"subaction_temperature={temp_now:.6f}, smoothing_eps={smoothing_now:.6f}"
  )

  vecenv = build_vecenv(
    trainer_args,
    backend=pufferlib.vector.Serial,
    num_envs=1,
    seed=seed,
  )
  runtime_fingerprint = _compute_runtime_fingerprint(vecenv)
  resume_config_fingerprint = _resume_config_fingerprint(trainer_args)
  base_env = _unwrap_base_env(vecenv.envs[0])
  policy = build_policy(vecenv, trainer_args)
  use_rnn = bool(trainer_args["train"].get("use_rnn", True))

  # Warm-up forward pass to initialize normalizer/lazy buffers.
  vecenv.async_reset(seed=seed)
  warm_obs, _, _, _, _, _, warm_masks = vecenv.recv()
  warm_state = {"mask": torch.as_tensor(warm_masks, device=device)}
  if use_rnn:
    warm_state["lstm_h"] = torch.zeros(vecenv.num_agents, policy.hidden_size, device=device)
    warm_state["lstm_c"] = torch.zeros(vecenv.num_agents, policy.hidden_size, device=device)
  with torch.no_grad():
    policy.forward_eval(torch.as_tensor(warm_obs, device=device), warm_state)

  if checkpoint is not None:
    _validate_resume_metadata(checkpoint, runtime_fingerprint, resume_config_fingerprint)
    _load_model_weights(policy, checkpoint, device=device, strict=False)
  policy.eval()

  rng = np.random.default_rng(seed + 17)
  wins = 0
  losses = 0
  draws = 0
  timeouts = 0
  lengths = []
  reward_totals = []
  positive_reward_episodes = 0

  try:
    for episode_idx in range(episodes):
      if use_rnn:
        state = {
          "lstm_h": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
          "lstm_c": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
        }
      else:
        state = {}

      vecenv.async_reset(seed=seed + episode_idx)
      obs, _, _, _, _, _, masks = vecenv.recv()
      done = False
      step_count = 0
      episode_reward = 0.0

      while not done and step_count < max_steps:
        active_index = int(getattr(base_env, "_active_player_index", 0))
        if active_index == policy_seat:
          obs_tensor = torch.as_tensor(obs, device=device)
          step_state = {"mask": torch.as_tensor(masks, device=device)}
          if use_rnn:
            step_state["lstm_h"] = state["lstm_h"]
            step_state["lstm_c"] = state["lstm_c"]

          with torch.no_grad():
            logits, _ = policy.forward_eval(obs_tensor, step_state)
            actions, _, _ = pufferlib.pytorch.sample_logits(logits)
          action_np = actions.cpu().numpy().astype(np.int32, copy=True)

          if use_rnn:
            state["lstm_h"] = step_state["lstm_h"]
            state["lstm_c"] = step_state["lstm_c"]
        else:
          action_np = np.zeros((vecenv.num_agents, 4), dtype=np.int32)
          action_np[active_index] = _random_legal_action(base_env, rng)

        vecenv.send(action_np)
        obs, rewards, _, _, _, _, masks = vecenv.recv()
        episode_reward += float(rewards[policy_seat])
        step_count += 1
        done = vecenv.envs[0].done

      lengths.append(step_count)
      reward_totals.append(episode_reward)
      if episode_reward > 0.0:
        positive_reward_episodes += 1
      if not done and step_count >= max_steps:
        draws += 1
        timeouts += 1
        continue
      terminal_info = base_env.infos.get(policy_seat, {})
      win_flag = float(terminal_info.get("win", 0.0))
      if win_flag >= 0.5:
        wins += 1
      else:
        opp_info = base_env.infos.get(1 - policy_seat, {})
        opp_win_flag = float(opp_info.get("win", 0.0))
        if opp_win_flag >= 0.5:
          losses += 1
        else:
          draws += 1
  finally:
    vecenv.close()

  total = max(wins + losses + draws, 1)
  return {
    "episodes": int(total),
    "policy_seat": int(policy_seat),
    "wins": int(wins),
    "losses": int(losses),
    "draws": int(draws),
    "timeouts": int(timeouts),
    "win_rate": float(wins / total),
    "avg_episode_length": float(np.mean(lengths) if lengths else 0.0),
    "avg_cumulative_reward": float(np.mean(reward_totals) if reward_totals else 0.0),
    "positive_reward_rate": float(positive_reward_episodes / total),
  }


def parse_args():
  parser = argparse.ArgumentParser(
    description="Evaluate a checkpoint against a random-legal baseline."
  )
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--episodes", type=int, default=200)
  parser.add_argument("--policy-seat", type=int, choices=[0, 1], default=0)
  parser.add_argument("--device", type=str, default="cuda")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--max-steps", type=int, default=400)
  return parser.parse_args()


def main():
  args = parse_args()
  result = evaluate(
    config_path=args.config,
    checkpoint=args.checkpoint,
    episodes=args.episodes,
    policy_seat=args.policy_seat,
    device=args.device,
    seed=args.seed,
    max_steps=args.max_steps,
  )
  print(result)


if __name__ == "__main__":
  main()
