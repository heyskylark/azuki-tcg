from __future__ import annotations

import argparse
import os
from pathlib import Path

import azk_puffer.pytorch as azk_pytorch
import azk_puffer.vector as azk_vector
import numpy as np
import torch

from evaluate_checkpoint import _apply_checkpoint_resume_policy_config
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


def main():
  parser = argparse.ArgumentParser(
    description="Roll deck-building self-play episodes from a checkpoint and dump deck snapshots."
  )
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--episodes", type=int, default=120)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--device", type=str, default="cuda")
  parser.add_argument("--seed", type=int, default=4242)
  parser.add_argument("--max-steps", type=int, default=700)
  args = parser.parse_args()

  trainer_args = load_training_config(args.config, [])
  trainer_args["train"]["device"] = args.device
  env_cfg = trainer_args.setdefault("env", {})
  env_cfg["deck_building_enabled"] = True
  _apply_checkpoint_resume_policy_config(trainer_args, args.checkpoint)
  env_cfg["deck_snapshot_dir"] = str(args.out_dir)
  env_cfg["deck_snapshot_every"] = 1
  install_tcg_sampler()

  sampler_cfg = _build_sampler_anneal_config(
    trainer_args,
    total_timesteps=int(trainer_args.get("train", {}).get("total_timesteps", 0)),
  )
  step_hint = _peek_resume_global_step(args.checkpoint, None) if args.checkpoint else None
  _apply_sampler_anneal(sampler_cfg, global_step=0 if step_hint is None else int(step_hint))

  vecenv = build_vecenv(trainer_args, backend=azk_vector.Serial, num_envs=1, seed=args.seed)
  policy = build_policy(vecenv, trainer_args)
  use_rnn = bool(trainer_args["train"].get("use_rnn", True))
  device = args.device

  vecenv.async_reset(seed=args.seed)
  obs, *_rest = vecenv.recv()
  masks = _rest[-1]
  warm_state = {"mask": torch.as_tensor(masks, device=device)}
  if use_rnn:
    warm_state["lstm_h"] = torch.zeros(vecenv.num_agents, policy.hidden_size, device=device)
    warm_state["lstm_c"] = torch.zeros(vecenv.num_agents, policy.hidden_size, device=device)
  with torch.no_grad():
    policy.forward_eval(torch.as_tensor(obs, device=device), warm_state)
  if args.checkpoint is not None:
    _load_model_weights(policy, args.checkpoint, device=device, strict=False)
  policy.eval()

  episodes_done = 0
  try:
    while episodes_done < args.episodes:
      state = {}
      if use_rnn:
        state = {
          "lstm_h": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
          "lstm_c": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
        }
      vecenv.async_reset(seed=args.seed + 7919 * (episodes_done + 1))
      obs, *_rest = vecenv.recv()
      masks = _rest[-1]
      done = False
      steps = 0
      while not done and steps < args.max_steps:
        step_state = {"mask": torch.as_tensor(masks, device=device)}
        if use_rnn:
          step_state["lstm_h"] = state["lstm_h"]
          step_state["lstm_c"] = state["lstm_c"]
        with torch.no_grad():
          logits, _ = policy.forward_eval(torch.as_tensor(obs, device=device), step_state)
          actions, _, _ = azk_pytorch.sample_logits(logits)
        if use_rnn:
          state["lstm_h"] = step_state["lstm_h"]
          state["lstm_c"] = step_state["lstm_c"]
        vecenv.send(actions.cpu().numpy().astype(np.int32, copy=True))
        obs, *_rest = vecenv.recv()
        masks = _rest[-1]
        steps += 1
        done = vecenv.envs[0].done
      episodes_done += 1
      if episodes_done % 20 == 0:
        print(f"[dump] {episodes_done}/{args.episodes} episodes")
  finally:
    vecenv.close()
  print(f"[dump] wrote snapshots for {episodes_done} episodes to {args.out_dir}")


if __name__ == "__main__":
  main()
