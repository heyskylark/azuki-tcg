from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
      records.append({"gate": gate_code, "win": win, "steps": steps, "timeout": timeout})
      per_gate[gate_code]["games"] += 1
      per_gate[gate_code]["wins"] += win
  finally:
    vecenv.close()

  games = max(len(records), 1)
  result = {
    "episodes": len(records),
    "drafter_seat": drafter_seat,
    "fixed_seat": int(fixed_seat),
    "drafter_win_rate": sum(r["win"] for r in records) / games,
    "timeout_rate": sum(1 for r in records if r["timeout"]) / games,
    "avg_steps": sum(r["steps"] for r in records) / games,
    "per_gate": {
      gate: {"games": stats["games"], "win_rate": stats["wins"] / max(stats["games"], 1)}
      for gate, stats in sorted(per_gate.items())
    },
  }
  return result


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
  parser.add_argument("--json", type=Path, default=None)
  args = parser.parse_args()

  seats = [0, 1] if args.fixed_seat == "both" else [int(args.fixed_seat)]
  results = []
  for seat in seats:
    result = run_eval(
      config_path=args.config,
      checkpoint=args.checkpoint,
      episodes=args.episodes // len(seats),
      fixed_seat=seat,
      device=args.device,
      seed=args.seed + seat * 99991,
      max_steps=args.max_steps,
      argmax=args.argmax,
    )
    results.append(result)
    print(json.dumps(result, indent=2))

  if len(results) == 2:
    total_games = sum(r["episodes"] for r in results)
    combined = sum(r["drafter_win_rate"] * r["episodes"] for r in results) / max(total_games, 1)
    print(f"combined drafter win rate (seat-fair): {combined:.4f} over {total_games} episodes")
  if args.json:
    args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
  main()
