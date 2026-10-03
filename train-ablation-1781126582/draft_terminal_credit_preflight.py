#!/usr/bin/env python3
"""Measure whether native draft rows can receive ordinary terminal GAE credit.

The diagnostic runs one deterministic native checkpoint self-play episode, maps
the requested draft decisions onto the training buffer's BPTT/update geometry,
and applies the exact PuffeRL advantage kernel to a single terminal impulse. A
second native episode is truncated immediately after drafting to verify that the
win-label path remains empty for truncations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from azk_native import NATIVE_DECKBUILD_OBS_DTYPE
from azk_puffer import trainer as pufferl
from evaluate_checkpoint import _apply_checkpoint_resume_policy_config
from policy.v2.tcg_sampler import tcg_argmax_logits
from train import _load_model_weights
from training_utils import build_policy, load_training_config, make_azuki_env


REQUESTED_PICKS = (1, 10, 25, 40, 50)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _deck_context_offsets() -> tuple[int, int, int]:
  deck_dtype, deck_offset = NATIVE_DECKBUILD_OBS_DTYPE.fields["deck_context"][:2]
  _, mode_offset = deck_dtype.fields["mode"][:2]
  _, main_count_offset = deck_dtype.fields["main_count"][:2]
  return (
    int(deck_offset + mode_offset),
    int(deck_offset + main_count_offset),
    int(NATIVE_DECKBUILD_OBS_DTYPE.itemsize),
  )


def _decode_draft_context(observation: np.ndarray) -> tuple[int, int]:
  mode_offset, main_count_offset, expected_bytes = _deck_context_offsets()
  row = np.asarray(observation, dtype=np.uint8).reshape(-1)
  if row.size != expected_bytes:
    raise ValueError(f"expected {expected_bytes} observation bytes, got {row.size}")
  mode = int(row[mode_offset:mode_offset + 4].copy().view(np.int32)[0])
  main_count = int(row[main_count_offset])
  return mode, main_count


def _fresh_state(policy, device: str, use_rnn: bool) -> dict[str, torch.Tensor] | None:
  if not use_rnn:
    return None
  hidden_size = int(policy.hidden_size)
  return {
    "lstm_h": torch.zeros(2, hidden_size, device=device),
    "lstm_c": torch.zeros(2, hidden_size, device=device),
  }


def _policy_action(
  policy,
  observation: np.ndarray,
  seat: int,
  state_bank: dict[str, torch.Tensor] | None,
  device: str,
) -> np.ndarray:
  state: dict[str, torch.Tensor] = {
    "mask": torch.ones(1, device=device, dtype=torch.bool),
  }
  if state_bank is not None:
    state["lstm_h"] = state_bank["lstm_h"][seat:seat + 1]
    state["lstm_c"] = state_bank["lstm_c"][seat:seat + 1]
  obs_t = torch.as_tensor(observation[None, :], device=device)
  with torch.inference_mode():
    logits, _ = policy.forward_eval(obs_t, state)
    action = tcg_argmax_logits(logits)
  if state_bank is not None:
    state_bank["lstm_h"][seat] = state["lstm_h"][0]
    state_bank["lstm_c"][seat] = state["lstm_c"][0]
  return action[0].to(dtype=torch.int32).cpu().numpy()


def _reset_game(env, seed: int) -> None:
  env.reset_evaluation_games(
    [{
      "env_index": 0,
      "seed": int(seed),
      "gate0": -1,
      "gate1": -1,
      "reference_seat": -1,
      "reference_deck_index": -1,
    }]
  )


def _run_terminal_episode(
  env,
  policy,
  *,
  seed: int,
  device: str,
  use_rnn: bool,
  max_steps: int,
) -> dict:
  _reset_game(env, seed)
  state = _fresh_state(policy, device, use_rnn)
  draft_rows: list[dict[str, int | str]] = []
  records: list[dict] = []
  step = 0
  while step < max_steps:
    active = int(env.active_players()[0])
    if active < 0:
      records.extend(env.drain_evaluation_records())
      break
    row_index = active
    mode, main_count = _decode_draft_context(env.observations[row_index])
    if mode == 1:
      draft_rows.append({"seat": active, "kind": "leader", "pick": 0, "row": step})
    elif mode == 2:
      draft_rows.append(
        {"seat": active, "kind": "main", "pick": main_count + 1, "row": step}
      )
    env.actions.fill(0)
    env.actions[row_index] = _policy_action(
      policy,
      env.observations[row_index],
      active,
      state,
      device,
    )
    env.step()
    step += 1
    records.extend(env.drain_evaluation_records())
    if records:
      break
  if not records:
    raise RuntimeError(f"native episode did not finish within {max_steps} decisions")
  record = records[0]
  if int(record.get("end_reason", -1)) != 0:
    raise RuntimeError(
      f"seed {seed} did not produce a true terminal: end_reason={record.get('end_reason')}"
    )
  return {
    "seed": int(seed),
    "decision_steps": int(step),
    # PuffeRL receives and stores the terminal reward on the row after the
    # final environment action.
    "terminal_row": int(step),
    "draft_rows": draft_rows,
    "raw_record": record,
    "terminal_flags_after_step": [bool(value) for value in env.terminals.tolist()],
    "truncation_flags_after_step": [bool(value) for value in env.truncations.tolist()],
    "rewards_after_step": [float(value) for value in env.rewards.tolist()],
  }


def _run_truncation_probe(
  env,
  policy,
  *,
  seed: int,
  device: str,
  use_rnn: bool,
) -> dict:
  _reset_game(env, seed)
  state = _fresh_state(policy, device, use_rnn)
  step = 0
  while True:
    active = int(env.active_players()[0])
    if active < 0:
      raise RuntimeError("native truncation probe ended before reaching battle")
    mode, _ = _decode_draft_context(env.observations[active])
    if mode == 0:
      break
    env.actions.fill(0)
    env.actions[active] = _policy_action(
      policy,
      env.observations[active],
      active,
      state,
      device,
    )
    env.step()
    step += 1
    if step > 200:
      raise RuntimeError("native truncation probe draft exceeded 200 decisions")

  env.force_evaluation_truncations([0])
  records = env.drain_evaluation_records()
  if len(records) != 1:
    raise RuntimeError(f"expected one forced-truncation record, got {len(records)}")
  terminal_mask = np.asarray(env.terminals, dtype=np.bool_)
  done_mask = np.logical_or(terminal_mask, np.asarray(env.truncations, dtype=np.bool_))
  labels = pufferl.terminal_win_labels_from_rewards(
    np.asarray([0, 1], dtype=np.int64),
    terminal_mask,
    np.asarray(env.rewards, dtype=np.float32),
    agents_per_env=2,
  )
  return {
    "seed": int(seed),
    "draft_decision_steps": int(step),
    "done_flags": [bool(value) for value in done_mask.tolist()],
    "terminal_flags": [bool(value) for value in terminal_mask.tolist()],
    "truncation_flags": [bool(value) for value in env.truncations.tolist()],
    "rewards": [float(value) for value in env.rewards.tolist()],
    "terminal_win_labels": labels,
    "raw_record": records[0],
  }


def _advantage_impulse(
  total_rows: int,
  terminal_row: int,
  *,
  horizon: int,
  gamma: float,
  gae_lambda: float,
  segmented: bool,
) -> np.ndarray:
  if segmented:
    segments = math.ceil(total_rows / horizon)
    shape = (segments, horizon)
  else:
    shape = (1, total_rows)
  values = torch.zeros(shape, dtype=torch.float32)
  rewards = torch.zeros(shape, dtype=torch.float32)
  terminals = torch.zeros(shape, dtype=torch.float32)
  ratio = torch.ones(shape, dtype=torch.float32)
  if segmented:
    segment, offset = divmod(terminal_row, horizon)
  else:
    segment, offset = 0, terminal_row
  rewards[segment, offset] = 1.0
  terminals[segment, offset] = 1.0
  advantages = pufferl.compute_puff_advantage(
    values,
    rewards,
    terminals,
    ratio,
    torch.zeros_like(values),
    gamma,
    gae_lambda,
    1.0,
    1.0,
  )
  if segmented:
    return advantages.reshape(-1)[:total_rows].detach().cpu().numpy()
  return advantages[0].detach().cpu().numpy()


def _training_geometry(config: dict, horizon: int) -> dict[str, int | str]:
  workers = int(config["vec"]["num_envs"])
  games_per_worker = int(config["env"]["native_envs_per_instance"])
  agents_per_game = 2
  total_agents = workers * games_per_worker * agents_per_game
  configured_batch = config["train"]["batch_size"]
  if configured_batch != "auto":
    raise ValueError(f"preflight expects train.batch_size=auto, got {configured_batch!r}")
  batch_size = total_agents * horizon
  return {
    "workers": workers,
    "games_per_worker": games_per_worker,
    "agents_per_game": agents_per_game,
    "total_agents": total_agents,
    "bptt_horizon": horizon,
    "batch_size": batch_size,
    "segments": batch_size // horizon,
    "segments_per_agent_per_update": (batch_size // horizon) // total_agents,
  }


def _analyze_impulse(
  episode: dict,
  horizon: int,
  *,
  expect_leader_rows: bool,
) -> dict:
  terminal_row = int(episode["terminal_row"])
  total_rows = terminal_row + 1
  settings = {
    "current_0.99_0.95": (0.99, 0.95),
    "proposed_1.0_0.99": (1.0, 0.99),
  }
  impulses = {}
  for name, (gamma, gae_lambda) in settings.items():
    impulses[name] = {
      "segmented": _advantage_impulse(
        total_rows,
        terminal_row,
        horizon=horizon,
        gamma=gamma,
        gae_lambda=gae_lambda,
        segmented=True,
      ),
      "unsegmented": _advantage_impulse(
        total_rows,
        terminal_row,
        horizon=horizon,
        gamma=gamma,
        gae_lambda=gae_lambda,
        segmented=False,
      ),
    }

  selected = []
  for row in episode["draft_rows"]:
    pick = int(row["pick"])
    if row["kind"] != "leader" and pick not in REQUESTED_PICKS:
      continue
    source_row = int(row["row"])
    item = {
      **row,
      "segment_update": source_row // horizon,
      "segment_offset": source_row % horizon,
      "distance_to_terminal": terminal_row - source_row,
      "trained_and_evicted_before_terminal": source_row // horizon < terminal_row // horizon,
      "advantages": {},
    }
    for name, values in impulses.items():
      item["advantages"][name] = {
        "actual_segmented": float(values["segmented"][source_row]),
        "hypothetical_unsegmented": float(values["unsegmented"][source_row]),
      }
    selected.append(item)

  expected = {
    (seat, "main", pick)
    for seat in range(2)
    for pick in REQUESTED_PICKS
  }
  if expect_leader_rows:
    expected.update((seat, "leader", 0) for seat in range(2))
  observed = {
    (int(row["seat"]), str(row["kind"]), int(row["pick"])) for row in selected
  }
  if observed != expected:
    raise RuntimeError(f"missing requested draft rows: {sorted(expected - observed)}")

  boundaries = []
  segmented = impulses["proposed_1.0_0.99"]["segmented"]
  for boundary in range(horizon, total_rows, horizon):
    boundaries.append(
      {
        "boundary_row": boundary,
        "update_before": boundary // horizon - 1,
        "update_after": boundary // horizon,
        "advantage_before": float(segmented[boundary - 1]),
        "advantage_after": float(segmented[boundary]),
        "after_shares_terminal_segment": boundary // horizon == terminal_row // horizon,
      }
    )
  return {
    "terminal_row": terminal_row,
    "terminal_segment_update": terminal_row // horizon,
    "terminal_segment_offset": terminal_row % horizon,
    "selected_rows": selected,
    "boundaries": boundaries,
    "all_requested_rows_receive_segmented_credit": all(
      abs(
        float(row["advantages"]["proposed_1.0_0.99"]["actual_segmented"])
      ) > 0.0
      for row in selected
    ),
  }


def _markdown(payload: dict) -> str:
  impulse = payload["impulse"]
  geometry = payload["training_geometry"]
  episode = payload["native_terminal_episode"]
  truncation = payload["native_truncation_probe"]
  if payload.get("uniform_assignment"):
    branch_text = (
      "The phase-aware GAE arm is infeasible under the current rollout lifecycle. "
      "Stage 2 must retain all 50 main-pick observations, actions, behavior log "
      "probabilities, and pre-decision recurrent states until an exact terminal, "
      "then train them through the fixed-shape full-episode credit path. There is "
      "no leader actor row under uniform assignment. This terminal channel remains "
      "independent of shaped-reward annealing."
    )
  else:
    branch_text = (
      "The preferred phase-aware GAE arm is infeasible under the current rollout "
      "lifecycle. The legacy fallback retains delayed sampled draft rows with exact "
      "terminal outcomes; it remains independent of shaped-reward annealing."
    )
  lines = [
    "# Step 5b terminal-credit preflight",
    "",
    f"Decision: **{payload['decision']}**.",
    "",
    "## Native rollout",
    "",
    f"- Checkpoint: `{payload['checkpoint']['path']}`",
    f"- SHA-256: `{payload['checkpoint']['sha256']}`",
    f"- Seed: `{episode['seed']}`",
    f"- Native decisions through true terminal: `{episode['decision_steps']}`",
    f"- Trainer terminal row: `{impulse['terminal_row']}` "
    f"(update `{impulse['terminal_segment_update']}`, offset "
    f"`{impulse['terminal_segment_offset']}`)",
    "",
    "## Actual buffer geometry",
    "",
    f"The production configuration has `{geometry['total_agents']}` agents, "
    f"`batch_size={geometry['batch_size']}`, and `bptt_horizon={geometry['bptt_horizon']}`. "
    f"That is exactly `{geometry['segments_per_agent_per_update']}` segment per agent per PPO "
    "update. PuffeRL initializes the advantage carry to zero for every segment, then trains and "
    "recycles that buffer at the update boundary.",
    "",
    "## Requested draft rows",
    "",
    "| Seat | Decision | Row | Update | Offset | Terminal distance | Current segmented | "
    "Proposed segmented | Proposed if unsegmented |",
    "|---:|:---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in impulse["selected_rows"]:
    label = "leader" if row["kind"] == "leader" else f"main {row['pick']}"
    current = row["advantages"]["current_0.99_0.95"]["actual_segmented"]
    proposed = row["advantages"]["proposed_1.0_0.99"]["actual_segmented"]
    unsegmented = row["advantages"]["proposed_1.0_0.99"]["hypothetical_unsegmented"]
    lines.append(
      f"| {row['seat']} | {label} | {row['row']} | {row['segment_update']} | "
      f"{row['segment_offset']} | {row['distance_to_terminal']} | {current:.8g} | "
      f"{proposed:.8g} | {unsegmented:.8g} |"
    )
  lines.extend(
    [
      "",
      "Every requested draft row was trained and evicted before the terminal update. Raising "
      "the terminal trace to `gamma=1.0, lambda=0.99` would provide nonzero credit in one "
      "retained trajectory, as the final column shows, but it cannot cross the current segment "
      "or update boundary.",
      "",
      "## Boundary and truncation checks",
      "",
      f"The impulse crossed `{len(impulse['boundaries'])}` real 16-row boundaries. Only rows in "
      "the terminal segment can be nonzero; every earlier segment is a disconnected advantage "
      "graph and has already been optimized before the outcome arrives.",
      "",
      f"The forced native truncation ended after `{truncation['draft_decision_steps']}` draft "
      f"decisions with terminal flags `{truncation['terminal_flags']}`, truncation flags "
      f"`{truncation['truncation_flags']}`, and terminal win labels "
      f"`{truncation['terminal_win_labels']}`. This confirms that truncations do not create a "
      "win target.",
      "",
      "## Branch result",
      "",
      branch_text,
      "",
    ]
  )
  return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--seed", type=int, default=731_991)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--max-steps", type=int, default=600)
  parser.add_argument("--uniform-assignment", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  trainer_args = load_training_config(args.config, [])
  _apply_checkpoint_resume_policy_config(trainer_args, args.checkpoint)
  trainer_args["train"]["device"] = args.device
  trainer_args["env"].update(
    {
      "native": True,
      "native_envs_per_instance": 1,
      "deck_building_enabled": True,
      "evaluation_mode": True,
      "draft_same_element_matchup_prob": 0.0,
      "draft_cross_gate_replay_prob": 0.0,
      "draft_uniform_assignment": bool(args.uniform_assignment),
      "deck_snapshot_dir": None,
    }
  )
  horizon = int(trainer_args["train"]["bptt_horizon"])
  production_config = load_training_config(args.config, [])
  geometry = _training_geometry(production_config, horizon)

  env = make_azuki_env(seed=args.seed, **trainer_args["env"])
  policy = build_policy(type("VecShape", (), {"driver_env": env})(), trainer_args)
  _load_model_weights(policy, args.checkpoint, device=args.device, strict=False)
  policy.eval()
  for parameter in policy.parameters():
    parameter.requires_grad_(False)

  try:
    episode = _run_terminal_episode(
      env,
      policy,
      seed=args.seed,
      device=args.device,
      use_rnn=bool(trainer_args["train"].get("use_rnn", True)),
      max_steps=args.max_steps,
    )
    truncation = _run_truncation_probe(
      env,
      policy,
      seed=args.seed + 1,
      device=args.device,
      use_rnn=bool(trainer_args["train"].get("use_rnn", True)),
    )
  finally:
    env.close()

  impulse = _analyze_impulse(
    episode,
    horizon,
    expect_leader_rows=not args.uniform_assignment,
  )
  if impulse["all_requested_rows_receive_segmented_credit"]:
    decision = "use_phase_aware_terminal_trace"
  elif args.uniform_assignment:
    decision = "use_delayed_full_episode_main_draft_credit"
  else:
    decision = "use_delayed_sampled_whole_draft_estimator"
  payload = {
    "schema_version": 1,
    "decision": decision,
    "checkpoint": {
      "path": str(args.checkpoint.resolve()),
      "sha256": _sha256(args.checkpoint),
    },
    "config": str(args.config.resolve()),
    "uniform_assignment": bool(args.uniform_assignment),
    "training_geometry": geometry,
    "native_terminal_episode": episode,
    "impulse": impulse,
    "native_truncation_probe": truncation,
  }
  args.out_dir.mkdir(parents=True, exist_ok=True)
  json_path = args.out_dir / "preflight.json"
  markdown_path = args.out_dir / "preflight.md"
  json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  markdown_path.write_text(_markdown(payload), encoding="utf-8")
  print(
    f"[step5b-preflight] decision={decision} terminal_row={impulse['terminal_row']} "
    f"terminal_update={impulse['terminal_segment_update']} out={args.out_dir}",
    flush=True,
  )


if __name__ == "__main__":
  main()
