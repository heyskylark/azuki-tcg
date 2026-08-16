#!/usr/bin/env python3
"""Batched native interventional draft KL for gate and leader context."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

import azk_puffer.pytorch as azk_pytorch
import azk_puffer.vector as azk_vector
from evaluate_checkpoint import _apply_checkpoint_resume_policy_config
from observation import DECKBUILD_OBSERVATION_CTYPE
from policy.v2 import tcg_sampler
from probe_context_kl import _group, _markdown, _quartile, _summary
from probe_gate_kl import GATE_CODE_PAIRS, OPPONENT_GATE, kl, tv
from train import _load_model_weights
from training_utils import (
  build_policy,
  build_vecenv,
  install_tcg_sampler,
  load_training_config,
  make_azuki_env,
)


DEFAULT_CONFIG = Path("python/config/azuki_deckbuild_native_3090.ini")
MAIN_PICKS = 50


def _legal_row_probs(
  logits: torch.Tensor,
  legal_count: int,
  *,
  temperature: float,
  smoothing_eps: float,
) -> np.ndarray:
  probabilities = torch.softmax(logits[:legal_count] / temperature, dim=-1)
  if smoothing_eps > 0.0:
    probabilities = (
      (1.0 - smoothing_eps) * probabilities
      + smoothing_eps / float(legal_count)
    )
  return probabilities.detach().float().cpu().numpy()


@dataclass(frozen=True)
class CompareSpec:
  index: int
  intervention: str
  element: str
  fixed_context: str
  source_gate: str
  target_gate: str
  source_leader: str
  target_leader: str
  history_index: int
  seed: int


def _build_policy(checkpoint: Path, trainer_args: dict, device: str):
  policy_args = copy.deepcopy(trainer_args)
  _apply_checkpoint_resume_policy_config(policy_args, checkpoint)
  policy_args["train"]["device"] = device
  policy_args["env"]["native"] = True
  policy_args["env"]["native_envs_per_instance"] = 1
  policy_args["env"]["draft_uniform_assignment"] = True
  vecenv = build_vecenv(
    policy_args,
    backend=azk_vector.Serial,
    num_envs=1,
    seed=7,
  )
  try:
    policy = build_policy(vecenv, policy_args)
    _load_model_weights(policy, checkpoint, device=device, strict=False)
  finally:
    vecenv.close()
  policy.eval()
  for parameter in policy.parameters():
    parameter.requires_grad_(False)
  return policy


def _state_bank(policy, batch_envs: int, device: str, use_rnn: bool):
  if not use_rnn:
    return None
  return {
    "lstm_h": torch.zeros(batch_envs, policy.hidden_size, device=device),
    "lstm_c": torch.zeros(batch_envs, policy.hidden_size, device=device),
  }


def _context_for_phase(spec: CompareSpec, phase: str) -> tuple[str, str]:
  if phase == "target":
    return spec.target_gate, spec.target_leader
  if phase in ("source", "control"):
    return spec.source_gate, spec.source_leader
  raise ValueError(f"Unknown replay phase: {phase}")


def _run_phase(
  *,
  trainer_args: dict,
  policy,
  specs: list[CompareSpec],
  catalog,
  opponent_gate_id: int,
  opponent_leader_id: int,
  batch_envs: int,
  device: str,
  phase: str,
  source_actions: dict[tuple[int, int], np.ndarray],
  sampling_seed: int,
  temperature: float,
  smoothing_eps: float,
) -> dict[int, dict[str, list[np.ndarray]]]:
  if phase not in ("source", "target", "control"):
    raise ValueError(f"Unknown replay phase: {phase}")
  outputs: dict[int, dict[str, list[np.ndarray]]] = {}
  packed_dtype = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
  use_rnn = bool(trainer_args.get("train", {}).get("use_rnn", True))
  for chunk_start in range(0, len(specs), int(batch_envs)):
    chunk = specs[chunk_start:chunk_start + int(batch_envs)]
    env_count = len(chunk)
    env_cfg = dict(trainer_args.get("env", {}) or {})
    env_cfg.update({
      "native": True,
      "native_envs_per_instance": env_count,
      "deck_building_enabled": True,
      "evaluation_mode": True,
      "draft_uniform_assignment": True,
      "draft_same_element_matchup_prob": 0.0,
      "draft_cross_gate_replay_prob": 0.0,
      "deck_snapshot_dir": None,
    })
    env = make_azuki_env(seed=int(chunk[0].seed), **env_cfg)
    candidate_state = _state_bank(policy, env_count, device, use_rnn)
    opponent_state = _state_bank(policy, env_count, device, use_rnn)
    active = np.ones(env_count, dtype=np.bool_)
    decision_steps = np.zeros(env_count, dtype=np.int16)
    resets: list[dict[str, int]] = []
    for env_index, spec in enumerate(chunk):
      gate_code, leader_code = _context_for_phase(spec, phase)
      resets.append({
        "env_index": env_index,
        "seed": int(spec.seed),
        "gate0": int(catalog.records_by_code[gate_code].card_def_id),
        "gate1": int(opponent_gate_id),
        "leader0": int(catalog.records_by_code[leader_code].card_def_id),
        "leader1": int(opponent_leader_id),
      })
      outputs[spec.index] = {"probs": [], "candidates": []}
    try:
      torch.manual_seed(int(sampling_seed) + chunk_start)
      env.reset_evaluation_games(resets)
      while bool(active.any()):
        packed = env.observations.view(packed_dtype).reshape(-1)
        for env_index in np.nonzero(active)[0].tolist():
          rows = packed[2 * env_index:2 * env_index + 2]
          modes = np.asarray(rows["deck_context"]["mode"], dtype=np.int32)
          counts = np.asarray(rows["deck_context"]["main_count"], dtype=np.int16)
          if bool(np.all(modes == 0) and np.all(counts == MAIN_PICKS)):
            active[env_index] = False
        if not bool(active.any()):
          break

        active_players = env.active_players()
        running_envs = np.asarray([
          env_index
          for env_index in np.nonzero(active)[0].tolist()
          if int(active_players[env_index]) >= 0
        ], dtype=np.int64)
        if running_envs.size == 0:
          raise RuntimeError(f"Native {phase} replay has no runnable environments")
        active_seats = active_players[running_envs].astype(np.int64)
        row_indices = np.column_stack(
          (2 * running_envs, 2 * running_envs + 1)
        ).reshape(-1)
        forwarded_envs = np.repeat(running_envs, 2)
        candidate_role = np.tile(np.asarray([True, False]), running_envs.size)
        obs_t = torch.as_tensor(env.observations[row_indices], device=device)
        state = {
          "mask": torch.ones(row_indices.size, device=device, dtype=torch.bool)
        }
        env_t = torch.as_tensor(forwarded_envs, device=device, dtype=torch.long)
        role_t = torch.as_tensor(candidate_role, device=device, dtype=torch.bool)
        if candidate_state is not None and opponent_state is not None:
          state["lstm_h"] = torch.where(
            role_t[:, None],
            candidate_state["lstm_h"][env_t],
            opponent_state["lstm_h"][env_t],
          )
          state["lstm_c"] = torch.where(
            role_t[:, None],
            candidate_state["lstm_c"][env_t],
            opponent_state["lstm_c"][env_t],
          )
        with torch.inference_mode():
          logits, _ = policy.forward_eval(obs_t, state)
          sampled_actions = None
          if phase == "source":
            sampled_actions, _, _ = azk_pytorch.sample_logits(logits)

        candidate_locals = 2 * np.nonzero(active_seats == 0)[0]
        for local_index_raw in candidate_locals.tolist():
          local_index = int(local_index_raw)
          env_index = int(running_envs[local_index // 2])
          spec = chunk[env_index]
          legal_count = int(logits.legal_action_count[local_index])
          legal_count = max(legal_count, 1)
          probabilities = _legal_row_probs(
            logits.legal_action_logits[local_index],
            legal_count,
            temperature=temperature,
            smoothing_eps=smoothing_eps,
          )
          row = packed[int(row_indices[local_index])]["deck_context"]
          candidate_count = int(row["candidate_count"])
          candidates = np.asarray(
            row["candidate_card_def_ids"][:candidate_count],
            dtype=np.int16,
          ).copy()
          outputs[spec.index]["probs"].append(probabilities)
          outputs[spec.index]["candidates"].append(candidates)

        if candidate_state is not None and opponent_state is not None:
          candidate_envs = env_t[role_t]
          opponent_envs = env_t[~role_t]
          candidate_state["lstm_h"][candidate_envs] = state["lstm_h"][role_t]
          candidate_state["lstm_c"][candidate_envs] = state["lstm_c"][role_t]
          opponent_state["lstm_h"][opponent_envs] = state["lstm_h"][~role_t]
          opponent_state["lstm_c"][opponent_envs] = state["lstm_c"][~role_t]

        env.actions.fill(0)
        if phase == "source":
          if sampled_actions is None:
            raise RuntimeError("Source replay did not sample actions")
          actions_np = sampled_actions.to(dtype=torch.int32).cpu().numpy()
          env.actions[row_indices] = actions_np
          for local_index, env_index_raw in enumerate(running_envs.tolist()):
            env_index = int(env_index_raw)
            spec = chunk[env_index]
            source_actions[(spec.index, int(decision_steps[env_index]))] = (
              actions_np[2 * local_index:2 * local_index + 2].copy()
            )
        else:
          replay_actions = []
          for env_index_raw in running_envs.tolist():
            env_index = int(env_index_raw)
            spec = chunk[env_index]
            key = (spec.index, int(decision_steps[env_index]))
            if key not in source_actions:
              raise RuntimeError(f"Missing source action for comparison {key}")
            replay_actions.append(source_actions[key])
          env.actions[row_indices] = np.asarray(
            replay_actions, dtype=np.int32
          ).reshape(-1, 4)
        env.step()
        decision_steps[running_envs] += 1
        if bool(np.any(decision_steps[running_envs] > 120)):
          raise RuntimeError(f"Native {phase} replay exceeded 120 decisions")
    finally:
      env.close()

    for env_index, spec in enumerate(chunk):
      output = outputs[spec.index]
      if len(output["probs"]) != MAIN_PICKS:
        raise RuntimeError(
          f"Comparison {spec.index} {phase} recorded "
          f"{len(output['probs'])} candidate picks"
        )
      if int(decision_steps[env_index]) != 2 * MAIN_PICKS:
        raise RuntimeError(
          f"Comparison {spec.index} {phase} used "
          f"{int(decision_steps[env_index])} decisions"
        )
  return outputs


def _build_specs(catalog, histories: int) -> list[CompareSpec]:
  code_by_id = {
    int(card_id): record.card_code
    for card_id, record in catalog.records_by_def_id.items()
  }
  leaders_by_element = {
    element: tuple(
      code_by_id[int(leader_id)]
      for leader_id in catalog.leader_def_ids_by_element[element]
    )
    for element in GATE_CODE_PAIRS
  }
  raw: list[dict[str, object]] = []
  for element, (gate_a, gate_b) in GATE_CODE_PAIRS.items():
    leader_a, leader_b = leaders_by_element[element]
    for leader in (leader_a, leader_b):
      for source_gate, target_gate in ((gate_a, gate_b), (gate_b, gate_a)):
        for history_index in range(histories):
          raw.append({
            "intervention": "gate_given_leader",
            "element": element,
            "fixed_context": leader,
            "source_gate": source_gate,
            "target_gate": target_gate,
            "source_leader": leader,
            "target_leader": leader,
            "history_index": history_index,
          })
    for gate in (gate_a, gate_b):
      for source_leader, target_leader in (
        (leader_a, leader_b),
        (leader_b, leader_a),
      ):
        for history_index in range(histories):
          raw.append({
            "intervention": "leader_given_gate",
            "element": element,
            "fixed_context": gate,
            "source_gate": gate,
            "target_gate": gate,
            "source_leader": source_leader,
            "target_leader": target_leader,
            "history_index": history_index,
          })
  return [
    CompareSpec(
      index=index,
      seed=83_000_021 + 100_003 * int(item["history_index"]),
      **item,
    )
    for index, item in enumerate(raw)
  ]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--histories", type=int, default=4)
  parser.add_argument("--temperature", type=float, default=1.2)
  parser.add_argument("--smoothing-eps", type=float, default=0.05)
  parser.add_argument("--batch-envs", type=int, default=128)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--json", type=Path, required=True)
  parser.add_argument("--md", type=Path, required=True)
  args = parser.parse_args()
  if args.histories < 1:
    raise ValueError("--histories must be positive")
  if args.batch_envs < 1:
    raise ValueError("--batch-envs must be positive")

  install_tcg_sampler()
  tcg_sampler.set_sampling_params(
    subaction_temperature=args.temperature,
    smoothing_eps=args.smoothing_eps,
    legal_row_temperature=args.temperature,
    deck_pick_smoothing_eps=args.smoothing_eps,
  )
  trainer_args = load_training_config(args.config, [])
  trainer_args["train"]["device"] = args.device
  trainer_args["env"]["draft_uniform_assignment"] = True
  policy = _build_policy(args.checkpoint, trainer_args, args.device)
  from deck_building import build_deck_build_catalog
  from training_deck_pool import load_training_deck_pool

  pool = load_training_deck_pool(trainer_args["env"].get("deck_pool_path"))
  catalog = build_deck_build_catalog(pool)
  specs = _build_specs(catalog, args.histories)
  opponent_gate_id = int(catalog.records_by_code[OPPONENT_GATE].card_def_id)
  opponent_leader_id = int(catalog.leader_def_ids_by_element["WATER"][0])
  source_actions: dict[tuple[int, int], np.ndarray] = {}
  common = {
    "trainer_args": trainer_args,
    "policy": policy,
    "specs": specs,
    "catalog": catalog,
    "opponent_gate_id": opponent_gate_id,
    "opponent_leader_id": opponent_leader_id,
    "batch_envs": args.batch_envs,
    "device": args.device,
    "source_actions": source_actions,
    "sampling_seed": 83_000_021,
    "temperature": args.temperature,
    "smoothing_eps": args.smoothing_eps,
  }
  source = _run_phase(phase="source", **common)
  target = _run_phase(phase="target", **common)
  control = _run_phase(phase="control", **common)

  rows: list[dict[str, object]] = []
  for spec in specs:
    source_output = source[spec.index]
    target_output = target[spec.index]
    control_output = control[spec.index]
    for pick_index in range(MAIN_PICKS):
      source_candidates = source_output["candidates"][pick_index]
      if not np.array_equal(
        source_candidates, target_output["candidates"][pick_index]
      ):
        raise RuntimeError("Intervention changed the main-card candidate set")
      if not np.array_equal(
        source_candidates, control_output["candidates"][pick_index]
      ):
        raise RuntimeError("Determinism replay changed the candidate set")
      source_probs = source_output["probs"][pick_index]
      target_probs = target_output["probs"][pick_index]
      control_probs = control_output["probs"][pick_index]
      forward_kl = kl(source_probs, target_probs)
      reverse_kl = kl(target_probs, source_probs)
      rows.append({
        "intervention": spec.intervention,
        "element": spec.element,
        "fixed_context": spec.fixed_context,
        "source_gate": spec.source_gate,
        "target_gate": spec.target_gate,
        "source_leader": spec.source_leader,
        "target_leader": spec.target_leader,
        "direction": (
          f"{spec.source_gate}:{spec.source_leader}->"
          f"{spec.target_gate}:{spec.target_leader}"
        ),
        "history_index": spec.history_index,
        "seed": spec.seed,
        "main_pick": pick_index + 1,
        "quartile": _quartile(pick_index + 1),
        "kl_source_to_target": forward_kl,
        "kl_target_to_source": reverse_kl,
        "symmetric_kl": 0.5 * (forward_kl + reverse_kl),
        "total_variation": tv(source_probs, target_probs),
        "determinism_kl": kl(source_probs, control_probs),
      })

  by_intervention = {
    intervention: [row for row in rows if row["intervention"] == intervention]
    for intervention in ("gate_given_leader", "leader_given_gate")
  }
  payload: dict[str, object] = {
    "schema_version": 1,
    "checkpoint": str(args.checkpoint.resolve()),
    "lifecycle": "uniform_gate_uniform_compatible_leader_50_main_picks",
    "histories_per_direction": args.histories,
    "sampling": {
      "temperature": args.temperature,
      "smoothing_eps": args.smoothing_eps,
    },
    "evaluator": {
      "version": "native-batched-context-replay-v1",
      "batch_envs": int(args.batch_envs),
      "sampling_seed": 83_000_021,
    },
    "aggregate": {
      intervention: _summary(selected)
      for intervention, selected in by_intervention.items()
    },
    "by_element": {
      intervention: _group(selected, "element")
      for intervention, selected in by_intervention.items()
    },
    "by_direction": {
      intervention: _group(selected, "direction")
      for intervention, selected in by_intervention.items()
    },
    "by_quartile": {
      intervention: _group(selected, "quartile")
      for intervention, selected in by_intervention.items()
    },
    "rows": rows,
  }
  control_max = max(
    summary["determinism_kl_max"]
    for summary in payload["aggregate"].values()
  )
  if float(control_max) > 1e-7:
    raise RuntimeError(f"Determinism control KL exceeded tolerance: {control_max}")
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  args.md.parent.mkdir(parents=True, exist_ok=True)
  args.md.write_text(_markdown(payload), encoding="utf-8")
  print(
    "[context-kl-native] "
    + " ".join(
      f"{name}={summary['symmetric_kl_mean']:.8f}"
      for name, summary in payload["aggregate"].items()
    ),
    flush=True,
  )


if __name__ == "__main__":
  main()
