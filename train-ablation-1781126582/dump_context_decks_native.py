#!/usr/bin/env python3
"""Batched native dump of greedy and sampled decks for all uniform contexts."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

import azk_puffer.pytorch as azk_pytorch
import azk_puffer.vector as azk_vector
from analyze_decks import GATE_NAMES
from dump_context_decks import (
  _card_entries,
  _card_metadata,
  _deck_summary,
  _distribution_summary,
  _markdown,
)
from evaluate_checkpoint import _apply_checkpoint_resume_policy_config
from observation import DECKBUILD_OBSERVATION_CTYPE
from policy.v2 import tcg_sampler
from policy.v2.tcg_sampler import tcg_argmax_logits
from probe_gate_kl import GATE_CODE_PAIRS, OPPONENT_GATE
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


@dataclass(frozen=True)
class DraftSpec:
  context_index: int
  draft_index: int
  seed: int
  gate_id: int
  leader_id: int


def _build_contexts(catalog) -> list[dict[str, object]]:
  contexts: list[dict[str, object]] = []
  for element, gate_codes in GATE_CODE_PAIRS.items():
    leader_ids = catalog.leader_def_ids_by_element[element]
    for gate_code in gate_codes:
      for leader_id_raw in leader_ids:
        leader_id = int(leader_id_raw)
        leader_code = catalog.records_by_def_id[leader_id].card_code
        contexts.append({
          "context_id": f"{element}:{gate_code}:{leader_code}",
          "element": element,
          "gate_code": gate_code,
          "gate_id": int(catalog.records_by_code[gate_code].card_def_id),
          "leader_code": leader_code,
          "leader_id": leader_id,
        })
  if len(contexts) != 16:
    raise RuntimeError(f"Expected 16 contexts, found {len(contexts)}")
  return contexts


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


def _zero_state(bank, env_index: int) -> None:
  if bank is None:
    return
  bank["lstm_h"][env_index] = 0
  bank["lstm_c"][env_index] = 0


def _forward_shared(
  policy,
  observations: np.ndarray,
  env_indices: np.ndarray,
  row_indices: np.ndarray,
  candidate_role: np.ndarray,
  candidate_state,
  opponent_state,
  *,
  device: str,
  sampled: bool,
) -> np.ndarray:
  obs_t = torch.as_tensor(observations[row_indices], device=device)
  state = {"mask": torch.ones(row_indices.size, device=device, dtype=torch.bool)}
  env_t = torch.as_tensor(env_indices, device=device, dtype=torch.long)
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
    if sampled:
      actions, _, _ = azk_pytorch.sample_logits(logits)
    else:
      actions = tcg_argmax_logits(logits)
  if candidate_state is not None and opponent_state is not None:
    candidate_envs = env_t[role_t]
    opponent_envs = env_t[~role_t]
    candidate_state["lstm_h"][candidate_envs] = state["lstm_h"][role_t]
    candidate_state["lstm_c"][candidate_envs] = state["lstm_c"][role_t]
    opponent_state["lstm_h"][opponent_envs] = state["lstm_h"][~role_t]
    opponent_state["lstm_c"][opponent_envs] = state["lstm_c"][~role_t]
  return actions.to(dtype=torch.int32).cpu().numpy()


def _run_drafts(
  *,
  trainer_args: dict,
  policy,
  specs: list[DraftSpec],
  opponent_gate_id: int,
  opponent_leader_id: int,
  batch_envs: int,
  device: str,
  sampled: bool,
  sampling_seed: int,
) -> dict[tuple[int, int], Counter[int]]:
  if not specs:
    return {}
  env_count = min(int(batch_envs), len(specs))
  if env_count < 1:
    raise ValueError("batch_envs must be positive")
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
  env = make_azuki_env(seed=int(specs[0].seed), **env_cfg)
  use_rnn = bool(trainer_args.get("train", {}).get("use_rnn", True))
  candidate_state = _state_bank(policy, env_count, device, use_rnn)
  opponent_state = _state_bank(policy, env_count, device, use_rnn)
  packed_dtype = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
  queue_index = 0
  active_specs: list[DraftSpec | None] = [None] * env_count
  decision_steps = np.zeros(env_count, dtype=np.int16)
  results: dict[tuple[int, int], Counter[int]] = {}

  def assign(env_indices: list[int]) -> None:
    nonlocal queue_index
    resets: list[dict[str, int]] = []
    for env_index in env_indices:
      if queue_index >= len(specs):
        active_specs[env_index] = None
        continue
      spec = specs[queue_index]
      queue_index += 1
      active_specs[env_index] = spec
      decision_steps[env_index] = 0
      _zero_state(candidate_state, env_index)
      _zero_state(opponent_state, env_index)
      resets.append({
        "env_index": env_index,
        "seed": int(spec.seed),
        "gate0": int(spec.gate_id),
        "gate1": int(opponent_gate_id),
        "leader0": int(spec.leader_id),
        "leader1": int(opponent_leader_id),
      })
    if resets:
      env.reset_evaluation_games(resets)

  try:
    torch.manual_seed(int(sampling_seed))
    assign(list(range(env_count)))
    while len(results) < len(specs):
      packed = env.observations.view(packed_dtype).reshape(-1)
      completed: list[int] = []
      for env_index, spec in enumerate(active_specs):
        if spec is None:
          continue
        rows = packed[2 * env_index:2 * env_index + 2]
        modes = np.asarray(rows["deck_context"]["mode"], dtype=np.int32)
        counts = np.asarray(rows["deck_context"]["main_count"], dtype=np.int16)
        if bool(np.all(modes == 0) and np.all(counts == MAIN_PICKS)):
          main_ids = np.asarray(
            rows[0]["deck_context"]["main_card_def_ids"],
            dtype=np.int16,
          )
          if bool(np.any(main_ids < 0)):
            raise RuntimeError("Completed native draft contains an invalid card id")
          results[(spec.context_index, spec.draft_index)] = Counter(
            int(value) for value in main_ids.tolist()
          )
          completed.append(env_index)
      if completed:
        assign(completed)
        if len(results) >= len(specs):
          break

      active_players = env.active_players()
      running_envs = np.asarray([
        env_index
        for env_index, player in enumerate(active_players.tolist())
        if player >= 0 and active_specs[env_index] is not None
      ], dtype=np.int64)
      if running_envs.size == 0:
        raise RuntimeError("Native draft queue has no runnable environments")
      # EpisodeRunner forwards both seats on every decision, even though the
      # deck builder consumes only the active seat's action. Preserve that
      # recurrent-state history so this is a batched execution of the same
      # diagnostic rather than a subtly different policy evaluation.
      row_indices = np.column_stack(
        (2 * running_envs, 2 * running_envs + 1)
      ).reshape(-1)
      forwarded_envs = np.repeat(running_envs, 2)
      candidate_role = np.tile(np.asarray([True, False]), running_envs.size)
      env.actions.fill(0)
      env.actions[row_indices] = _forward_shared(
        policy,
        env.observations,
        forwarded_envs,
        row_indices,
        candidate_role,
        candidate_state,
        opponent_state,
        device=device,
        sampled=sampled,
      )
      env.step()
      decision_steps[running_envs] += 1
      if bool(np.any(decision_steps[running_envs] > 120)):
        raise RuntimeError("Native draft did not complete within 120 decisions")
  finally:
    env.close()
  return results


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--drafts-per-context", type=int, default=24)
  parser.add_argument("--temperature", type=float, default=1.2)
  parser.add_argument("--smoothing-eps", type=float, default=0.05)
  parser.add_argument("--batch-envs", type=int, default=48)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--json", type=Path, required=True)
  parser.add_argument("--md", type=Path, required=True)
  args = parser.parse_args()
  if args.drafts_per_context < 2:
    raise ValueError("--drafts-per-context must be at least 2")

  install_tcg_sampler()
  trainer_args = load_training_config(args.config, [])
  trainer_args["train"]["device"] = args.device
  trainer_args["env"]["draft_uniform_assignment"] = True
  policy = _build_policy(args.checkpoint, trainer_args, args.device)
  from deck_building import build_deck_build_catalog
  from training_deck_pool import load_training_deck_pool

  pool = load_training_deck_pool(trainer_args["env"].get("deck_pool_path"))
  catalog = build_deck_build_catalog(pool)
  contexts = _build_contexts(catalog)
  code_by_id = {
    int(card_id): record.card_code
    for card_id, record in catalog.records_by_def_id.items()
  }
  opponent_gate_id = int(catalog.records_by_code[OPPONENT_GATE].card_def_id)
  opponent_leader_id = int(catalog.leader_def_ids_by_element["WATER"][0])

  greedy_specs = [
    DraftSpec(
      context_index=index,
      draft_index=-1,
      seed=71_000_003,
      gate_id=int(context["gate_id"]),
      leader_id=int(context["leader_id"]),
    )
    for index, context in enumerate(contexts)
  ]
  stochastic_specs = [
    DraftSpec(
      context_index=index,
      draft_index=draft_index,
      seed=73_000_019 + 100_003 * draft_index,
      gate_id=int(context["gate_id"]),
      leader_id=int(context["leader_id"]),
    )
    for index, context in enumerate(contexts)
    for draft_index in range(args.drafts_per_context)
  ]

  tcg_sampler.set_sampling_params(
    subaction_temperature=1e-6,
    smoothing_eps=0.0,
    legal_row_temperature=1e-6,
    deck_pick_smoothing_eps=0.0,
  )
  greedy_by_key = _run_drafts(
    trainer_args=trainer_args,
    policy=policy,
    specs=greedy_specs,
    opponent_gate_id=opponent_gate_id,
    opponent_leader_id=opponent_leader_id,
    batch_envs=min(args.batch_envs, len(greedy_specs)),
    device=args.device,
    sampled=False,
    sampling_seed=71_000_003,
  )
  tcg_sampler.set_sampling_params(
    subaction_temperature=args.temperature,
    smoothing_eps=args.smoothing_eps,
    legal_row_temperature=args.temperature,
    deck_pick_smoothing_eps=args.smoothing_eps,
  )
  stochastic_by_key = _run_drafts(
    trainer_args=trainer_args,
    policy=policy,
    specs=stochastic_specs,
    opponent_gate_id=opponent_gate_id,
    opponent_leader_id=opponent_leader_id,
    batch_envs=args.batch_envs,
    device=args.device,
    sampled=True,
    sampling_seed=73_000_019,
  )

  metadata = _card_metadata()
  output_contexts: list[dict[str, object]] = []
  for context_index, context in enumerate(contexts):
    greedy_ids = greedy_by_key[(context_index, -1)]
    greedy = Counter({code_by_id[card_id]: count for card_id, count in greedy_ids.items()})
    stochastic_decks: list[Counter[str]] = []
    raw_decks: list[dict[str, object]] = []
    aggregate: Counter[str] = Counter()
    for draft_index in range(args.drafts_per_context):
      ids = stochastic_by_key[(context_index, draft_index)]
      deck = Counter({code_by_id[card_id]: count for card_id, count in ids.items()})
      if sum(deck.values()) != MAIN_PICKS:
        raise RuntimeError("Native context draft did not contain 50 main cards")
      stochastic_decks.append(deck)
      aggregate.update(deck)
      raw_decks.append({
        "draft_index": draft_index,
        "seed": 73_000_019 + 100_003 * draft_index,
        "summary": _deck_summary(deck, metadata),
        "cards": _card_entries(deck, metadata),
      })
    mean_deck = Counter({
      code: count / args.drafts_per_context for code, count in aggregate.items()
    })
    element = str(context["element"])
    gate_code = str(context["gate_code"])
    leader_code = str(context["leader_code"])
    output_contexts.append({
      "context_id": str(context["context_id"]),
      "element": element,
      "gate_code": gate_code,
      "gate_name": GATE_NAMES.get(gate_code, gate_code),
      "leader_code": leader_code,
      "leader_name": str(metadata[leader_code]["name"]),
      "greedy": {
        "summary": _deck_summary(greedy, metadata),
        "cards": _card_entries(greedy, metadata),
      },
      "stochastic": {
        "distribution": _distribution_summary(stochastic_decks),
        "mean_cards": _card_entries(mean_deck, metadata),
        "decks": raw_decks,
      },
    })
    print(
      f"[context-decks-native] {element} {gate_code} {leader_code} "
      f"greedy_unique={len(greedy)} stochastic_unique_mean="
      f"{_distribution_summary(stochastic_decks)['main_unique_mean']:.2f}",
      flush=True,
    )

  payload: dict[str, object] = {
    "schema_version": 1,
    "checkpoint": str(args.checkpoint.resolve()),
    "lifecycle": "uniform_gate_uniform_compatible_leader_50_main_picks",
    "stochastic_drafts_per_context": args.drafts_per_context,
    "sampling": {
      "temperature": args.temperature,
      "smoothing_eps": args.smoothing_eps,
    },
    "evaluator": {
      "version": "native-batched-draft-v1",
      "batch_envs": int(args.batch_envs),
      "sampling_seed": 73_000_019,
    },
    "contexts": output_contexts,
  }
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  args.md.parent.mkdir(parents=True, exist_ok=True)
  args.md.write_text(_markdown(payload), encoding="utf-8")
  print(f"[context-decks-native] wrote {args.json} and {args.md}", flush=True)


if __name__ == "__main__":
  main()
