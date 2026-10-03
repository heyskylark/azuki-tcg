#!/usr/bin/env python3
"""Paired behavior traces: candidate policy (specialist / u8223 / u9305) vs a fixed u8223 opponent.

Adapted from strategy_causal_followup_v1/evaluate_followup.py (CrossoverPolicy, assigned_states)
and run_prefix_paired_eval.py. Only the active seat advances its own recurrent state; each seat
has private draft/battle sampling streams derived from the world seed, so every arm sees the same
world seed, deck shuffles, seats and opponent sampling stream (trajectories diverge once actions do).

Suites:
  fixed       candidate plays a curated deck of its element; opponent plays a panel deck.
  free_draft  candidate drafts 50 cards under a (gate, leader) context of its element; opponent
              plays a panel deck.

Run from the snapshot root with PYTHONPATH=build/python/src:python/src:<this dir>.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from deck_building import PlayerDeckBuildState, build_deck_build_catalog
from evaluate_checkpoint import _apply_checkpoint_resume_policy_config
from play_selfplay_games import GameLogger
from policy.tcg_distribution import TCGLegalActionDistribution
from policy.v2 import tcg_sampler
from probe_gate_kl import EpisodeRunner
from train import _load_model_weights
from training_deck_pool import load_training_deck_pool
from training_utils import build_policy, load_training_config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RT = Path("/home/skylark/git/azuki-tcg-specialist-runtime")
POOL = RT / "train-specialist/decks/curated_deck_pool.json"
CONFIG = ROOT / "behavior_eval.ini"
_A = Path("/home/skylark/git/azuki-tcg/train-ablation-1781126582/results/control_strategy_scale_30m_v1/runs/control/artifacts")
_S = RT / "train-specialist/runs"
CHECKPOINTS = {
  "u8223": _A / "azuki_local_control_strategy_scale_30m_s43_179038406361/model_azuki_local_008223.pt",
  "u9305": _A / "azuki_local_control_strategy_scale_30m_s43_179039793722/model_azuki_local_009305.pt",
  "water": _S / "water/artifacts/azuki_local_specialist_water_s43_179083097333/model_azuki_local_011361.pt",
  "earth": _S / "earth/artifacts/azuki_local_specialist_earth_s43_179084927273/model_azuki_local_011422.pt",
  "lightning": _S / "lightning/artifacts/azuki_local_specialist_lightning_s43_179087061766/model_azuki_local_011502.pt",
  "fire": _S / "fire/artifacts/azuki_local_specialist_fire_s43_179089277894/model_azuki_local_011477.pt",
  # Earth continuation (runs/earth_c1): +75M (best paired strength, 106M total) and final +100M (130M total).
  "earth_c75": _S / "earth_c1/artifacts/azuki_local_specialist_earth_s43_179098246978/model_azuki_local_019500.pt",
  "earth_c100": _S / "earth_c1/artifacts/azuki_local_specialist_earth_s43_179098246978/model_azuki_local_022062.pt",
}
OPPONENT = "u8223"
OPP_PANEL = (0, 2, 3, 4, 11, 12, 13, 14)  # Rushfire, Ragefire, Surge, Stormchain, Hydromancy x2, Stonehaven Goro/Bobu
FIXED_CANDIDATE_DECKS = {  # every curated deck of the element; Fire capped to 3 Rushfire + 3 Ragefire
  "FIRE": (0, 1, 5, 2, 16, 17),
  "LIGHTNING": (3, 4, 9, 10),
  "WATER": (11, 12),
  "EARTH": (13, 14, 15),
}
REPLICATES = {"fixed": 4, "free_draft": 3}
STEP_CAP = 1200


def world_seed(*parts) -> int:
  value = "azuki.specialist_behavior.v1:" + ":".join(str(p) for p in parts)
  return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") % (2**31 - 1)


def stream_seed(seed: int, seat: int, phase: str) -> int:
  value = f"azuki.specialist_behavior.v1:{seed}:{seat}:{phase}"
  return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**63 - 1)


def build_tasks() -> list[dict]:
  pool_json = json.loads(POOL.read_text())
  catalog = build_deck_build_catalog(load_training_deck_pool(POOL))
  rec = catalog.records_by_def_id
  gates = sorted({int(g) for g in catalog.gate_def_id_population})
  tasks = []
  for element in ("FIRE", "LIGHTNING", "WATER", "EARTH"):
    for rep in range(REPLICATES["fixed"]):
      for cand in FIXED_CANDIDATE_DECKS[element]:
        deck = pool_json["decks"][cand]
        for opp in OPP_PANEL:
          block = f"fixed:{element}:r{rep}:cand{cand}:opp{opp}"
          seed = world_seed(block)
          for seat in (0, 1):
            tasks.append({
              "task_id": f"{block}:seat{seat}", "block_id": block, "suite": "fixed", "element": element,
              "seed": seed, "candidate_seat": seat, "candidate_deck_index": cand,
              "candidate_gate": deck["gate_card_id"], "candidate_leader": deck["leader_card_id"],
              "opponent_deck_index": opp, "opponent_gate": pool_json["decks"][opp]["gate_card_id"],
              "opponent_leader": pool_json["decks"][opp]["leader_card_id"], "replicate": rep,
            })
    contexts = [
      (rec[g].card_code, rec[l].card_code)
      for g in gates if rec[g].element == element
      for l in catalog.leader_def_ids_by_element[element]
    ]
    for rep in range(REPLICATES["free_draft"]):
      for gate, leader in contexts:
        for opp in OPP_PANEL:
          block = f"free_draft:{element}:r{rep}:{gate}:{leader}:opp{opp}"
          seed = world_seed(block)
          for seat in (0, 1):
            tasks.append({
              "task_id": f"{block}:seat{seat}", "block_id": block, "suite": "free_draft", "element": element,
              "seed": seed, "candidate_seat": seat, "candidate_deck_index": None,
              "candidate_gate": gate, "candidate_leader": leader,
              "opponent_deck_index": opp, "opponent_gate": pool_json["decks"][opp]["gate_card_id"],
              "opponent_leader": pool_json["decks"][opp]["leader_card_id"], "replicate": rep,
            })
  return tasks


class SeatPolicy:
  """Active-seat dispatcher (GameLogger-compatible); never shares hidden state across seats."""

  def __init__(self, runner, policies, seed, mode, device):
    self.runner = runner
    self.policies = policies
    self.mode = mode
    self.device = device
    self.states = [{n: torch.zeros(1, p.hidden_size, device=device) for n in ("lstm_h", "lstm_c")} for p in policies]
    self.seeds = [[stream_seed(seed, seat, phase) for phase in ("draft", "battle")] for seat in range(2)]
    self.rng = [[torch.Generator().manual_seed(v).get_state() for v in pair] for pair in self.seeds]
    self.counts = [[0, 0], [0, 0]]

  def forward_eval(self, observations, state):
    seat = int(self.runner.base_env._active_player_index)
    phase = 0 if self.runner.base_env._building else 1
    row = observations[seat:seat + 1].to(self.device)
    sub_state = {n: v.expand(2, -1).contiguous() for n, v in self.states[seat].items()}
    sub_state["mask"] = state["mask"][seat:seat + 1].repeat(2).to(self.device)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
      torch.set_rng_state(self.rng[seat][phase])
      out, _ = self.policies[seat].forward_eval(row.expand(2, *row.shape[1:]), sub_state)
      # Sample on CPU so the per-seat/phase CPU generator streams stay authoritative.
      logits = TCGLegalActionDistribution(out.legal_action_logits.float().cpu(), out.legal_actions.cpu(),
                                          out.legal_action_count.cpu())
      if self.mode == "sample":
        actions, _, _ = tcg_sampler.tcg_sample_logits(logits)
      else:
        actions = tcg_sampler.tcg_argmax_logits(logits)
      self.rng[seat][phase] = torch.get_rng_state()
    self.states[seat] = {n: sub_state[n][:1].clone() for n in ("lstm_h", "lstm_c")}
    self.counts[seat][phase] += 1
    chosen = torch.zeros(2, 1, 4, dtype=torch.long)
    chosen[seat, 0] = actions[0]
    return TCGLegalActionDistribution(torch.zeros(2, 1), chosen, torch.ones(2, dtype=torch.long)), torch.zeros(2)


def load_policy(runner, checkpoint: Path, device: str):
  config = load_training_config(CONFIG, [])
  config["train"]["device"] = device
  _apply_checkpoint_resume_policy_config(config, checkpoint)
  policy = build_policy(runner.vecenv, config)
  runner.vecenv.async_reset(seed=7)
  obs, _, _, _, _, _, masks = runner.vecenv.recv()
  state = {n: torch.zeros(2, policy.hidden_size, device=device) for n in ("lstm_h", "lstm_c")}
  state["mask"] = torch.as_tensor(masks, device=device)
  with torch.no_grad():
    policy.forward_eval(torch.as_tensor(obs, device=device), state)
  _load_model_weights(policy, checkpoint, device=device, strict=True)
  policy.eval()
  policy.requires_grad_(False)
  return policy


def initial_states(runner, task, pool):
  states = []
  for seat in range(2):
    if seat == task["candidate_seat"]:
      if task["candidate_deck_index"] is not None:
        st = runner.base_env._fixed_state_from_deck(pool[task["candidate_deck_index"]])
      else:
        st = PlayerDeckBuildState.create(runner.code_to_def[task["candidate_gate"]])
        st.leader_card_def_id = runner.code_to_def[task["candidate_leader"]]
      prefix = "candidate"
    else:
      st = runner.base_env._fixed_state_from_deck(pool[task["opponent_deck_index"]])
      prefix = "opponent"
    rec = runner.catalog.records_by_def_id
    if (rec[st.gate_card_def_id].card_code, rec[st.leader_card_def_id].card_code) != (task[prefix + "_gate"], task[prefix + "_leader"]):
      raise ValueError(f"context mismatch {task['task_id']} seat{seat}")
    states.append(st)
  return states


def validate(record, task):
  errors = []
  out = record["outcome"]
  if not out.get("terminated") or out.get("truncated"):
    errors.append("incomplete_game")
  decks = record.get("decks")
  if not decks:
    return errors + ["missing_decks"]
  for seat in range(2):
    prefix = "candidate" if seat == task["candidate_seat"] else "opponent"
    if (decks[seat]["gate"], decks[seat]["leader"]) != (task[prefix + "_gate"], task[prefix + "_leader"]) or len(decks[seat]["main"]) != 50:
      errors.append(f"seat{seat}_context_or_size")
  if any(step["a"] not in step["legal"] for step in record["steps"]):
    errors.append("illegal_battle_action")
  return errors


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--element", required=True)
  ap.add_argument("--arm", required=True, help="checkpoint key for the candidate seat")
  ap.add_argument("--mode", choices=("argmax", "sample"), required=True)
  ap.add_argument("--suites", default="fixed,free_draft")
  ap.add_argument("--shards", type=int, default=1)
  ap.add_argument("--shard-index", type=int, default=0)
  ap.add_argument("--limit", type=int)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--device", default="cuda")
  args = ap.parse_args()
  torch.set_num_threads(1)
  suites = set(args.suites.split(","))
  tasks = [t for t in build_tasks() if t["element"] == args.element.upper() and t["suite"] in suites]
  blocks = list(dict.fromkeys(t["block_id"] for t in tasks))
  selected = set(blocks[args.shard_index::args.shards])
  tasks = [t for t in tasks if t["block_id"] in selected][: args.limit]
  runner = EpisodeRunner(CONFIG, CHECKPOINTS[OPPONENT], args.device, uniform_assignment=True)
  try:
    opponent = load_policy(runner, CHECKPOINTS[OPPONENT], args.device)
    candidate = opponent if args.arm == OPPONENT else load_policy(runner, CHECKPOINTS[args.arm], args.device)
    runner.policy = None  # free the EpisodeRunner's own copy
    if args.device.startswith("cuda"):
      torch.cuda.empty_cache()
    tcg_sampler.set_sampling_params(primary_temperature=1.0, subaction_temperature=1.0, smoothing_eps=0.0,
                                    legal_row_temperature=1.0, deck_pick_smoothing_eps=0.0)
    runner.use_rnn = False  # dispatcher owns recurrent state
    pool = load_training_deck_pool(POOL)
    logger = GameLogger(runner, log_legal_actions=True, action_mode=args.mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.out.open("w", encoding="utf-8") as stream:
      for index, task in enumerate(tasks):
        init = initial_states(runner, task, pool)
        runner.base_env._initial_states = lambda init=init: copy.deepcopy(init)
        policies = [None, None]
        policies[task["candidate_seat"]] = candidate
        policies[1 - task["candidate_seat"]] = opponent
        dispatcher = SeatPolicy(runner, policies, task["seed"], args.mode, args.device)
        runner.policy = dispatcher
        record = logger.play_game(index, task["seed"], STEP_CAP)
        errors = validate(record, task)
        winner = record["outcome"]["winner"]
        record["paired_eval"] = {
          **task, "arm": args.arm, "mode": args.mode, "opponent_id": OPPONENT,
          "candidate_checkpoint": str(CHECKPOINTS[args.arm]),
          "candidate_score": 0.5 if winner == -1 else float(winner == task["candidate_seat"]),
          "policy_action_counts": dispatcher.counts, "validation_errors": errors,
          "sampling": tcg_sampler.get_sampling_params(),
        }
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        stream.flush()
        if index % 20 == 0:
          print(json.dumps({"done": index + 1, "of": len(tasks), "errors": errors,
                            "elapsed": round(time.monotonic() - started, 1)}), flush=True)
  finally:
    runner.vecenv.close()


if __name__ == "__main__":
  main()
