from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import pufferlib.pytorch
import pufferlib.vector


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


@dataclass
class MatchResult:
  wins_a: int
  wins_b: int
  draws: int
  episodes: int

  @property
  def win_rate_a(self) -> float:
    if self.episodes <= 0:
      return 0.0
    return float(self.wins_a) / float(self.episodes)

  @property
  def score_a(self) -> float:
    if self.episodes <= 0:
      return 0.5
    return (float(self.wins_a) + 0.5 * float(self.draws)) / float(self.episodes)


@dataclass
class MatchRequest:
  episodes: int
  max_steps: int
  seed: int
  device: str


class LeagueEvaluator:
  def evaluate(self, trainer_args: dict, *, policy_a, policy_b, request: MatchRequest) -> MatchResult:  # pragma: no cover - interface
    raise NotImplementedError


class InlineLeagueEvaluator(LeagueEvaluator):
  def evaluate(self, trainer_args: dict, *, policy_a, policy_b, request: MatchRequest) -> MatchResult:
    from training_utils import build_vecenv

    episodes = int(request.episodes)
    max_steps = int(request.max_steps)
    device = str(request.device)
    seed = int(request.seed)
    if episodes <= 0:
      return MatchResult(wins_a=0, wins_b=0, draws=0, episodes=0)

    vecenv = build_vecenv(
      trainer_args,
      backend=pufferlib.vector.Serial,
      num_envs=1,
      seed=seed,
    )
    base_env = _unwrap_base_env(vecenv.envs[0])
    num_agents = int(vecenv.num_agents)
    use_rnn = bool(trainer_args.get("train", {}).get("use_rnn", True))

    # Preserve caller mode (learner policy may be in train mode) and restore on exit.
    policy_a_was_training = bool(policy_a.training)
    policy_b_was_training = bool(policy_b.training)
    policy_a.eval()
    policy_b.eval()
    wins_a = 0
    wins_b = 0
    draws = 0

    try:
      for ep in range(episodes):
        seat_of_a = 0 if ep % 2 == 0 else 1
        seat_of_b = 1 - seat_of_a
        lstm_state_a = {}
        lstm_state_b = {}
        if use_rnn:
          lstm_state_a["lstm_h"] = torch.zeros(num_agents, policy_a.hidden_size, device=device)
          lstm_state_a["lstm_c"] = torch.zeros(num_agents, policy_a.hidden_size, device=device)
          lstm_state_b["lstm_h"] = torch.zeros(num_agents, policy_b.hidden_size, device=device)
          lstm_state_b["lstm_c"] = torch.zeros(num_agents, policy_b.hidden_size, device=device)

        vecenv.async_reset(seed=seed + ep)
        obs, _, _, _, _, _, masks = vecenv.recv()
        done = False
        steps = 0

        while not done and steps < max_steps:
          active = int(getattr(base_env, "_active_player_index", 0))
          acting_policy = policy_a if active == seat_of_a else policy_b
          acting_state = lstm_state_a if active == seat_of_a else lstm_state_b

          step_state = {"mask": torch.as_tensor(masks, device=device)}
          if use_rnn:
            step_state["lstm_h"] = acting_state["lstm_h"]
            step_state["lstm_c"] = acting_state["lstm_c"]
          obs_t = torch.as_tensor(obs, device=device)
          with torch.no_grad():
            logits, _ = acting_policy.forward_eval(obs_t, step_state)
            actions, _, _ = pufferlib.pytorch.sample_logits(logits)

          if use_rnn:
            acting_state["lstm_h"] = step_state["lstm_h"]
            acting_state["lstm_c"] = step_state["lstm_c"]

          action_np = np.zeros((num_agents, 4), dtype=np.int32)
          action_np[active] = actions[active].detach().cpu().numpy().astype(np.int32)
          vecenv.send(action_np)
          obs, _, _, _, _, _, masks = vecenv.recv()
          done = bool(vecenv.envs[0].done)
          steps += 1

        info_a = base_env.infos.get(seat_of_a, {}) if hasattr(base_env, "infos") else {}
        info_b = base_env.infos.get(seat_of_b, {}) if hasattr(base_env, "infos") else {}
        win_a = float(info_a.get("win", 0.0))
        win_b = float(info_b.get("win", 0.0))
        if win_a >= 0.5:
          wins_a += 1
        elif win_b >= 0.5:
          wins_b += 1
        else:
          draws += 1
    finally:
      if policy_a_was_training:
        policy_a.train()
      else:
        policy_a.eval()
      if policy_b_was_training:
        policy_b.train()
      else:
        policy_b.eval()
      vecenv.close()

    return MatchResult(wins_a=wins_a, wins_b=wins_b, draws=draws, episodes=episodes)


def make_league_evaluator(mode: str) -> LeagueEvaluator:
  normalized = str(mode).strip().lower()
  if normalized in {"inline", ""}:
    return InlineLeagueEvaluator()
  raise ValueError(
    f"Unsupported league evaluator mode '{mode}'. "
    "Use 'inline' for now (external worker mode can be added later)."
  )
