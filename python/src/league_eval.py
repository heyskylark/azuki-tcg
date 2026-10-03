from __future__ import annotations

from dataclasses import dataclass, field
import time

import azk_puffer.pytorch as azk_pytorch
import azk_puffer.vector as azk_vector
import numpy as np
import torch

from league_promotion import PromotionGameRecord, PromotionGameSpec
from policy.v2.tcg_sampler import tcg_argmax_logits


def _unwrap_base_env(env):
  current = getattr(env, "env", env)
  seen = set()
  while hasattr(current, "env"):
    # Stop at the deck-building wrapper (class attr — wrapper __getattr__
    # forwarding blocks underscore attrs like _active_player_index, and the
    # battle env below it holds a STALE active player during the draft phase).
    if getattr(type(current), "is_deck_building_wrapper", False):
      break
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
  records: tuple[PromotionGameRecord, ...] = ()
  raw_records: tuple[dict, ...] = ()
  wall_time_seconds: float = 0.0
  timings: dict[str, float] = field(default_factory=dict)

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
  batch_envs: int = 12


class LeagueEvaluator:
  def evaluate(self, trainer_args: dict, *, policy_a, policy_b, request: MatchRequest) -> MatchResult:  # pragma: no cover - interface
    raise NotImplementedError

  def evaluate_schedule(
    self,
    trainer_args: dict,
    *,
    policy_a,
    policy_b,
    request: MatchRequest,
    games: list[PromotionGameSpec],
  ) -> MatchResult:
    raise NotImplementedError


class InlineLeagueEvaluator(LeagueEvaluator):
  @staticmethod
  def _rebuild_for_eval(policy, vecenv, eval_args, device: str, use_rnn: bool):
    import numpy as np
    import torch

    from training_utils import build_policy

    fresh = build_policy(vecenv, eval_args)
    # Warm forward so lazily-created modules/buffers exist before the copy.
    warm_obs = torch.as_tensor(
      np.zeros((vecenv.num_agents, *vecenv.single_observation_space.shape),
               dtype=vecenv.single_observation_space.dtype),
      device=device,
    )
    warm_state = {"mask": torch.zeros(vecenv.num_agents, device=device)}
    if use_rnn:
      warm_state["lstm_h"] = torch.zeros(vecenv.num_agents, fresh.hidden_size, device=device)
      warm_state["lstm_c"] = torch.zeros(vecenv.num_agents, fresh.hidden_size, device=device)
    with torch.no_grad():
      fresh.forward_eval(warm_obs, warm_state)

    from train import _materialize_scalar_norm_buffers_from_state_dict, _strip_module_prefix

    source = _strip_module_prefix(policy.state_dict())
    _materialize_scalar_norm_buffers_from_state_dict(fresh, source)
    missing, unexpected = fresh.load_state_dict(source, strict=False)
    real_missing = [k for k in missing if "scalar_normalizer" not in k]
    if real_missing or unexpected:
      raise RuntimeError(
        f"League eval policy rebuild mismatch: missing={real_missing[:5]} "
        f"unexpected={list(unexpected)[:5]}"
      )
    return fresh

  def evaluate(self, trainer_args: dict, *, policy_a, policy_b, request: MatchRequest) -> MatchResult:
    from training_utils import build_vecenv

    episodes = int(request.episodes)
    max_steps = int(request.max_steps)
    device = str(request.device)
    seed = int(request.seed)
    if episodes <= 0:
      return MatchResult(wins_a=0, wins_b=0, draws=0, episodes=0)

    # The inline evaluator drives seats through the legacy wrapper chain
    # (_active_player_index, per-seat infos, .done). Force the legacy env
    # path even when training runs native — checkpoints work on both.
    eval_args = dict(trainer_args)
    eval_env_cfg = dict(eval_args.get("env", {}) or {})
    eval_env_cfg["native"] = False
    eval_env_cfg.pop("native_envs_per_instance", None)
    eval_args["env"] = eval_env_cfg

    vecenv = build_vecenv(
      eval_args,
      backend=azk_vector.Serial,
      num_envs=1,
      seed=seed,
    )
    base_env = _unwrap_base_env(vecenv.envs[0])
    num_agents = int(vecenv.num_agents)
    use_rnn = bool(trainer_args.get("train", {}).get("use_rnn", True))

    # The provided policies were built against the TRAINING vecenv; when
    # training runs the native packed layout, their observation decode does
    # not match this legacy eval env. Rebuild fresh policies against the eval
    # env's spec and copy the (layout-agnostic) weights over.
    policy_a = self._rebuild_for_eval(policy_a, vecenv, eval_args, device, use_rnn)
    policy_b = self._rebuild_for_eval(policy_b, vecenv, eval_args, device, use_rnn)

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
            actions, _, _ = azk_pytorch.sample_logits(logits)

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


_END_REASON_NAMES = {
  0: "gameover",
  1: "timeout",
  2: "auto_tick_truncation",
  3: "zero_legal_action_truncation",
}


class NativeLeagueEvaluator(LeagueEvaluator):
  """Batched deterministic runner for explicit promotion schedules."""

  evaluator_version = "native-paired-v1"

  @staticmethod
  def _state_bank(policy, num_envs: int, device: str, use_rnn: bool):
    if not use_rnn:
      return None
    hidden_size = int(policy.hidden_size)
    return {
      "lstm_h": torch.zeros(num_envs, hidden_size, device=device),
      "lstm_c": torch.zeros(num_envs, hidden_size, device=device),
    }

  @staticmethod
  def _zero_state(bank, env_index: int) -> None:
    if bank is None:
      return
    bank["lstm_h"][env_index] = 0
    bank["lstm_c"][env_index] = 0

  @staticmethod
  def _winner_from_record(raw: dict) -> int:
    players = raw.get("players")
    if not isinstance(players, list) or len(players) != 2:
      return -1
    if float(players[0].get("win", 0.0)) >= 0.5:
      return 0
    if float(players[1].get("win", 0.0)) >= 0.5:
      return 1
    return -1

  def _record_from_raw(
    self,
    raw: dict,
    spec: PromotionGameSpec,
    *,
    decision_steps: int,
    elapsed: float,
  ) -> PromotionGameRecord:
    players = raw["players"]
    candidate_gate = int(players[spec.candidate_seat]["gate"])
    opponent_gate = int(players[1 - spec.candidate_seat]["gate"])
    candidate_leader = int(players[spec.candidate_seat].get("leader", -1))
    opponent_leader = int(players[1 - spec.candidate_seat].get("leader", -1))
    return PromotionGameRecord(
      game_id=spec.game_id,
      block_id=spec.block_id,
      phase=spec.phase,
      opponent_id=spec.opponent_id,
      seed=spec.seed,
      candidate_seat=spec.candidate_seat,
      candidate_gate=candidate_gate,
      opponent_gate=opponent_gate,
      winner_seat=self._winner_from_record(raw),
      steps=int(decision_steps),
      end_reason=_END_REASON_NAMES.get(int(raw.get("end_reason", -1)), "unknown"),
      schedule_version=spec.schedule_version,
      reference_seat=int(raw.get("ref_seat", -1)),
      reference_deck_index=int(raw.get("ref_deck_index", -1)),
      world_seed=int(raw.get("seed", -1)),
      starting_player=int(raw.get("starting_player", -1)),
      evaluator_version=self.evaluator_version,
      wall_time_seconds=float(elapsed),
      candidate_leader=candidate_leader,
      opponent_leader=opponent_leader,
    )

  @staticmethod
  def _forward_role(
    policy,
    observations: np.ndarray,
    env_indices: np.ndarray,
    row_indices: np.ndarray,
    state_bank,
    *,
    device: str,
  ) -> np.ndarray:
    if row_indices.size == 0:
      return np.zeros((0, 4), dtype=np.int32)
    obs_t = torch.as_tensor(observations[row_indices], device=device)
    state = {"mask": torch.ones(row_indices.size, device=device, dtype=torch.bool)}
    if state_bank is not None:
      env_t = torch.as_tensor(env_indices, device=device, dtype=torch.long)
      state["lstm_h"] = state_bank["lstm_h"][env_t]
      state["lstm_c"] = state_bank["lstm_c"][env_t]
    with torch.inference_mode():
      logits, _ = policy.forward_eval(obs_t, state)
      actions = tcg_argmax_logits(logits)
    if state_bank is not None:
      state_bank["lstm_h"][env_t] = state["lstm_h"]
      state_bank["lstm_c"][env_t] = state["lstm_c"]
    return actions.to(dtype=torch.int32).cpu().numpy()

  @staticmethod
  def _forward_shared_policy(
    policy,
    observations: np.ndarray,
    env_indices: np.ndarray,
    row_indices: np.ndarray,
    candidate_role: np.ndarray,
    state_a,
    state_b,
    *,
    device: str,
  ) -> np.ndarray:
    obs_t = torch.as_tensor(observations[row_indices], device=device)
    state = {"mask": torch.ones(row_indices.size, device=device, dtype=torch.bool)}
    env_t = torch.as_tensor(env_indices, device=device, dtype=torch.long)
    role_t = torch.as_tensor(candidate_role, device=device, dtype=torch.bool)
    if state_a is not None and state_b is not None:
      state["lstm_h"] = torch.where(
        role_t[:, None],
        state_a["lstm_h"][env_t],
        state_b["lstm_h"][env_t],
      )
      state["lstm_c"] = torch.where(
        role_t[:, None],
        state_a["lstm_c"][env_t],
        state_b["lstm_c"][env_t],
      )
    with torch.inference_mode():
      logits, _ = policy.forward_eval(obs_t, state)
      actions = tcg_argmax_logits(logits)
    if state_a is not None and state_b is not None:
      candidate_envs = env_t[role_t]
      opponent_envs = env_t[~role_t]
      state_a["lstm_h"][candidate_envs] = state["lstm_h"][role_t]
      state_a["lstm_c"][candidate_envs] = state["lstm_c"][role_t]
      state_b["lstm_h"][opponent_envs] = state["lstm_h"][~role_t]
      state_b["lstm_c"][opponent_envs] = state["lstm_c"][~role_t]
    return actions.to(dtype=torch.int32).cpu().numpy()

  def _override_actions(
    self,
    *,
    env,
    running_envs: np.ndarray,
    seats: np.ndarray,
    row_indices: np.ndarray,
    candidate_role: np.ndarray,
    active_specs: list[PromotionGameSpec | None],
  ) -> None:
    """Evaluation-only extension point; the production evaluator is a no-op."""

  def evaluate(self, trainer_args: dict, *, policy_a, policy_b, request: MatchRequest) -> MatchResult:
    raise ValueError(
      "NativeLeagueEvaluator requires an explicit paired schedule; use evaluate_schedule"
    )

  def evaluate_schedule(
    self,
    trainer_args: dict,
    *,
    policy_a,
    policy_b,
    request: MatchRequest,
    games: list[PromotionGameSpec],
  ) -> MatchResult:
    from training_utils import make_azuki_env

    if not games:
      return MatchResult(wins_a=0, wins_b=0, draws=0, episodes=0)

    device = str(request.device)
    batch_envs = min(max(1, int(request.batch_envs)), len(games))
    max_steps = int(request.max_steps)
    if max_steps < 1:
      raise ValueError("Evaluation max_steps must be >= 1")

    env_cfg = dict(trainer_args.get("env", {}) or {})
    env_cfg.update(
      {
        "native": True,
        "native_envs_per_instance": batch_envs,
        "deck_building_enabled": True,
        "evaluation_mode": True,
        "draft_same_element_matchup_prob": 0.0,
        "draft_cross_gate_replay_prob": 0.0,
        "deck_snapshot_dir": None,
      }
    )
    env = make_azuki_env(seed=int(request.seed), **env_cfg)
    use_rnn = bool(trainer_args.get("train", {}).get("use_rnn", True))
    state_a = self._state_bank(policy_a, batch_envs, device, use_rnn)
    state_b = self._state_bank(policy_b, batch_envs, device, use_rnn)
    policy_a_was_training = bool(policy_a.training)
    policy_b_was_training = bool(policy_b.training)
    policy_a.eval()
    policy_b.eval()

    queue_index = 0
    active_specs: list[PromotionGameSpec | None] = [None] * batch_envs
    decision_steps = np.zeros(batch_envs, dtype=np.int32)
    started_at = np.zeros(batch_envs, dtype=np.float64)
    records: list[PromotionGameRecord] = []
    captured_raw_records: list[dict] = []
    overall_start = time.perf_counter()
    timings = {
      "control_seconds": 0.0,
      "reset_seconds": 0.0,
      "candidate_forward_seconds": 0.0,
      "opponent_forward_seconds": 0.0,
      "env_step_seconds": 0.0,
      "record_drain_seconds": 0.0,
    }

    def assign(env_indices: list[int]) -> None:
      nonlocal queue_index
      resets = []
      for env_index in env_indices:
        if queue_index >= len(games):
          active_specs[env_index] = None
          continue
        spec = games[queue_index]
        queue_index += 1
        active_specs[env_index] = spec
        decision_steps[env_index] = 0
        started_at[env_index] = time.perf_counter()
        self._zero_state(state_a, env_index)
        self._zero_state(state_b, env_index)
        resets.append(
          {
            "env_index": env_index,
            "seed": spec.seed,
            "gate0": spec.gate0,
            "gate1": spec.gate1,
            "leader0": spec.leader0,
            "leader1": spec.leader1,
            "reference_seat": spec.reference_seat,
            "reference_deck_index": spec.reference_deck_index,
          }
        )
      if resets:
        started = time.perf_counter()
        env.reset_evaluation_games(resets)
        timings["reset_seconds"] += time.perf_counter() - started

    try:
      assign(list(range(batch_envs)))
      while len(records) < len(games):
        started = time.perf_counter()
        active_players = env.active_players()
        timings["control_seconds"] += time.perf_counter() - started
        running_envs = np.asarray(
          [
            index
            for index, player in enumerate(active_players.tolist())
            if player >= 0 and active_specs[index] is not None
          ],
          dtype=np.int64,
        )
        if running_envs.size == 0:
          raise RuntimeError("Native promotion evaluator has queued games but no runnable env")

        seats = active_players[running_envs].astype(np.int64)
        row_indices = 2 * running_envs + seats
        candidate_role = np.asarray(
          [
            int(seat) == int(active_specs[int(env_index)].candidate_seat)
            for env_index, seat in zip(running_envs.tolist(), seats.tolist())
          ],
          dtype=np.bool_,
        )
        env.actions.fill(0)
        if policy_a is policy_b:
          started = time.perf_counter()
          env.actions[row_indices] = self._forward_shared_policy(
            policy_a,
            env.observations,
            running_envs,
            row_indices,
            candidate_role,
            state_a,
            state_b,
            device=device,
          )
          timings["candidate_forward_seconds"] += time.perf_counter() - started
        else:
          candidate_envs = running_envs[candidate_role]
          candidate_rows = row_indices[candidate_role]
          opponent_envs = running_envs[~candidate_role]
          opponent_rows = row_indices[~candidate_role]
          if candidate_rows.size:
            started = time.perf_counter()
            env.actions[candidate_rows] = self._forward_role(
              policy_a,
              env.observations,
              candidate_envs,
              candidate_rows,
              state_a,
              device=device,
            )
            timings["candidate_forward_seconds"] += time.perf_counter() - started
          if opponent_rows.size:
            started = time.perf_counter()
            env.actions[opponent_rows] = self._forward_role(
              policy_b,
              env.observations,
              opponent_envs,
              opponent_rows,
              state_b,
              device=device,
            )
            timings["opponent_forward_seconds"] += time.perf_counter() - started

        self._override_actions(
          env=env,
          running_envs=running_envs,
          seats=seats,
          row_indices=row_indices,
          candidate_role=candidate_role,
          active_specs=active_specs,
        )

        started = time.perf_counter()
        env.step()
        timings["env_step_seconds"] += time.perf_counter() - started
        decision_steps[running_envs] += 1
        started = time.perf_counter()
        raw_records = env.drain_evaluation_records()
        timings["record_drain_seconds"] += time.perf_counter() - started
        completed_envs: list[int] = []
        for raw in raw_records:
          env_index = int(raw["env_index"])
          spec = active_specs[env_index]
          if spec is None:
            raise RuntimeError(f"Received an evaluation record for idle env {env_index}")
          captured = dict(raw)
          captured.update(
            {
              "game_id": spec.game_id,
              "block_id": spec.block_id,
              "phase": spec.phase,
              "opponent_id": spec.opponent_id,
              "candidate_seat": spec.candidate_seat,
            }
          )
          captured_raw_records.append(captured)
          records.append(
            self._record_from_raw(
              raw,
              spec,
              decision_steps=int(decision_steps[env_index]),
              elapsed=time.perf_counter() - started_at[env_index],
            )
          )
          completed_envs.append(env_index)
          active_specs[env_index] = None

        if completed_envs:
          assign(completed_envs)

        timed_out = [
          int(env_index)
          for env_index in running_envs.tolist()
          if active_specs[int(env_index)] is not None
          and decision_steps[int(env_index)] >= max_steps
        ]
        if timed_out:
          started = time.perf_counter()
          env.force_evaluation_truncations(timed_out)
          forced_records = env.drain_evaluation_records()
          timings["record_drain_seconds"] += time.perf_counter() - started
          forced_completed: list[int] = []
          for raw in forced_records:
            env_index = int(raw["env_index"])
            spec = active_specs[env_index]
            if spec is None:
              raise RuntimeError(f"Received a forced record for idle env {env_index}")
            captured = dict(raw)
            captured.update(
              {
                "game_id": spec.game_id,
                "block_id": spec.block_id,
                "phase": spec.phase,
                "opponent_id": spec.opponent_id,
                "candidate_seat": spec.candidate_seat,
              }
            )
            captured_raw_records.append(captured)
            records.append(
              self._record_from_raw(
                raw,
                spec,
                decision_steps=int(decision_steps[env_index]),
                elapsed=time.perf_counter() - started_at[env_index],
              )
            )
            forced_completed.append(env_index)
            active_specs[env_index] = None
          if forced_completed:
            assign(forced_completed)
    finally:
      if policy_a_was_training:
        policy_a.train()
      else:
        policy_a.eval()
      if policy_b_was_training:
        policy_b.train()
      else:
        policy_b.eval()
      env.close()

    order = {game.game_id: index for index, game in enumerate(games)}
    records.sort(key=lambda record: order[record.game_id])
    captured_raw_records.sort(key=lambda record: order[str(record["game_id"])])
    wins_a = sum(record.candidate_score == 1.0 for record in records)
    wins_b = sum(record.candidate_score == 0.0 for record in records)
    draws = len(records) - wins_a - wins_b
    return MatchResult(
      wins_a=wins_a,
      wins_b=wins_b,
      draws=draws,
      episodes=len(records),
      records=tuple(records),
      raw_records=tuple(captured_raw_records),
      wall_time_seconds=time.perf_counter() - overall_start,
      timings=timings,
    )


def make_league_evaluator(mode: str) -> LeagueEvaluator:
  normalized = str(mode).strip().lower()
  if normalized in {"inline", ""}:
    return InlineLeagueEvaluator()
  if normalized in {"native_panel", "native-paired", "native_paired"}:
    return NativeLeagueEvaluator()
  raise ValueError(
    f"Unsupported league evaluator mode '{mode}'. "
    "Use 'inline' or 'native_panel'."
  )
