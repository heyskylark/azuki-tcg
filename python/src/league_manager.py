from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from league_eval import MatchRequest, make_league_evaluator
from league_ratings import apply_match_result, rank_table
from league_state import (
  LeaguePolicyEntry,
  LeagueState,
  classify_and_prune,
  load_league_state,
  register_policy,
  save_league_state,
)


@dataclass
class LeagueManagerConfig:
  enabled: bool = False
  state_path: str = "experiments/league_state.json"
  checkpoint_add_interval: int = 1
  eval_interval: int = 1
  eval_mode: str = "inline"
  quick_eval_interval: int = 3
  quick_eval_episodes: int = 8
  full_eval_interval: int = 12
  full_eval_episodes: int = 48
  eval_episodes: int = 8
  eval_max_steps: int = 400
  keep_recent: int = 6
  keep_mid: int = 4
  keep_old: int = 3
  min_candidate_epoch_gap: int = 1
  promotion_min_winrate_vs_champion: float = 0.55
  promotion_baseline_count: int = 2
  promotion_min_winrate_vs_baseline: float = 0.50
  promotion_min_games_vs_champion: int = 16
  promotion_min_games_vs_baseline: int = 8
  promotion_wilson_confidence_z: float = 1.28


def parse_league_manager_config(trainer_args: dict) -> LeagueManagerConfig:
  league = trainer_args.get("league")
  if not isinstance(league, dict):
    league = {}
  legacy_eval_episodes = int(league.get("eval_episodes", 8))
  cfg = LeagueManagerConfig(
    enabled=bool(league.get("enable", False)),
    state_path=str(league.get("state_path", "experiments/league_state.json")),
    checkpoint_add_interval=int(league.get("checkpoint_add_interval", 1)),
    eval_interval=int(league.get("eval_interval", 1)),
    eval_mode=str(league.get("eval_mode", "inline")),
    quick_eval_interval=int(league.get("quick_eval_interval", 3)),
    quick_eval_episodes=int(league.get("quick_eval_episodes", legacy_eval_episodes)),
    full_eval_interval=int(league.get("full_eval_interval", 12)),
    full_eval_episodes=int(league.get("full_eval_episodes", max(48, legacy_eval_episodes))),
    eval_episodes=int(league.get("eval_episodes", 8)),
    eval_max_steps=int(league.get("eval_max_steps", 400)),
    keep_recent=int(league.get("keep_recent", 6)),
    keep_mid=int(league.get("keep_mid", 4)),
    keep_old=int(league.get("keep_old", 3)),
    min_candidate_epoch_gap=int(league.get("min_candidate_epoch_gap", 1)),
    promotion_min_winrate_vs_champion=float(league.get("promotion_min_winrate_vs_champion", 0.55)),
    promotion_baseline_count=int(league.get("promotion_baseline_count", 2)),
    promotion_min_winrate_vs_baseline=float(league.get("promotion_min_winrate_vs_baseline", 0.50)),
    promotion_min_games_vs_champion=int(league.get("promotion_min_games_vs_champion", 16)),
    promotion_min_games_vs_baseline=int(league.get("promotion_min_games_vs_baseline", 8)),
    promotion_wilson_confidence_z=float(league.get("promotion_wilson_confidence_z", 1.28)),
  )
  _validate_config(cfg)
  return cfg


def _validate_config(cfg: LeagueManagerConfig) -> None:
  if cfg.checkpoint_add_interval < 1:
    raise ValueError("league.checkpoint_add_interval must be >= 1")
  if cfg.quick_eval_interval < 1:
    raise ValueError("league.quick_eval_interval must be >= 1")
  if cfg.full_eval_interval < 1:
    raise ValueError("league.full_eval_interval must be >= 1")
  if cfg.quick_eval_episodes < 1:
    raise ValueError("league.quick_eval_episodes must be >= 1")
  if cfg.full_eval_episodes < cfg.quick_eval_episodes:
    raise ValueError("league.full_eval_episodes must be >= league.quick_eval_episodes")
  if cfg.eval_max_steps < 1:
    raise ValueError("league.eval_max_steps must be >= 1")
  if cfg.keep_recent < 0 or cfg.keep_mid < 0 or cfg.keep_old < 0:
    raise ValueError("league.keep_recent/mid/old must be >= 0")
  if cfg.min_candidate_epoch_gap < 1:
    raise ValueError("league.min_candidate_epoch_gap must be >= 1")
  if cfg.promotion_min_winrate_vs_champion < 0.0 or cfg.promotion_min_winrate_vs_champion > 1.0:
    raise ValueError("league.promotion_min_winrate_vs_champion must be in [0,1]")
  if cfg.promotion_min_winrate_vs_baseline < 0.0 or cfg.promotion_min_winrate_vs_baseline > 1.0:
    raise ValueError("league.promotion_min_winrate_vs_baseline must be in [0,1]")
  if cfg.promotion_min_games_vs_champion < 1 or cfg.promotion_min_games_vs_baseline < 1:
    raise ValueError("league.promotion_min_games_* must be >= 1")
  if cfg.promotion_wilson_confidence_z <= 0.0:
    raise ValueError("league.promotion_wilson_confidence_z must be > 0")


class LeagueManager:
  def __init__(self, config: LeagueManagerConfig):
    self.config = config
    self.state_path = Path(config.state_path)
    self.state: LeagueState = load_league_state(self.state_path)
    self.latest_metrics: dict[str, float | str] = {}
    self.current_candidate_id: str | None = self.state.current_candidate_policy_id
    self.evaluator = make_league_evaluator(config.eval_mode)

  def save(self) -> None:
    save_league_state(self.state_path, self.state)

  def ensure_seed_policies(self, checkpoint_paths: list[Path], *, created_epoch: int = 0) -> None:
    for path in checkpoint_paths:
      if not path.exists():
        continue
      register_policy(
        self.state,
        checkpoint_path=path,
        created_epoch=created_epoch,
        source="seed",
      )
    self._prune()
    self.save()

  def attach_learner_identity(self, learner_id: str) -> None:
    if not learner_id:
      return
    self.state.learner_policy_id = str(learner_id)
    self.save()

  def _prune(self) -> list[str]:
    return classify_and_prune(
      self.state,
      keep_recent=self.config.keep_recent,
      keep_mid=self.config.keep_mid,
      keep_old=self.config.keep_old,
    )

  def _active_entries(self) -> list[LeaguePolicyEntry]:
    return [entry for entry in self.state.policies.values() if entry.active and Path(entry.checkpoint_path).exists()]

  def _entry_by_id(self, policy_id: str | None) -> LeaguePolicyEntry | None:
    if policy_id is None:
      return None
    return self.state.policies.get(policy_id)

  def _load_policy(self, *, entry: LeaguePolicyEntry, vecenv, trainer_args: dict, build_policy_fn, load_weights_fn):
    policy = build_policy_fn(vecenv, trainer_args)
    load_weights_fn(policy, Path(entry.checkpoint_path), device=str(trainer_args["train"].get("device", "cpu")), strict=False)
    policy.eval()
    for param in policy.parameters():
      param.requires_grad_(False)
    return policy

  def opponent_entries_for_training(self, *, exclude_policy_id: str | None = None) -> list[LeaguePolicyEntry]:
    entries = self._active_entries()
    if exclude_policy_id is not None:
      entries = [entry for entry in entries if entry.policy_id != exclude_policy_id]
    return entries

  def pool_metrics(self) -> dict[str, float]:
    entries = self._active_entries()
    total = float(len(entries))
    if total <= 0:
      return {
        "league/pool_size_active": 0.0,
        "league/pool_recent_frac": 0.0,
        "league/pool_mid_frac": 0.0,
        "league/pool_old_frac": 0.0,
      }
    recent = sum(1 for e in entries if e.bucket == "recent")
    mid = sum(1 for e in entries if e.bucket == "mid")
    old = sum(1 for e in entries if e.bucket == "old")
    return {
      "league/pool_size_active": total,
      "league/pool_recent_frac": float(recent / total),
      "league/pool_mid_frac": float(mid / total),
      "league/pool_old_frac": float(old / total),
    }

  def maybe_add_checkpoint(self, checkpoint_path: Path, *, epoch: int) -> LeaguePolicyEntry | None:
    if self.config.checkpoint_add_interval <= 0:
      return None
    if epoch % self.config.checkpoint_add_interval != 0:
      return None
    current_learner_id = self.state.learner_policy_id
    recent_candidates = sorted(
      [
        entry.created_epoch
        for entry in self.state.policies.values()
        if entry.source == "checkpoint"
        and (
          current_learner_id is None
          or entry.created_by_learner_id is None
          or entry.created_by_learner_id == current_learner_id
        )
      ],
      reverse=True,
    )
    if recent_candidates:
      latest_epoch = int(recent_candidates[0])
      if (int(epoch) - latest_epoch) < self.config.min_candidate_epoch_gap:
        return None

    entry = register_policy(
      self.state,
      checkpoint_path=checkpoint_path,
      created_epoch=epoch,
      source="checkpoint",
      created_by_learner_id=self.state.learner_policy_id,
    )
    self.current_candidate_id = entry.policy_id
    self.state.current_candidate_policy_id = entry.policy_id
    self.state.history.append(
      {
        "event": "checkpoint_ingested",
        "epoch": int(epoch),
        "policy_id": entry.policy_id,
        "checkpoint_path": entry.checkpoint_path,
        "learner_policy_id": self.state.learner_policy_id,
      }
    )
    self._prune()
    self.save()
    return entry

  @staticmethod
  def _wilson_lower_bound(wins: int, games: int, z: float) -> float:
    if games <= 0:
      return 0.0
    p = float(wins) / float(games)
    n = float(games)
    denom = 1.0 + (z * z) / n
    center = p + (z * z) / (2.0 * n)
    margin = z * ((p * (1.0 - p) + (z * z) / (4.0 * n)) / n) ** 0.5
    return (center - margin) / denom

  def _baseline_entries(self, *, exclude_ids: set[str]) -> list[LeaguePolicyEntry]:
    ranked = rank_table([entry.rating for entry in self._active_entries() if entry.policy_id not in exclude_ids])
    out: list[LeaguePolicyEntry] = []
    for rating in ranked[: max(self.config.promotion_baseline_count, 0)]:
      entry = self._entry_by_id(rating.policy_id)
      if entry is not None:
        out.append(entry)
    return out

  def maybe_evaluate_and_promote(
    self,
    *,
    epoch: int,
    trainer_args: dict,
    vecenv,
    build_policy_fn,
    load_weights_fn,
    learner_policy: torch.nn.Module,
  ) -> dict[str, float | str]:
    if self.current_candidate_id is None:
      return {}
    run_quick = epoch % self.config.quick_eval_interval == 0
    run_full = epoch % self.config.full_eval_interval == 0
    if self.config.eval_interval > 1 and (epoch % self.config.eval_interval != 0):
      run_quick = False
      run_full = False

    candidate = self._entry_by_id(self.current_candidate_id)
    champion = self._entry_by_id(self.state.champion_policy_id)
    if candidate is None or champion is None or not Path(champion.checkpoint_path).exists():
      return {}

    # Candidate can be current learner weights if checkpoint paths match.
    if Path(candidate.checkpoint_path).resolve() == Path(champion.checkpoint_path).resolve():
      return {}

    device = str(trainer_args["train"].get("device", "cpu"))
    champion_policy = self._load_policy(
      entry=champion,
      vecenv=vecenv,
      trainer_args=trainer_args,
      build_policy_fn=build_policy_fn,
      load_weights_fn=load_weights_fn,
    )

    if not run_quick:
      return {}

    quick_req = MatchRequest(
      episodes=self.config.quick_eval_episodes,
      max_steps=self.config.eval_max_steps,
      seed=epoch * 1000 + 7,
      device=device,
    )
    h2h = self.evaluator.evaluate(
      trainer_args,
      policy_a=learner_policy,
      policy_b=champion_policy,
      request=quick_req,
    )
    apply_match_result(candidate.rating, champion.rating, score_a=h2h.score_a)

    promotion_reasons = []
    if h2h.episodes < self.config.promotion_min_games_vs_champion:
      promotion_reasons.append("insufficient_games_vs_champion")
    if h2h.win_rate_a < self.config.promotion_min_winrate_vs_champion:
      promotion_reasons.append("winrate_vs_champion_too_low")
    h2h_wilson = self._wilson_lower_bound(
      wins=h2h.wins_a,
      games=h2h.episodes,
      z=self.config.promotion_wilson_confidence_z,
    )
    if h2h_wilson < self.config.promotion_min_winrate_vs_champion:
      promotion_reasons.append("wilson_lower_bound_too_low")

    if run_full and not promotion_reasons:
      full_req = MatchRequest(
        episodes=self.config.full_eval_episodes,
        max_steps=self.config.eval_max_steps,
        seed=epoch * 1000 + 113,
        device=device,
      )
      h2h = self.evaluator.evaluate(
        trainer_args,
        policy_a=learner_policy,
        policy_b=champion_policy,
        request=full_req,
      )
      apply_match_result(candidate.rating, champion.rating, score_a=h2h.score_a)
      h2h_wilson = self._wilson_lower_bound(
        wins=h2h.wins_a,
        games=h2h.episodes,
        z=self.config.promotion_wilson_confidence_z,
      )
      if h2h.episodes < self.config.promotion_min_games_vs_champion:
        promotion_reasons.append("insufficient_games_vs_champion_full")
      if h2h.win_rate_a < self.config.promotion_min_winrate_vs_champion:
        promotion_reasons.append("winrate_vs_champion_too_low_full")
      if h2h_wilson < self.config.promotion_min_winrate_vs_champion:
        promotion_reasons.append("wilson_lower_bound_too_low_full")

    baseline_ok = True
    baseline_min = 1.0
    baseline_wilson_min = 1.0
    eval_baseline_samples = 0
    baselines = self._baseline_entries(exclude_ids={candidate.policy_id, champion.policy_id})
    for idx, baseline in enumerate(baselines):
      baseline_policy = self._load_policy(
        entry=baseline,
        vecenv=vecenv,
        trainer_args=trainer_args,
        build_policy_fn=build_policy_fn,
        load_weights_fn=load_weights_fn,
      )
      baseline_episodes = max(self.config.promotion_min_games_vs_baseline, self.config.quick_eval_episodes // 2)
      result = self.evaluator.evaluate(
        trainer_args,
        policy_a=learner_policy,
        policy_b=baseline_policy,
        request=MatchRequest(
          episodes=baseline_episodes,
          max_steps=self.config.eval_max_steps,
          seed=epoch * 2000 + 17 + idx,
          device=device,
        ),
      )
      eval_baseline_samples += result.episodes
      baseline_min = min(baseline_min, result.win_rate_a)
      baseline_wilson = self._wilson_lower_bound(
        wins=result.wins_a,
        games=result.episodes,
        z=self.config.promotion_wilson_confidence_z,
      )
      baseline_wilson_min = min(baseline_wilson_min, baseline_wilson)
      apply_match_result(candidate.rating, baseline.rating, score_a=result.score_a)
      if result.win_rate_a < self.config.promotion_min_winrate_vs_baseline:
        baseline_ok = False
      if result.episodes < self.config.promotion_min_games_vs_baseline:
        baseline_ok = False

    if baselines and baseline_wilson_min < self.config.promotion_min_winrate_vs_baseline:
      baseline_ok = False
      promotion_reasons.append("baseline_wilson_too_low")
    if baselines and baseline_min < self.config.promotion_min_winrate_vs_baseline:
      promotion_reasons.append("baseline_winrate_too_low")

    promote = (not promotion_reasons) and baseline_ok
    reject_reason = "ok" if promote else (";".join(promotion_reasons) if promotion_reasons else "gate_not_met")
    if promote:
      prev = self.state.champion_policy_id
      self.state.champion_policy_id = candidate.policy_id
      self.state.current_candidate_policy_id = None
      self.state.history.append(
        {
          "event": "promotion",
          "epoch": int(epoch),
          "new_champion": candidate.policy_id,
          "old_champion": prev,
          "h2h_win_rate": float(h2h.win_rate_a),
          "baseline_min_win_rate": float(baseline_min),
        }
      )
    else:
      self.state.history.append(
        {
          "event": "promotion_rejected",
          "epoch": int(epoch),
          "candidate": candidate.policy_id,
          "champion": champion.policy_id,
          "h2h_win_rate": float(h2h.win_rate_a),
          "h2h_wilson": float(h2h_wilson),
          "baseline_min_win_rate": float(baseline_min if baselines else 1.0),
          "baseline_wilson_min": float(baseline_wilson_min if baselines else 1.0),
          "reason": reject_reason,
        }
      )

    self._prune()
    self.save()

    ranked = rank_table([entry.rating for entry in self._active_entries()])
    rank_position = 1
    for idx, rating in enumerate(ranked, start=1):
      if rating.policy_id == candidate.policy_id:
        rank_position = idx
        break

    self.latest_metrics = {
      "league/champion_policy_id": str(self.state.champion_policy_id or ""),
      "league/candidate_policy_id": str(candidate.policy_id),
      "league/candidate_winrate_vs_champion": float(h2h.win_rate_a),
      "league/candidate_wilson_vs_champion": float(h2h_wilson),
      "league/candidate_baseline_min_winrate": float(baseline_min if baselines else 1.0),
      "league/candidate_baseline_wilson_min": float(baseline_wilson_min if baselines else 1.0),
      "league/candidate_elo": float(candidate.rating.elo),
      "league/champion_elo": float(champion.rating.elo),
      "league/candidate_rank": float(rank_position),
      "league/promotion_accepted": 1.0 if promote else 0.0,
      "league/active_pool_size": float(len(self._active_entries())),
      "league/eval_h2h_samples": float(h2h.episodes),
      "league/eval_baseline_samples": float(eval_baseline_samples),
      "league/reject_reason": reject_reason,
      "league/reject_reason_code": float(0.0 if promote else 1.0),
      "league/eval_mode": self.config.eval_mode,
      "league/quick_eval_interval": float(self.config.quick_eval_interval),
      "league/full_eval_interval": float(self.config.full_eval_interval),
      "league/quick_eval_episodes": float(self.config.quick_eval_episodes),
      "league/full_eval_episodes": float(self.config.full_eval_episodes),
    }
    return dict(self.latest_metrics)
