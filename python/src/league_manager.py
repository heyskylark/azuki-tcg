from __future__ import annotations

import copy
import contextlib
from dataclasses import asdict
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from deck_building import build_deck_build_catalog
from league_eval import MatchRequest, make_league_evaluator
from league_archive import (
  PanelManifest,
  PanelMember,
  PromotionArchiveState,
  QualityArchiveEntry,
  active_panel,
  admit_quality_policy,
  ensure_panel_manifest,
  ensure_production_anchor,
  load_promotion_archive_state,
  protected_policy_ids,
  record_payoff_results,
  save_promotion_archive_state,
)
from league_promotion import (
  CONFIRMATION_PHASE,
  SCREEN_PHASE,
  PromotionDecision,
  PromotionGameRecord,
  PromotionThresholds,
  build_panel_schedule,
  build_reference_schedule,
  compare_panel_records,
  compare_reference_records,
  decide_promotion,
  screen_passes,
  summarize_panel,
)
from league_promotion_store import (
  checkpoint_sha256,
  deck_sha256,
  load_records,
  schedule_sha256,
  write_immutable_json,
  write_promotion_run,
)
from league_ratings import apply_match_result, rank_table
from league_state import (
  LeaguePolicyEntry,
  LeagueState,
  classify_and_prune,
  load_league_state,
  register_policy,
  save_league_state,
)
from training_deck_pool import load_training_deck_pool


EXTERNAL_STRENGTH_WINDOW_SIZE = 3


def summarize_external_strength_window(
  history: list[dict],
  current: dict,
  *,
  window_size: int = EXTERNAL_STRENGTH_WINDOW_SIZE,
) -> dict:
  if int(window_size) < 1:
    raise ValueError("external strength window size must be positive")
  schedule_hash = str(current.get("schedule_hash", ""))
  if not schedule_hash:
    raise ValueError("external strength observation requires a schedule hash")
  previous = [
    item
    for item in history
    if item.get("event") == "external_strength_observation"
    and str(item.get("schedule_hash", "")) == schedule_hash
  ]
  selected = (previous + [current])[-int(window_size):]
  epochs = np.asarray([int(item["epoch"]) for item in selected], dtype=np.float64)
  scores = np.asarray(
    [float(item["candidate_score"]) for item in selected],
    dtype=np.float64,
  )
  deltas = np.asarray([float(item["delta"]) for item in selected], dtype=np.float64)
  lower_bounds = np.asarray(
    [float(item["paired_lcb"]) for item in selected],
    dtype=np.float64,
  )
  slope = 0.0
  if len(selected) >= 2 and len(set(epochs.tolist())) >= 2:
    slope = float(np.polyfit(epochs, scores, 1)[0] * 100.0)
  return {
    "version": "external-strength-window-v1",
    "schedule_hash": schedule_hash,
    "target_size": int(window_size),
    "count": len(selected),
    "epochs": [int(value) for value in epochs],
    "candidate_ids": [str(item["candidate_id"]) for item in selected],
    "score_mean": float(scores.mean()),
    "score_min": float(scores.min()),
    "score_max": float(scores.max()),
    "score_latest": float(scores[-1]),
    "delta_mean": float(deltas.mean()),
    "delta_latest": float(deltas[-1]),
    "paired_lcb_min": float(lower_bounds.min()),
    "paired_lcb_latest": float(lower_bounds[-1]),
    "score_slope_per_100_updates": slope,
  }


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
  promotion_mode: str = "legacy"
  promotion_shadow_mode: bool = True
  promotion_state_path: str = ""
  promotion_records_dir: str = ""
  production_anchor_checkpoint: str = ""
  promotion_anchor_in_training_pool: bool = False
  promotion_archive_affects_training_pool: bool = False
  promotion_bootstrap_panel_path: str = ""
  promotion_panel_size: int = 4
  promotion_panel_refresh_epochs: int = 600
  promotion_archive_max_active: int = 6
  promotion_eval_batch_envs: int = 12
  promotion_schedule_seed: int = 42_001_701
  promotion_reference_seed: int = 42_009_919
  promotion_reference_seeds: tuple[int, ...] = ()
  promotion_reference_deck_indices: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14, 16)
  promotion_reference_holdout_deck_indices: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 13, 15, 17)
  promotion_require_reference: bool = True
  promotion_screen_pooled_min: float = 0.48
  promotion_screen_opponent_floor: float = 0.25
  promotion_screen_seat_floor: float = 0.35
  promotion_pooled_min: float = 0.52
  promotion_opponent_quorum_min: float = 0.45
  promotion_opponent_quorum_count: int = 3
  promotion_opponent_break_even_count: int = 2
  promotion_opponent_floor: float = 0.35
  promotion_seat_floor: float = 0.40
  promotion_paired_lcb_min: float = 0.48
  promotion_bootstrap_confidence: float = 0.80
  promotion_bootstrap_samples: int = 10_000
  promotion_reference_candidate_floor: float = 0.50
  promotion_reference_noninferiority: float = -0.15
  promotion_reference_paired_lcb_min: float = -0.18
  promotion_reference_leap: float = 0.05
  promotion_max_timeout_rate: float = 0.0
  promotion_anchored_screen_pooled_floor: float = 0.35
  promotion_anchored_screen_opponent_floor: float = 0.15
  promotion_anchored_screen_seat_floor: float = 0.25
  promotion_anchored_screen_relative_pooled_min: float = -0.08
  promotion_anchored_screen_relative_opponent_floor: float = -0.20
  promotion_anchored_screen_relative_seat_floor: float = -0.15
  promotion_anchored_pooled_floor: float = 0.40
  promotion_anchored_opponent_floor: float = 0.20
  promotion_anchored_seat_floor: float = 0.35
  promotion_anchored_relative_pooled_min: float = -0.02
  promotion_anchored_relative_opponent_quorum_min: float = -0.05
  promotion_anchored_relative_opponent_quorum_count: int = 3
  promotion_anchored_relative_break_even_count: int = 2
  promotion_anchored_relative_opponent_floor: float = -0.10
  promotion_anchored_relative_seat_floor: float = -0.08
  promotion_anchored_relative_paired_lcb_min: float = -0.05
  promotion_anchored_leap_relative_pooled_min: float = -0.05
  promotion_anchored_max_timeout_rate: float = 0.01
  promotion_anchored_max_timeout_delta: float = 0.01


def _parse_int_tuple(value, *, default: tuple[int, ...]) -> tuple[int, ...]:
  if value is None or value == "":
    return default
  if isinstance(value, int):
    return (int(value),)
  if isinstance(value, (list, tuple)):
    return tuple(int(item) for item in value)
  return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def parse_league_manager_config(trainer_args: dict) -> LeagueManagerConfig:
  league = trainer_args.get("league")
  if not isinstance(league, dict):
    league = {}
  legacy_eval_episodes = int(league.get("eval_episodes", 8))
  eval_mode = str(league.get("eval_mode", "inline"))
  default_promotion_mode = "archive_panel" if eval_mode.strip().lower() in {
    "native_panel",
    "native-paired",
    "native_paired",
  } else "legacy"
  cfg = LeagueManagerConfig(
    enabled=bool(league.get("enable", False)),
    state_path=str(league.get("state_path", "experiments/league_state.json")),
    checkpoint_add_interval=int(league.get("checkpoint_add_interval", 1)),
    eval_interval=int(league.get("eval_interval", 1)),
    eval_mode=eval_mode,
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
    promotion_mode=str(league.get("promotion_mode", default_promotion_mode)),
    promotion_shadow_mode=bool(league.get("promotion_shadow_mode", True)),
    promotion_state_path=str(league.get("promotion_state_path", "")),
    promotion_records_dir=str(league.get("promotion_records_dir", "")),
    production_anchor_checkpoint=str(league.get("production_anchor_checkpoint", "")),
    promotion_anchor_in_training_pool=bool(
      league.get("promotion_anchor_in_training_pool", False)
    ),
    promotion_archive_affects_training_pool=bool(
      league.get("promotion_archive_affects_training_pool", False)
    ),
    promotion_bootstrap_panel_path=str(league.get("promotion_bootstrap_panel_path", "")),
    promotion_panel_size=int(league.get("promotion_panel_size", 4)),
    promotion_panel_refresh_epochs=int(league.get("promotion_panel_refresh_epochs", 600)),
    promotion_archive_max_active=int(league.get("promotion_archive_max_active", 6)),
    promotion_eval_batch_envs=int(league.get("promotion_eval_batch_envs", 12)),
    promotion_schedule_seed=int(league.get("promotion_schedule_seed", 42_001_701)),
    promotion_reference_seed=int(league.get("promotion_reference_seed", 42_009_919)),
    promotion_reference_seeds=_parse_int_tuple(
      league.get("promotion_reference_seeds"),
      default=(),
    ),
    promotion_reference_deck_indices=_parse_int_tuple(
      league.get("promotion_reference_deck_indices"),
      default=(0, 2, 4, 6, 8, 10, 12, 14, 16),
    ),
    promotion_reference_holdout_deck_indices=_parse_int_tuple(
      league.get("promotion_reference_holdout_deck_indices"),
      default=(1, 3, 5, 7, 9, 11, 13, 15, 17),
    ),
    promotion_require_reference=bool(league.get("promotion_require_reference", True)),
    promotion_screen_pooled_min=float(league.get("promotion_screen_pooled_min", 0.48)),
    promotion_screen_opponent_floor=float(league.get("promotion_screen_opponent_floor", 0.25)),
    promotion_screen_seat_floor=float(league.get("promotion_screen_seat_floor", 0.35)),
    promotion_pooled_min=float(league.get("promotion_pooled_min", 0.52)),
    promotion_opponent_quorum_min=float(league.get("promotion_opponent_quorum_min", 0.45)),
    promotion_opponent_quorum_count=int(league.get("promotion_opponent_quorum_count", 3)),
    promotion_opponent_break_even_count=int(league.get("promotion_opponent_break_even_count", 2)),
    promotion_opponent_floor=float(league.get("promotion_opponent_floor", 0.35)),
    promotion_seat_floor=float(league.get("promotion_seat_floor", 0.40)),
    promotion_paired_lcb_min=float(league.get("promotion_paired_lcb_min", 0.48)),
    promotion_bootstrap_confidence=float(league.get("promotion_bootstrap_confidence", 0.80)),
    promotion_bootstrap_samples=int(league.get("promotion_bootstrap_samples", 10_000)),
    promotion_reference_candidate_floor=float(
      league.get("promotion_reference_candidate_floor", 0.50)
    ),
    promotion_reference_noninferiority=float(
      league.get("promotion_reference_noninferiority", -0.15)
    ),
    promotion_reference_paired_lcb_min=float(
      league.get("promotion_reference_paired_lcb_min", -0.18)
    ),
    promotion_reference_leap=float(league.get("promotion_reference_leap", 0.05)),
    promotion_max_timeout_rate=float(league.get("promotion_max_timeout_rate", 0.0)),
    promotion_anchored_screen_pooled_floor=float(
      league.get("promotion_anchored_screen_pooled_floor", 0.35)
    ),
    promotion_anchored_screen_opponent_floor=float(
      league.get("promotion_anchored_screen_opponent_floor", 0.15)
    ),
    promotion_anchored_screen_seat_floor=float(
      league.get("promotion_anchored_screen_seat_floor", 0.25)
    ),
    promotion_anchored_screen_relative_pooled_min=float(
      league.get("promotion_anchored_screen_relative_pooled_min", -0.08)
    ),
    promotion_anchored_screen_relative_opponent_floor=float(
      league.get("promotion_anchored_screen_relative_opponent_floor", -0.20)
    ),
    promotion_anchored_screen_relative_seat_floor=float(
      league.get("promotion_anchored_screen_relative_seat_floor", -0.15)
    ),
    promotion_anchored_pooled_floor=float(
      league.get("promotion_anchored_pooled_floor", 0.40)
    ),
    promotion_anchored_opponent_floor=float(
      league.get("promotion_anchored_opponent_floor", 0.20)
    ),
    promotion_anchored_seat_floor=float(
      league.get("promotion_anchored_seat_floor", 0.35)
    ),
    promotion_anchored_relative_pooled_min=float(
      league.get("promotion_anchored_relative_pooled_min", -0.02)
    ),
    promotion_anchored_relative_opponent_quorum_min=float(
      league.get("promotion_anchored_relative_opponent_quorum_min", -0.05)
    ),
    promotion_anchored_relative_opponent_quorum_count=int(
      league.get("promotion_anchored_relative_opponent_quorum_count", 3)
    ),
    promotion_anchored_relative_break_even_count=int(
      league.get("promotion_anchored_relative_break_even_count", 2)
    ),
    promotion_anchored_relative_opponent_floor=float(
      league.get("promotion_anchored_relative_opponent_floor", -0.10)
    ),
    promotion_anchored_relative_seat_floor=float(
      league.get("promotion_anchored_relative_seat_floor", -0.08)
    ),
    promotion_anchored_relative_paired_lcb_min=float(
      league.get("promotion_anchored_relative_paired_lcb_min", -0.05)
    ),
    promotion_anchored_leap_relative_pooled_min=float(
      league.get("promotion_anchored_leap_relative_pooled_min", -0.05)
    ),
    promotion_anchored_max_timeout_rate=float(
      league.get("promotion_anchored_max_timeout_rate", 0.01)
    ),
    promotion_anchored_max_timeout_delta=float(
      league.get("promotion_anchored_max_timeout_delta", 0.01)
    ),
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
  if cfg.promotion_mode not in {"legacy", "archive_panel"}:
    raise ValueError("league.promotion_mode must be 'legacy' or 'archive_panel'")
  if cfg.promotion_mode == "archive_panel" and cfg.eval_mode.strip().lower() not in {
    "native_panel",
    "native-paired",
    "native_paired",
  }:
    raise ValueError("league.promotion_mode=archive_panel requires league.eval_mode=native_panel")
  if cfg.promotion_panel_size != 4:
    raise ValueError("league.promotion_panel_size must be 4 for paired-v1")
  if cfg.promotion_panel_refresh_epochs < 1:
    raise ValueError("league.promotion_panel_refresh_epochs must be >= 1")
  if not 4 <= cfg.promotion_archive_max_active <= 6:
    raise ValueError("league.promotion_archive_max_active must be in [4,6]")
  if cfg.promotion_eval_batch_envs < 1:
    raise ValueError("league.promotion_eval_batch_envs must be >= 1")
  effective_reference_seeds = (
    cfg.promotion_reference_seeds
    if cfg.promotion_reference_seeds
    else (cfg.promotion_reference_seed,)
  )
  if len(set(effective_reference_seeds)) != len(effective_reference_seeds):
    raise ValueError("league.promotion_reference_seeds must be unique")
  if any(seed < 0 or seed > 0xFFFFFFFF for seed in effective_reference_seeds):
    raise ValueError("league.promotion_reference_seeds must contain uint32 values")
  if len(set(cfg.promotion_reference_deck_indices)) != len(cfg.promotion_reference_deck_indices):
    raise ValueError("league.promotion_reference_deck_indices must be unique")
  if any(index < 0 for index in cfg.promotion_reference_deck_indices):
    raise ValueError("league.promotion_reference_deck_indices must be non-negative")
  if len(set(cfg.promotion_reference_holdout_deck_indices)) != len(
    cfg.promotion_reference_holdout_deck_indices
  ):
    raise ValueError("league.promotion_reference_holdout_deck_indices must be unique")
  if any(index < 0 for index in cfg.promotion_reference_holdout_deck_indices):
    raise ValueError("league.promotion_reference_holdout_deck_indices must be non-negative")
  if set(cfg.promotion_reference_deck_indices).intersection(
    cfg.promotion_reference_holdout_deck_indices
  ):
    raise ValueError("league promotion reference and holdout deck splits must be disjoint")
  bounded = {
    "promotion_screen_pooled_min": cfg.promotion_screen_pooled_min,
    "promotion_screen_opponent_floor": cfg.promotion_screen_opponent_floor,
    "promotion_screen_seat_floor": cfg.promotion_screen_seat_floor,
    "promotion_pooled_min": cfg.promotion_pooled_min,
    "promotion_opponent_quorum_min": cfg.promotion_opponent_quorum_min,
    "promotion_opponent_floor": cfg.promotion_opponent_floor,
    "promotion_seat_floor": cfg.promotion_seat_floor,
    "promotion_paired_lcb_min": cfg.promotion_paired_lcb_min,
    "promotion_bootstrap_confidence": cfg.promotion_bootstrap_confidence,
    "promotion_reference_candidate_floor": cfg.promotion_reference_candidate_floor,
    "promotion_reference_leap": cfg.promotion_reference_leap,
    "promotion_max_timeout_rate": cfg.promotion_max_timeout_rate,
    "promotion_anchored_screen_pooled_floor": cfg.promotion_anchored_screen_pooled_floor,
    "promotion_anchored_screen_opponent_floor": cfg.promotion_anchored_screen_opponent_floor,
    "promotion_anchored_screen_seat_floor": cfg.promotion_anchored_screen_seat_floor,
    "promotion_anchored_pooled_floor": cfg.promotion_anchored_pooled_floor,
    "promotion_anchored_opponent_floor": cfg.promotion_anchored_opponent_floor,
    "promotion_anchored_seat_floor": cfg.promotion_anchored_seat_floor,
    "promotion_anchored_max_timeout_rate": cfg.promotion_anchored_max_timeout_rate,
    "promotion_anchored_max_timeout_delta": cfg.promotion_anchored_max_timeout_delta,
  }
  for name, value in bounded.items():
    if not 0.0 <= value <= 1.0:
      raise ValueError(f"league.{name} must be in [0,1]")
  if not -1.0 <= cfg.promotion_reference_noninferiority <= 0.0:
    raise ValueError("league.promotion_reference_noninferiority must be in [-1,0]")
  relative_bounds = {
    "promotion_reference_paired_lcb_min": cfg.promotion_reference_paired_lcb_min,
    "promotion_anchored_screen_relative_pooled_min": (
      cfg.promotion_anchored_screen_relative_pooled_min
    ),
    "promotion_anchored_screen_relative_opponent_floor": (
      cfg.promotion_anchored_screen_relative_opponent_floor
    ),
    "promotion_anchored_screen_relative_seat_floor": (
      cfg.promotion_anchored_screen_relative_seat_floor
    ),
    "promotion_anchored_relative_pooled_min": cfg.promotion_anchored_relative_pooled_min,
    "promotion_anchored_relative_opponent_quorum_min": (
      cfg.promotion_anchored_relative_opponent_quorum_min
    ),
    "promotion_anchored_relative_opponent_floor": (
      cfg.promotion_anchored_relative_opponent_floor
    ),
    "promotion_anchored_relative_seat_floor": cfg.promotion_anchored_relative_seat_floor,
    "promotion_anchored_relative_paired_lcb_min": (
      cfg.promotion_anchored_relative_paired_lcb_min
    ),
    "promotion_anchored_leap_relative_pooled_min": (
      cfg.promotion_anchored_leap_relative_pooled_min
    ),
  }
  for name, value in relative_bounds.items():
    if not -1.0 <= value <= 0.0:
      raise ValueError(f"league.{name} must be in [-1,0]")
  if cfg.promotion_opponent_quorum_count < 0 or cfg.promotion_opponent_quorum_count > 4:
    raise ValueError("league.promotion_opponent_quorum_count must be in [0,4]")
  if cfg.promotion_opponent_break_even_count < 0 or cfg.promotion_opponent_break_even_count > 4:
    raise ValueError("league.promotion_opponent_break_even_count must be in [0,4]")
  if not 0 <= cfg.promotion_anchored_relative_opponent_quorum_count <= 4:
    raise ValueError(
      "league.promotion_anchored_relative_opponent_quorum_count must be in [0,4]"
    )
  if not 0 <= cfg.promotion_anchored_relative_break_even_count <= 4:
    raise ValueError("league.promotion_anchored_relative_break_even_count must be in [0,4]")
  if not 0.5 < cfg.promotion_bootstrap_confidence < 1.0:
    raise ValueError("league.promotion_bootstrap_confidence must be in (0.5,1)")
  if cfg.promotion_bootstrap_samples < 100:
    raise ValueError("league.promotion_bootstrap_samples must be >= 100")


@contextlib.contextmanager
def preserve_training_rng_state():
  """Keep synchronous evaluation from changing subsequent PPO sampling."""
  python_state = random.getstate()
  numpy_state = np.random.get_state()
  torch_cpu_state = torch.get_rng_state()
  torch_cuda_states = (
    torch.cuda.get_rng_state_all()
    if torch.cuda.is_available() and torch.cuda.is_initialized()
    else None
  )
  try:
    yield
  finally:
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_cpu_state)
    if torch_cuda_states is not None:
      torch.cuda.set_rng_state_all(torch_cuda_states)


class LeagueManager:
  def __init__(self, config: LeagueManagerConfig):
    self.config = config
    self.state_path = Path(config.state_path)
    self.state: LeagueState = load_league_state(self.state_path)
    default_archive_path = self.state_path.with_name(
      f"{self.state_path.stem}_promotion{self.state_path.suffix or '.json'}"
    )
    self.archive_state_path = Path(config.promotion_state_path) if config.promotion_state_path else default_archive_path
    self.archive_state: PromotionArchiveState = load_promotion_archive_state(self.archive_state_path)
    self.promotion_records_dir = (
      Path(config.promotion_records_dir)
      if config.promotion_records_dir
      else self.archive_state_path.parent / "promotion_records"
    )
    self.latest_metrics: dict[str, float | str] = {}
    self.current_candidate_id: str | None = self.state.current_candidate_policy_id
    self.evaluator = make_league_evaluator(config.eval_mode)
    self._checkpoint_hash_cache: dict[str, str] = dict(self.archive_state.checkpoint_hashes)

  def save(self) -> None:
    save_league_state(self.state_path, self.state)
    if self.config.promotion_mode == "archive_panel":
      self.archive_state.checkpoint_hashes = dict(self._checkpoint_hash_cache)
      save_promotion_archive_state(self.archive_state_path, self.archive_state)

  def ensure_seed_policies(self, checkpoint_paths: list[Path], *, created_epoch: int = 0) -> None:
    anchor_entry = None
    anchor_policy_id = None
    configured_anchor_path = None
    if self.config.promotion_mode == "archive_panel" and self.config.production_anchor_checkpoint:
      anchor_path = Path(self.config.production_anchor_checkpoint).expanduser()
      if not anchor_path.is_absolute():
        anchor_path = Path.cwd() / anchor_path
      if not anchor_path.exists():
        raise FileNotFoundError(f"Production anchor checkpoint not found: {anchor_path}")
      configured_anchor_path = anchor_path.resolve()
      anchor_entry = next(
        (
          entry
          for entry in self.state.policies.values()
          if Path(entry.checkpoint_path).resolve() == configured_anchor_path
        ),
        None,
      )
      add_anchor_to_training = (
        self.config.promotion_anchor_in_training_pool
        or not self.config.promotion_bootstrap_panel_path
      )
      if anchor_entry is None and add_anchor_to_training:
        anchor_entry = register_policy(
          self.state,
          checkpoint_path=anchor_path,
          created_epoch=created_epoch,
          source="production_anchor",
        )
      if anchor_entry is not None:
        anchor_policy_id = anchor_entry.policy_id
      else:
        anchor_hash = checkpoint_sha256(anchor_path)
        anchor_policy_id = f"external_anchor_{anchor_hash[:16]}"
        self._checkpoint_hash_cache[anchor_policy_id] = anchor_hash
    for path in checkpoint_paths:
      if not path.exists():
        continue
      entry = register_policy(
        self.state,
        checkpoint_path=path,
        created_epoch=created_epoch,
        source="seed",
      )
      if (
        configured_anchor_path is not None
        and Path(entry.checkpoint_path).resolve() == configured_anchor_path
      ):
        anchor_entry = entry
        anchor_policy_id = entry.policy_id
      elif anchor_policy_id is None:
        anchor_entry = entry
        anchor_policy_id = entry.policy_id
    if self.config.promotion_mode == "archive_panel" and anchor_policy_id is not None:
      ensure_production_anchor(
        self.archive_state,
        policy_id=anchor_policy_id,
        epoch=created_epoch,
      )
      if self.archive_state.production_anchor_policy_id != anchor_policy_id:
        raise ValueError(
          "Configured production anchor does not match the existing promotion archive state: "
          f"configured={anchor_policy_id}, "
          f"state={self.archive_state.production_anchor_policy_id}"
        )
      # Compatibility only: archive admission never changes this pointer. An
      # external evaluation anchor is not a training-pool champion; leaving a
      # stale pool id here would protect it from normal retention pruning.
      self.state.champion_policy_id = (
        anchor_entry.policy_id if anchor_entry is not None else None
      )
      self._maybe_bootstrap_panel()
    self._prune()
    self.save()

  def attach_learner_identity(self, learner_id: str) -> None:
    if not learner_id:
      return
    self.state.learner_policy_id = str(learner_id)
    self.save()

  def _prune(self) -> list[str]:
    protected = (
      protected_policy_ids(self.archive_state)
      if (
        self.config.promotion_mode == "archive_panel"
        and self.config.promotion_archive_affects_training_pool
      )
      else set()
    )
    return classify_and_prune(
      self.state,
      keep_recent=self.config.keep_recent,
      keep_mid=self.config.keep_mid,
      keep_old=self.config.keep_old,
      protected_ids=protected,
    )

  def _checkpoint_hash(self, policy_id: str) -> str:
    cached = self._checkpoint_hash_cache.get(policy_id)
    if cached:
      return cached
    entry = self._promotion_entry(policy_id)
    if entry is None:
      raise KeyError(f"Unknown policy id for checkpoint hash: {policy_id}")
    digest = checkpoint_sha256(Path(entry.checkpoint_path))
    self._checkpoint_hash_cache[policy_id] = digest
    return digest

  def _promotion_entry(self, policy_id: str | None) -> LeaguePolicyEntry | None:
    entry = self._entry_by_id(policy_id)
    if entry is not None or policy_id is None:
      return entry
    for panel in reversed(self.archive_state.panels):
      for member in panel.members:
        if member.policy_id != policy_id:
          continue
        checkpoint_path = Path(member.checkpoint_path)
        return LeaguePolicyEntry(
          policy_id=member.policy_id,
          checkpoint_path=str(checkpoint_path),
          created_epoch=int(member.selection_evidence.get("epoch", 0)),
          created_ts=0.0,
          source="promotion_panel",
          active=True,
          bucket="promotion_panel",
        )
    return None

  def _promotion_entries(self) -> dict[str, LeaguePolicyEntry]:
    entries = dict(self.state.policies)
    for panel in self.archive_state.panels:
      for member in panel.members:
        if member.policy_id not in entries:
          entry = self._promotion_entry(member.policy_id)
          if entry is not None:
            entries[member.policy_id] = entry
    return entries

  def _maybe_bootstrap_panel(self) -> None:
    raw_path = self.config.promotion_bootstrap_panel_path
    if not raw_path or self.archive_state.panels:
      return
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
      path = Path.cwd() / path
    if not path.exists():
      raise FileNotFoundError(f"Promotion bootstrap panel manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_members = payload.get("members")
    if not isinstance(raw_members, list) or len(raw_members) != self.config.promotion_panel_size:
      raise ValueError("Promotion bootstrap panel must contain exactly four members")
    source_ids = [str(item.get("policy_id", "")) for item in raw_members]
    if any(not policy_id for policy_id in source_ids) or len(set(source_ids)) != len(source_ids):
      raise ValueError("Promotion bootstrap panel policy ids must be non-empty and unique")
    source_anchor = str(payload.get("anchor_id", source_ids[0]))
    anchor_id = self.archive_state.production_anchor_policy_id
    if anchor_id is None:
      raise ValueError("Production anchor must be initialized before the bootstrap panel")

    members = []
    mapped_ids: set[str] = set()
    for item in raw_members:
      source_id = str(item["policy_id"])
      policy_id = anchor_id if source_id == source_anchor else f"historical:{source_id}"
      if policy_id in mapped_ids:
        raise ValueError("Promotion bootstrap panel aliases are not unique")
      mapped_ids.add(policy_id)
      checkpoint_path = Path(str(item["checkpoint_path"])).expanduser()
      if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
      if not checkpoint_path.exists():
        raise FileNotFoundError(f"Bootstrap panel checkpoint not found: {checkpoint_path}")
      if source_id == source_anchor:
        anchor_entry = self._entry_by_id(anchor_id)
        expected_anchor_path = (
          Path(anchor_entry.checkpoint_path).resolve()
          if anchor_entry is not None
          else Path(self.config.production_anchor_checkpoint).expanduser().resolve()
        )
        if checkpoint_path.resolve() != expected_anchor_path:
          raise ValueError("Bootstrap panel anchor checkpoint does not match production anchor")
      actual_hash = checkpoint_sha256(checkpoint_path)
      expected_hash = str(item.get("checkpoint_hash", ""))
      if expected_hash and actual_hash != expected_hash:
        raise ValueError(f"Bootstrap panel checkpoint hash mismatch: {source_id}")
      self._checkpoint_hash_cache[policy_id] = actual_hash
      evidence = dict(item.get("selection_evidence", item.get("evidence", {})) or {})
      evidence["source_policy_id"] = source_id
      if "epoch" in item:
        evidence["epoch"] = int(item["epoch"])
      members.append(
        PanelMember(
          policy_id=policy_id,
          role=str(item.get("role", "historical")),
          checkpoint_path=str(checkpoint_path.resolve()),
          checkpoint_hash=actual_hash,
          selection_evidence=evidence,
        )
      )
      if bool(item.get("quality_qualified", False)) and not any(
        archive_item.policy_id == policy_id for archive_item in self.archive_state.quality_archive
      ):
        self.archive_state.quality_archive.append(
          QualityArchiveEntry(
            policy_id=policy_id,
            admitted_epoch=0,
            run_id="retrospective-v1",
            route="retrospective_bootstrap",
          )
        )

    if members[0].policy_id != anchor_id:
      raise ValueError("Production anchor must be the first bootstrap panel member")
    version = int(payload.get("panel_version", 1))
    manifest = PanelManifest(
      version=version,
      activated_epoch=int(payload.get("activated_epoch", 0)),
      schedule_seed=int(payload.get("schedule_seed", self.config.promotion_schedule_seed)),
      members=members,
    )
    self.archive_state.panels.append(manifest)
    self.archive_state.active_panel_version = version
    self.archive_state.history.append(
      {
        "event": "panel_bootstrapped_from_retrospective",
        "version": version,
        "manifest_path": str(path.resolve()),
        "members": [member.policy_id for member in members],
        "ts": float(time.time()),
      }
    )

  def _promotion_thresholds(self) -> PromotionThresholds:
    cfg = self.config
    return PromotionThresholds(
      screen_pooled_min=cfg.promotion_screen_pooled_min,
      screen_opponent_floor=cfg.promotion_screen_opponent_floor,
      screen_seat_floor=cfg.promotion_screen_seat_floor,
      pooled_min=cfg.promotion_pooled_min,
      opponent_quorum_min=cfg.promotion_opponent_quorum_min,
      opponent_quorum_count=cfg.promotion_opponent_quorum_count,
      opponent_break_even_count=cfg.promotion_opponent_break_even_count,
      opponent_floor=cfg.promotion_opponent_floor,
      seat_floor=cfg.promotion_seat_floor,
      paired_lcb_min=cfg.promotion_paired_lcb_min,
      bootstrap_confidence=cfg.promotion_bootstrap_confidence,
      bootstrap_samples=cfg.promotion_bootstrap_samples,
      reference_candidate_floor=cfg.promotion_reference_candidate_floor,
      reference_noninferiority=cfg.promotion_reference_noninferiority,
      reference_paired_lcb_min=cfg.promotion_reference_paired_lcb_min,
      reference_leap=cfg.promotion_reference_leap,
      max_timeout_rate=cfg.promotion_max_timeout_rate,
      anchored_screen_pooled_floor=cfg.promotion_anchored_screen_pooled_floor,
      anchored_screen_opponent_floor=cfg.promotion_anchored_screen_opponent_floor,
      anchored_screen_seat_floor=cfg.promotion_anchored_screen_seat_floor,
      anchored_screen_relative_pooled_min=cfg.promotion_anchored_screen_relative_pooled_min,
      anchored_screen_relative_opponent_floor=(
        cfg.promotion_anchored_screen_relative_opponent_floor
      ),
      anchored_screen_relative_seat_floor=cfg.promotion_anchored_screen_relative_seat_floor,
      anchored_pooled_floor=cfg.promotion_anchored_pooled_floor,
      anchored_opponent_floor=cfg.promotion_anchored_opponent_floor,
      anchored_seat_floor=cfg.promotion_anchored_seat_floor,
      anchored_relative_pooled_min=cfg.promotion_anchored_relative_pooled_min,
      anchored_relative_opponent_quorum_min=(
        cfg.promotion_anchored_relative_opponent_quorum_min
      ),
      anchored_relative_opponent_quorum_count=(
        cfg.promotion_anchored_relative_opponent_quorum_count
      ),
      anchored_relative_break_even_count=cfg.promotion_anchored_relative_break_even_count,
      anchored_relative_opponent_floor=cfg.promotion_anchored_relative_opponent_floor,
      anchored_relative_seat_floor=cfg.promotion_anchored_relative_seat_floor,
      anchored_relative_paired_lcb_min=cfg.promotion_anchored_relative_paired_lcb_min,
      anchored_leap_relative_pooled_min=cfg.promotion_anchored_leap_relative_pooled_min,
      anchored_max_timeout_rate=cfg.promotion_anchored_max_timeout_rate,
      anchored_max_timeout_delta=cfg.promotion_anchored_max_timeout_delta,
    )

  def _active_entries(self) -> list[LeaguePolicyEntry]:
    return [entry for entry in self.state.policies.values() if entry.active and Path(entry.checkpoint_path).exists()]

  def _entry_by_id(self, policy_id: str | None) -> LeaguePolicyEntry | None:
    if policy_id is None:
      return None
    return self.state.policies.get(policy_id)

  def _load_policy(self, *, entry: LeaguePolicyEntry, vecenv, trainer_args: dict, build_policy_fn, load_weights_fn):
    from evaluate_checkpoint import _apply_checkpoint_resume_policy_config

    policy_args = copy.deepcopy(trainer_args)
    checkpoint_path = Path(entry.checkpoint_path)
    _apply_checkpoint_resume_policy_config(policy_args, checkpoint_path)
    policy = build_policy_fn(vecenv, policy_args)
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
    if (
      self.config.promotion_mode == "archive_panel"
      and self.archive_state.production_anchor_policy_id not in self.state.policies
    ):
      self.state.champion_policy_id = None
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

  def _maybe_evaluate_archive_panel(
    self,
    *,
    epoch: int,
    trainer_args: dict,
    vecenv,
    build_policy_fn,
    load_weights_fn,
    learner_policy: torch.nn.Module,
  ) -> dict[str, float | str]:
    run_screen = epoch % self.config.quick_eval_interval == 0
    run_confirmation = epoch % self.config.full_eval_interval == 0
    if self.config.eval_interval > 1 and epoch % self.config.eval_interval != 0:
      return {}
    if run_confirmation:
      run_screen = True
    if not run_screen or self.current_candidate_id is None:
      return {}

    candidate = self._entry_by_id(self.current_candidate_id)
    if candidate is None or not Path(candidate.checkpoint_path).exists():
      return {}
    anchor_id = self.archive_state.production_anchor_policy_id
    anchor = self._promotion_entry(anchor_id)
    if anchor is None or not Path(anchor.checkpoint_path).exists():
      return {
        "league/reject_reason": "production_anchor_missing",
        "league/reject_reason_code": 1.0,
      }
    if candidate.policy_id == anchor.policy_id:
      return {}
    if any(
      item.active and item.policy_id == candidate.policy_id
      for item in self.archive_state.quality_archive
    ):
      return {}

    panel = ensure_panel_manifest(
      self.archive_state,
      self._promotion_entries(),
      epoch=epoch,
      panel_size=self.config.promotion_panel_size,
      refresh_epochs=self.config.promotion_panel_refresh_epochs,
      base_seed=self.config.promotion_schedule_seed,
      checkpoint_hash=self._checkpoint_hash,
      exclude_ids={candidate.policy_id},
    )
    if panel is None:
      self.archive_state.history.append(
        {
          "event": "promotion_evaluation_skipped",
          "epoch": int(epoch),
          "candidate": candidate.policy_id,
          "reason": "panel_has_fewer_than_four_unique_policies",
          "ts": float(time.time()),
        }
      )
      save_promotion_archive_state(self.archive_state_path, self.archive_state)
      return {
        "league/promotion_accepted": 0.0,
        "league/promotion_decision_admitted": 0.0,
        "league/reject_reason": "panel_has_fewer_than_four_unique_policies",
        "league/reject_reason_code": 1.0,
      }

    panel_ids = [member.policy_id for member in panel.members]
    if len(panel_ids) != 4 or len(set(panel_ids)) != 4:
      raise RuntimeError(f"Promotion panel v{panel.version} is not four unique policies")
    panel_entries = []
    for policy_id in panel_ids:
      entry = self._promotion_entry(policy_id)
      if entry is None or not Path(entry.checkpoint_path).exists():
        raise FileNotFoundError(f"Promotion panel policy is unavailable: {policy_id}")
      panel_entries.append(entry)

    env_cfg = trainer_args.get("env", {})
    deck_pool_path = env_cfg.get("deck_pool_path") if isinstance(env_cfg, dict) else None
    deck_pool = load_training_deck_pool(deck_pool_path)
    catalog = build_deck_build_catalog(deck_pool)
    uniform_leader_ids = (
      catalog.leader_def_ids_by_element
      if isinstance(env_cfg, dict) and bool(env_cfg.get("draft_uniform_assignment", False))
      else None
    )
    all_reference_indices = (
      self.config.promotion_reference_deck_indices
      + self.config.promotion_reference_holdout_deck_indices
    )
    if all_reference_indices and max(all_reference_indices) >= len(deck_pool):
      raise ValueError(
        "Promotion reference deck index exceeds the configured deck pool: "
        f"max={max(all_reference_indices)}, pool={len(deck_pool)}"
      )
    reference_seeds = (
      self.config.promotion_reference_seeds
      if self.config.promotion_reference_seeds
      else (self.config.promotion_reference_seed,)
    )
    reference_manifest = {
      "version": "reference-split-v1",
      "controller_mode": "candidate_controls_both_seats",
      "assignment_contract": (
        "uniform_gate_same_element_leader"
        if uniform_leader_ids is not None
        else "policy_selects_leader"
      ),
      "seeds": list(reference_seeds),
      "promotion": [
        {"deck_index": index, "deck_hash": deck_sha256(deck_pool[index])}
        for index in self.config.promotion_reference_deck_indices
      ],
      "holdout": [
        {"deck_index": index, "deck_hash": deck_sha256(deck_pool[index])}
        for index in self.config.promotion_reference_holdout_deck_indices
      ],
    }

    thresholds = self._promotion_thresholds()
    device = str(trainer_args["train"].get("device", "cpu"))
    request_seed = panel.schedule_seed
    full_schedule = build_panel_schedule(
      panel_ids,
      catalog.records_by_code,
      base_seed=panel.schedule_seed,
      include_confirmation=True,
      leader_ids_by_element=uniform_leader_ids,
    )
    screen_schedule = [game for game in full_schedule if game.phase == SCREEN_PHASE]
    confirmation_schedule = [
      game for game in full_schedule if game.phase == CONFIRMATION_PHASE
    ]
    timings = {
      "policy_load_seconds": 0.0,
      "panel_anchor_seconds": 0.0,
      "screen_seconds": 0.0,
      "confirmation_seconds": 0.0,
      "reference_seconds": 0.0,
      "serialization_seconds": 0.0,
      "evaluator_forward_seconds": 0.0,
      "evaluator_env_step_seconds": 0.0,
      "evaluator_reset_seconds": 0.0,
      "evaluator_control_seconds": 0.0,
    }
    panel_records: list[PromotionGameRecord] = []
    anchor_policy = None

    def add_evaluator_timings(result) -> None:
      timings["evaluator_forward_seconds"] += float(
        result.timings.get("candidate_forward_seconds", 0.0)
        + result.timings.get("opponent_forward_seconds", 0.0)
      )
      timings["evaluator_env_step_seconds"] += float(
        result.timings.get("env_step_seconds", 0.0)
      )
      timings["evaluator_reset_seconds"] += float(
        result.timings.get("reset_seconds", 0.0)
      )
      timings["evaluator_control_seconds"] += float(
        result.timings.get("control_seconds", 0.0)
        + result.timings.get("record_drain_seconds", 0.0)
      )

    def load_panel_policy(entry: LeaguePolicyEntry):
      nonlocal anchor_policy
      if entry.policy_id == anchor.policy_id and anchor_policy is not None:
        return anchor_policy, False
      started = time.perf_counter()
      policy = self._load_policy(
        entry=entry,
        vecenv=vecenv,
        trainer_args=trainer_args,
        build_policy_fn=build_policy_fn,
        load_weights_fn=load_weights_fn,
      )
      timings["policy_load_seconds"] += time.perf_counter() - started
      if entry.policy_id == anchor.policy_id:
        anchor_policy = policy
        return policy, False
      return policy, True

    def evaluate_policy_phase(
      candidate_policy,
      phase_schedule,
      timing_key: str,
      destination: list[PromotionGameRecord],
    ) -> None:
      for entry in panel_entries:
        games = [game for game in phase_schedule if game.opponent_id == entry.policy_id]
        policy, release = load_panel_policy(entry)
        started = time.perf_counter()
        result = self.evaluator.evaluate_schedule(
          trainer_args,
          policy_a=candidate_policy,
          policy_b=policy,
          request=MatchRequest(
            episodes=len(games),
            max_steps=self.config.eval_max_steps,
            seed=request_seed,
            device=device,
            batch_envs=self.config.promotion_eval_batch_envs,
          ),
          games=games,
        )
        timings[timing_key] += time.perf_counter() - started
        add_evaluator_timings(result)
        destination.extend(result.records)
        if release:
          del policy
          gc.collect()

    evaluator_version = str(getattr(self.evaluator, "evaluator_version", "unknown"))
    panel_schedule_hash = schedule_sha256(full_schedule)
    panel_hash_suffix = "-".join(member.checkpoint_hash[:8] for member in panel.members)
    anchor_hash = self._checkpoint_hash(anchor.policy_id)
    panel_cache_key = (
      f"panel{panel.version}-{anchor_hash[:16]}-{panel_schedule_hash[:16]}-"
      f"{panel_hash_suffix}-m{self.config.eval_max_steps}-{evaluator_version}"
    ).replace("/", "_")
    panel_cache_raw = self.archive_state.panel_cache.get(panel_cache_key)
    panel_cache_path = (
      Path(panel_cache_raw)
      if panel_cache_raw
      else self.promotion_records_dir / "panel_cache" / f"{panel_cache_key}.json"
    )
    if panel_cache_path.exists():
      anchor_panel_records = load_records(panel_cache_path)
    else:
      if anchor_policy is None:
        anchor_policy, _ = load_panel_policy(anchor)
      anchor_panel_records: list[PromotionGameRecord] = []
      evaluate_policy_phase(
        anchor_policy,
        full_schedule,
        "panel_anchor_seconds",
        anchor_panel_records,
      )
      write_immutable_json(
        panel_cache_path,
        {
          "schema_version": 1,
          "kind": "production_anchor_panel_cache",
          "panel_version": panel.version,
          "anchor_policy_id": anchor.policy_id,
          "anchor_checkpoint": anchor.checkpoint_path,
          "anchor_checkpoint_hash": anchor_hash,
          "opponent_checkpoint_hashes": {
            member.policy_id: member.checkpoint_hash for member in panel.members
          },
          "schedule_hash": panel_schedule_hash,
          "evaluator_version": evaluator_version,
          "eval_max_steps": self.config.eval_max_steps,
          "schedule": [game.to_dict() for game in full_schedule],
          "games": [record.to_dict() for record in anchor_panel_records],
        },
      )
    self.archive_state.panel_cache[panel_cache_key] = str(panel_cache_path.resolve())

    evaluate_policy_phase(
      learner_policy,
      screen_schedule,
      "screen_seconds",
      panel_records,
    )
    screen_records = [record for record in panel_records if record.phase == SCREEN_PHASE]
    anchor_screen_records = [
      record for record in anchor_panel_records if record.phase == SCREEN_PHASE
    ]
    screen_summary = summarize_panel(
      screen_records,
      thresholds=thresholds,
      bootstrap_seed=panel.schedule_seed + 17,
    )
    screen_comparison = compare_panel_records(
      screen_records,
      anchor_screen_records,
      confidence=thresholds.bootstrap_confidence,
      samples=thresholds.bootstrap_samples,
      seed=panel.schedule_seed + 19,
    )
    passed_screen, screen_reasons = screen_passes(
      screen_summary,
      thresholds,
      screen_comparison,
    )

    reference = None
    external_strength_observation = None
    external_strength_window = None
    reference_schedule_hash = None
    reference_records: list[PromotionGameRecord] = []
    anchor_reference_records: list[PromotionGameRecord] = []
    evaluated_schedule = list(screen_schedule)
    decision: PromotionDecision
    if not passed_screen:
      decision = PromotionDecision(
        admitted=False,
        route="screen_rejected",
        reasons=screen_reasons,
        summary=screen_summary,
        reference=None,
        panel_comparison=screen_comparison,
      )
    elif not run_confirmation:
      decision = PromotionDecision(
        admitted=False,
        route="screen_passed",
        reasons=("confirmation_not_scheduled",),
        summary=screen_summary,
        reference=None,
        panel_comparison=screen_comparison,
      )
    else:
      evaluate_policy_phase(
        learner_policy,
        confirmation_schedule,
        "confirmation_seconds",
        panel_records,
      )
      evaluated_schedule.extend(confirmation_schedule)

      if self.config.promotion_require_reference:
        reference_schedule = [
          game
          for seed_index, reference_seed in enumerate(reference_seeds)
          for game in build_reference_schedule(
            self.config.promotion_reference_deck_indices,
            catalog.records_by_code,
            base_seed=reference_seed,
            schedule_id=(
              f"seed{seed_index:02d}" if len(reference_seeds) > 1 else ""
            ),
            leader_ids_by_element=uniform_leader_ids,
            leader_assignment_offset=seed_index % 2,
          )
        ]
        evaluated_schedule.extend(reference_schedule)
        if anchor_policy is None:
          anchor_policy, _ = load_panel_policy(anchor)
        reference_schedule_hash = schedule_sha256(reference_schedule)
        cache_key = (
          f"{anchor.policy_id}-{anchor_hash[:16]}-{reference_schedule_hash[:16]}-"
          f"m{self.config.eval_max_steps}-{evaluator_version}"
        ).replace("/", "_")
        cache_raw = self.archive_state.reference_cache.get(cache_key)
        cache_path = Path(cache_raw) if cache_raw else self.promotion_records_dir / "reference_cache" / f"{cache_key}.json"
        if cache_path.exists():
          anchor_reference_records = load_records(cache_path)
        else:
          started = time.perf_counter()
          anchor_result = self.evaluator.evaluate_schedule(
            trainer_args,
            policy_a=anchor_policy,
            policy_b=anchor_policy,
            request=MatchRequest(
              episodes=len(reference_schedule),
              max_steps=self.config.eval_max_steps,
              seed=reference_seeds[0],
              device=device,
              batch_envs=self.config.promotion_eval_batch_envs,
            ),
            games=reference_schedule,
          )
          timings["reference_seconds"] += time.perf_counter() - started
          add_evaluator_timings(anchor_result)
          anchor_reference_records = list(anchor_result.records)
          write_immutable_json(
            cache_path,
            {
              "schema_version": 1,
              "kind": "production_anchor_reference_cache",
              "anchor_policy_id": anchor.policy_id,
              "anchor_checkpoint": anchor.checkpoint_path,
              "anchor_checkpoint_hash": anchor_hash,
              "schedule_hash": reference_schedule_hash,
              "evaluator_version": evaluator_version,
              "eval_max_steps": self.config.eval_max_steps,
              "reference_manifest": reference_manifest,
              "schedule": [game.to_dict() for game in reference_schedule],
              "games": [record.to_dict() for record in anchor_reference_records],
            },
          )
          self.archive_state.reference_cache[cache_key] = str(cache_path.resolve())

        started = time.perf_counter()
        candidate_reference_result = self.evaluator.evaluate_schedule(
          trainer_args,
          policy_a=learner_policy,
          policy_b=learner_policy,
          request=MatchRequest(
            episodes=len(reference_schedule),
            max_steps=self.config.eval_max_steps,
            seed=reference_seeds[0],
            device=device,
            batch_envs=self.config.promotion_eval_batch_envs,
          ),
          games=reference_schedule,
        )
        timings["reference_seconds"] += time.perf_counter() - started
        add_evaluator_timings(candidate_reference_result)
        reference_records = list(candidate_reference_result.records)
        reference = compare_reference_records(
          reference_records,
          anchor_reference_records,
          confidence=thresholds.bootstrap_confidence,
          samples=thresholds.bootstrap_samples,
          seed=reference_seeds[0] + 29,
        )
        external_strength_observation = {
          "event": "external_strength_observation",
          "epoch": int(epoch),
          "candidate_id": candidate.policy_id,
          "candidate_checkpoint_hash": self._checkpoint_hash(candidate.policy_id),
          "schedule_hash": reference_schedule_hash,
          "candidate_score": float(reference.candidate_score),
          "anchor_score": float(reference.anchor_score),
          "delta": float(reference.delta),
          "paired_lcb": float(reference.paired_lcb),
          "games": int(reference.games),
        }
        external_strength_window = summarize_external_strength_window(
          self.archive_state.history,
          external_strength_observation,
        )

      decision = decide_promotion(
        panel_records,
        thresholds=thresholds,
        reference=reference,
        require_reference=self.config.promotion_require_reference,
        bootstrap_seed=panel.schedule_seed + 31,
        panel_comparison=compare_panel_records(
          panel_records,
          anchor_panel_records,
          confidence=thresholds.bootstrap_confidence,
          samples=thresholds.bootstrap_samples,
          seed=panel.schedule_seed + 37,
        ),
      )

    schedule_hash = schedule_sha256(evaluated_schedule)
    run_id = (
      f"epoch{int(epoch):06d}-{candidate.policy_id}-panel{panel.version:03d}-"
      f"{decision.route}-{schedule_hash[:10]}-{time.time_ns()}"
    )
    opponent_hashes = {member.policy_id: member.checkpoint_hash for member in panel.members}
    candidate_hash = self._checkpoint_hash(candidate.policy_id)
    for deck_index in self.config.promotion_reference_deck_indices:
      opponent_hashes[f"reference:{deck_index}"] = candidate_hash
    serialized_records = panel_records + reference_records
    started = time.perf_counter()
    run_path = write_promotion_run(
      self.promotion_records_dir,
      run_id=run_id,
      candidate_id=candidate.policy_id,
      candidate_checkpoint=candidate.checkpoint_path,
      candidate_checkpoint_hash=candidate_hash,
      panel_version=panel.version,
      schedule=evaluated_schedule,
      records=serialized_records,
      opponent_checkpoint_hashes=opponent_hashes,
      metadata={
        "epoch": int(epoch),
        "shadow_mode": bool(self.config.promotion_shadow_mode),
        "training_rng_isolated": True,
        "archive_affects_training_pool": bool(
          self.config.promotion_archive_affects_training_pool
        ),
        "production_anchor_policy_id": anchor.policy_id,
        "production_anchor_checkpoint_hash": anchor_hash,
        "panel": asdict(panel),
        "thresholds": asdict(thresholds),
        "reference_manifest": reference_manifest,
        "external_strength_window": external_strength_window,
        "timings": dict(timings),
        "anchor_panel_cache": str(panel_cache_path.resolve()),
        "anchor_reference_cache": None
        if not anchor_reference_records
        else self.archive_state.reference_cache,
      },
      screen_result={
        "passed": passed_screen,
        "reasons": list(screen_reasons),
        "summary": screen_summary.to_dict(),
        "panel_comparison": screen_comparison.to_dict(),
      },
      decision=decision,
    )
    timings["serialization_seconds"] += time.perf_counter() - started

    record_payoff_results(
      self.archive_state,
      run_id=run_id,
      candidate_id=candidate.policy_id,
      records=panel_records,
    )
    if external_strength_observation is not None:
      self.archive_state.history.append({
        **external_strength_observation,
        "run_id": run_id,
        "window": external_strength_window,
        "ts": float(time.time()),
      })
    if decision.admitted and not self.config.promotion_shadow_mode:
      admit_quality_policy(
        self.archive_state,
        policy_id=candidate.policy_id,
        epoch=epoch,
        run_id=run_id,
        route=decision.route,
        max_active=self.config.promotion_archive_max_active,
      )
      self._prune()
      save_league_state(self.state_path, self.state)
    else:
      self.archive_state.history.append(
        {
          "event": "promotion_shadow_decision"
          if self.config.promotion_shadow_mode
          else "quality_archive_rejected",
          "epoch": int(epoch),
          "candidate": candidate.policy_id,
          "admitted_by_rule": bool(decision.admitted),
          "route": decision.route,
          "reasons": list(decision.reasons),
          "run_id": run_id,
          "record_path": str(run_path.resolve()),
          "ts": float(time.time()),
        }
      )
    self.archive_state.checkpoint_hashes = dict(self._checkpoint_hash_cache)
    save_promotion_archive_state(self.archive_state_path, self.archive_state)

    summary = decision.summary
    reasons = "ok" if decision.admitted else ";".join(decision.reasons)
    metrics: dict[str, float | str] = {
      "league/champion_policy_id": str(self.state.champion_policy_id or ""),
      "league/production_anchor_policy_id": anchor.policy_id,
      "league/candidate_policy_id": candidate.policy_id,
      "league/promotion_panel_version": float(panel.version),
      "league/promotion_panel_games": float(summary.games),
      "league/promotion_panel_pooled_score": float(summary.pooled_score),
      "league/promotion_panel_paired_lcb": float(summary.paired_lcb),
      "league/promotion_panel_timeout_rate": float(summary.timeout_rate),
      "league/promotion_decision_admitted": 1.0 if decision.admitted else 0.0,
      "league/promotion_accepted": 1.0
      if decision.admitted and not self.config.promotion_shadow_mode
      else 0.0,
      "league/promotion_shadow_mode": 1.0 if self.config.promotion_shadow_mode else 0.0,
      "league/promotion_archive_affects_training_pool": (
        1.0 if self.config.promotion_archive_affects_training_pool else 0.0
      ),
      "league/promotion_route": decision.route,
      "league/reject_reason": reasons,
      "league/reject_reason_code": 0.0 if decision.admitted else 1.0,
      "league/eval_mode": self.config.eval_mode,
      "league/eval_panel_anchor_seconds": float(timings["panel_anchor_seconds"]),
      "league/eval_screen_seconds": float(timings["screen_seconds"]),
      "league/eval_confirmation_seconds": float(timings["confirmation_seconds"]),
      "league/eval_reference_seconds": float(timings["reference_seconds"]),
      "league/eval_policy_load_seconds": float(timings["policy_load_seconds"]),
      "league/eval_serialization_seconds": float(timings["serialization_seconds"]),
      "league/eval_forward_seconds": float(timings["evaluator_forward_seconds"]),
      "league/eval_env_step_seconds": float(timings["evaluator_env_step_seconds"]),
      "league/eval_reset_seconds": float(timings["evaluator_reset_seconds"]),
      "league/eval_control_seconds": float(timings["evaluator_control_seconds"]),
      "league/eval_raw_record_path": str(run_path.resolve()),
    }
    for opponent_id, score in summary.opponent_scores.items():
      metrics[f"league/promotion_opponent_score/{opponent_id}"] = float(score)
    for seat, score in summary.seat_scores.items():
      metrics[f"league/promotion_seat_score/{seat}"] = float(score)
    for leader, score in summary.leader_scores.items():
      metrics[f"league/promotion_leader_score/{leader}"] = float(score)
    for context, score in summary.context_scores.items():
      metrics[f"league/promotion_context_score/{context}"] = float(score)
    if decision.panel_comparison is not None:
      comparison = decision.panel_comparison
      metrics["league/promotion_panel_anchor_score"] = float(comparison.anchor_score)
      metrics["league/promotion_panel_relative_delta"] = float(comparison.delta)
      metrics["league/promotion_panel_relative_lcb"] = float(comparison.paired_lcb)
      metrics["league/promotion_panel_timeout_delta"] = float(comparison.timeout_delta)
      for opponent_id, delta in comparison.opponent_deltas.items():
        metrics[f"league/promotion_opponent_delta/{opponent_id}"] = float(delta)
      for seat, delta in comparison.seat_deltas.items():
        metrics[f"league/promotion_seat_delta/{seat}"] = float(delta)
      for leader, delta in comparison.leader_deltas.items():
        metrics[f"league/promotion_leader_delta/{leader}"] = float(delta)
      for context, delta in comparison.context_deltas.items():
        metrics[f"league/promotion_context_delta/{context}"] = float(delta)
    if reference is not None:
      metrics["league/promotion_reference_candidate_score"] = float(reference.candidate_score)
      metrics["league/promotion_reference_anchor_score"] = float(reference.anchor_score)
      metrics["league/promotion_reference_delta"] = float(reference.delta)
      metrics["league/promotion_reference_paired_lcb"] = float(reference.paired_lcb)
    if external_strength_window is not None:
      metrics["league/promotion_reference_window_count"] = float(
        external_strength_window["count"]
      )
      metrics["league/promotion_reference_window_score_mean"] = float(
        external_strength_window["score_mean"]
      )
      metrics["league/promotion_reference_window_score_min"] = float(
        external_strength_window["score_min"]
      )
      metrics["league/promotion_reference_window_delta_mean"] = float(
        external_strength_window["delta_mean"]
      )
      metrics["league/promotion_reference_window_slope_per_100_updates"] = float(
        external_strength_window["score_slope_per_100_updates"]
      )
    self.latest_metrics = metrics
    return dict(metrics)

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
    with preserve_training_rng_state():
      return self._maybe_evaluate_and_promote_isolated(
        epoch=epoch,
        trainer_args=trainer_args,
        vecenv=vecenv,
        build_policy_fn=build_policy_fn,
        load_weights_fn=load_weights_fn,
        learner_policy=learner_policy,
      )

  def _maybe_evaluate_and_promote_isolated(
    self,
    *,
    epoch: int,
    trainer_args: dict,
    vecenv,
    build_policy_fn,
    load_weights_fn,
    learner_policy: torch.nn.Module,
  ) -> dict[str, float | str]:
    if self.config.promotion_mode == "archive_panel":
      return self._maybe_evaluate_archive_panel(
        epoch=epoch,
        trainer_args=trainer_args,
        vecenv=vecenv,
        build_policy_fn=build_policy_fn,
        load_weights_fn=load_weights_fn,
        learner_policy=learner_policy,
      )
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
