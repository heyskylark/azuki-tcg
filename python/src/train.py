from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import glob
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import azk_puffer.pytorch as azk_pytorch
import azk_puffer.trainer as pufferl
import azk_puffer.vector as azk_vector
import torch

from production_runtime import (
    CheckpointPolicy,
    active_league_updates,
    apply_process_environment,
    filter_numeric_metrics,
    finalize_checkpoint_state,
    parse_patterns,
    prune_checkpoints,
)
from policy.v2 import tcg_sampler
from league_manager import LeagueManager, parse_league_manager_config
from league_training import (
    LeagueConfig,
    LeaguePuffeRL,
    compute_frozen_matchup_ratio,
    compute_league_active,
)
from playback import run_playback
from training_deck_pool import resolve_training_deck_pool_path
from training_utils import (
    DEFAULT_CONFIG_PATH,
    build_policy,
    build_vecenv,
    install_tcg_sampler,
    load_training_config,
)


RESUME_COMPLETED_EPISODES_ENV = "AZK_RESUME_COMPLETED_EPISODES"
RESUME_ALLOW_SCHEDULE_REWIND_ENV = "AZK_RESUME_ALLOW_SCHEDULE_REWIND"
RESUME_COMPLETED_EPISODE_KEYS = (
    "0/azk_completed_episodes",
    "1/azk_completed_episodes",
    "environment/0/azk_completed_episodes",
    "environment/1/azk_completed_episodes",
    "environment/completed_episodes",
)
RESUME_SCHEDULE_ENV_VARS = (
    "AZK_MAX_AUTO_TICKS_PER_STEP",
    "AZK_MAX_TICKS_PER_EPISODE",
    "AZK_MAX_TICKS_CURRICULUM",
    "AZK_MAX_TICKS_CURRICULUM_INITIAL",
    "AZK_MAX_TICKS_CURRICULUM_FINAL",
    "AZK_MAX_TICKS_CURRICULUM_WARMUP_EPISODES",
    "AZK_MAX_TICKS_CURRICULUM_RAMP_EPISODES",
    "AZK_MAX_TICKS_CURRICULUM_LONG_EPISODE_EVERY",
    "AZK_MAX_TICKS_CURRICULUM_LONG_EPISODE_CAP",
    "AZK_REWARD_SHAPING_ANNEAL",
    "AZK_REWARD_SHAPING_ANNEAL_INITIAL",
    "AZK_REWARD_SHAPING_ANNEAL_FINAL",
    "AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES",
    "AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES",
    "AZK_TRAINER_SHAPED_REWARD_ANNEAL",
    "AZK_TRAINER_SHAPED_REWARD_ANNEAL_START_EPOCH",
    "AZK_TRAINER_SHAPED_REWARD_ANNEAL_END_EPOCH",
)
RESUME_REWARD_ENV_VARS = (
    "AZK_REWARD_LEADER_DELTA_WEIGHT",
    "AZK_REWARD_BOARD_DELTA_WEIGHT",
    "AZK_REWARD_NOOP_PENALTY",
    "AZK_TRUNCATION_BOARD_EDGE_WEIGHT",
    "AZK_REWARD_UNTAPPED_IKZ_WEIGHT",
    "AZK_PORTAL_GP_BONUS",
    "AZK_PORTAL_OUTCOME_BONUS",
    "AZK_EARLY_TEMPO_BONUS",
    "AZK_EARLY_TEMPO_CAP",
    "AZK_EARLY_TEMPO_TURNS",
    "AZK_EARLY_TEMPO_DEDUP_PORTAL_ABILITIES",
    "AZK_DMG_MITIGATION_BONUS",
    "AZK_DMG_MITIGATION_CAP",
    "AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP",
    "AZK_ENTITY_DAMAGE_EXCHANGE_STEP_CAP",
    "AZK_GENERATED_IKZ_CONVERSION_BONUS",
    "AZK_GENERATED_IKZ_CONVERSION_STEP_CAP",
    "AZK_TEMP_CHARGE_REALIZATION_BONUS",
    "AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE",
    "AZK_TEMP_ATTACK_REALIZATION_DAMAGE_CAP",
    "AZK_CONTEXTUAL_RESPONSE_RESERVE_BONUS",
    "AZK_DRAFT_VBOOT_COEF",
    "AZK_DRAFT_SIBDIFF_COEF",
    "AZK_DRAFT_SIBDIFF_CAP",
    "AZK_DRAFT_AUX_ANNEAL",
    "AZK_LEADER_TERMINAL_CREDIT_COEF",
    "AZK_LEADER_TERMINAL_CREDIT_CLIP",
    "AZK_LEADER_TERMINAL_CREDIT_GATE_CODES",
    "AZK_LEADER_TERMINAL_CREDIT_UPDATE_INTERVAL",
    "AZK_LEADER_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS",
    "AZK_DRAFT_TERMINAL_CREDIT_COEF",
    "AZK_DRAFT_TERMINAL_CREDIT_CLIP",
    "AZK_DRAFT_TERMINAL_CREDIT_UPDATE_INTERVAL",
    "AZK_DRAFT_TERMINAL_CREDIT_BATCH_SIZE",
    "AZK_DRAFT_TERMINAL_CREDIT_SEED",
    "AZK_DRAFT_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS",
    "AZK_DRAFT_TERMINAL_CREDIT_GRAD_PROBE",
    "AZK_DRAFT_EPISODE_CREDIT_COEF",
    "AZK_DRAFT_EPISODE_CREDIT_CLIP",
    "AZK_DRAFT_EPISODE_CREDIT_BATCH_DRAFTS",
    "AZK_DRAFT_EPISODE_CREDIT_UPDATE_INTERVAL",
    "AZK_DRAFT_EPISODE_CREDIT_BASELINE_COEF",
    "AZK_DRAFT_EPISODE_CREDIT_SEED",
    "AZK_DRAFT_EPISODE_CREDIT_LABEL_WARMUP_EPOCHS",
    "AZK_DRAFT_PREFIX_OUTCOME_MODEL",
    "AZK_DRAFT_PREFIX_OUTCOME_SHA256",
    "AZK_DRAFT_PREFIX_OUTCOME_COEF",
    "AZK_DRAFT_PREFIX_LENGTHS",
    "AZK_DRAFT_PREFIX_PROBS",
    "AZK_DRAFT_PREFIX_SEED",
)
RESUME_SOURCE_HASH_TARGETS = (
    "python/src/azk_puffer/trainer.py",
    "python/src/league_training.py",
    "python/src/draft_prefix_outcome.py",
    "python/src/policy/v2/tcg_policy.py",
    "python/src/policy/v2/tcg_sampler.py",
    "python/src/v2/tcg.py",
    "python/src/v2/tcg_parallel.py",
    "python/src/v2/observation.py",
    "python/src/deck_building.py",
    "python/src/tcg.h",
    "python/src/production_runtime.py",
    "python/src/train.py",
    "python/src/training_deck_pool.py",
    "python/src/training_utils.py",
    ".codex/docs/azuki_garden_arena_2026-08-15_decks.json",
)


def _seed_training_process(train_config: dict) -> int | None:
    """Optionally seed process RNGs before environments and policies are built."""
    if not bool(train_config.get("seed_process_rngs", False)):
        return None

    seed = int(train_config.get("seed", 0))
    if not 0 <= seed <= (1 << 32) - 1:
        raise ValueError("train.seed must be in [0, 2**32 - 1] when process seeding is enabled")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


class _JsonlLogger:
    """Windowed local metrics logger with a production-safe allowlist."""

    def __init__(self, trainer_args: dict, path: Path):
        tag = trainer_args.get("tag")
        suffix = str(int(100 * time.time()))
        self.run_id = f"{tag}_{suffix}" if tag else suffix
        path = path if path.suffix == ".jsonl" else path / f"{self.run_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", buffering=1, encoding="utf-8")
        logging_cfg = trainer_args.get("logging")
        if not isinstance(logging_cfg, dict):
            logging_cfg = {}
        self._interval = max(1, int(logging_cfg.get("interval_updates", 1)))
        self._patterns = parse_patterns(logging_cfg.get("metric_patterns"))
        self._sum_patterns = parse_patterns(logging_cfg.get("sum_patterns"), default=())
        self._max_patterns = parse_patterns(logging_cfg.get("max_patterns"), default=())
        self._pending: dict[str, list[float]] = {}
        self._pending_updates = 0
        self._last_step = 0

        config_snapshot: dict[str, object] = {}
        for section in (
            "train",
            "vec",
            "env",
            "policy",
            "league",
            "resume",
            "logging",
            "artifacts",
            "launch_gates",
            "process_env",
        ):
            section_cfg = trainer_args.get(section)
            if isinstance(section_cfg, dict):
                for key, value in section_cfg.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        config_snapshot[f"{section}.{key}"] = value
        self._handle.write(
            json.dumps(
                {"_event": "start", "run_id": self.run_id, "ts": round(time.time(), 2), "config": config_snapshot}
            )
            + "\n"
        )
        print(
            f"[jsonl-logger] writing compact metrics to {path} "
            f"every {self._interval} updates"
        )

    @staticmethod
    def _matches(key: str, patterns: tuple[str, ...]) -> bool:
        from fnmatch import fnmatchcase

        return any(fnmatchcase(key, pattern) for pattern in patterns)

    def _flush(self) -> None:
        if self._pending_updates == 0:
            return
        row: dict[str, object] = {
            "_step": self._last_step,
            "_window_updates": self._pending_updates,
            "ts": round(time.time(), 2),
        }
        for key, values in self._pending.items():
            total, minimum, maximum, last, count = values
            if key in {"agent_steps", "uptime", "epoch", "learning_rate"}:
                row[key] = last
            elif self._matches(key, self._sum_patterns):
                row[key] = total
            elif self._matches(key, self._max_patterns):
                row[key] = maximum
            else:
                row[key] = total / count
        self._handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._pending.clear()
        self._pending_updates = 0

    def log(self, logs, step):
        selected = filter_numeric_metrics(logs, self._patterns)
        for key, value in selected.items():
            values = self._pending.get(key)
            if values is None:
                self._pending[key] = [value, value, value, value, 1.0]
            else:
                values[0] += value
                values[1] = min(values[1], value)
                values[2] = max(values[2], value)
                values[3] = value
                values[4] += 1.0
        self._last_step = int(step)
        self._pending_updates += 1
        if self._pending_updates >= self._interval:
            self._flush()

    def close(self, model_path=None):
        try:
            self._flush()
            self._handle.write(
                json.dumps({"_event": "close", "model_path": str(model_path or ""), "ts": round(time.time(), 2)})
                + "\n"
            )
            self._handle.close()
        except Exception:
            pass


@dataclass(frozen=True)
class _LinearAnneal:
    initial: float
    final: float
    start_step: int
    end_step: int

    def value_at(self, step: int) -> float:
        if self.end_step <= self.start_step:
            return float(self.final)
        if step <= self.start_step:
            return float(self.initial)
        if step >= self.end_step:
            return float(self.final)
        progress = (float(step) - float(self.start_step)) / (float(self.end_step) - float(self.start_step))
        return float(self.initial + progress * (self.final - self.initial))


@dataclass(frozen=True)
class _SamplerAnnealConfig:
    subaction_temperature: _LinearAnneal
    smoothing_eps: _LinearAnneal


@dataclass(frozen=True)
class _EntCoefAnnealConfig:
    ent_coef: _LinearAnneal


@dataclass(frozen=True)
class _TorchProfilerConfig:
    output_dir: Path
    warmup_epochs: int
    active_epochs: int
    row_limit: int
    record_shapes: bool
    profile_memory: bool
    with_stack: bool
    export_trace: bool
    stop_after_capture: bool

    @property
    def total_profile_epochs(self) -> int:
        return int(self.warmup_epochs + self.active_epochs)


def parse_script_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="PuffeRL trainer for the Azuki TCG environment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the PuffeRL-compatible .ini file to load.",
    )
    parser.add_argument(
        "--render-playback-interval",
        type=int,
        default=0,
        help="If >0, run a rendered playback every N epochs using the latest checkpoint.",
    )
    parser.add_argument(
        "--render-playback-final",
        action="store_true",
        help="Run a rendered playback once after training finishes.",
    )
    parser.add_argument(
        "--render-playback-episodes",
        type=int,
        default=1,
        help="Number of episodes to roll out when playback runs.",
    )
    parser.add_argument(
        "--render-playback-steps",
        type=int,
        default=200,
        help="Maximum steps per episode during playback (guards log size).",
    )
    parser.add_argument(
        "--render-playback-dir",
        type=Path,
        help="Optional directory to write ANSI frames; prints to stdout if omitted.",
    )
    parser.add_argument(
        "--render-playback-device",
        type=str,
        default="cpu",
        help="Device to run playback inference on (cpu or cuda).",
    )
    parser.add_argument(
        "--render-playback-cast",
        action="store_true",
        help="Also emit an asciicast v2 (.cast) alongside the ANSI output.",
    )
    parser.add_argument(
        "--render-playback-no-clear-frames",
        action="store_true",
        help="Disable clear-screen control codes between frames in playback outputs.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help=(
            "Resume from a checkpoint file or checkpoint directory. "
            "If a directory is given, the newest model_*.pt file is used."
        ),
    )
    parser.add_argument(
        "--resume-load-optimizer",
        action="store_true",
        help="When resuming, also restore optimizer/global_step/epoch from trainer_state.pt if available.",
    )
    parser.add_argument(
        "--resume-restart-lr-schedule",
        action="store_true",
        help=(
            "After an optimizer restore, start a new cosine schedule over only the "
            "remaining updates, using train.learning_rate as its peak."
        ),
    )
    parser.add_argument(
        "--resume-strict",
        action="store_true",
        help="Require an exact key match when loading model weights from a checkpoint.",
    )
    parser.add_argument(
        "--resume-allow-deck-pool-migration",
        action="store_true",
        help=(
            "Allow only the deck_pool_path resume fingerprint to change. "
            "Use for an explicit corpus-migration experiment; all other config fields remain strict."
        ),
    )
    parser.add_argument(
        "--resume-reset-critic",
        action="store_true",
        help="When resuming, reinitialize the critic/value head to avoid unstable value-loss spikes.",
    )
    parser.add_argument(
        "--resume-auto-reset-critic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically reset critic on resume for continuity robustness.",
    )
    parser.add_argument(
        "--league-enable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable league mode (learner vs mixed latest/frozen opponents).",
    )
    parser.add_argument(
        "--league-opponent-dir",
        type=Path,
        help="Directory of opponent checkpoints (model_*.pt) for league training.",
    )
    parser.add_argument(
        "--league-opponent-checkpoints",
        type=str,
        default="",
        help="Comma-separated opponent checkpoint file paths for league training.",
    )
    parser.add_argument(
        "--league-frozen-ratio",
        type=float,
        default=None,
        help=(
            "Fraction of rollout rows that should be frozen league weights. "
            "With 2-player games, 0.20 means roughly 20%% frozen rows / 80%% trainable rows."
        ),
    )
    parser.add_argument(
        "--league-latest-ratio",
        type=float,
        default=None,
        help=(
            "Legacy league knob: fraction of matchups using latest learner as opponent. "
            "Prefer --league-frozen-ratio for row-level freeze control."
        ),
    )
    parser.add_argument(
        "--league-randomize-learner-seat",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Randomize learner seat each episode in league mode.",
    )
    parser.add_argument(
        "--league-activate-after-steps",
        type=int,
        default=None,
        help="Run pure self-play until this many learner steps, then activate league mixing/evals.",
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="Capture a torch.profiler trace for a short training window on the active Azuki trainer path.",
    )
    parser.add_argument(
        "--torch-profile-dir",
        type=Path,
        help="Directory for torch.profiler traces and summaries. Defaults under experiments/ when profiling is enabled.",
    )
    parser.add_argument(
        "--torch-profile-warmup-epochs",
        type=int,
        default=1,
        help="Number of epochs to use for profiler warmup before active capture begins.",
    )
    parser.add_argument(
        "--torch-profile-active-epochs",
        type=int,
        default=1,
        help="Number of epochs to capture with torch.profiler after warmup.",
    )
    parser.add_argument(
        "--torch-profile-row-limit",
        type=int,
        default=40,
        help="How many operator rows to keep in the saved top-op summaries.",
    )
    parser.add_argument(
        "--torch-profile-record-shapes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record operator input shapes in the torch.profiler trace.",
    )
    parser.add_argument(
        "--torch-profile-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record CPU/GPU memory usage in the torch.profiler trace.",
    )
    parser.add_argument(
        "--torch-profile-with-stack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include Python stacks in the torch.profiler trace. This is slower and larger.",
    )
    parser.add_argument(
        "--torch-profile-export-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write TensorBoard-compatible trace files. Disabled by default to keep profiling lightweight.",
    )
    parser.add_argument(
        "--torch-profile-stop-after-capture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop training once the configured profiler warmup + active epochs have completed.",
    )
    return parser.parse_known_args()


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def _materialize_scalar_norm_buffers_from_state_dict(
    policy: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> int:
    base_policy = getattr(policy, "policy", policy)
    scalar_norm = getattr(base_policy, "scalar_normalizer", None)
    ensure_buffers = getattr(scalar_norm, "_ensure_buffers", None)
    if scalar_norm is None or ensure_buffers is None:
        return 0
    try:
        policy_device = next(base_policy.parameters()).device
    except StopIteration:
        policy_device = torch.device("cpu")

    created = 0
    for key, value in state_dict.items():
        if not key.endswith("_mean") or not torch.is_tensor(value):
            continue

        norm_key = None
        if key.startswith("policy.scalar_normalizer._rms_"):
            norm_key = key[len("policy.scalar_normalizer._rms_") : -len("_mean")]
        elif key.startswith("scalar_normalizer._rms_"):
            norm_key = key[len("scalar_normalizer._rms_") : -len("_mean")]

        if not norm_key:
            continue

        feature_shape = tuple(int(dim) for dim in value.shape)
        ensure_buffers(norm_key, feature_shape, policy_device)
        created += 1

    return created


def _select_latest_model_file(checkpoint_dir: Path) -> Path:
    candidates = sorted(glob.glob(str(checkpoint_dir / "model_*.pt")))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt files found in checkpoint directory: {checkpoint_dir}")
    return Path(candidates[-1])


def _resolve_resume_artifacts(resume_checkpoint: Path) -> tuple[Path, Path | None]:
    def _trainer_state_for_model(model_path: Path) -> Path | None:
        stem = model_path.stem
        suffix = stem.rsplit("_", 1)[-1] if "_" in stem else ""
        candidates: list[Path] = []
        if suffix.isdigit():
            candidates.append(model_path.parent / f"trainer_state_{suffix}.pt")
        candidates.append(model_path.parent / "trainer_state.pt")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    if resume_checkpoint.is_dir():
        model_path = _select_latest_model_file(resume_checkpoint)
        return model_path, _trainer_state_for_model(model_path)
    if not resume_checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint path not found: {resume_checkpoint}")
    return resume_checkpoint, _trainer_state_for_model(resume_checkpoint)


def _coerce_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        as_int = int(value)
        return as_int if as_int >= 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = int(stripped)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _resolve_anneal_steps(
    policy_cfg: dict[str, object],
    *,
    key_prefix: str,
    total_timesteps: int,
    default_start_frac: float,
    default_end_frac: float,
) -> tuple[int, int]:
    start_key_step = f"{key_prefix}_anneal_start_step"
    end_key_step = f"{key_prefix}_anneal_end_step"
    start_key_frac = f"{key_prefix}_anneal_start_frac"
    end_key_frac = f"{key_prefix}_anneal_end_frac"

    start_step = _coerce_nonnegative_int(policy_cfg.get(start_key_step))
    end_step = _coerce_nonnegative_int(policy_cfg.get(end_key_step))

    start_frac_raw = _coerce_float(policy_cfg.get(start_key_frac))
    end_frac_raw = _coerce_float(policy_cfg.get(end_key_frac))
    start_frac = _clamp(default_start_frac if start_frac_raw is None else start_frac_raw, 0.0, 1.0)
    end_frac = _clamp(default_end_frac if end_frac_raw is None else end_frac_raw, 0.0, 1.0)

    safe_total = max(0, int(total_timesteps))
    if start_step is None:
        start_step = int(round(start_frac * safe_total))
    if end_step is None:
        end_step = int(round(end_frac * safe_total))
    start_step = max(0, int(start_step))
    end_step = max(start_step, int(end_step))
    return start_step, end_step


def _build_sampler_anneal_config(
    trainer_args: dict,
    *,
    total_timesteps: int,
) -> _SamplerAnnealConfig:
    policy_cfg = trainer_args.get("policy")
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}

    temp_initial_raw = _coerce_float(policy_cfg.get("subaction_temperature_initial"))
    temp_initial = (
        float(tcg_sampler.DEFAULT_SUBACTION_TEMPERATURE)
        if temp_initial_raw is None
        else temp_initial_raw
    )
    temp_initial = max(temp_initial, 1e-6)

    temp_final_raw = _coerce_float(policy_cfg.get("subaction_temperature_final"))
    temp_final = temp_initial if temp_final_raw is None else max(temp_final_raw, 1e-6)

    smooth_initial_raw = _coerce_float(policy_cfg.get("smoothing_eps_initial"))
    smooth_initial = (
        float(tcg_sampler.DEFAULT_SMOOTHING_EPS)
        if smooth_initial_raw is None
        else smooth_initial_raw
    )
    smooth_initial = _clamp(smooth_initial, 0.0, 1.0)

    smooth_final_raw = _coerce_float(policy_cfg.get("smoothing_eps_final"))
    smooth_final = smooth_initial if smooth_final_raw is None else _clamp(smooth_final_raw, 0.0, 1.0)

    temp_start_step, temp_end_step = _resolve_anneal_steps(
        policy_cfg,
        key_prefix="subaction_temperature",
        total_timesteps=total_timesteps,
        default_start_frac=0.0,
        default_end_frac=1.0,
    )
    smooth_start_step, smooth_end_step = _resolve_anneal_steps(
        policy_cfg,
        key_prefix="smoothing_eps",
        total_timesteps=total_timesteps,
        default_start_frac=0.0,
        default_end_frac=1.0,
    )

    return _SamplerAnnealConfig(
        subaction_temperature=_LinearAnneal(
            initial=float(temp_initial),
            final=float(temp_final),
            start_step=int(temp_start_step),
            end_step=int(temp_end_step),
        ),
        smoothing_eps=_LinearAnneal(
            initial=float(smooth_initial),
            final=float(smooth_final),
            start_step=int(smooth_start_step),
            end_step=int(smooth_end_step),
        ),
    )


def _build_ent_coef_anneal_config(
    trainer_args: dict,
    *,
    total_timesteps: int,
) -> _EntCoefAnnealConfig:
    train_cfg = trainer_args.get("train")
    if not isinstance(train_cfg, dict):
        train_cfg = {}

    base_ent_coef_raw = _coerce_float(train_cfg.get("ent_coef"))
    base_ent_coef = 0.0 if base_ent_coef_raw is None else base_ent_coef_raw
    base_ent_coef = max(float(base_ent_coef), 0.0)

    initial_raw = _coerce_float(train_cfg.get("ent_coef_anneal_initial"))
    initial = base_ent_coef if initial_raw is None else max(float(initial_raw), 0.0)

    final_raw = _coerce_float(train_cfg.get("ent_coef_anneal_final"))
    final = initial if final_raw is None else max(float(final_raw), 0.0)

    start_step, end_step = _resolve_anneal_steps(
        train_cfg,
        key_prefix="ent_coef",
        total_timesteps=total_timesteps,
        default_start_frac=0.0,
        default_end_frac=1.0,
    )

    return _EntCoefAnnealConfig(
        ent_coef=_LinearAnneal(
            initial=float(initial),
            final=float(final),
            start_step=int(start_step),
            end_step=int(end_step),
        ),
    )


def _apply_sampler_anneal(config: _SamplerAnnealConfig, *, global_step: int) -> tuple[float, float]:
    step = max(0, int(global_step))
    subaction_temperature = config.subaction_temperature.value_at(step)
    smoothing_eps = config.smoothing_eps.value_at(step)
    tcg_sampler.set_sampling_params(
        subaction_temperature=subaction_temperature,
        smoothing_eps=smoothing_eps,
    )
    return float(subaction_temperature), float(smoothing_eps)


def _apply_ent_coef_anneal(trainer, config: _EntCoefAnnealConfig, *, global_step: int) -> float:
    step = max(0, int(global_step))
    ent_coef = config.ent_coef.value_at(step)
    trainer.config["ent_coef"] = float(ent_coef)
    return float(ent_coef)


def _iter_numeric_values(value: object):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            yield numeric
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_numeric_values(item)


def _extract_completed_episodes_from_mapping(mapping: object) -> int | None:
    if not isinstance(mapping, dict):
        return None

    candidates: list[float] = []
    for key in RESUME_COMPLETED_EPISODE_KEYS:
        if key not in mapping:
            continue
        candidates.extend(_iter_numeric_values(mapping[key]))

    if not candidates:
        return None
    # Native vector metrics report the mean counter across environments. Round
    # upward so a resume never makes curriculum or reward shaping denser again;
    # this advances an aggregate by less than one episode at most.
    return int(math.ceil(max(candidates)))


def _update_completed_episode_tracker(current_value: int | None, source: object) -> int | None:
    candidate = _extract_completed_episodes_from_mapping(source)
    if candidate is None:
        return current_value
    if current_value is None:
        return candidate
    return max(current_value, candidate)


def _resume_env_var_fingerprint(names: tuple[str, ...]) -> dict[str, str]:
    return {name: os.getenv(name, "") for name in names}


def _source_hash_fingerprint() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    out: dict[str, str] = {}
    for rel_path in RESUME_SOURCE_HASH_TARGETS:
        target = repo_root / rel_path
        if not target.exists():
            out[rel_path] = "<missing>"
            continue
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        out[rel_path] = digest.hexdigest()
    return out


def _env_flag(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    normalized = raw.strip().lower()
    if not normalized:
        return False
    return normalized not in {"0", "false", "no", "off"}


def _env_nonnegative_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    parsed = _coerce_nonnegative_int(raw)
    if parsed is None:
        return default
    return parsed


def _episode_length_upper_bound_from_env() -> int | None:
    candidates: list[int] = []

    explicit_cap = _env_nonnegative_int("AZK_MAX_TICKS_PER_EPISODE", default=0)
    if explicit_cap is not None and explicit_cap > 0:
        candidates.append(explicit_cap)

    curriculum_enabled = _env_flag("AZK_MAX_TICKS_CURRICULUM")
    if curriculum_enabled:
        final_cap_default = explicit_cap if explicit_cap is not None and explicit_cap > 0 else 1000
        final_cap = _env_nonnegative_int("AZK_MAX_TICKS_CURRICULUM_FINAL", default=final_cap_default)
        if final_cap is None or final_cap <= 0:
            final_cap = final_cap_default

        initial_cap_default = 300 if final_cap > 300 else final_cap
        initial_cap = _env_nonnegative_int("AZK_MAX_TICKS_CURRICULUM_INITIAL", default=initial_cap_default)
        if initial_cap is None or initial_cap <= 0:
            initial_cap = initial_cap_default

        long_cap_default = max(final_cap + 400, 1600)
        long_cap = _env_nonnegative_int("AZK_MAX_TICKS_CURRICULUM_LONG_EPISODE_CAP", default=long_cap_default)
        if long_cap is None or long_cap <= 0:
            long_cap = final_cap

        candidates.extend([initial_cap, final_cap, long_cap])

    if not candidates:
        return None
    return int(max(candidates))


def _infer_completed_episodes_from_global_step(
    global_step: int, *, num_envs_hint: int
) -> int | None:
    if global_step <= 0:
        return None
    episode_upper_bound = _episode_length_upper_bound_from_env()
    if episode_upper_bound is None or episode_upper_bound <= 0:
        return None
    parallel_envs = max(1, int(num_envs_hint))
    per_env_steps = global_step // parallel_envs
    inferred = max(1, int(per_env_steps // episode_upper_bound))
    return inferred


def _enabled_episode_schedule_completion() -> int | None:
    """Return the first episode where every enabled monotonic schedule is final."""
    completions: list[int] = []

    if _env_flag("AZK_MAX_TICKS_CURRICULUM"):
        warmup = _env_nonnegative_int(
            "AZK_MAX_TICKS_CURRICULUM_WARMUP_EPISODES", default=0
        )
        ramp = _env_nonnegative_int(
            "AZK_MAX_TICKS_CURRICULUM_RAMP_EPISODES", default=3000
        )
        completions.append(int(warmup or 0) + int(ramp or 0))

    if _env_flag("AZK_REWARD_SHAPING_ANNEAL"):
        warmup = _env_nonnegative_int(
            "AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES", default=2000
        )
        ramp = _env_nonnegative_int(
            "AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES", default=30000
        )
        completions.append(int(warmup or 0) + int(ramp or 0))

    if not completions:
        return None
    return max(completions)


def _select_resume_completed_episodes(
    *,
    saved_completed_episodes: int | None,
    manual_completed_episodes: int | None,
    allow_schedule_rewind: bool,
) -> int | None:
    if saved_completed_episodes is None:
        return manual_completed_episodes
    if manual_completed_episodes is None:
        return saved_completed_episodes
    if allow_schedule_rewind or manual_completed_episodes >= saved_completed_episodes:
        return manual_completed_episodes
    return saved_completed_episodes


def _flatten_config_fingerprint(payload: dict[str, object]) -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flattened[f"{key}.{sub_key}"] = sub_value
        else:
            flattened[key] = value
    return flattened


def _resume_cfg_mismatches(
    saved_resume_cfg: dict[str, object], current_resume_cfg: dict[str, object]
) -> list[tuple[str, object, object]]:
    saved_flat = _flatten_config_fingerprint(saved_resume_cfg)
    current_flat = _flatten_config_fingerprint(current_resume_cfg)
    mismatches: list[tuple[str, object, object]] = []
    for key in sorted(set(saved_flat) & set(current_flat)):
        if saved_flat[key] != current_flat[key]:
            mismatches.append((key, saved_flat[key], current_flat[key]))
    return mismatches


def _peek_resume_env_completed_episodes(
    model_path: Path | None,
    trainer_state_path: Path | None,
    *,
    num_envs_hint: int,
) -> int | None:
    exact_candidates: list[tuple[str, int]] = []
    global_step_candidates: list[int] = []

    if trainer_state_path is not None and trainer_state_path.exists():
        try:
            trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
            if isinstance(trainer_state, dict):
                candidate = _coerce_nonnegative_int(trainer_state.get("env_completed_episodes"))
                if candidate is not None:
                    exact_candidates.append(("trainer_state", candidate))
                global_step_candidate = _coerce_nonnegative_int(
                    trainer_state.get("global_step", trainer_state.get("agent_step"))
                )
                if global_step_candidate is not None:
                    global_step_candidates.append(global_step_candidate)
        except Exception as exc:
            print(f"[resume] warning: failed reading trainer state for env progression restore: {exc}")

    if model_path is not None:
        metadata_path = _checkpoint_metadata_path(model_path)
        if metadata_path.exists():
            try:
                payload = json.loads(metadata_path.read_text())
                if isinstance(payload, dict):
                    candidate = _coerce_nonnegative_int(payload.get("env_completed_episodes"))
                    if candidate is not None:
                        exact_candidates.append(("checkpoint_metadata", candidate))
                    global_step_candidate = _coerce_nonnegative_int(payload.get("global_step"))
                    if global_step_candidate is not None:
                        global_step_candidates.append(global_step_candidate)
            except Exception as exc:
                print(f"[resume] warning: failed reading checkpoint metadata for env progression restore: {exc}")

    if exact_candidates:
        selected_source, selected_value = max(exact_candidates, key=lambda item: item[1])
        distinct_values = sorted({value for _, value in exact_candidates})
        if len(distinct_values) > 1:
            details = ", ".join(f"{source}={value}" for source, value in exact_candidates)
            print(
                "[resume] warning: env progression sidecars disagree; "
                f"using monotonic maximum {selected_source}={selected_value} ({details})"
            )
        return selected_value

    schedule_completion = _enabled_episode_schedule_completion()
    if model_path is not None and schedule_completion is not None:
        global_step_detail = (
            f", checkpoint_global_step={max(global_step_candidates)}"
            if global_step_candidates
            else ""
        )
        print(
            "[resume] legacy checkpoint has no exact env progression; "
            "pinning enabled episode schedules at their final phase to prevent replay: "
            f"completed_episodes={schedule_completion}{global_step_detail}"
        )
        return schedule_completion

    if global_step_candidates:
        global_step_candidate = max(global_step_candidates)
        inferred = _infer_completed_episodes_from_global_step(
            global_step_candidate, num_envs_hint=num_envs_hint
        )
        if inferred is not None:
            print(
                "[resume] inferred env progression from global_step for legacy checkpoint: "
                f"global_step={global_step_candidate}, num_envs={max(1, int(num_envs_hint))}, "
                f"inferred_completed_episodes={inferred}"
            )
            return inferred

    return None


def _peek_resume_global_step(
    model_path: Path | None,
    trainer_state_path: Path | None,
) -> int | None:
    if trainer_state_path is not None and trainer_state_path.exists():
        try:
            trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
            if isinstance(trainer_state, dict):
                candidate = _coerce_nonnegative_int(
                    trainer_state.get("global_step", trainer_state.get("agent_step"))
                )
                if candidate is not None:
                    return candidate
        except Exception as exc:
            print(f"[resume] warning: failed reading trainer state for global_step restore hint: {exc}")

    if model_path is not None:
        metadata_path = _checkpoint_metadata_path(model_path)
        if metadata_path.exists():
            try:
                payload = json.loads(metadata_path.read_text())
                if isinstance(payload, dict):
                    candidate = _coerce_nonnegative_int(payload.get("global_step"))
                    if candidate is not None:
                        return candidate
            except Exception as exc:
                print(f"[resume] warning: failed reading checkpoint metadata for global_step hint: {exc}")

    return None


def _load_saved_resume_config_fingerprint(
    model_path: Path | None, trainer_state_path: Path | None
) -> dict[str, object] | None:
    if trainer_state_path is not None and trainer_state_path.exists():
        try:
            trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
            if isinstance(trainer_state, dict):
                saved = trainer_state.get("resume_config_fingerprint")
                if isinstance(saved, dict):
                    return saved
        except Exception as exc:
            print(f"[resume] warning: failed reading trainer state resume fingerprint: {exc}")

    if model_path is not None:
        metadata_path = _checkpoint_metadata_path(model_path)
        if metadata_path.exists():
            try:
                payload = json.loads(metadata_path.read_text())
                if isinstance(payload, dict):
                    saved = payload.get("resume_config_fingerprint")
                    if isinstance(saved, dict):
                        return saved
            except Exception as exc:
                print(f"[resume] warning: failed reading checkpoint metadata resume fingerprint: {exc}")

    return None


def _apply_saved_env_group(
    saved_resume_config: dict[str, object] | None,
    *,
    fingerprint_key: str,
    names: tuple[str, ...],
    keep_current_env: str,
    label: str,
) -> None:
    if not isinstance(saved_resume_config, dict):
        return
    if _env_flag(keep_current_env):
        print(
            f"[resume] keeping current {label} env vars per "
            f"{keep_current_env}=1 (saved values not applied)"
        )
        return
    saved_env = saved_resume_config.get(fingerprint_key)
    if not isinstance(saved_env, dict):
        return

    changes: list[tuple[str, str, str]] = []
    for name in names:
        if name not in saved_env:
            continue
        saved_value_raw = saved_env.get(name)
        saved_value = saved_value_raw if isinstance(saved_value_raw, str) else str(saved_value_raw)
        current_value = os.getenv(name, "")
        if current_value == saved_value:
            continue
        if saved_value == "":
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved_value
        changes.append((name, current_value, saved_value))

    if changes:
        details = ", ".join(f"{name}: current={curr!r} -> saved={saved!r}" for name, curr, saved in changes)
        print(f"[resume] applied saved {label} env vars from checkpoint metadata: " + details)


def _apply_saved_schedule_env(saved_resume_config: dict[str, object] | None) -> None:
    _apply_saved_env_group(
        saved_resume_config,
        fingerprint_key="schedule_env",
        names=RESUME_SCHEDULE_ENV_VARS,
        keep_current_env="AZK_RESUME_KEEP_CURRENT_SCHEDULE_ENV",
        label="schedule",
    )


def _apply_saved_reward_env(saved_resume_config: dict[str, object] | None) -> None:
    _apply_saved_env_group(
        saved_resume_config,
        fingerprint_key="reward_env",
        names=RESUME_REWARD_ENV_VARS,
        keep_current_env="AZK_RESUME_KEEP_CURRENT_REWARD_ENV",
        label="reward",
    )


def _compute_runtime_fingerprint(vecenv) -> dict[str, object]:
    out: dict[str, object] = {}
    try:
        emulated = getattr(vecenv.driver_env, "emulated", {})
        obs_dtype = emulated.get("emulated_observation_dtype")
        if obs_dtype is not None:
            out["obs_dtype_descr"] = repr(obs_dtype.descr)
            out["obs_dtype_itemsize"] = int(obs_dtype.itemsize)
    except Exception:
        pass

    try:
        import binding  # type: ignore

        binding_path = Path(binding.__file__).resolve()
        stat = binding_path.stat()
        out["binding_path"] = str(binding_path)
        out["binding_mtime_ns"] = int(stat.st_mtime_ns)
        out["binding_size"] = int(stat.st_size)
    except Exception:
        pass
    return out


def _resume_config_fingerprint(trainer_args: dict) -> dict[str, object]:
    train_cfg = trainer_args.get("train")
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    env_cfg = trainer_args.get("env")
    if not isinstance(env_cfg, dict):
        env_cfg = {}
    policy_cfg = trainer_args.get("policy")
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}

    return {
        "use_rnn": bool(train_cfg.get("use_rnn", False)),
        "direct_parallel": bool(env_cfg.get("direct_parallel", False)),
        "deck_building_enabled": bool(env_cfg.get("deck_building_enabled", False)),
        "draft_uniform_assignment": bool(env_cfg.get("draft_uniform_assignment", False)),
        "deck_pool_path": str(resolve_training_deck_pool_path(env_cfg.get("deck_pool_path"))),
        "policy_model_version": str(policy_cfg.get("model_version", "metadata_v1")),
        "policy_actor_head_type": str(policy_cfg.get("actor_head_type", "legal_action_scorer")),
        "policy_legal_action_scorer_use_references": bool(
            policy_cfg.get("legal_action_scorer_use_references", True)
        ),
        "policy_critic_head_type": str(policy_cfg.get("critic_head_type", "full_lstm_mlp")),
        "policy_privileged_critic_enabled": bool(policy_cfg.get("privileged_critic_enabled", False)),
        "policy_privileged_critic_embed_dim": int(policy_cfg.get("privileged_critic_embed_dim", 64)),
        "policy_privileged_critic_deck_heads": int(policy_cfg.get("privileged_critic_deck_heads", 4)),
        "policy_privileged_critic_deck_layers": int(policy_cfg.get("privileged_critic_deck_layers", 2)),
        "policy_privileged_critic_deck_ff_size": int(policy_cfg.get("privileged_critic_deck_ff_size", 256)),
        "policy_privileged_critic_fusion_hidden_size": int(
            policy_cfg.get("privileged_critic_fusion_hidden_size", 512)
        ),
        "policy_privileged_critic_fusion_projection_size": int(
            policy_cfg.get("privileged_critic_fusion_projection_size", 128)
        ),
        "policy_privileged_critic_feature_scale": float(
            policy_cfg.get("privileged_critic_feature_scale", 1.0)
        ),
        "policy_win_prob_aux_enabled": bool(policy_cfg.get("win_prob_aux_enabled", False)),
        "policy_win_prob_aux_coef": float(policy_cfg.get("win_prob_aux_coef", 0.1)),
        "policy_split_value_heads_enabled": bool(policy_cfg.get("split_value_heads_enabled", False)),
        "policy_split_value_component_coef": float(policy_cfg.get("split_value_component_coef", 0.5)),
        "policy_gate_id_embedding_enabled": bool(policy_cfg.get("gate_id_embedding_enabled", False)),
        "schedule_env": _resume_env_var_fingerprint(RESUME_SCHEDULE_ENV_VARS),
        "reward_env": _resume_env_var_fingerprint(RESUME_REWARD_ENV_VARS),
        "source_hashes": _source_hash_fingerprint(),
    }


def _checkpoint_metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".meta.json")


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _default_torch_profile_dir() -> Path:
    return Path("experiments") / f"torch_profile_{int(time.time())}"


def _build_torch_profiler_config(script_args: argparse.Namespace) -> _TorchProfilerConfig | None:
    enabled = bool(script_args.torch_profile or script_args.torch_profile_dir is not None)
    if not enabled:
        return None
    if script_args.torch_profile_warmup_epochs < 0:
        raise ValueError("--torch-profile-warmup-epochs must be >= 0")
    if script_args.torch_profile_active_epochs <= 0:
        raise ValueError("--torch-profile-active-epochs must be > 0")
    if script_args.torch_profile_row_limit <= 0:
        raise ValueError("--torch-profile-row-limit must be > 0")
    return _TorchProfilerConfig(
        output_dir=script_args.torch_profile_dir or _default_torch_profile_dir(),
        warmup_epochs=int(script_args.torch_profile_warmup_epochs),
        active_epochs=int(script_args.torch_profile_active_epochs),
        row_limit=int(script_args.torch_profile_row_limit),
        record_shapes=bool(script_args.torch_profile_record_shapes),
        profile_memory=bool(script_args.torch_profile_memory),
        with_stack=bool(script_args.torch_profile_with_stack),
        export_trace=bool(script_args.torch_profile_export_trace),
        stop_after_capture=bool(script_args.torch_profile_stop_after_capture),
    )


def _profiler_event_payload(event) -> dict[str, object]:
    return {
        "key": str(getattr(event, "key", "")),
        "count": int(getattr(event, "count", 0)),
        "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0)),
        "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0)),
        "self_cuda_time_total_us": float(getattr(event, "self_cuda_time_total", 0.0)),
        "cuda_time_total_us": float(getattr(event, "cuda_time_total", 0.0)),
        "self_cpu_memory_usage": int(getattr(event, "self_cpu_memory_usage", 0)),
        "cpu_memory_usage": int(getattr(event, "cpu_memory_usage", 0)),
        "self_cuda_memory_usage": int(getattr(event, "self_cuda_memory_usage", 0)),
        "cuda_memory_usage": int(getattr(event, "cuda_memory_usage", 0)),
    }


def _top_profiler_events(rows: list[dict[str, object]], key: str, *, limit: int) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: float(row.get(key, 0.0)), reverse=True)[:limit]


def _numeric_last_stats_snapshot(trainer) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    last_stats = getattr(trainer, "last_stats", {}) or {}
    if not isinstance(last_stats, dict):
        return snapshot
    for key, value in last_stats.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            snapshot[str(key)] = float(value)
    return snapshot


def _write_torch_profiler_outputs(
    profiler,
    profile_cfg: _TorchProfilerConfig,
    *,
    trainer,
    config_path: Path,
) -> None:
    profile_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    averages = profiler.key_averages()
    rows = [_profiler_event_payload(event) for event in averages]

    cpu_table = averages.table(sort_by="self_cpu_time_total", row_limit=profile_cfg.row_limit)
    (profile_cfg.output_dir / "top_cpu_ops.txt").write_text(cpu_table)

    cuda_sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    cuda_table = averages.table(sort_by=cuda_sort_key, row_limit=profile_cfg.row_limit)
    (profile_cfg.output_dir / "top_cuda_ops.txt").write_text(cuda_table)

    summary = {
        "config": str(config_path.resolve()),
        "profile_output_dir": str(profile_cfg.output_dir.resolve()),
        "trace_dir": (
            str((profile_cfg.output_dir / "traces").resolve())
            if profile_cfg.export_trace
            else None
        ),
        "warmup_epochs": profile_cfg.warmup_epochs,
        "active_epochs": profile_cfg.active_epochs,
        "row_limit": profile_cfg.row_limit,
        "record_shapes": profile_cfg.record_shapes,
        "profile_memory": profile_cfg.profile_memory,
        "with_stack": profile_cfg.with_stack,
        "export_trace": profile_cfg.export_trace,
        "trainer_epoch": int(getattr(trainer, "epoch", 0)),
        "global_step": int(getattr(trainer, "global_step", 0)),
        "last_stats": _numeric_last_stats_snapshot(trainer),
        "top_ops_by_self_cpu_time_us": _top_profiler_events(
            rows, "self_cpu_time_total_us", limit=profile_cfg.row_limit
        ),
        "top_ops_by_self_cuda_time_us": _top_profiler_events(
            rows, "self_cuda_time_total_us", limit=profile_cfg.row_limit
        ),
    }
    _write_json_atomic(profile_cfg.output_dir / "summary.json", summary)


def _save_checkpoint_metadata(model_path: Path, payload: dict) -> None:
    _write_json_atomic(_checkpoint_metadata_path(model_path), payload)


def _trainer_shaped_reward_schedule_state(trainer) -> dict[str, object] | None:
    getter = getattr(trainer, "trainer_shaped_reward_schedule_state", None)
    if not callable(getter):
        return None
    state = getter()
    return state if isinstance(state, dict) else None


def _save_per_checkpoint_trainer_state(
    trainer,
    checkpoint_path: Path,
    runtime_fingerprint: dict[str, object],
    resume_config_fingerprint: dict[str, object],
    env_completed_episodes: int | None = None,
) -> Path:
    stem = checkpoint_path.stem
    suffix = stem.rsplit("_", 1)[-1] if "_" in stem else "latest"
    state_path = checkpoint_path.parent / f"trainer_state_{suffix}.pt"
    state = {
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "global_step": int(trainer.global_step),
        "agent_step": int(trainer.global_step),
        "update": int(trainer.epoch),
        "model_name": checkpoint_path.name,
        "run_id": getattr(trainer.logger, "run_id", ""),
        "runtime_fingerprint": runtime_fingerprint,
        "resume_config_fingerprint": resume_config_fingerprint,
    }
    shaped_reward_schedule = _trainer_shaped_reward_schedule_state(trainer)
    if shaped_reward_schedule is not None:
        state["trainer_shaped_reward_schedule"] = shaped_reward_schedule
    if env_completed_episodes is not None:
        state["env_completed_episodes"] = int(env_completed_episodes)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(state_path)
    return state_path


def _extract_checkpoint_state_dict(raw: object, *, source: str) -> dict[str, torch.Tensor]:
    state_dict = raw.get("state_dict") if isinstance(raw, dict) and "state_dict" in raw else raw
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format (expected state_dict dict): {source}")
    return _strip_module_prefix(state_dict)


def _state_dict_diff_summary(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]
) -> dict[str, object]:
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    shape_mismatches: list[str] = []
    max_abs_diff = 0.0
    max_abs_diff_key = ""

    for key in sorted(set(expected) & set(actual)):
        lhs = expected[key]
        rhs = actual[key]
        if not torch.is_tensor(lhs) or not torch.is_tensor(rhs):
            continue
        if lhs.shape != rhs.shape:
            shape_mismatches.append(key)
            continue

        lhs_cpu = lhs.detach().to(device="cpu")
        rhs_cpu = rhs.detach().to(device="cpu")
        if lhs_cpu.is_floating_point() or rhs_cpu.is_floating_point():
            diff = float((lhs_cpu.float() - rhs_cpu.float()).abs().max().item())
        elif lhs_cpu.dtype == torch.bool and rhs_cpu.dtype == torch.bool:
            diff = float((lhs_cpu != rhs_cpu).any().item())
        else:
            diff = float((lhs_cpu.long() - rhs_cpu.long()).abs().max().item())
        if diff > max_abs_diff:
            max_abs_diff = diff
            max_abs_diff_key = key

    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch_keys": shape_mismatches,
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_key": max_abs_diff_key,
    }


def _checkpoint_parity_guard(trainer, checkpoint_path: Path) -> dict[str, object]:
    live_state = _strip_module_prefix(trainer.policy.state_dict())
    loaded_raw = torch.load(checkpoint_path, map_location="cpu")
    loaded_state = _extract_checkpoint_state_dict(loaded_raw, source=str(checkpoint_path))
    diff = _state_dict_diff_summary(live_state, loaded_state)

    has_key_mismatch = bool(diff["missing_keys"] or diff["unexpected_keys"] or diff["shape_mismatch_keys"])
    has_value_mismatch = float(diff["max_abs_diff"]) > 0.0
    rewritten = False

    if has_key_mismatch or has_value_mismatch:
        rewritten = True
        print(
            "[checkpoint] parity mismatch detected; rewriting checkpoint from live policy: "
            f"path={checkpoint_path}, missing={len(diff['missing_keys'])}, "
            f"unexpected={len(diff['unexpected_keys'])}, "
            f"shape_mismatch={len(diff['shape_mismatch_keys'])}, "
            f"max_abs_diff={diff['max_abs_diff']:.6g}, max_abs_diff_key={diff['max_abs_diff_key'] or '<none>'}"
        )
        tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(trainer.policy.state_dict(), tmp)
        tmp.replace(checkpoint_path)

        reloaded_raw = torch.load(checkpoint_path, map_location="cpu")
        reloaded_state = _extract_checkpoint_state_dict(reloaded_raw, source=str(checkpoint_path))
        diff = _state_dict_diff_summary(live_state, reloaded_state)
        has_key_mismatch = bool(diff["missing_keys"] or diff["unexpected_keys"] or diff["shape_mismatch_keys"])
        has_value_mismatch = float(diff["max_abs_diff"]) > 0.0
        if has_key_mismatch or has_value_mismatch:
            raise RuntimeError(
                "[checkpoint] parity verification failed after rewrite: "
                f"path={checkpoint_path}, missing={len(diff['missing_keys'])}, "
                f"unexpected={len(diff['unexpected_keys'])}, "
                f"shape_mismatch={len(diff['shape_mismatch_keys'])}, "
                f"max_abs_diff={diff['max_abs_diff']:.6g}, max_abs_diff_key={diff['max_abs_diff_key'] or '<none>'}"
            )

    print(
        "[checkpoint] parity verified: "
        f"path={checkpoint_path}, rewritten={rewritten}, "
        f"max_abs_diff={diff['max_abs_diff']:.6g}, max_abs_diff_key={diff['max_abs_diff_key'] or '<none>'}"
    )
    return {
        "rewritten": rewritten,
        "missing_key_count": len(diff["missing_keys"]),
        "unexpected_key_count": len(diff["unexpected_keys"]),
        "shape_mismatch_key_count": len(diff["shape_mismatch_keys"]),
        "max_abs_diff": float(diff["max_abs_diff"]),
        "max_abs_diff_key": str(diff["max_abs_diff_key"]),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trusted_parent_source_drift(
    trainer_args: dict,
    model_path: Path,
    trainer_state_path: Path | None,
) -> bool:
    raw_manifest = trainer_args.get("parent_manifest")
    if not isinstance(raw_manifest, str) or not raw_manifest.strip():
        return False
    manifest_path = Path(raw_manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = Path(__file__).resolve().parents[2] / manifest_path
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError(f"Unsupported production parent manifest: {manifest_path}")
    model_entry = payload.get("model")
    if not isinstance(model_entry, dict):
        raise ValueError("Production parent manifest has no model entry")
    recorded_model_path = Path(str(model_entry.get("path", ""))).expanduser()
    if not recorded_model_path.is_absolute():
        recorded_model_path = Path(__file__).resolve().parents[2] / recorded_model_path
    if recorded_model_path.resolve() != model_path.resolve():
        return False

    expected = {
        "model": model_path,
        "metadata": _checkpoint_metadata_path(model_path),
    }
    if trainer_state_path is not None:
        expected["trainer"] = trainer_state_path
    for key, actual_path in expected.items():
        entry = payload.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"Production parent manifest has no {key} entry")
        recorded_path = Path(str(entry.get("path", ""))).expanduser()
        if not recorded_path.is_absolute():
            recorded_path = Path(__file__).resolve().parents[2] / recorded_path
        if recorded_path.resolve() != actual_path.resolve():
            raise ValueError(
                f"Production parent manifest {key} path mismatch: "
                f"recorded={recorded_path} requested={actual_path}"
            )
        recorded_hash = entry.get("sha256")
        if not isinstance(recorded_hash, str) or _file_sha256(actual_path) != recorded_hash:
            raise ValueError(f"Production parent manifest {key} hash mismatch: {actual_path}")
    print(
        "[resume] trusted atomic parent verified; source-only drift is allowed for "
        f"the initial migration: {manifest_path}"
    )
    return True

def _approved_resume_cfg_mismatch_prefixes(
    *,
    allow_trusted_source_drift: bool,
    allow_deck_pool_migration: bool,
) -> tuple[str, ...]:
    prefixes: list[str] = []
    if allow_trusted_source_drift or os.environ.get("AZK_RESUME_ALLOW_SOURCE_DRIFT") == "1":
        prefixes.append("source_hashes.")
    if _env_flag("AZK_RESUME_KEEP_CURRENT_SCHEDULE_ENV"):
        prefixes.append("schedule_env.")
    if _env_flag("AZK_RESUME_KEEP_CURRENT_REWARD_ENV"):
        prefixes.append("reward_env.")
    if allow_deck_pool_migration:
        prefixes.append("deck_pool_path")
    return tuple(prefixes)



def _validate_resume_metadata(
    model_path: Path,
    runtime_fingerprint: dict[str, object],
    resume_config_fingerprint: dict[str, object],
    allow_trusted_source_drift: bool = False,
    allow_deck_pool_migration: bool = False,
) -> None:
    metadata_path = _checkpoint_metadata_path(model_path)
    if not metadata_path.exists():
        binding_mtime = runtime_fingerprint.get("binding_mtime_ns")
        if isinstance(binding_mtime, int):
            model_mtime = int(model_path.stat().st_mtime_ns)
            if binding_mtime > model_mtime:
                print(
                    "[resume] warning: checkpoint has no runtime metadata and current binding is newer than checkpoint. "
                    "This can cause silent behavior drift across resumes."
                )
        return

    try:
        payload = json.loads(metadata_path.read_text())
    except Exception as exc:
        print(f"[resume] warning: failed to parse checkpoint metadata {metadata_path}: {exc}")
        return

    saved_fp = payload.get("runtime_fingerprint")
    if not isinstance(saved_fp, dict):
        return

    mismatches = []
    keys = ("obs_dtype_descr", "obs_dtype_itemsize", "binding_size")
    for key in keys:
        if key in saved_fp and key in runtime_fingerprint and saved_fp[key] != runtime_fingerprint[key]:
            mismatches.append((key, saved_fp[key], runtime_fingerprint[key]))

    if mismatches:
        details = ", ".join(f"{k}: saved={a!r} current={b!r}" for k, a, b in mismatches)
        binding_only = all(key == "binding_size" for key, _, _ in mismatches)
        if binding_only and os.environ.get("AZK_RESUME_ALLOW_BINDING_MISMATCH") == "1":
            # binding_size fires on ANY rebuild of the .so, semantic or not;
            # the obs dtype fields above remain strict.
            print(
                "[resume] warning: binding changed since checkpoint "
                f"({details}); continuing per AZK_RESUME_ALLOW_BINDING_MISMATCH=1"
            )
        else:
            raise RuntimeError(
                "[resume] runtime/checkpoint incompatibility detected. "
                f"Checkpoint={model_path}. Mismatched fields: {details}"
            )

    saved_resume_cfg = payload.get("resume_config_fingerprint")
    if isinstance(saved_resume_cfg, dict):
        cfg_mismatches = _resume_cfg_mismatches(saved_resume_cfg, resume_config_fingerprint)
        excusable_prefixes = _approved_resume_cfg_mismatch_prefixes(
            allow_trusted_source_drift=allow_trusted_source_drift,
            allow_deck_pool_migration=allow_deck_pool_migration,
        )
        if cfg_mismatches and excusable_prefixes:
            excused = [m for m in cfg_mismatches if m[0].startswith(tuple(excusable_prefixes))]
            cfg_mismatches = [m for m in cfg_mismatches if not m[0].startswith(tuple(excusable_prefixes))]
            if excused:
                details = ", ".join(f"{k}" for k, _, _ in excused)
                print(f"[resume] warning: approved mismatches excused: {details}")
        if cfg_mismatches:
            details = ", ".join(f"{k}: saved={a!r} current={b!r}" for k, a, b in cfg_mismatches)
            raise RuntimeError(
                "[resume] config/checkpoint incompatibility detected. "
                f"Checkpoint={model_path}. Mismatched fields: {details}. "
                "Resume with matching config or start a new run."
            )


def _load_model_weights(policy: torch.nn.Module, model_path: Path, *, device: str, strict: bool) -> None:
  raw = torch.load(model_path, map_location=device)
  state_dict = raw.get("state_dict") if isinstance(raw, dict) and "state_dict" in raw else raw
  if not isinstance(state_dict, dict):
    raise ValueError(f"Unsupported checkpoint format (expected state_dict dict): {model_path}")
  cleaned = _strip_module_prefix(state_dict)
  static_card_prefixes = (
    "policy.static_",
    "static_",
  )
  stale_static_keys = [
    key for key in cleaned.keys()
    if key.startswith(static_card_prefixes)
  ]
  for key in stale_static_keys:
    cleaned.pop(key, None)
  materialized = _materialize_scalar_norm_buffers_from_state_dict(policy, cleaned)
  scalar_norm_keys = sum(1 for key in cleaned.keys() if "scalar_normalizer._rms_" in key)
  missing, unexpected = policy.load_state_dict(cleaned, strict=strict)
  base_policy = getattr(policy, "policy", policy)
  invalidate_text_cache = getattr(base_policy, "_invalidate_text_feature_table", None)
  if callable(invalidate_text_cache):
    invalidate_text_cache()
  print(
    "[resume] loaded model checkpoint: "
    f"path={model_path}, strict={strict}, missing_keys={len(missing)}, "
    f"unexpected_keys={len(unexpected)}, "
    f"materialized_scalar_norm_features={materialized}, scalar_norm_state_keys={scalar_norm_keys}, "
    f"filtered_static_card_keys={len(stale_static_keys)}"
  )


def _maybe_reset_critic_head(policy: torch.nn.Module) -> bool:
    base_policy = getattr(policy, "policy", policy)
    reset_critic_head = getattr(base_policy, "reset_critic_head", None)
    if callable(reset_critic_head):
        reset_critic_head()
        return True
    value_head = getattr(base_policy, "value_fn", None)
    if value_head is None:
        return False
    weight = getattr(value_head, "weight", None)
    bias = getattr(value_head, "bias", None)
    if weight is None or bias is None:
        return False
    torch.nn.init.orthogonal_(weight, gain=1.0)
    torch.nn.init.constant_(bias, 0.0)
    return True


def _sync_optimizer_lr_from_scheduler(trainer) -> list[float]:
    scheduler_lrs = [float(v) for v in trainer.scheduler.get_last_lr()]
    if not scheduler_lrs:
        scheduler_lrs = [float(trainer.config.get("learning_rate", 0.0))]
    for idx, group in enumerate(trainer.optimizer.param_groups):
        group["lr"] = scheduler_lrs[min(idx, len(scheduler_lrs) - 1)]
    return [float(group.get("lr", 0.0)) for group in trainer.optimizer.param_groups]


def _optimizer_group_lrs(trainer) -> list[float]:
    return [float(group.get("lr", 0.0)) for group in trainer.optimizer.param_groups]


def _restart_lr_schedule_for_remaining_epochs(trainer) -> tuple[int, list[float]]:
    remaining_epochs = int(trainer.total_epochs) - int(trainer.epoch)
    if remaining_epochs < 1:
        raise ValueError("Cannot restart the LR schedule without remaining training epochs")
    peak_lr = float(trainer.config.get("learning_rate", 0.0))
    if peak_lr <= 0.0:
        raise ValueError("train.learning_rate must be > 0 for a resumed LR schedule")

    for group in trainer.optimizer.param_groups:
        group["lr"] = peak_lr
        group["initial_lr"] = peak_lr
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=remaining_epochs,
    )
    lrs = _optimizer_group_lrs(trainer)
    print(
        "[resume] restarted LR schedule: "
        f"remaining_epochs={remaining_epochs}, peak_lrs={lrs}, final_lr=0.0"
    )
    return remaining_epochs, lrs


def _maybe_restore_trainer_state(
    trainer,
    trainer_state_path: Path | None,
    *,
    expected_model_path: Path | None = None,
    expected_resume_config_fingerprint: dict[str, object] | None = None,
    allow_trusted_source_drift: bool = False,
    allow_deck_pool_migration: bool = False,
) -> bool:
    if trainer_state_path is None or not trainer_state_path.exists():
        return False

    # trainer_state.pt is produced locally by this trainer and includes optimizer
    # internals (e.g. defaultdict), which require full pickle loading.
    trainer_state = torch.load(
        trainer_state_path, map_location="cpu", weights_only=False
    )
    if not isinstance(trainer_state, dict):
        raise ValueError(f"Unsupported trainer_state format: {trainer_state_path}")

    if expected_model_path is not None:
        saved_model_name_raw = trainer_state.get("model_name")
        if isinstance(saved_model_name_raw, str) and saved_model_name_raw:
            saved_model_name = Path(saved_model_name_raw).name
            expected_model_name = expected_model_path.name
            if saved_model_name != expected_model_name:
                print(
                    "[resume] trainer_state/model mismatch; skipping optimizer+step restore: "
                    f"trainer_state_model={saved_model_name}, loaded_model={expected_model_name}. "
                    "Use a matching checkpoint+trainer_state pair when passing --resume-load-optimizer."
                )
                return False

    if expected_resume_config_fingerprint is not None:
        saved_resume_cfg = trainer_state.get("resume_config_fingerprint")
        if isinstance(saved_resume_cfg, dict):
            cfg_mismatches = _resume_cfg_mismatches(saved_resume_cfg, expected_resume_config_fingerprint)
            # Same excusals as _validate_resume_metadata: intentional source
            # patches / caller-pinned schedule env vars must not silently
            # downgrade to model-only resume (loses optimizer + global_step).
            excusable_prefixes = _approved_resume_cfg_mismatch_prefixes(
                allow_trusted_source_drift=allow_trusted_source_drift,
                allow_deck_pool_migration=allow_deck_pool_migration,
            )
            if cfg_mismatches and excusable_prefixes:
                excused = [m for m in cfg_mismatches if m[0].startswith(tuple(excusable_prefixes))]
                cfg_mismatches = [m for m in cfg_mismatches if not m[0].startswith(tuple(excusable_prefixes))]
                if excused:
                    details = ", ".join(k for k, _, _ in excused)
                    print(f"[resume] trainer_state approved mismatches excused: {details}")
            if cfg_mismatches:
                details = ", ".join(f"{k}: saved={a!r} current={b!r}" for k, a, b in cfg_mismatches)
                print(
                    "[resume] trainer_state/config mismatch; skipping optimizer+step restore: "
                    f"{details}. Resume model-only with matching config."
                )
                return False

    optimizer_state = trainer_state.get("optimizer_state_dict")
    optimizer_restored = False
    if optimizer_state is not None:
        try:
            trainer.optimizer.load_state_dict(optimizer_state)
            optimizer_restored = True
        except Exception as exc:
            print(
                "[resume] optimizer restore skipped due to incompatibility: "
                f"path={trainer_state_path}, error={type(exc).__name__}: {exc}"
            )

    scheduler_state = trainer_state.get("scheduler_state_dict")
    scheduler_restored = False
    if scheduler_state is not None:
        try:
            trainer.scheduler.load_state_dict(scheduler_state)
            scheduler_restored = True
        except Exception as exc:
            print(
                "[resume] scheduler restore skipped due to incompatibility: "
                f"path={trainer_state_path}, error={type(exc).__name__}: {exc}"
            )

    restored_global_step = int(trainer_state.get("global_step", trainer.global_step))
    restored_epoch = int(trainer_state.get("update", trainer.epoch))
    trainer.global_step = restored_global_step
    trainer.epoch = restored_epoch
    trainer.last_log_step = restored_global_step
    trainer.last_log_time = time.time()
    trainer.start_time = time.time()
    if not scheduler_restored:
        trainer.scheduler.last_epoch = max(restored_epoch - 1, -1)

    # Keep optimizer-restored LRs exact; only derive from scheduler when
    # optimizer state is unavailable.
    if optimizer_restored:
        synced_lrs = _optimizer_group_lrs(trainer)
        if hasattr(trainer.scheduler, "_last_lr"):
            trainer.scheduler._last_lr = list(synced_lrs)
    else:
        synced_lrs = _sync_optimizer_lr_from_scheduler(trainer)
    print(
        "[resume] restored trainer state: "
        f"path={trainer_state_path}, global_step={restored_global_step}, "
        f"epoch={restored_epoch}, optimizer_restored={optimizer_restored}, "
        f"scheduler_restored={scheduler_restored}, optimizer_lrs={synced_lrs}"
    )
    return True


def _rollout_health_snapshot(trainer) -> dict[str, float]:
    actions = getattr(trainer, "actions", None)
    if actions is None or not torch.is_tensor(actions) or actions.numel() == 0:
        return {}

    noop_selected_rate = float((actions[..., 0] == 0).float().mean().item())
    mb_obs = trainer.observations
    mb_actions = trainer.actions
    if not bool(trainer.config.get("use_rnn", False)):
        mb_obs = mb_obs.reshape(-1, *trainer.vecenv.single_observation_space.shape)

    amp_context = getattr(trainer, "amp_context", None)
    if amp_context is None:
        amp_context = contextlib.nullcontext()

    # Forward in slices: a full-batch pass materializes [batch, 1024, 64]
    # deck-candidate tensors (>5GB at batch 23040) and OOMs on resume.
    # Dim 0 is segments under use_rnn (xTT rows each), flat rows otherwise.
    bptt = max(1, int(trainer.config.get("bptt_horizon", 1)))
    chunk = 2048 if not bool(trainer.config.get("use_rnn", False)) else max(1, 2048 // bptt)
    entropy_sum = 0.0
    entropy_count = 0
    with torch.no_grad(), amp_context:
        for start in range(0, mb_obs.shape[0], chunk):
            obs_chunk = mb_obs[start : start + chunk]
            action_chunk = mb_actions[start : start + chunk]
            state = dict(action=action_chunk, lstm_h=None, lstm_c=None)
            logits, _ = trainer.policy(obs_chunk, state)
            _, _, entropy = azk_pytorch.sample_logits(
                logits, action=action_chunk.reshape(-1, action_chunk.shape[-1])
            )
            entropy_sum += float(entropy.sum().item())
            entropy_count += int(entropy.numel())

    return {
        "entropy": entropy_sum / max(entropy_count, 1),
        "noop_selected_rate": noop_selected_rate,
    }


def _resume_reset_start_probe(
    trainer_args: dict,
    policy: torch.nn.Module,
    *,
    seed: int = 0,
    max_steps: int = 256,
) -> dict[str, float]:
    """Probe checkpoint behavior from fresh resets to separate load bugs from policy quality."""
    probe_vecenv = None
    was_training = bool(policy.training)
    train_cfg = trainer_args.get("train")
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    device = str(train_cfg.get("device", "cpu"))
    use_rnn = bool(train_cfg.get("use_rnn", False))
    precision = str(train_cfg.get("precision", "float32"))
    amp_context = contextlib.nullcontext()
    if device.startswith("cuda") and precision in ("float16", "bfloat16"):
        amp_context = torch.amp.autocast(device_type="cuda", dtype=getattr(torch, precision))

    try:
        policy.eval()
        probe_vecenv = build_vecenv(
            trainer_args,
            backend=azk_vector.Serial,
            num_envs=1,
            seed=seed,
        )

        lstm_h = None
        lstm_c = None
        if use_rnn:
            lstm_h = torch.zeros(probe_vecenv.num_agents, policy.hidden_size, device=device)
            lstm_c = torch.zeros(probe_vecenv.num_agents, policy.hidden_size, device=device)

        probe_vecenv.async_reset(seed=seed)
        obs, _, _, _, _, _, masks = probe_vecenv.recv()

        done = False
        steps = 0
        total_actions = 0
        noop_actions = 0
        entropy_total = 0.0
        entropy_batches = 0

        while not done and steps < max_steps:
            obs_t = torch.as_tensor(obs, device=device)
            state = {"mask": torch.as_tensor(masks, device=device)}
            if use_rnn:
                state["lstm_h"] = lstm_h
                state["lstm_c"] = lstm_c

            with torch.no_grad(), amp_context:
                logits, _ = policy.forward_eval(obs_t, state)
                action, _, entropy = azk_pytorch.sample_logits(logits)

            if use_rnn:
                lstm_h = state["lstm_h"]
                lstm_c = state["lstm_c"]

            action_np = action.detach().cpu().numpy()
            primary = action_np[:, 0] if action_np.ndim >= 2 else action_np
            noop_actions += int((primary == 0).sum())
            total_actions += int(primary.size)
            entropy_total += float(entropy.mean().item())
            entropy_batches += 1

            probe_vecenv.send(action_np)
            obs, _, _, _, _, _, masks = probe_vecenv.recv()
            done = bool(getattr(probe_vecenv.envs[0], "done", False))
            steps += 1

        return {
            "entropy": (entropy_total / entropy_batches) if entropy_batches > 0 else 0.0,
            "noop_selected_rate": (noop_actions / total_actions) if total_actions > 0 else 0.0,
            "steps": float(steps),
            "actions": float(total_actions),
        }
    finally:
        if probe_vecenv is not None:
            probe_vecenv.close()
        if was_training:
            policy.train()


def _collect_league_checkpoint_paths(script_args: argparse.Namespace, trainer_args: dict) -> list[Path]:
    league_cfg = trainer_args.get("league")
    if not isinstance(league_cfg, dict):
        league_cfg = {}

    paths: list[Path] = []
    if script_args.league_opponent_dir is not None:
        paths.extend(sorted(Path(script_args.league_opponent_dir).glob("model_*.pt")))
    elif isinstance(league_cfg.get("opponent_dir"), str) and league_cfg.get("opponent_dir"):
        paths.extend(sorted(Path(league_cfg["opponent_dir"]).glob("model_*.pt")))

    raw_list = script_args.league_opponent_checkpoints.strip()
    if not raw_list and isinstance(league_cfg.get("opponent_checkpoints"), str):
        raw_list = league_cfg["opponent_checkpoints"].strip()
    if raw_list:
        for item in raw_list.split(","):
            stripped = item.strip()
            if not stripped:
                continue
            candidate = Path(stripped)
            if candidate:
                paths.append(candidate)

    deduped = []
    seen = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _is_league_enabled(script_args: argparse.Namespace, trainer_args: dict) -> bool:
    if script_args.league_enable is not None:
        return bool(script_args.league_enable)
    league_cfg = trainer_args.get("league")
    if isinstance(league_cfg, dict):
        return bool(league_cfg.get("enable", False))
    return False


def _league_cfg(script_args: argparse.Namespace, trainer_args: dict) -> LeagueConfig:
    league_cfg = trainer_args.get("league")
    if not isinstance(league_cfg, dict):
        league_cfg = {}

    frozen_ratio = script_args.league_frozen_ratio
    if frozen_ratio is None:
        raw_frozen_ratio = league_cfg.get("frozen_ratio")
        if raw_frozen_ratio is not None:
            frozen_ratio = float(raw_frozen_ratio)
    if frozen_ratio is not None:
        frozen_ratio = max(0.0, min(0.5, float(frozen_ratio)))

    latest_ratio = script_args.league_latest_ratio
    if latest_ratio is None and frozen_ratio is None:
        latest_ratio = float(league_cfg.get("latest_ratio", 0.85))
    if latest_ratio is not None:
        latest_ratio = max(0.0, min(1.0, float(latest_ratio)))

    randomize_learner_seat = script_args.league_randomize_learner_seat
    if randomize_learner_seat is None:
        randomize_learner_seat = bool(league_cfg.get("randomize_learner_seat", True))

    activate_after_steps = script_args.league_activate_after_steps
    if activate_after_steps is None:
        activate_after_steps = int(league_cfg.get("activate_after_steps", 0))
    activate_after_steps = int(max(0, activate_after_steps))

    train_cfg = trainer_args.get("train", {})
    seed = int(train_cfg.get("seed", 0)) if isinstance(train_cfg, dict) else 0

    return LeagueConfig(
        enabled=True,
        frozen_ratio=frozen_ratio,
        latest_ratio=latest_ratio,
        randomize_learner_seat=bool(randomize_learner_seat),
        seed=seed,
        activate_after_steps=activate_after_steps,
        frozen_window_epochs=int(league_cfg.get("frozen_window_epochs", 0) or 0),
        max_distinct_frozen=int(league_cfg.get("max_distinct_frozen", 1) or 1),
    )


def run_training(script_args: argparse.Namespace, forwarded_cli):
    config_path = script_args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    torch_profile_cfg = _build_torch_profiler_config(script_args)
    trainer_args = load_training_config(config_path, forwarded_cli)
    process_seed = _seed_training_process(trainer_args["train"])
    if process_seed is not None:
        print(f"[reproducibility] seeded Python, NumPy, and Torch RNGs with {process_seed}")
    applied_process_env = apply_process_environment(trainer_args)
    if applied_process_env:
        print(
            "[config] applied process environment: "
            + ", ".join(sorted(applied_process_env))
        )
    resume_cfg = trainer_args.get("resume")
    if not isinstance(resume_cfg, dict):
        resume_cfg = {}
    if not script_args.resume_load_optimizer:
        script_args.resume_load_optimizer = bool(resume_cfg.get("load_optimizer", False))
    if not script_args.resume_restart_lr_schedule:
        script_args.resume_restart_lr_schedule = bool(
            resume_cfg.get("restart_lr_schedule", False)
        )
    if not script_args.resume_strict:
        script_args.resume_strict = bool(resume_cfg.get("strict", False))
    if "auto_reset_critic" in resume_cfg:
        script_args.resume_auto_reset_critic = bool(resume_cfg["auto_reset_critic"])

    checkpoint_policy = CheckpointPolicy.from_config(trainer_args)
    if checkpoint_policy is not None:
        trainer_args["train"]["checkpoint_interval"] = checkpoint_policy.scheduler_interval
        print(
            "[artifacts] tiered checkpoints: "
            f"scheduler={checkpoint_policy.scheduler_interval}, "
            f"recovery={checkpoint_policy.recovery_interval}, "
            f"evaluation={checkpoint_policy.evaluation_interval}, "
            f"milestone={checkpoint_policy.milestone_interval}"
        )
    logging_cfg = trainer_args.get("logging")
    if not isinstance(logging_cfg, dict):
        logging_cfg = {}
    stdout_interval = max(1, int(logging_cfg.get("stdout_interval_updates", 1)))
    stdout_patterns = parse_patterns(logging_cfg.get("stdout_metric_patterns"))
    dashboard_enabled = bool(logging_cfg.get("dashboard", True))
    trainer_args["train"]["env"] = trainer_args.get("env_name", "azuki_local")
    logger = None

    env_contract = trainer_args.get("env", {})
    if isinstance(env_contract, dict) and bool(
        env_contract.get("deck_building_enabled", False)
    ):
        uniform_assignment = bool(
            env_contract.get("draft_uniform_assignment", False)
        )
        print(
            "[draft] lifecycle: "
            f"uniform_assignment={uniform_assignment} "
            f"policy_picks_per_seat={50 if uniform_assignment else 51}"
        )

    install_tcg_sampler()

    static_policy_cfg = trainer_args.get("policy")
    if isinstance(static_policy_cfg, dict):
        static_pick_eps = _coerce_float(static_policy_cfg.get("deck_pick_smoothing_eps"))
        static_row_temp = _coerce_float(static_policy_cfg.get("legal_row_temperature"))
        if static_pick_eps is not None or static_row_temp is not None:
            tcg_sampler.set_sampling_params(
                deck_pick_smoothing_eps=static_pick_eps,
                legal_row_temperature=static_row_temp,
            )
            print(
                "[sampler] legal-row overrides: "
                f"deck_pick_smoothing_eps={static_pick_eps}, legal_row_temperature={static_row_temp}"
            )

    resume_checkpoint = script_args.resume_checkpoint
    if resume_checkpoint is None:
        configured_resume = resume_cfg.get("checkpoint")
        if isinstance(configured_resume, str) and configured_resume:
            resume_checkpoint = Path(configured_resume)
        else:
            fallback_resume = trainer_args.get("load_model_path")
            if isinstance(fallback_resume, str) and fallback_resume and fallback_resume != "latest":
                resume_checkpoint = Path(fallback_resume)

    model_resume_path: Path | None = None
    trainer_state_path: Path | None = None
    allow_trusted_source_drift = False
    if resume_checkpoint is not None:
        model_resume_path, trainer_state_path = _resolve_resume_artifacts(resume_checkpoint)
        saved_resume_config = _load_saved_resume_config_fingerprint(
            model_resume_path, trainer_state_path
        )
        _apply_saved_schedule_env(saved_resume_config)
        _apply_saved_reward_env(saved_resume_config)
        allow_trusted_source_drift = _trusted_parent_source_drift(
            trainer_args,
            model_resume_path,
            trainer_state_path if script_args.resume_load_optimizer else None,
        )

    vec_cfg = trainer_args.get("vec")
    num_envs_hint = 1
    if isinstance(vec_cfg, dict):
        parsed_num_envs = _coerce_nonnegative_int(vec_cfg.get("num_envs"))
        if parsed_num_envs is not None and parsed_num_envs > 0:
            num_envs_hint = parsed_num_envs

    resume_env_completed_episodes: int | None = None
    if model_resume_path is not None:
        saved_resume_completed = _peek_resume_env_completed_episodes(
            model_resume_path,
            trainer_state_path,
            num_envs_hint=num_envs_hint,
        )
        manual_resume_completed = _coerce_nonnegative_int(
            os.getenv(RESUME_COMPLETED_EPISODES_ENV)
        )
        allow_schedule_rewind = _env_flag(RESUME_ALLOW_SCHEDULE_REWIND_ENV)
        resume_env_completed_episodes = _select_resume_completed_episodes(
            saved_completed_episodes=saved_resume_completed,
            manual_completed_episodes=manual_resume_completed,
            allow_schedule_rewind=allow_schedule_rewind,
        )
        if manual_resume_completed is not None:
            if (
                saved_resume_completed is not None
                and manual_resume_completed < saved_resume_completed
            ):
                if allow_schedule_rewind:
                    print(
                        "[resume] warning: caller explicitly rewound env progression per "
                        f"{RESUME_ALLOW_SCHEDULE_REWIND_ENV}=1: "
                        f"saved={saved_resume_completed}, requested={manual_resume_completed}"
                    )
                else:
                    print(
                        "[resume] warning: ignoring caller env progression below saved state: "
                        f"saved={saved_resume_completed}, requested={manual_resume_completed}. "
                        f"Set {RESUME_ALLOW_SCHEDULE_REWIND_ENV}=1 only for an intentional rewind."
                    )
            else:
                print(
                    "[resume] using caller-provided env progression override: "
                    f"{RESUME_COMPLETED_EPISODES_ENV}={manual_resume_completed}"
                )
    if resume_env_completed_episodes is not None:
        os.environ[RESUME_COMPLETED_EPISODES_ENV] = str(resume_env_completed_episodes)
        print(
            "[resume] restoring env progression: "
            f"{RESUME_COMPLETED_EPISODES_ENV}={resume_env_completed_episodes}"
        )
    else:
        os.environ.pop(RESUME_COMPLETED_EPISODES_ENV, None)
        if model_resume_path is not None:
            print(
                "[resume] warning: no saved env progression found in trainer state/metadata. "
                "No episode-driven schedule is enabled; the diagnostic episode counter will restart from zero."
            )

    vecenv = build_vecenv(trainer_args)
    runtime_fingerprint = _compute_runtime_fingerprint(vecenv)
    resume_config_fingerprint = _resume_config_fingerprint(trainer_args)
    train_cfg = trainer_args["train"]

    configured_batch_size = train_cfg.get("batch_size", "auto")
    configured_bptt = train_cfg.get("bptt_horizon", "auto")
    if configured_batch_size == "auto":
        if configured_bptt == "auto":
            raise ValueError("Both train.batch_size and train.bptt_horizon are auto; cannot infer epoch size")
        effective_batch_size = int(vecenv.num_agents * int(configured_bptt))
    else:
        effective_batch_size = int(configured_batch_size)

    requested_total_timesteps = int(train_cfg.get("total_timesteps", 0))
    align_total_timesteps_up = bool(train_cfg.pop("align_total_timesteps_up", False))
    remainder = requested_total_timesteps % effective_batch_size if effective_batch_size > 0 else 0
    if effective_batch_size <= 0:
        raise ValueError(f"Invalid effective batch size {effective_batch_size}")
    if remainder != 0:
        rounded_down = requested_total_timesteps - remainder
        rounded_up = rounded_down + effective_batch_size
        if align_total_timesteps_up:
            train_cfg["total_timesteps"] = rounded_up
            print(
                "[train] aligning total_timesteps up to epoch boundary: "
                f"requested={requested_total_timesteps}, epoch_batch={effective_batch_size}, aligned={rounded_up}"
            )
        else:
            print(
                "[train] total_timesteps is not divisible by epoch batch and will stop early on epoch boundary: "
                f"requested={requested_total_timesteps}, epoch_batch={effective_batch_size}, "
                f"effective={rounded_down}, dropped={requested_total_timesteps - rounded_down}. "
                "Set train.align_total_timesteps_up=True to round up instead."
            )

    pretrainer_total_timesteps = int(train_cfg.get("total_timesteps", requested_total_timesteps))
    sampler_anneal_config = _build_sampler_anneal_config(
        trainer_args,
        total_timesteps=pretrainer_total_timesteps,
    )
    resume_global_step_hint = _peek_resume_global_step(model_resume_path, trainer_state_path)
    sampler_hint_step = 0 if resume_global_step_hint is None else int(resume_global_step_hint)
    initial_temp, initial_smoothing = _apply_sampler_anneal(
        sampler_anneal_config,
        global_step=sampler_hint_step,
    )
    print(
        "[sampler] schedule configured: "
        f"subaction_temperature={sampler_anneal_config.subaction_temperature.initial:.6f}"
        f"->{sampler_anneal_config.subaction_temperature.final:.6f} "
        f"(steps {sampler_anneal_config.subaction_temperature.start_step}"
        f"->{sampler_anneal_config.subaction_temperature.end_step}), "
        f"smoothing_eps={sampler_anneal_config.smoothing_eps.initial:.6f}"
        f"->{sampler_anneal_config.smoothing_eps.final:.6f} "
        f"(steps {sampler_anneal_config.smoothing_eps.start_step}"
        f"->{sampler_anneal_config.smoothing_eps.end_step}), "
        f"applied_step={sampler_hint_step}, "
        f"subaction_temperature_now={initial_temp:.6f}, smoothing_eps_now={initial_smoothing:.6f}"
    )

    if trainer_args.get("wandb"):
        logger = pufferl.WandbLogger(trainer_args)
    elif trainer_args.get("neptune"):
        logger = pufferl.NeptuneLogger(trainer_args)
    elif trainer_args.get("jsonl_log"):
        logger = _JsonlLogger(trainer_args, Path(str(trainer_args["jsonl_log"])))

    policy = build_policy(vecenv, trainer_args)
    if model_resume_path is not None:
        _validate_resume_metadata(
            model_resume_path,
            runtime_fingerprint,
            resume_config_fingerprint,
            allow_trusted_source_drift=allow_trusted_source_drift,
            allow_deck_pool_migration=bool(
                script_args.resume_allow_deck_pool_migration
            ),
        )
        _load_model_weights(
            policy,
            model_resume_path,
            device=str(train_cfg.get("device", "cpu")),
            strict=bool(script_args.resume_strict),
        )
        should_reset_critic = bool(script_args.resume_reset_critic) or (
            bool(script_args.resume_auto_reset_critic) and not bool(script_args.resume_load_optimizer)
        )
        if should_reset_critic:
            if _maybe_reset_critic_head(policy):
                reason = "manual flag" if script_args.resume_reset_critic else "auto default"
                print(f"[resume] critic head reset after checkpoint load ({reason})")
            else:
                print("[resume] critic head reset requested but no value head was found")

        env_cfg_probe = trainer_args.get("env", {})
        if bool(env_cfg_probe.get("native")) and bool(env_cfg_probe.get("deck_building_enabled")):
            # The probe's mask wiring doesn't cover the native draft phase; a
            # sampled NOOP on a draft step hits the C-side abort and kills the
            # resume (2026-07-07). Skip until the probe feeds draft masks.
            print("[resume] reset-start probe skipped (native deck-building env)")
        else:
            reset_probe = _resume_reset_start_probe(trainer_args, policy)
            print(
                "[resume] reset-start probe: "
                f"entropy={reset_probe['entropy']:.6f}, "
                f"noop_selected_rate={reset_probe['noop_selected_rate']:.6f}, "
                f"actions={int(reset_probe['actions'])}, steps={int(reset_probe['steps'])}"
            )
            if reset_probe["entropy"] < 0.3 and reset_probe["noop_selected_rate"] > 0.9:
                print(
                    "[resume] warning: checkpoint policy is collapse-like from fresh reset starts. "
                    "Model weight load parity passed; this usually indicates checkpoint behavior degradation "
                    "on opening states rather than a missing-weights resume bug."
                )

    league_enabled = _is_league_enabled(script_args, trainer_args)
    league_cfg = _league_cfg(script_args, trainer_args)
    league_manager = LeagueManager(parse_league_manager_config(trainer_args)) if league_enabled else None
    opponent_policies = []
    opponent_paths: list[Path] = []
    if league_enabled:
        opponent_paths = _collect_league_checkpoint_paths(script_args, trainer_args)
        if league_manager is not None:
            league_manager.ensure_seed_policies(opponent_paths, created_epoch=0)
            opponent_paths = [
                Path(entry.checkpoint_path) for entry in league_manager.opponent_entries_for_training()
            ]
        missing = [path for path in opponent_paths if not path.exists()]
        if missing:
            missing_str = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"League opponent checkpoints not found: {missing_str}")
        for checkpoint_path in opponent_paths:
            opponent_policy = build_policy(vecenv, trainer_args)
            _load_model_weights(
                opponent_policy,
                checkpoint_path,
                device=str(train_cfg.get("device", "cpu")),
                strict=False,
            )
            opponent_policy.eval()
            opponent_policies.append(opponent_policy)
        agents_per_match = int(
            getattr(
                vecenv.driver_env,
                "agents_per_match",
                vecenv.driver_env.num_agents,
            )
        )
        print(
            "[league] enabled: "
            f"opponents={len(opponent_policies)}, "
            f"frozen_ratio_target="
            f"{(0.0 if league_cfg.frozen_ratio is None else league_cfg.frozen_ratio):.3f}, "
            f"frozen_matchup_ratio="
            f"{(compute_frozen_matchup_ratio(frozen_row_ratio=league_cfg.frozen_ratio, agents_per_env=agents_per_match) if league_cfg.frozen_ratio is not None else max(0.0, min(1.0, 1.0 - float(league_cfg.latest_ratio)))):.3f}, "
            f"activate_after_steps={league_cfg.activate_after_steps}"
        )

    if league_enabled:
        trainer = LeaguePuffeRL(
            trainer_args["train"],
            vecenv,
            policy,
            opponent_policies=opponent_policies,
            opponent_keys=[str(path.resolve()) for path in opponent_paths],
            league_cfg=league_cfg,
            logger=logger,
        )
    else:
        trainer = pufferl.PuffeRL(trainer_args["train"], vecenv, policy, logger=logger)
    if not dashboard_enabled:
        trainer.print_dashboard = lambda *args, **kwargs: None


    if torch_profile_cfg is not None and trainer.total_epochs < torch_profile_cfg.total_profile_epochs:
        raise ValueError(
            "Profiler capture window exceeds available epochs: "
            f"requested={torch_profile_cfg.total_profile_epochs}, available={trainer.total_epochs}. "
            "Increase train.total_timesteps or reduce --torch-profile-warmup-epochs / --torch-profile-active-epochs."
        )

    resume_completed_episode_tracker = _coerce_nonnegative_int(resume_env_completed_episodes)
    original_save_checkpoint = trainer.save_checkpoint
    checkpoint_cache: dict[int, Path] = {}
    force_checkpoint = False
    finalized_checkpoints: set[int] = set()

    def _save_checkpoint_with_metadata():
        nonlocal force_checkpoint, resume_completed_episode_tracker
        update = int(trainer.epoch)
        done_training = bool(
            force_checkpoint
            or trainer.global_step >= trainer.config["total_timesteps"]
            or trainer.epoch >= trainer.total_epochs
        )
        if checkpoint_policy is not None and not checkpoint_policy.due(
            update, done=done_training
        ):
            return None
        cached = checkpoint_cache.get(update)
        if cached is not None:
            return str(cached)

        resume_completed_episode_tracker = _update_completed_episode_tracker(
            resume_completed_episode_tracker, getattr(trainer, "last_stats", None)
        )
        resume_completed_episode_tracker = _update_completed_episode_tracker(
            resume_completed_episode_tracker, getattr(trainer, "stats", None)
        )
        checkpoint_raw = original_save_checkpoint()
        if not checkpoint_raw:
            return checkpoint_raw
        checkpoint_path = Path(checkpoint_raw)
        parity_summary = _checkpoint_parity_guard(trainer, checkpoint_path)
        state_path = _save_per_checkpoint_trainer_state(
            trainer,
            checkpoint_path,
            runtime_fingerprint,
            resume_config_fingerprint,
            env_completed_episodes=resume_completed_episode_tracker,
        )
        metadata_payload = {
            "model_name": checkpoint_path.name,
            "global_step": int(trainer.global_step),
            "update": update,
            "trainer_state_path": str(state_path.name),
            "runtime_fingerprint": runtime_fingerprint,
            "resume_config_fingerprint": resume_config_fingerprint,
            "checkpoint_parity": parity_summary,
        }
        shaped_reward_schedule = _trainer_shaped_reward_schedule_state(trainer)
        if shaped_reward_schedule is not None:
            metadata_payload["trainer_shaped_reward_schedule"] = shaped_reward_schedule
        if resume_completed_episode_tracker is not None:
            metadata_payload["env_completed_episodes"] = int(resume_completed_episode_tracker)
        _save_checkpoint_metadata(checkpoint_path, metadata_payload)
        checkpoint_cache[update] = checkpoint_path
        return str(checkpoint_path)

    def _finalize_checkpoint(checkpoint_path: Path) -> None:
        update = int(trainer.epoch)
        if update in finalized_checkpoints:
            return
        league_state_path = league_manager.state_path if league_manager is not None else None
        promotion_state_path = (
            league_manager.archive_state_path if league_manager is not None else None
        )
        roles = checkpoint_policy.roles(update) if checkpoint_policy is not None else ()
        finalize_checkpoint_state(
            checkpoint_path,
            league_state_path=league_state_path,
            promotion_state_path=promotion_state_path,
            config_path=config_path,
            roles=roles,
        )
        finalized_checkpoints.add(update)
        if checkpoint_policy is not None:
            protected = (
                active_league_updates(
                    league_manager.state_path,
                    checkpoint_dir=checkpoint_path.parent,
                )
                if league_manager is not None
                else set()
            )
            removed = prune_checkpoints(
                checkpoint_path.parent,
                checkpoint_policy,
                protected_updates=protected,
            )
            if removed:
                print(f"[artifacts] pruned {len(removed)} obsolete checkpoint files")

    trainer.save_checkpoint = _save_checkpoint_with_metadata

    trainer_state_restored = False
    if script_args.resume_load_optimizer:
        if trainer_state_path is not None:
            trainer_state_restored = _maybe_restore_trainer_state(
                trainer,
                trainer_state_path,
                expected_model_path=model_resume_path,
                expected_resume_config_fingerprint=resume_config_fingerprint,
                allow_trusted_source_drift=allow_trusted_source_drift,
                allow_deck_pool_migration=bool(
                    script_args.resume_allow_deck_pool_migration
                ),
            )
        else:
            print("[resume] --resume-load-optimizer requested but trainer_state.pt was not found")
    if script_args.resume_restart_lr_schedule:
        if not script_args.resume_load_optimizer or not trainer_state_restored:
            raise RuntimeError(
                "--resume-restart-lr-schedule requires a successful optimizer/trainer-state restore"
            )
        _restart_lr_schedule_for_remaining_epochs(trainer)

    anneal_step_offset = 0
    if resume_global_step_hint is not None:
        hint_step = int(resume_global_step_hint)
        current_step = int(trainer.global_step)
        if hint_step > current_step:
            anneal_step_offset = hint_step - current_step
            print(
                "[resume] applying anneal step offset from checkpoint progress: "
                f"current_global_step={current_step}, checkpoint_global_step={hint_step}, "
                f"anneal_offset={anneal_step_offset}"
            )

    if league_enabled and league_manager is not None:
        learner_id = f"learner_{trainer.logger.run_id}"
        league_manager.attach_learner_identity(learner_id)
        league_active = compute_league_active(
            global_step=int(trainer.global_step),
            activate_after_steps=int(league_cfg.activate_after_steps),
        )
        trainer.stats["league/active"].append(1.0 if league_active else 0.0)
        if league_active:
            pool_metrics = league_manager.pool_metrics()
            for key, value in pool_metrics.items():
                trainer.stats[key].append(float(value))

    effective_total_timesteps = int(trainer.total_epochs * trainer.config["batch_size"])
    sampler_anneal_config = _build_sampler_anneal_config(
        trainer_args,
        total_timesteps=effective_total_timesteps,
    )
    ent_coef_anneal_config = _build_ent_coef_anneal_config(
        trainer_args,
        total_timesteps=effective_total_timesteps,
    )
    initial_anneal_step = int(trainer.global_step) + int(anneal_step_offset)
    current_temp, current_smoothing = _apply_sampler_anneal(
        sampler_anneal_config,
        global_step=initial_anneal_step,
    )
    current_ent_coef = _apply_ent_coef_anneal(
        trainer,
        ent_coef_anneal_config,
        global_step=initial_anneal_step,
    )
    print(
      "[train] epoch plan: "
      f"batch_size={trainer.config['batch_size']}, total_epochs={trainer.total_epochs}, "
      f"effective_total_timesteps={effective_total_timesteps}, "
      f"sampler_temp={current_temp:.6f}, sampler_smoothing={current_smoothing:.6f}, "
      f"ent_coef={current_ent_coef:.6f}"
    )
    print(
        "[anneal] ent_coef schedule: "
        f"{ent_coef_anneal_config.ent_coef.initial:.6f}->{ent_coef_anneal_config.ent_coef.final:.6f} "
        f"(steps {ent_coef_anneal_config.ent_coef.start_step}->{ent_coef_anneal_config.ent_coef.end_step}), "
        f"applied_step={initial_anneal_step}"
    )

    is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    torch_profiler = None
    torch_record_function = None
    torch_profile_context = contextlib.nullcontext()
    profiled_epochs = 0
    if torch_profile_cfg is not None:
        from torch.profiler import ProfilerActivity, profile as torch_profile, record_function, schedule, tensorboard_trace_handler

        torch_profile_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        trace_handler = None
        if torch_profile_cfg.export_trace:
            trace_dir = torch_profile_cfg.output_dir / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_handler = tensorboard_trace_handler(str(trace_dir))
        activities = [ProfilerActivity.CPU]
        if str(train_cfg.get("device", "cpu")) == "cuda" and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        torch_record_function = record_function
        torch_profile_context = torch_profile(
            activities=activities,
            schedule=schedule(wait=0, warmup=torch_profile_cfg.warmup_epochs, active=torch_profile_cfg.active_epochs, repeat=1),
            on_trace_ready=trace_handler,
            record_shapes=torch_profile_cfg.record_shapes,
            profile_memory=torch_profile_cfg.profile_memory,
            with_stack=torch_profile_cfg.with_stack,
        )
        print(
            "[profile] torch profiler enabled: "
            f"warmup_epochs={torch_profile_cfg.warmup_epochs}, "
            f"active_epochs={torch_profile_cfg.active_epochs}, "
            f"output_dir={torch_profile_cfg.output_dir}"
        )

    def maybe_record(name: str):
        if torch_record_function is None:
            return contextlib.nullcontext()
        return torch_record_function(name)

    def maybe_run_playback(reason: str, checkpoint_path: Path):
        if not is_main_process:
            return
        output_path = None
        cast_output_path = None
        if script_args.render_playback_dir:
            script_args.render_playback_dir.mkdir(parents=True, exist_ok=True)
            output_path = script_args.render_playback_dir / f"{checkpoint_path.stem}_{reason}.ansi"
            if script_args.render_playback_cast:
                cast_output_path = script_args.render_playback_dir / f"{checkpoint_path.stem}_{reason}.cast"
        try:
            result = run_playback(
                checkpoint=checkpoint_path,
                config_path=config_path,
                episodes=script_args.render_playback_episodes,
                max_steps=script_args.render_playback_steps,
                device=script_args.render_playback_device,
                output_path=output_path,
                asciicast=script_args.render_playback_cast,
                cast_output_path=cast_output_path,
                clear_frames=not script_args.render_playback_no_clear_frames,
            )
            frames = result.get("frames", "unknown") if isinstance(result, dict) else "unknown"
            target = output_path if output_path else "stdout"
            cast_target = result.get("cast") if isinstance(result, dict) else None
            if cast_target:
                print(f"[render playback] {reason}: frames={frames} -> {target}, cast -> {cast_target}")
            else:
                print(f"[render playback] {reason}: frames={frames} -> {target}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[render playback] Skipped ({reason}) due to error: {exc}")

    resume_health_pending = model_resume_path is not None

    with torch_profile_context as torch_profiler:
        try:
            while trainer.epoch < trainer.total_epochs:
                anneal_step = int(trainer.global_step) + int(anneal_step_offset)
                current_temp, current_smoothing = _apply_sampler_anneal(
                    sampler_anneal_config,
                    global_step=anneal_step,
                )
                current_ent_coef = _apply_ent_coef_anneal(
                    trainer,
                    ent_coef_anneal_config,
                    global_step=anneal_step,
                )
                trainer.stats["sampler/subaction_temperature"].append(float(current_temp))
                trainer.stats["sampler/smoothing_eps"].append(float(current_smoothing))
                trainer.stats["anneal/ent_coef"].append(float(current_ent_coef))
                with maybe_record("azk.evaluate"):
                    trainer.evaluate()
                resume_completed_episode_tracker = _update_completed_episode_tracker(
                    resume_completed_episode_tracker, getattr(trainer, "stats", None)
                )
                if resume_health_pending:
                    snapshot = _rollout_health_snapshot(trainer)
                    if snapshot:
                        print(
                            "[resume] rollout health: "
                            f"entropy={snapshot['entropy']:.6f}, "
                            f"noop_selected_rate={snapshot['noop_selected_rate']:.6f}"
                        )
                        if snapshot["entropy"] < 0.2 and snapshot["noop_selected_rate"] > 0.9:
                            print(
                                "[resume] warning: rollout appears collapse-like before the first optimizer step. "
                                "This usually indicates an incompatible or degraded checkpoint."
                            )
                    resume_health_pending = False
                with maybe_record("azk.train"):
                    logs = trainer.train()
                if logs is not None:
                    resume_completed_episode_tracker = _update_completed_episode_tracker(
                        resume_completed_episode_tracker, logs
                    )
                    done_training = trainer.epoch >= trainer.total_epochs
                    if trainer.epoch % stdout_interval == 0 or done_training:
                        console = filter_numeric_metrics(logs, stdout_patterns)
                        print(
                            f"[epoch {trainer.epoch}] "
                            + json.dumps(console, sort_keys=True, separators=(",", ":"))
                        )
                league_active = compute_league_active(
                    global_step=int(trainer.global_step),
                    activate_after_steps=int(league_cfg.activate_after_steps),
                )
                trainer.stats["league/active"].append(1.0 if league_active else 0.0)
                if league_enabled and league_manager is not None:
                    if league_active:
                        pool_metrics = league_manager.pool_metrics()
                        for key, value in pool_metrics.items():
                            trainer.stats[key].append(float(value))
                if league_enabled and league_manager is not None:
                    checkpoint_interval = int(trainer.config.get("checkpoint_interval", 0))
                    done_training = trainer.epoch >= trainer.total_epochs
                    if league_active and checkpoint_interval > 0 and (
                        trainer.epoch % checkpoint_interval == 0 or done_training
                    ):
                        checkpoint_raw = trainer.save_checkpoint()
                        if checkpoint_raw:
                            checkpoint_path = Path(checkpoint_raw)
                            added = league_manager.maybe_add_checkpoint(checkpoint_path, epoch=trainer.epoch)
                            if added is not None:
                                metrics = league_manager.maybe_evaluate_and_promote(
                                    epoch=trainer.epoch,
                                    trainer_args=trainer_args,
                                    vecenv=vecenv,
                                    build_policy_fn=build_policy,
                                    load_weights_fn=_load_model_weights,
                                    learner_policy=policy,
                                )
                                if metrics:
                                    for key, value in metrics.items():
                                        if isinstance(value, (int, float)):
                                            trainer.stats[key].append(float(value))
                                        else:
                                            print(f"[league] {key}={value}")

                                refreshed_paths = [
                                    Path(entry.checkpoint_path)
                                    for entry in league_manager.opponent_entries_for_training()
                                ]
                                resident_by_key = {
                                    key: resident_policy
                                    for key, resident_policy in zip(
                                        getattr(trainer, "opponent_keys", []),
                                        trainer.opponent_policies,
                                    )
                                }
                                refreshed_policies = []
                                for opp_path in refreshed_paths:
                                    opponent_key = str(opp_path.resolve())
                                    opp = resident_by_key.get(opponent_key)
                                    if opp is None:
                                        opp = build_policy(vecenv, trainer_args)
                                        _load_model_weights(
                                            opp,
                                            opp_path,
                                            device=str(train_cfg.get("device", "cpu")),
                                            strict=False,
                                        )
                                        opp.eval()
                                    refreshed_policies.append(opp)
                                trainer.set_opponent_policies(
                                    refreshed_policies,
                                    opponent_keys=[str(path.resolve()) for path in refreshed_paths],
                                )
                                print(
                                    "[league] pool refreshed: "
                                    f"sampling_opponents={len(refreshed_policies)}, "
                                    f"resident_opponents={len(trainer.opponent_policies)}, "
                                    f"champion={league_manager.state.champion_policy_id}"
                                )
                current_checkpoint = checkpoint_cache.get(int(trainer.epoch))
                if current_checkpoint is not None:
                    _finalize_checkpoint(current_checkpoint)
                if (
                    script_args.render_playback_interval > 0
                    and trainer.epoch % script_args.render_playback_interval == 0
                ):
                    checkpoint_raw = trainer.save_checkpoint()
                    if checkpoint_raw:
                        checkpoint_path = Path(checkpoint_raw)
                        maybe_run_playback(f"epoch{trainer.epoch:06d}", checkpoint_path)

                if torch_profiler is not None:
                    torch_profiler.step()
                    profiled_epochs += 1
                    if (
                        torch_profile_cfg is not None
                        and torch_profile_cfg.stop_after_capture
                        and profiled_epochs >= torch_profile_cfg.total_profile_epochs
                    ):
                        print(
                            "[profile] capture complete; stopping early after "
                            f"{profiled_epochs} profiled epochs."
                        )
                        break
        finally:
            trainer.print_dashboard()
            force_checkpoint = True
            model_path_raw = trainer.close()
            current_checkpoint = checkpoint_cache.get(int(trainer.epoch))
            if current_checkpoint is not None:
                _finalize_checkpoint(current_checkpoint)
            model_path = Path(model_path_raw) if model_path_raw else None
            if logger is not None:
                logger.close(str(model_path) if model_path else None)
            if script_args.render_playback_final and model_path is not None:
                maybe_run_playback("final", model_path)

    if torch_profiler is not None and torch_profile_cfg is not None:
        _write_torch_profiler_outputs(
            torch_profiler,
            torch_profile_cfg,
            trainer=trainer,
            config_path=config_path,
        )
        print(f"[profile] wrote trace and summaries to {torch_profile_cfg.output_dir}")


def main():
    # Expandable segments avoid fragmentation OOMs from the mixed large/small
    # transients (bucket-1024 decode vs everything else). The allocator reads
    # this at the first CUDA allocation, so a set-if-unset here is effective.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    script_args, forwarded_cli = parse_script_args()
    run_training(script_args, forwarded_cli)


if __name__ == "__main__":
    main()
