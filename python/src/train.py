from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import glob
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import pufferlib.pytorch
import pufferlib.vector
from pufferlib import pufferl

from policy import tcg_sampler
from league_manager import LeagueManager, parse_league_manager_config
from league_training import LeagueConfig, LeaguePuffeRL, compute_league_active
from playback import run_playback
from training_utils import (
    DEFAULT_CONFIG_PATH,
    build_policy,
    build_vecenv,
    install_tcg_sampler,
    load_training_config,
)


RESUME_COMPLETED_EPISODES_ENV = "AZK_RESUME_COMPLETED_EPISODES"
RESUME_COMPLETED_EPISODE_KEYS = (
    "0/azk_completed_episodes",
    "1/azk_completed_episodes",
    "environment/0/azk_completed_episodes",
    "environment/1/azk_completed_episodes",
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
)
RESUME_SOURCE_HASH_TARGETS = (
    "python/src/policy/tcg_policy.py",
    "python/src/policy/tcg_sampler.py",
    "python/src/tcg.h",
    "python/src/train.py",
    "python/src/training_utils.py",
)


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
        "--resume-strict",
        action="store_true",
        help="Require an exact key match when loading model weights from a checkpoint.",
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
        "--league-latest-ratio",
        type=float,
        default=None,
        help="Fraction of matchups using latest learner as opponent (e.g. 0.85).",
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
        ensure_buffers(norm_key, feature_shape)
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
    return int(max(candidates))


def _update_completed_episode_tracker(current_value: int | None, source: object) -> int | None:
    candidate = _extract_completed_episodes_from_mapping(source)
    if candidate is None:
        return current_value
    if current_value is None:
        return candidate
    return max(current_value, candidate)


def _resume_env_var_fingerprint() -> dict[str, str]:
    return {name: os.getenv(name, "") for name in RESUME_SCHEDULE_ENV_VARS}


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
    global_step_candidate: int | None = None

    if trainer_state_path is not None and trainer_state_path.exists():
        try:
            trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
            if isinstance(trainer_state, dict):
                candidate = _coerce_nonnegative_int(trainer_state.get("env_completed_episodes"))
                if candidate is not None:
                    return candidate
                global_step_candidate = _coerce_nonnegative_int(
                    trainer_state.get("global_step", trainer_state.get("agent_step"))
                )
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
                        return candidate
                    if global_step_candidate is None:
                        global_step_candidate = _coerce_nonnegative_int(payload.get("global_step"))
            except Exception as exc:
                print(f"[resume] warning: failed reading checkpoint metadata for env progression restore: {exc}")

    if global_step_candidate is not None:
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


def _apply_saved_schedule_env(
    saved_resume_config: dict[str, object] | None,
) -> None:
    if not isinstance(saved_resume_config, dict):
        return
    schedule_env = saved_resume_config.get("schedule_env")
    if not isinstance(schedule_env, dict):
        return

    changes: list[tuple[str, str, str]] = []
    for name in RESUME_SCHEDULE_ENV_VARS:
        if name not in schedule_env:
            continue
        saved_value_raw = schedule_env.get(name)
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
        print("[resume] applied saved schedule env vars from checkpoint metadata: " + details)


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

    return {
        "use_rnn": bool(train_cfg.get("use_rnn", False)),
        "direct_parallel": bool(env_cfg.get("direct_parallel", False)),
        "schedule_env": _resume_env_var_fingerprint(),
        "source_hashes": _source_hash_fingerprint(),
    }


def _checkpoint_metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".meta.json")


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _save_checkpoint_metadata(model_path: Path, payload: dict) -> None:
    _write_json_atomic(_checkpoint_metadata_path(model_path), payload)


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


def _validate_resume_metadata(
    model_path: Path,
    runtime_fingerprint: dict[str, object],
    resume_config_fingerprint: dict[str, object],
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
        raise RuntimeError(
            "[resume] runtime/checkpoint incompatibility detected. "
            f"Checkpoint={model_path}. Mismatched fields: {details}"
        )

    saved_resume_cfg = payload.get("resume_config_fingerprint")
    if isinstance(saved_resume_cfg, dict):
        cfg_mismatches = _resume_cfg_mismatches(saved_resume_cfg, resume_config_fingerprint)
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
  materialized = _materialize_scalar_norm_buffers_from_state_dict(policy, cleaned)
  scalar_norm_keys = sum(1 for key in cleaned.keys() if "scalar_normalizer._rms_" in key)
  missing, unexpected = policy.load_state_dict(cleaned, strict=strict)
  print(
    "[resume] loaded model checkpoint: "
    f"path={model_path}, strict={strict}, missing_keys={len(missing)}, "
    f"unexpected_keys={len(unexpected)}, "
    f"materialized_scalar_norm_features={materialized}, scalar_norm_state_keys={scalar_norm_keys}"
  )


def _maybe_reset_critic_head(policy: torch.nn.Module) -> bool:
    base_policy = getattr(policy, "policy", policy)
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


def _maybe_restore_trainer_state(
    trainer,
    trainer_state_path: Path | None,
    *,
    expected_model_path: Path | None = None,
    expected_resume_config_fingerprint: dict[str, object] | None = None,
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

    state = dict(action=mb_actions, lstm_h=None, lstm_c=None)
    amp_context = getattr(trainer, "amp_context", None)
    if amp_context is None:
        amp_context = contextlib.nullcontext()

    with torch.no_grad(), amp_context:
        logits, _ = trainer.policy(mb_obs, state)
        _, _, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

    return {
        "entropy": float(entropy.mean().item()),
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
            backend=pufferlib.vector.Serial,
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
                action, _, entropy = pufferlib.pytorch.sample_logits(logits)

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

    latest_ratio = script_args.league_latest_ratio
    if latest_ratio is None:
        latest_ratio = float(league_cfg.get("latest_ratio", 0.85))
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
        latest_ratio=latest_ratio,
        randomize_learner_seat=bool(randomize_learner_seat),
        seed=seed,
        activate_after_steps=activate_after_steps,
    )


def run_training(script_args: argparse.Namespace, forwarded_cli):
    config_path = script_args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    trainer_args = load_training_config(config_path, forwarded_cli)
    trainer_args["train"]["env"] = trainer_args.get("env_name", "azuki_local")
    logger = None

    install_tcg_sampler()

    resume_checkpoint = script_args.resume_checkpoint
    if resume_checkpoint is None:
        fallback_resume = trainer_args.get("load_model_path")
        if isinstance(fallback_resume, str) and fallback_resume and fallback_resume != "latest":
            resume_checkpoint = Path(fallback_resume)

    model_resume_path: Path | None = None
    trainer_state_path: Path | None = None
    if resume_checkpoint is not None:
        model_resume_path, trainer_state_path = _resolve_resume_artifacts(resume_checkpoint)
        _apply_saved_schedule_env(
            _load_saved_resume_config_fingerprint(model_resume_path, trainer_state_path)
        )

    vec_cfg = trainer_args.get("vec")
    num_envs_hint = 1
    if isinstance(vec_cfg, dict):
        parsed_num_envs = _coerce_nonnegative_int(vec_cfg.get("num_envs"))
        if parsed_num_envs is not None and parsed_num_envs > 0:
            num_envs_hint = parsed_num_envs

    manual_resume_completed = _coerce_nonnegative_int(os.getenv(RESUME_COMPLETED_EPISODES_ENV))
    if manual_resume_completed is not None:
        resume_env_completed_episodes = manual_resume_completed
        print(
            "[resume] using caller-provided env progression override: "
            f"{RESUME_COMPLETED_EPISODES_ENV}={manual_resume_completed}"
        )
    else:
        resume_env_completed_episodes = _peek_resume_env_completed_episodes(
            model_resume_path,
            trainer_state_path,
            num_envs_hint=num_envs_hint,
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
                "Curriculum- and reward-anneal episode counters will restart from zero."
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

    policy = build_policy(vecenv, trainer_args)
    if model_resume_path is not None:
        _validate_resume_metadata(model_resume_path, runtime_fingerprint, resume_config_fingerprint)
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
        print(
            "[league] enabled: "
            f"opponents={len(opponent_policies)}, "
            f"latest_ratio={league_cfg.latest_ratio:.3f}, "
            f"activate_after_steps={league_cfg.activate_after_steps}"
        )

    if league_enabled:
        trainer = LeaguePuffeRL(
            trainer_args["train"],
            vecenv,
            policy,
            opponent_policies=opponent_policies,
            league_cfg=league_cfg,
            logger=logger,
        )
    else:
        trainer = pufferl.PuffeRL(trainer_args["train"], vecenv, policy, logger=logger)

    resume_completed_episode_tracker = _coerce_nonnegative_int(resume_env_completed_episodes)
    original_save_checkpoint = trainer.save_checkpoint

    def _save_checkpoint_with_metadata():
        nonlocal resume_completed_episode_tracker
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
            "update": int(trainer.epoch),
            "trainer_state_path": str(state_path.name),
            "runtime_fingerprint": runtime_fingerprint,
            "resume_config_fingerprint": resume_config_fingerprint,
            "checkpoint_parity": parity_summary,
        }
        if resume_completed_episode_tracker is not None:
            metadata_payload["env_completed_episodes"] = int(resume_completed_episode_tracker)
        _save_checkpoint_metadata(
            checkpoint_path,
            metadata_payload,
        )
        return checkpoint_raw

    trainer.save_checkpoint = _save_checkpoint_with_metadata

    if script_args.resume_load_optimizer:
        if trainer_state_path is not None:
            _maybe_restore_trainer_state(
                trainer,
                trainer_state_path,
                expected_model_path=model_resume_path,
                expected_resume_config_fingerprint=resume_config_fingerprint,
            )
        else:
            print("[resume] --resume-load-optimizer requested but trainer_state.pt was not found")

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
            logs = trainer.train()
            if logs is not None:
                resume_completed_episode_tracker = _update_completed_episode_tracker(
                    resume_completed_episode_tracker, logs
                )
                print(f"[epoch {trainer.epoch}] {logs}")
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
                            refreshed_policies = []
                            for opp_path in refreshed_paths:
                                opp = build_policy(vecenv, trainer_args)
                                _load_model_weights(
                                    opp,
                                    opp_path,
                                    device=str(train_cfg.get("device", "cpu")),
                                    strict=False,
                                )
                                opp.eval()
                                refreshed_policies.append(opp)
                            trainer.set_opponent_policies(refreshed_policies)
                            print(
                                "[league] pool refreshed: "
                                f"opponents={len(refreshed_policies)}, champion={league_manager.state.champion_policy_id}"
                            )
            if (
                script_args.render_playback_interval > 0
                and trainer.epoch % script_args.render_playback_interval == 0
            ):
                checkpoint_raw = trainer.save_checkpoint()
                if checkpoint_raw:
                    checkpoint_path = Path(checkpoint_raw)
                    maybe_run_playback(f"epoch{trainer.epoch:06d}", checkpoint_path)
    finally:
        trainer.print_dashboard()
        model_path_raw = trainer.close()
        model_path = Path(model_path_raw) if model_path_raw else None
        if logger is not None:
            logger.close(str(model_path) if model_path else None)
        if script_args.render_playback_final and model_path is not None:
            maybe_run_playback("final", model_path)


def main():
    script_args, forwarded_cli = parse_script_args()
    run_training(script_args, forwarded_cli)


if __name__ == "__main__":
    main()
