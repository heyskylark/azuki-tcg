from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import torch
from pufferlib import pufferl

from league_manager import LeagueManager, parse_league_manager_config
from league_training import LeagueConfig, LeaguePuffeRL
from playback import run_playback
from training_utils import (
    DEFAULT_CONFIG_PATH,
    build_policy,
    build_vecenv,
    install_tcg_sampler,
    load_training_config,
)


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
    if resume_checkpoint.is_dir():
        model_path = _select_latest_model_file(resume_checkpoint)
        trainer_state_path = resume_checkpoint / "trainer_state.pt"
        return model_path, trainer_state_path if trainer_state_path.exists() else None
    if not resume_checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint path not found: {resume_checkpoint}")
    trainer_state_path = resume_checkpoint.parent / "trainer_state.pt"
    return resume_checkpoint, trainer_state_path if trainer_state_path.exists() else None


def _load_model_weights(policy: torch.nn.Module, model_path: Path, *, device: str, strict: bool) -> None:
    raw = torch.load(model_path, map_location=device)
    state_dict = raw.get("state_dict") if isinstance(raw, dict) and "state_dict" in raw else raw
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format (expected state_dict dict): {model_path}")
    cleaned = _strip_module_prefix(state_dict)
    materialized = _materialize_scalar_norm_buffers_from_state_dict(policy, cleaned)
    missing, unexpected = policy.load_state_dict(cleaned, strict=strict)
    print(
        "[resume] loaded model checkpoint: "
        f"path={model_path}, strict={strict}, missing_keys={len(missing)}, "
        f"unexpected_keys={len(unexpected)}, materialized_scalar_norm_buffers={materialized}"
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


def _maybe_restore_trainer_state(trainer, trainer_state_path: Path | None) -> bool:
    if trainer_state_path is None or not trainer_state_path.exists():
        return False

    # trainer_state.pt is produced locally by this trainer and includes optimizer
    # internals (e.g. defaultdict), which require full pickle loading.
    trainer_state = torch.load(
        trainer_state_path, map_location="cpu", weights_only=False
    )
    if not isinstance(trainer_state, dict):
        raise ValueError(f"Unsupported trainer_state format: {trainer_state_path}")

    optimizer_state = trainer_state.get("optimizer_state_dict")
    if optimizer_state is not None:
        trainer.optimizer.load_state_dict(optimizer_state)

    restored_global_step = int(trainer_state.get("global_step", trainer.global_step))
    restored_epoch = int(trainer_state.get("update", trainer.epoch))
    trainer.global_step = restored_global_step
    trainer.epoch = restored_epoch
    trainer.last_log_step = restored_global_step
    trainer.last_log_time = time.time()
    trainer.start_time = time.time()
    trainer.scheduler.last_epoch = max(restored_epoch - 1, -1)
    synced_lrs = _sync_optimizer_lr_from_scheduler(trainer)
    print(
        "[resume] restored trainer state: "
        f"path={trainer_state_path}, global_step={restored_global_step}, "
        f"epoch={restored_epoch}, optimizer_lrs={synced_lrs}"
    )
    return True


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

    train_cfg = trainer_args.get("train", {})
    seed = int(train_cfg.get("seed", 0)) if isinstance(train_cfg, dict) else 0

    return LeagueConfig(
        enabled=True,
        latest_ratio=latest_ratio,
        randomize_learner_seat=bool(randomize_learner_seat),
        seed=seed,
    )


def run_training(script_args: argparse.Namespace, forwarded_cli):
    config_path = script_args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    trainer_args = load_training_config(config_path, forwarded_cli)
    trainer_args["train"]["env"] = trainer_args.get("env_name", "azuki_local")
    logger = None

    install_tcg_sampler()

    vecenv = build_vecenv(trainer_args)
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

    if trainer_args.get("wandb"):
        logger = pufferl.WandbLogger(trainer_args)
    elif trainer_args.get("neptune"):
        logger = pufferl.NeptuneLogger(trainer_args)

    resume_checkpoint = script_args.resume_checkpoint
    if resume_checkpoint is None:
        fallback_resume = trainer_args.get("load_model_path")
        if isinstance(fallback_resume, str) and fallback_resume and fallback_resume != "latest":
            resume_checkpoint = Path(fallback_resume)

    model_resume_path: Path | None = None
    trainer_state_path: Path | None = None
    if resume_checkpoint is not None:
        model_resume_path, trainer_state_path = _resolve_resume_artifacts(resume_checkpoint)

    policy = build_policy(vecenv, trainer_args)
    if model_resume_path is not None:
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
            f"latest_ratio={league_cfg.latest_ratio:.3f}"
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

    if league_enabled and league_manager is not None:
        learner_id = f"learner_{trainer.logger.run_id}"
        league_manager.attach_learner_identity(learner_id)
        pool_metrics = league_manager.pool_metrics()
        for key, value in pool_metrics.items():
            trainer.stats[key].append(float(value))
    if script_args.resume_load_optimizer and trainer_state_path is not None:
        _maybe_restore_trainer_state(trainer, trainer_state_path)

    effective_total_timesteps = int(trainer.total_epochs * trainer.config["batch_size"])
    print(
      "[train] epoch plan: "
      f"batch_size={trainer.config['batch_size']}, total_epochs={trainer.total_epochs}, "
      f"effective_total_timesteps={effective_total_timesteps}"
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

    try:
        while trainer.epoch < trainer.total_epochs:
            trainer.evaluate()
            logs = trainer.train()
            if logs is not None:
                print(f"[epoch {trainer.epoch}] {logs}")
            if league_enabled and league_manager is not None:
                pool_metrics = league_manager.pool_metrics()
                for key, value in pool_metrics.items():
                    trainer.stats[key].append(float(value))
            if league_enabled and league_manager is not None:
                checkpoint_interval = int(trainer.config.get("checkpoint_interval", 0))
                done_training = trainer.epoch >= trainer.total_epochs
                if checkpoint_interval > 0 and (
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
