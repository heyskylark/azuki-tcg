from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import torch
from pufferlib import pufferl

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
    return parser.parse_known_args()


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


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
    missing, unexpected = policy.load_state_dict(cleaned, strict=strict)
    print(
        "[resume] loaded model checkpoint: "
        f"path={model_path}, strict={strict}, missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
    )


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
    print(
        "[resume] restored trainer state: "
        f"path={trainer_state_path}, global_step={restored_global_step}, epoch={restored_epoch}"
    )
    return True


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

    trainer = pufferl.PuffeRL(trainer_args["train"], vecenv, policy, logger=logger)
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
