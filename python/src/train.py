from __future__ import annotations

import argparse
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
    return parser.parse_known_args()


def run_training(script_args: argparse.Namespace, forwarded_cli):
    config_path = script_args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    trainer_args = load_training_config(config_path, forwarded_cli)
    trainer_args["train"]["env"] = trainer_args.get("env_name", "azuki_local")
    logger = None
    if trainer_args.get("wandb"):
        logger = pufferl.WandbLogger(trainer_args)
    elif trainer_args.get("neptune"):
        logger = pufferl.NeptuneLogger(trainer_args)

    install_tcg_sampler()

    vecenv = build_vecenv(trainer_args)
    policy = build_policy(vecenv, trainer_args)
    trainer = pufferl.PuffeRL(trainer_args["train"], vecenv, policy, logger=logger)

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
