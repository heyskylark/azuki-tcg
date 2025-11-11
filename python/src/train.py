from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path
from typing import Sequence

import torch
from pettingzoo.utils.conversions import turn_based_aec_to_parallel
import pufferlib.models
import pufferlib.vector
from pufferlib import MultiagentEpisodeStats, emulation, pufferl

# Ensure the compiled binding in build/python/src is importable before we pull in tcg.
REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_PYTHON_DIR = REPO_ROOT / "build" / "python" / "src"
if str(BUILD_PYTHON_DIR) not in sys.path and BUILD_PYTHON_DIR.exists():
    sys.path.insert(0, str(BUILD_PYTHON_DIR))

from tcg import AzukiTCG  # noqa: E402  (imports binding)

DEFAULT_CONFIG_PATH = REPO_ROOT / "python" / "config" / "azuki.ini"


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
    return parser.parse_known_args()


def load_training_config(config_path: Path, forwarded_cli: Sequence[str]) -> dict:
    parser = pufferl.make_parser()
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0], *forwarded_cli]
    try:
        return pufferl.load_config_file(str(config_path), fill_in_default=True, parser=parser)
    finally:
        sys.argv = original_argv


def make_azuki_env(**env_kwargs):
    """Instantiate the wrapped Azuki env in the same order as the CLI."""
    seed = env_kwargs.pop("seed", None)
    env = AzukiTCG(seed=seed)
    env = turn_based_aec_to_parallel(env)
    env = MultiagentEpisodeStats(env)
    env = emulation.PettingZooPufferEnv(env)
    return env


def build_vecenv(trainer_args: dict):
    env_kwargs = dict(trainer_args.get("env", {}))
    vec_kwargs = dict(trainer_args.get("vec", {}))
    return pufferlib.vector.make(
        functools.partial(make_azuki_env, **env_kwargs),
        **vec_kwargs,
    )


def build_policy(vecenv, trainer_args: dict) -> torch.nn.Module:
    policy_kwargs = dict(trainer_args.get("policy", {}))
    hidden_size = policy_kwargs.get("hidden_size", 256)
    policy = pufferlib.models.Default(vecenv.driver_env, hidden_size=hidden_size)
    return policy.to(trainer_args["train"]["device"])


def run_training(config_path: Path, forwarded_cli: Sequence[str]):
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    trainer_args = load_training_config(config_path, forwarded_cli)
    trainer_args["train"]["env"] = trainer_args.get("env_name", "azuki_local")

    vecenv = build_vecenv(trainer_args)
    policy = build_policy(vecenv, trainer_args)
    trainer = pufferl.PuffeRL(trainer_args["train"], vecenv, policy)

    try:
        while trainer.epoch < trainer.total_epochs:
            trainer.evaluate()
            logs = trainer.train()
            if logs is not None:
                print(f"[epoch {trainer.epoch}] {logs}")
    finally:
        trainer.print_dashboard()
        trainer.close()


def main():
    script_args, forwarded_cli = parse_script_args()
    run_training(script_args.config, forwarded_cli)


if __name__ == "__main__":
    main()
