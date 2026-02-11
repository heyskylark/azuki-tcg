from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Sequence

import torch
from pettingzoo.utils.conversions import turn_based_aec_to_parallel
import pufferlib.vector
from pufferlib import MultiagentEpisodeStats, emulation, pufferl
import pufferlib.pytorch

from policy.tcg_policy import TCG, TCGLSTM
from policy import tcg_sampler

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_PYTHON_DIR = REPO_ROOT / "build" / "python" / "src"
DEFAULT_CONFIG_PATH = REPO_ROOT / "python" / "config" / "azuki.ini"


def ensure_python_build_on_path() -> None:
  if str(BUILD_PYTHON_DIR) not in sys.path and BUILD_PYTHON_DIR.exists():
    sys.path.insert(0, str(BUILD_PYTHON_DIR))


ensure_python_build_on_path()

from tcg import AzukiTCG  # noqa: E402  (import after adjusting sys.path)


def install_tcg_sampler() -> None:
  """Use the custom sampler that understands the Azuki action layout."""
  tcg_sampler.set_fallback_sampler(pufferlib.pytorch.sample_logits)
  pufferlib.pytorch.sample_logits = tcg_sampler.tcg_sample_logits


def load_training_config(config_path: Path, forwarded_cli: Sequence[str]) -> dict:
  parser = pufferl.make_parser()
  original_argv = sys.argv[:]
  sys.argv = [sys.argv[0], *forwarded_cli]
  try:
    return pufferl.load_config_file(str(config_path), fill_in_default=True, parser=parser)
  finally:
    sys.argv = original_argv


def make_azuki_env(*, seed: int | None = None, buf=None, **env_kwargs):
  """Instantiate the wrapped Azuki env in the same order as training."""
  seed = seed if seed is not None else env_kwargs.pop("seed", None)
  env = AzukiTCG(seed=seed)
  env = turn_based_aec_to_parallel(env)
  env = MultiagentEpisodeStats(env)
  env = emulation.PettingZooPufferEnv(env, buf=buf, seed=seed)
  return env


def build_vecenv(trainer_args: dict, *, backend=None, num_envs: int | None = None, seed: int | None = None):
  env_kwargs = dict(trainer_args.get("env", {}))
  vec_kwargs = dict(trainer_args.get("vec", {}))
  if backend is not None:
    vec_kwargs["backend"] = backend
  if num_envs is not None:
    vec_kwargs["num_envs"] = num_envs
  if seed is not None:
    vec_kwargs["seed"] = seed
  chosen_backend = vec_kwargs.get("backend")
  if isinstance(chosen_backend, str):
    backend_attr = getattr(pufferlib.vector, chosen_backend, None)
    if backend_attr is not None:
      chosen_backend = backend_attr
      vec_kwargs["backend"] = chosen_backend
  if chosen_backend == pufferlib.vector.Serial or chosen_backend is pufferlib.vector.Serial:
    vec_kwargs.pop("num_workers", None)
    vec_kwargs["batch_size"] = vec_kwargs.get("num_envs", 1)
  return pufferlib.vector.make(
    functools.partial(make_azuki_env, **env_kwargs),
    **vec_kwargs,
  )


def build_policy(vecenv, trainer_args: dict) -> torch.nn.Module:
  base_policy = TCG(
    vecenv.driver_env,
  )
  policy = TCGLSTM(
    vecenv.driver_env,
    base_policy
  )
  return policy.to(trainer_args["train"]["device"])
