from __future__ import annotations

import functools
import sys
import ast
import configparser
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
from tcg_parallel import AzukiTCGParallel  # noqa: E402


def install_tcg_sampler() -> None:
  """Use the custom sampler that understands the Azuki action layout."""
  tcg_sampler.set_fallback_sampler(pufferlib.pytorch.sample_logits)
  pufferlib.pytorch.sample_logits = tcg_sampler.tcg_sample_logits


def load_training_config(config_path: Path, forwarded_cli: Sequence[str]) -> dict:
  def parse_value(raw: str):
    lowered = raw.lower()
    if lowered == "true":
      return True
    if lowered == "false":
      return False
    if lowered == "none":
      return None
    try:
      return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
      return raw

  def merge_ini(path: Path, target: dict) -> None:
    cfg = configparser.ConfigParser()
    read_files = cfg.read(path)
    if not read_files:
      raise FileNotFoundError(f"Failed to read config file: {path}")

    for section in cfg.sections():
      section_data = target.get(section)
      if not isinstance(section_data, dict):
        section_data = {}
        target[section] = section_data
      for key, value in cfg.items(section):
        section_data[key] = parse_value(value)

  parsed: dict = {}
  pufferl_default = Path(pufferl.__file__).resolve().parent / "config" / "default.ini"
  if pufferl_default.exists():
    merge_ini(pufferl_default, parsed)
  merge_ini(config_path, parsed)

  base_section = parsed.get("base", {})
  if isinstance(base_section, dict):
    for key, value in base_section.items():
      parsed[key] = value

  i = 0
  cli = list(forwarded_cli)
  while i < len(cli):
    token = cli[i]
    if not token.startswith("--"):
      i += 1
      continue

    key_token = token[2:]
    value_token = None
    if "=" in key_token:
      key_token, value_token = key_token.split("=", 1)
      i += 1
    elif i + 1 < len(cli) and not cli[i + 1].startswith("--"):
      value_token = cli[i + 1]
      i += 2
    else:
      value_token = "true"
      i += 1

    parts = key_token.replace("-", "_").split(".")
    value = parse_value(value_token)
    target = parsed
    for part in parts[:-1]:
      next_target = target.get(part)
      if not isinstance(next_target, dict):
        next_target = {}
        target[part] = next_target
      target = next_target
    target[parts[-1]] = value

  train_config = parsed.get("train")
  if isinstance(train_config, dict):
    # Azuki always builds a TCGLSTM policy. Keep RNN training enabled by
    # default unless the user explicitly disables it in config/CLI.
    if "use_rnn" in train_config:
      train_config["use_rnn"] = bool(train_config["use_rnn"])
    else:
      train_config["use_rnn"] = True
    if "wandb_project" not in parsed and "project" in train_config:
      parsed["wandb_project"] = train_config["project"]
    if "wandb_group" not in parsed:
      parsed["wandb_group"] = None
    if "tag" not in parsed:
      parsed["tag"] = None
    if "no_model_upload" not in parsed:
      parsed["no_model_upload"] = False

  return parsed


def make_azuki_env(*, seed: int | None = None, buf=None, **env_kwargs):
  """Instantiate the wrapped Azuki env in the same order as training."""
  seed = seed if seed is not None else env_kwargs.pop("seed", None)
  native = bool(env_kwargs.pop("native", False))
  env_kwargs.pop("native_envs_per_instance", None)
  env_kwargs.pop("native_log_interval", None)
  direct_parallel = bool(env_kwargs.pop("direct_parallel", False))
  if native:
    raise RuntimeError(
      "env.native is currently disabled: raw native packed observations use "
      "interleaved struct arrays that PufferLib tensor nativization cannot "
      "decode correctly yet."
    )
  if direct_parallel:
    env = AzukiTCGParallel(seed=seed)
    env = MultiagentEpisodeStats(env)
    env = emulation.PettingZooPufferEnv(env, buf=buf, seed=seed)
    return env
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
