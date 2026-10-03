from __future__ import annotations

import functools
import sys
import ast
import configparser
import os
from pathlib import Path
from typing import Sequence

import azk_puffer.emulation as emulation
import azk_puffer.pytorch as azk_pytorch
import azk_puffer.trainer as pufferl
import azk_puffer.vector as azk_vector
import torch
from pettingzoo.utils.conversions import turn_based_aec_to_parallel
from azk_puffer import MultiagentEpisodeStats

from deck_building import DeckBuildingParallelEnv
from policy.v2.tcg_policy import TCGLSTM, build_policy_model
from policy.v2 import tcg_sampler
from training_deck_pool import load_training_deck_pool

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_PYTHON_DIR = REPO_ROOT / "build" / "python" / "src"
DEFAULT_CONFIG_PATH = REPO_ROOT / "python" / "config" / "azuki.ini"


def _has_binding_artifact(path: Path) -> bool:
  return any(path.glob("binding*.so")) or any(path.glob("binding*.pyd"))


def _candidate_python_build_dirs() -> list[Path]:
  env_override = os.getenv("AZK_BUILD_PYTHON_DIR")
  candidates: list[Path] = []
  if env_override:
    candidates.append(Path(env_override).expanduser())

  discovered: list[Path] = [BUILD_PYTHON_DIR]
  for path in sorted(REPO_ROOT.glob("build*/python/src")):
    if path not in discovered:
      discovered.append(path)

  def _binding_mtime(path: Path) -> float:
    binding_candidates = list(path.glob("binding*.so")) + list(path.glob("binding*.pyd"))
    if not binding_candidates:
      return -1.0
    return max(candidate.stat().st_mtime for candidate in binding_candidates)

  for path in sorted(discovered, key=_binding_mtime, reverse=True):
    if path not in candidates:
      candidates.append(path)
  return candidates


def _existing_binding_dir_on_sys_path() -> Path | None:
  for entry in sys.path:
    if not entry:
      continue
    try:
      candidate = Path(entry).expanduser()
    except (TypeError, OSError):
      continue
    if not candidate.exists() or not candidate.is_dir():
      continue
    if _has_binding_artifact(candidate):
      return candidate
  return None


def ensure_python_build_on_path() -> None:
  env_override = os.getenv("AZK_BUILD_PYTHON_DIR")
  if env_override:
    candidate = Path(env_override).expanduser()
    if candidate.exists() and candidate.is_dir() and _has_binding_artifact(candidate):
      candidate_str = str(candidate)
      if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
      return

  existing = _existing_binding_dir_on_sys_path()
  if existing is not None:
    return

  for candidate in _candidate_python_build_dirs():
    if not candidate.exists():
      continue
    if not _has_binding_artifact(candidate):
      continue
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
      sys.path.insert(0, candidate_str)
    break


ensure_python_build_on_path()

from v2.tcg import AzukiTCG  # noqa: E402  (import after adjusting sys.path)
from v2.tcg_parallel import AzukiTCGParallel  # noqa: E402


def install_tcg_sampler() -> None:
  """Use the custom sampler that understands the Azuki action layout."""
  tcg_sampler.set_fallback_sampler(azk_pytorch.sample_logits)
  azk_pytorch.sample_logits = tcg_sampler.tcg_sample_logits


def _configure_pbrs_discount(trainer_args: dict) -> None:
  env_config = trainer_args.get("env", {})
  if not bool(env_config.get("native", False)):
    return
  gamma = float(trainer_args.get("train", {}).get("gamma", 0.99))
  if not 0.0 <= gamma <= 1.0:
    raise ValueError("train.gamma must be finite and in [0, 1]")
  # An explicit environment value is only a consistency assertion. Training
  # owns the discount used by both the return estimator and the potential.
  declared_gamma = env_config.get("pbrs_gamma")
  if declared_gamma is not None and float(declared_gamma) != gamma:
    raise ValueError("env.pbrs_gamma must match train.gamma; omit it to derive the discount")
  env_config["pbrs_gamma"] = gamma
  if env_config.get("pbrs_mode", "legacy") == "discounted":
    if env_config.get("pbrs_terminal_closure") is False:
      raise ValueError("discounted PBRS requires terminal closure")
    env_config["pbrs_terminal_closure"] = True


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

  _configure_pbrs_discount(parsed)
  return parsed


def make_azuki_env(*, seed: int | None = None, buf=None, **env_kwargs):
  """Instantiate the wrapped Azuki env in the same order as training."""
  seed = seed if seed is not None else env_kwargs.pop("seed", None)
  native = bool(env_kwargs.pop("native", False))
  native_envs_per_instance = env_kwargs.pop("native_envs_per_instance", None)
  env_kwargs.pop("native_log_interval", None)
  direct_parallel = bool(env_kwargs.pop("direct_parallel", False))
  deck_building_enabled = bool(env_kwargs.pop("deck_building_enabled", False))
  draft_same_element_matchup_prob = float(
    env_kwargs.pop("draft_same_element_matchup_prob", 0.0) or 0.0
  )
  deck_building_privileged_decks = bool(
    env_kwargs.pop("deck_building_privileged_decks", False)
  )
  draft_uniform_assignment = bool(
    env_kwargs.pop("draft_uniform_assignment", False)
  )
  evaluation_mode = bool(env_kwargs.pop("evaluation_mode", False))
  draft_cross_gate_replay_prob = float(
    env_kwargs.pop("draft_cross_gate_replay_prob", 0.0) or 0.0
  )
  reward_telemetry = bool(env_kwargs.pop("reward_telemetry", False))
  reward_decomposed_schedule = bool(
    env_kwargs.pop("reward_decomposed_schedule", False)
  )
  pbrs_mode = str(env_kwargs.pop("pbrs_mode", "legacy"))
  pbrs_gamma = float(env_kwargs.pop("pbrs_gamma", 0.99))
  pbrs_terminal_closure = env_kwargs.pop("pbrs_terminal_closure", None)
  prebuilt_curriculum = bool(env_kwargs.pop("prebuilt_curriculum", False))
  prebuilt_probability = float(env_kwargs.pop("prebuilt_probability", 0.0))
  from specialist import parse_learner_element

  learner_element = parse_learner_element(env_kwargs.pop("learner_element", "none"))
  if learner_element != "none" and not (native and deck_building_enabled):
    raise ValueError("env.learner_element requires env.native=true and deck_building_enabled=true")
  fixed_seats_raw = env_kwargs.pop("deck_building_fixed_seats", None)
  if fixed_seats_raw is None or fixed_seats_raw == "":
    fixed_deck_seats: tuple[int, ...] = ()
  elif isinstance(fixed_seats_raw, int):
    fixed_deck_seats = (int(fixed_seats_raw),)
  elif isinstance(fixed_seats_raw, (list, tuple)):
    fixed_deck_seats = tuple(int(seat) for seat in fixed_seats_raw)
  else:
    fixed_deck_seats = tuple(int(part) for part in str(fixed_seats_raw).split(",") if part.strip())
  deck_pool = env_kwargs.pop("deck_pool", None)
  deck_pool_path = env_kwargs.pop("deck_pool_path", None)
  if deck_pool is not None and deck_pool_path is not None:
    raise ValueError("Pass either env.deck_pool or env.deck_pool_path, not both")
  if deck_pool is None:
    deck_pool = load_training_deck_pool(deck_pool_path)
  prebuilt_deck_groups = None
  learner_prebuilt_deck_indices = None
  if prebuilt_curriculum:
    if not native or not deck_building_enabled or not deck_pool_path:
      raise ValueError("prebuilt_curriculum requires native deck building and an explicit deck_pool_path")
    if learner_element == "none":
      from prebuilt_deck_pool import load_prebuilt_deck_groups

      prebuilt_deck_groups = load_prebuilt_deck_groups(deck_pool_path)
    else:
      from prebuilt_deck_pool import load_specialist_deck_groups, specialist_learner_deck_indices

      prebuilt_deck_groups = load_specialist_deck_groups(deck_pool_path)
      learner_prebuilt_deck_indices = specialist_learner_deck_indices(deck_pool_path, learner_element)
  elif prebuilt_probability != 0.0:
    raise ValueError("prebuilt_probability requires prebuilt_curriculum=true")
  if native:
    if direct_parallel:
      raise ValueError("env.native is incompatible with direct_parallel")
    if deck_building_enabled and fixed_deck_seats:
      raise ValueError(
        "deck_building_fixed_seats is not supported on the native path; "
        "use the legacy path for fixed-seat evals"
      )
    from azk_native import AzukiNativeEnv

    return AzukiNativeEnv(
      num_envs=int(native_envs_per_instance or 1),
      deck_pool=deck_pool,
      buf=buf,
      seed=seed if seed is not None else 0,
      deck_building=deck_building_enabled,
      deck_snapshot_dir=env_kwargs.pop("deck_snapshot_dir", None),
      deck_snapshot_every=env_kwargs.pop("deck_snapshot_every", None),
      draft_same_element_matchup_prob=draft_same_element_matchup_prob,
      draft_cross_gate_replay_prob=draft_cross_gate_replay_prob,
      deck_building_privileged_decks=deck_building_privileged_decks,
      draft_uniform_assignment=draft_uniform_assignment,
      evaluation_mode=evaluation_mode,
      reward_telemetry=reward_telemetry,
      reward_decomposed_schedule=reward_decomposed_schedule,
      pbrs_mode=pbrs_mode,
      pbrs_gamma=pbrs_gamma,
      pbrs_terminal_closure=pbrs_terminal_closure,
      prebuilt_deck_groups=prebuilt_deck_groups,
      prebuilt_probability=prebuilt_probability,
      learner_element=learner_element,
      learner_prebuilt_deck_indices=learner_prebuilt_deck_indices,
    )
  if reward_telemetry:
    raise ValueError("env.reward_telemetry requires env.native=true")
  if reward_decomposed_schedule:
    raise ValueError(
      "env.reward_decomposed_schedule requires env.native=true"
    )
  if pbrs_mode != "legacy" or pbrs_terminal_closure:
    raise ValueError("PBRS configuration requires env.native=true")
  if deck_building_enabled:
    env = AzukiTCGParallel(seed=seed, deck_pool=deck_pool)
    env = DeckBuildingParallelEnv(
      env,
      deck_pool=deck_pool,
      seed=seed,
      fixed_deck_seats=fixed_deck_seats,
      snapshot_dir=env_kwargs.pop("deck_snapshot_dir", None),
      snapshot_every=env_kwargs.pop("deck_snapshot_every", None),
      same_element_matchup_prob=draft_same_element_matchup_prob,
      privileged_decks=deck_building_privileged_decks,
      uniform_assignment=draft_uniform_assignment,
    )
    env = MultiagentEpisodeStats(env)
    env = emulation.PettingZooPufferEnv(env, buf=buf, seed=seed)
    return env
  if direct_parallel:
    env = AzukiTCGParallel(seed=seed, deck_pool=deck_pool)
    env = MultiagentEpisodeStats(env)
    env = emulation.PettingZooPufferEnv(env, buf=buf, seed=seed)
    return env
  env = AzukiTCG(seed=seed, deck_pool=deck_pool)
  env = turn_based_aec_to_parallel(env)
  env = MultiagentEpisodeStats(env)
  env = emulation.PettingZooPufferEnv(env, buf=buf, seed=seed)
  return env


def build_vecenv(trainer_args: dict, *, backend=None, num_envs: int | None = None, seed: int | None = None):
  _configure_pbrs_discount(trainer_args)
  env_kwargs = dict(trainer_args.get("env", {}))
  vec_kwargs = dict(trainer_args.get("vec", {}))
  if backend is not None:
    vec_kwargs["backend"] = backend
  if num_envs is not None:
    vec_kwargs["num_envs"] = num_envs
  if seed is not None:
    vec_kwargs["seed"] = seed
  chosen_backend = vec_kwargs.get("backend")
  if isinstance(chosen_backend, str) and chosen_backend.lower() == "jax":
    from azk_puffer.jax_vector import JaxVecEnv

    if bool(env_kwargs.get("deck_building_enabled", False)):
      raise ValueError("vec.backend=Jax supports battle-only training; set env.deck_building_enabled=false")
    if env_kwargs.get("deck_pool") is not None:
      deck_pool = env_kwargs["deck_pool"]
    else:
      deck_pool = load_training_deck_pool(env_kwargs.get("deck_pool_path"))
    return JaxVecEnv(
      num_envs=int(vec_kwargs.get("num_envs", 1)),
      deck_pool=deck_pool,
      seed=int(vec_kwargs.get("seed", 0) or 0),
    )
  if isinstance(chosen_backend, str):
    backend_attr = getattr(azk_vector, chosen_backend, None)
    if backend_attr is not None:
      chosen_backend = backend_attr
      vec_kwargs["backend"] = chosen_backend
  if chosen_backend == azk_vector.Serial or chosen_backend is azk_vector.Serial:
    vec_kwargs.pop("num_workers", None)
    vec_kwargs["batch_size"] = vec_kwargs.get("num_envs", 1)
  return azk_vector.make(
    functools.partial(make_azuki_env, **env_kwargs),
    **vec_kwargs,
  )


def build_policy(vecenv, trainer_args: dict) -> torch.nn.Module:
  policy_config = trainer_args.get("policy", {})
  base_policy = build_policy_model(
    vecenv.driver_env,
    policy_config=policy_config,
  )
  policy = TCGLSTM(
    vecenv.driver_env,
    base_policy
  )
  return policy.to(trainer_args["train"]["device"])
