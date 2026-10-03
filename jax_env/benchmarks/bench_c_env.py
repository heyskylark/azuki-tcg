"""Benchmark the C environment stepping speed (no model).

Measures:
1. Raw binding path: env_step + reading raw struct bytes (what the C engine costs).
2. Full wrapper path: AzukiTCGParallel.step() with observation_to_dict (what
   training currently pays per step, before pufferlib emulation packing).
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "build/python/src")
sys.path.insert(0, "python/src")

import binding  # noqa: E402
from tcg_parallel import AzukiTCGParallel  # noqa: E402
from training_deck_pool import load_training_deck_pool  # noqa: E402


def bench_raw(env: AzukiTCGParallel, num_steps: int, rng: np.random.Generator) -> float:
  """Steps/sec submitting random legal actions, reading only raw mask bytes."""
  binding.env_reset(env.c_envs, int(rng.integers(0, 2**31 - 1)))
  start = time.perf_counter()
  steps = 0
  while steps < num_steps:
    active = binding.env_active_player(env.c_envs)
    if active < 0:
      binding.env_reset(env.c_envs, int(rng.integers(0, 2**31 - 1)))
      continue
    raw = env._raw_observation(active)
    mask = raw.action_mask
    count = int(mask.legal_action_count)
    if count == 0:
      binding.env_reset(env.c_envs, int(rng.integers(0, 2**31 - 1)))
      continue
    choice = int(rng.integers(0, count))
    env._actions[active] = (
      int(mask.legal_primary[choice]),
      int(mask.legal_sub1[choice]),
      int(mask.legal_sub2[choice]),
      int(mask.legal_sub3[choice]),
    )
    binding.env_step(env.c_envs)
    steps += 1
    if env._terminals.all() or env._truncations.all():
      binding.env_reset(env.c_envs, int(rng.integers(0, 2**31 - 1)))
  elapsed = time.perf_counter() - start
  return steps / elapsed


def bench_wrapper(env: AzukiTCGParallel, num_steps: int, rng: np.random.Generator) -> float:
  """Steps/sec through the full ParallelEnv.step path (dict observations)."""
  env.reset(seed=int(rng.integers(0, 2**31 - 1)))
  start = time.perf_counter()
  steps = 0
  while steps < num_steps:
    if not env.agents:
      env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    action = env.random_legal_action(rng)
    actions = {agent: action for agent in env.possible_agents}
    env.step(actions)
    steps += 1
  elapsed = time.perf_counter() - start
  return steps / elapsed


def main() -> None:
  num_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
  pool = load_training_deck_pool(".codex/docs/azuki_tcg_decks_final.json")
  rng = np.random.default_rng(0)
  env = AzukiTCGParallel(seed=1234, deck_pool=pool)

  raw_sps = bench_raw(env, num_steps, rng)
  print(f"raw C binding step:      {raw_sps:10.1f} steps/s ({1e6/raw_sps:8.1f} us/step)")

  wrapper_sps = bench_wrapper(env, num_steps, rng)
  print(f"full wrapper (dict obs): {wrapper_sps:10.1f} steps/s ({1e6/wrapper_sps:8.1f} us/step)")
  env.close()


if __name__ == "__main__":
  main()
