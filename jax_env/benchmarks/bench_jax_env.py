"""Benchmark the split JAX vector environment without model/training overhead.

This is the sim-only benchmark used for the parity project: it drives the same
`JaxVecEnv` split-action backend as training, chooses a random legal row from
the current JAX legal mask on the host, and reports env-steps/s after warm-up.
JIT compile/warm-up time is printed separately and excluded from SPS.

Usage:
  PYTHONPATH=build/python/src:python/src:jax_env \
    JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python jax_env/benchmarks/bench_jax_env.py [batch ...] --steps 1000 --warmup 25
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "jax_env", REPO / "python/src", REPO / "build/python/src"):
  if str(entry) not in sys.path:
    sys.path.insert(0, str(entry))


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "batch",
      nargs="*",
      type=int,
      default=[512, 2048, 8192, 16384],
      help="JAX vector env batch sizes to benchmark.",
  )
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--warmup", type=int, default=25)
  parser.add_argument(
      "--max-warmup",
      type=int,
      default=600,
      help="Adaptive warmup cap: keep warming until step latency stabilizes "
      "(no step in the trailing window exceeds 5x its median) or this many "
      "steps, so first-compiles never land in the timed region.",
  )
  parser.add_argument("--seed", type=int, default=1)
  return parser.parse_args()


# Sample from the same action-type set the parity gate drives
# (test_l3_abilities_batch4.DRIVER_TYPES_4): the SPS then measures the
# parity-verified trajectory distribution, and every kernel it touches is
# the gate-compiled set. (Full-mask sampling reaches an action family whose
# lazy generic kernel's XLA compile exceeds this box's RAM.)
DRIVER_TYPES = frozenset(
    {0, 1, 2, 6, 7, 8, 9, 11, 13, 14, 16, 25}
    | {10, 12, 18, 19, 20, 21, 22, 23, 24}
)


def _sample_legal_actions(env, rng: np.random.Generator) -> np.ndarray:
  legal = np.asarray(env._pending[4])
  count = np.asarray(env._pending[5]).astype(np.int32, copy=False)
  out = np.zeros((env.num_environments, 2, 4), dtype=np.int32)
  for env_idx, legal_count in enumerate(count):
    n = int(legal_count)
    if n > 0:
      rows = legal[env_idx, :n]
      driver = rows[np.isin(rows[:, 0], list(DRIVER_TYPES))]
      pick = driver if len(driver) > 0 else rows
      row = pick[int(rng.integers(0, len(pick)))]
      out[env_idx, 0] = row
      out[env_idx, 1] = row
  return out.reshape(env.num_agents, 4)


def _step_once(env, rng: np.random.Generator) -> None:
  from azk_puffer.jax_vector import SEND

  actions = _sample_legal_actions(env, rng)
  env.flag = SEND
  env.send(actions)


def _block_ready(env) -> None:
  env._jax.block_until_ready(env._states.zone)


def _bench_batch(
    batch: int, *, steps: int, warmup: int, max_warmup: int, seed: int
) -> None:
  from training_deck_pool import load_training_deck_pool
  from azk_puffer.jax_vector import JaxVecEnv

  deck_pool = load_training_deck_pool(str(REPO / ".codex/docs/azuki_tcg_decks_final.json"))
  env = JaxVecEnv(batch, deck_pool, seed=seed)
  rng = np.random.default_rng(seed)

  # Adaptive warmup: run until a trailing window of steps has no latency
  # spike (spike = a probable first-compile), so the timed region below
  # measures steady state only.
  window = 20
  t0 = time.perf_counter()
  env.async_reset(seed=seed)
  latencies: list[float] = []
  for i in range(max(max_warmup, warmup, window)):
    s0 = time.perf_counter()
    _step_once(env, rng)
    _block_ready(env)
    latencies.append(time.perf_counter() - s0)
    if i + 1 >= max(warmup, window):
      tail = latencies[-window:]
      if max(tail) <= 5.0 * float(np.median(tail)):
        break
  warmup_s = time.perf_counter() - t0
  warmup_steps = len(latencies)

  t0 = time.perf_counter()
  timed: list[float] = []
  for _ in range(steps):
    s0 = time.perf_counter()
    _step_once(env, rng)
    _block_ready(env)
    timed.append(time.perf_counter() - s0)
  elapsed = time.perf_counter() - t0
  arr = np.asarray(timed)
  median = float(np.median(arr))
  slow = int(np.sum(arr > 5.0 * median))
  sps = batch * steps / elapsed
  median_sps = batch / median
  print(
      f"batch {batch:6d}: {sps:12.0f} env-steps/s mean"
      f" ({elapsed / steps * 1e3:8.3f} ms/batch-step;"
      f" median {median * 1e3:8.3f} ms -> {median_sps:12.0f} env-steps/s;"
      f" p95 {float(np.percentile(arr, 95)) * 1e3:8.3f} ms;"
      f" {slow} slow steps >5x median;"
      f" warmup {warmup_steps} steps {warmup_s:.1f}s)",
      flush=True,
  )
  env.close()


def main() -> None:
  args = _parse_args()
  for batch in args.batch:
    _bench_batch(
        int(batch),
        steps=int(args.steps),
        warmup=int(args.warmup),
        max_warmup=int(args.max_warmup),
        seed=int(args.seed),
    )


if __name__ == "__main__":
  main()
