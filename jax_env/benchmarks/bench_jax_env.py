"""Benchmark the JAX env: vmapped random-legal-action stepping on GPU.

Measures env-only SPS at several batch sizes (paper B.1 protocol), using a
uniform-random legal action chosen on-device from the mask (index via
jax.random, matching the training-time action source shape).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "jax_env", REPO / "python/src", REPO / "build/python/src"):
  if str(entry) not in sys.path:
    sys.path.insert(0, str(entry))

import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
  from training_deck_pool import load_training_deck_pool

  from azuki_jax.env import init_state
  from azuki_jax.masks import build_mask
  from azuki_jax.setup import build_deck_pool_tables
  from azuki_jax.step import step_with_legal_count as env_step

  pool_native = load_training_deck_pool(
      str(REPO / ".codex/docs/azuki_tcg_decks_final.json")
  )
  pool = build_deck_pool_tables(pool_native)

  def one_env_step(state, key, prev_term, prev_trunc):
    legal, count, _ = build_mask(state)
    idx = jax.random.randint(key, (), 0, jnp.maximum(count.astype(jnp.int32), 1))
    action = legal[idx].astype(jnp.int32)
    actions = jnp.stack([action, action])
    state, rewards, terms, truncs = env_step(
        state, actions, prev_term, prev_trunc, pool, count
    )
    return state, terms, truncs

  batched_step = jax.jit(jax.vmap(one_env_step))
  # Fresh reset already lands at the first mulligan decision point. Keeping
  # stabilize() out of the reset jit avoids compiling the full auto-resolve
  # loop before the benchmarked step function.
  batched_init = jax.jit(jax.vmap(lambda seed: init_state(seed, pool)))

  batch_sizes = [int(x) for x in (sys.argv[1:] or ["32", "128", "512", "2048"])]
  steps = 200

  for batch in batch_sizes:
    seeds = jnp.arange(batch, dtype=jnp.uint32) + 1
    state = batched_init(seeds)
    terms = jnp.zeros((batch, 2), jnp.bool_)
    truncs = jnp.zeros((batch, 2), jnp.bool_)
    keys = jax.random.split(jax.random.PRNGKey(0), steps * batch).reshape(
        steps, batch, 2
    )

    # warmup + compile
    t0 = time.perf_counter()
    state, terms, truncs = batched_step(state, keys[0], terms, truncs)
    jax.block_until_ready(state.zone)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(1, steps):
      state, terms, truncs = batched_step(state, keys[i], terms, truncs)
    jax.block_until_ready(state.zone)
    elapsed = time.perf_counter() - t0
    sps = (steps - 1) * batch / elapsed
    print(
        f"batch {batch:5d}: {sps:12.0f} env-steps/s"
        f"  ({elapsed/(steps-1)*1e3:7.2f} ms/step, compile {compile_s:.1f}s)"
    )


if __name__ == "__main__":
  main()
