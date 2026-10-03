"""Lean batched throughput: vmap(engine_step) only (core sim, no obs/reward).
One batch size per run (each shape recompiles ~25min). Random legal action.
  python jax_env/benchmarks/bench_engine_step.py <batch> <steps>
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for e in (REPO / "jax_env", REPO / "python/src", REPO / "build/python/src"):
  sys.path.insert(0, str(e))

import jax
import jax.numpy as jnp

from azuki_jax.engine.step import engine_step, stabilize
from azuki_jax.env import init_state_with_decks
from azuki_jax.masks import build_mask
from azuki_jax.setup import deck_tables_from_card_lists
from training_deck_pool import load_training_deck_pool

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
pool = [list(d) for d in load_training_deck_pool(
    str(REPO / ".codex/docs/azuki_tcg_decks_final.json"))]
tables = deck_tables_from_card_lists(pool[0], pool[1])
print(f"batch={batch} steps={steps} device={jax.devices()}", flush=True)

base = stabilize(init_state_with_decks(12345, tables))
state = jax.tree.map(lambda x: jnp.broadcast_to(x, (batch,) + x.shape), base)


def one(state, key):
  legal, count, _ = build_mask(state)
  idx = jax.random.randint(key, (), 0, jnp.maximum(count.astype(jnp.int32), 1))
  a = legal[idx].astype(jnp.int32)
  return engine_step(state, a)


bstep = jax.jit(jax.vmap(one))
keys = jax.random.split(jax.random.PRNGKey(0), steps * batch).reshape(steps, batch, 2)

t0 = time.perf_counter()
state = bstep(state, keys[0])
jax.block_until_ready(state.zone)
print(f"compile+first: {time.perf_counter() - t0:.1f}s", flush=True)

t0 = time.perf_counter()
for i in range(1, steps):
  state = bstep(state, keys[i])
jax.block_until_ready(state.zone)
el = time.perf_counter() - t0
sps = (steps - 1) * batch / el
print(f"batch {batch}: {sps:12.0f} env-steps/s  "
      f"({el / (steps - 1) * 1e3:7.2f} ms/batch-step)", flush=True)
