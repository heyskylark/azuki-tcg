"""Measure XLA compile time for engine_step and build_mask on this box, and
whether the on-disk persistent cache makes a second process instant."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "build/python/src", REPO / "python/src", REPO / "jax_env"):
  if str(entry) not in sys.path:
    sys.path.insert(0, str(entry))

import jax  # noqa: E402

from azuki_jax.engine.step import engine_step, stabilize  # noqa: E402
from azuki_jax.env import init_state_with_decks  # noqa: E402
from azuki_jax.masks import NUM_CANDIDATES, MAX_SELECTION_ROWS, build_mask  # noqa: E402
from azuki_jax.setup import deck_tables_from_card_lists  # noqa: E402
from training_deck_pool import load_training_deck_pool  # noqa: E402

print(f"NUM_CANDIDATES={NUM_CANDIDATES} MAX_SELECTION_ROWS={MAX_SELECTION_ROWS}", flush=True)
print(f"devices={jax.devices()}", flush=True)

pool = [list(d) for d in load_training_deck_pool(
    str(REPO / ".codex/docs/azuki_tcg_decks_final.json"))]
tables = deck_tables_from_card_lists(pool[0], pool[1])
state = stabilize(init_state_with_decks(12345, tables))
action = np.zeros(4, np.int32)

jit_mask = jax.jit(build_mask)
jit_step = jax.jit(engine_step)

t0 = time.time()
m = jit_mask(state)
jax.block_until_ready(m)
t1 = time.time()
print(f"build_mask compile+run: {t1 - t0:.1f}s", flush=True)

t0 = time.time()
s2 = jit_step(state, action)
jax.block_until_ready(s2)
t1 = time.time()
print(f"engine_step compile+run: {t1 - t0:.1f}s", flush=True)

# steady-state run time (already compiled)
t0 = time.time()
for _ in range(20):
  s2 = jit_step(state, action)
jax.block_until_ready(s2)
t1 = time.time()
print(f"engine_step steady: {(t1 - t0) / 20 * 1000:.2f} ms/step", flush=True)
print("DONE", flush=True)
