"""Measure compile time of apply_user_action and micro_tick SEPARATELY.

The fused engine_step (apply + lax.while_loop(micro_tick)) takes ~2h to compile.
Hypothesis: the while_loop fusion of the 120-card dispatch is the blowup, and
jitting the two pieces alone (driving the auto_resolve loop in Python) compiles
in minutes. This times each piece's first-call (compile) wall-clock so we can
decide whether to pivot to a split-jit harness for fast diagnostics+verify.

Uses the raw (pre-stabilize) init state — compile time depends on shapes/dtypes,
not values, and this avoids compiling auto_resolve's while_loop.
"""
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

from test_l3_abilities_batch4 import DECK_GROUPS  # noqa: E402

from azuki_jax.engine.step import apply_user_action, micro_tick  # noqa: E402
from azuki_jax.env import init_state_with_decks  # noqa: E402
from azuki_jax.setup import deck_tables_from_card_lists  # noqa: E402


def stamp(msg):
  print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
  deck = DECK_GROUPS["V3"]
  tables = deck_tables_from_card_lists(deck, deck)
  state = init_state_with_decks(1234, tables)  # raw, pre-stabilize
  action = np.zeros(4, np.int32)

  stamp("start: jitting micro_tick (no action arg)")
  jit_tick = jax.jit(micro_tick)
  t0 = time.time()
  out = jit_tick(state)
  jax.block_until_ready(out)
  stamp(f"micro_tick compiled+ran in {time.time()-t0:.1f}s")

  stamp("start: jitting apply_user_action")
  jit_apply = jax.jit(apply_user_action)
  t0 = time.time()
  out = jit_apply(state, action)
  jax.block_until_ready(out)
  stamp(f"apply_user_action compiled+ran in {time.time()-t0:.1f}s")

  # warm calls (should be ~instant if compiled)
  t0 = time.time()
  for _ in range(5):
    out = jit_tick(out if False else state)
  jax.block_until_ready(out)
  stamp(f"5 warm micro_tick calls in {time.time()-t0:.3f}s")
  stamp("DONE")


if __name__ == "__main__":
  main()
