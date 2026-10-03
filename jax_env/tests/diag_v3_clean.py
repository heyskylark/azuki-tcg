"""Clean V3/1234 diag: semantic check each step, no inspect_gate (which
corrupted state). Gate internals come from the in-flow apply.py DBG_GATE print.
Run with DBG_GATE=1 JAX_PLATFORMS=cpu.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "build/python/src", REPO / "python/src", REPO / "jax_env"):
  if str(entry) not in sys.path:
    sys.path.insert(0, str(entry))

import jax  # noqa: E402

from conftest import CRef  # noqa: E402
from test_l2_vanilla import c_semantic_view, jax_semantic_view  # noqa: E402
from test_l3_abilities_batch3 import _selection_view_c, _selection_view_jax  # noqa: E402
from test_l3_abilities_batch4 import DECK_GROUPS, DRIVER_TYPES_4  # noqa: E402

from azuki_jax.engine.step import engine_step, stabilize  # noqa: E402
from azuki_jax.env import init_state_with_decks  # noqa: E402
from azuki_jax.setup import deck_tables_from_card_lists  # noqa: E402


def main():
  seed = 1234
  deck = DECK_GROUPS["V3"]
  cref = CRef(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck, deck)
  tables = deck_tables_from_card_lists(deck, deck)
  rng = np.random.default_rng(seed)
  with jax.disable_jit():
    state = stabilize(init_state_with_decks(seed, tables))
    for si in range(120):
      cv = c_semantic_view(cref); cv.update(_selection_view_c(cref))
      jv = jax_semantic_view(state); jv.update(_selection_view_jax(state))
      diverged = [k for k in cv if k != "winner" and jv.get(k) != cv[k]]
      if diverged:
        print(f"\n*** FIRST DIVERGENCE at step {si} ***", flush=True)
        for k in diverged[:10]:
          print(f"   {k}: C={cv[k]} JAX={jv[k]}")
        return
      active = cv["active"]
      c_rows = cref.legal_actions(active)
      driver = [r for r in c_rows if r[0] in DRIVER_TYPES_4]
      if not driver:
        print(f"step {si}: no actions"); return
      a = driver[int(rng.integers(0, len(driver)))]
      cref.step(np.asarray(a, np.int32))
      state = engine_step(state, np.asarray(a, np.int32))  # DBG_GATE print fires here
      if si % 15 == 0:
        print(f"  ..step {si}", flush=True)
      if cref.dones()[0] or cref.dones()[1]:
        print(f"step {si}: ended"); return
  print("no divergence in 120 steps")


if __name__ == "__main__":
  main()
