"""General evolution dump: replay a pool pair and print chosen semantic keys
for BOTH engines across a step window, flagging per-step divergence. Reusable
for any fullpool residual.

  python jax_env/tests/diag_evolution.py <i> <j> <seed> <start> <end> [key1 key2 ...]
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
from test_l3_abilities_batch4 import DRIVER_TYPES_4  # noqa: E402

from azuki_jax.engine.step import engine_step, stabilize  # noqa: E402
from azuki_jax.env import init_state_with_decks  # noqa: E402
from azuki_jax.setup import deck_tables_from_card_lists  # noqa: E402


def main():
  i, j, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
  start, end = int(sys.argv[4]), int(sys.argv[5])
  keys = sys.argv[6:] or ["phase", "active"]
  from training_deck_pool import load_training_deck_pool
  pool = [list(d) for d in load_training_deck_pool(
      str(REPO / ".codex/docs/azuki_tcg_decks_final.json"))]
  deck0, deck1 = pool[i], pool[j]
  cref = CRef(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck0, deck1)
  tables = deck_tables_from_card_lists(deck0, deck1)
  rng = np.random.default_rng(seed)
  with jax.disable_jit():
    state = stabilize(init_state_with_decks(seed, tables))
    for si in range(end + 1):
      cv = c_semantic_view(cref); cv.update(_selection_view_c(cref))
      jv = jax_semantic_view(state); jv.update(_selection_view_jax(state))
      if start <= si <= end:
        print(f"--- step {si}  phase={cv.get('phase')} active={cv.get('active')} "
              f"turn={cv.get('turn')} ---", flush=True)
        for k in keys:
          c, jx = cv.get(k), jv.get(k)
          mark = "  <<< DIFF" if c != jx else ""
          print(f"   {k}: C={c}")
          print(f"   {' '*len(k)}  J={jx}{mark}")
      active = cv["active"]
      c_rows = cref.legal_actions(active)
      if not c_rows:
        print(f"step {si}: no actions"); return
      driver = [r for r in c_rows if r[0] in DRIVER_TYPES_4]
      a = driver[int(rng.integers(0, len(driver)))]
      if start <= si <= end:
        print(f"   -> action {tuple(int(x) for x in a)}")
      cref.step(np.asarray(a, np.int32))
      state = engine_step(state, np.asarray(a, np.int32))
      if cref.dones()[0] or cref.dones()[1]:
        print(f"step {si}: ended"); return
  print("done")


if __name__ == "__main__":
  main()
