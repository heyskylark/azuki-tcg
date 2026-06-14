"""Fast C-only replay of V3/1234 (no JAX). Drives the C engine with the same
action sequence (rng + driver filter from C legal_actions) and lets the
DBG_GATE_C fprintf in azk01_124_validate reveal C's gate_power + portaled card
at each portal. Run with DBG_GATE_C=1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "build/python/src", REPO / "python/src", REPO / "jax_env"):
  if str(entry) not in sys.path:
    sys.path.insert(0, str(entry))

from conftest import CRef  # noqa: E402
from test_l2_vanilla import c_semantic_view  # noqa: E402
from test_l3_abilities_batch4 import DECK_GROUPS, DRIVER_TYPES_4  # noqa: E402


def main():
  group = sys.argv[1] if len(sys.argv) > 1 else "V3"
  seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1234
  deck = DECK_GROUPS[group]
  cref = CRef(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck, deck)
  rng = np.random.default_rng(seed)
  for si in range(135):
    cv = c_semantic_view(cref)
    active = cv["active"]
    c_rows = cref.legal_actions(active)
    if not c_rows:
      print(f"step {si}: no actions"); return
    driver = [r for r in c_rows if r[0] in DRIVER_TYPES_4]
    a = driver[int(rng.integers(0, len(driver)))]
    print(f"[STEP {si}] action={tuple(int(x) for x in a)} active={active} ab={cv.get('ab_phase')}",
          flush=True)
    cref.step(np.asarray(a, np.int32))
    if cref.dones()[0] or cref.dones()[1]:
      print(f"step {si}: ended"); return
  print("done 135 steps")


if __name__ == "__main__":
  main()
