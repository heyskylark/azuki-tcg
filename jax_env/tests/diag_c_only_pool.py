"""Fast C-only replay of a pool pair (no JAX), driving with the same rng+driver
filter the eager diag uses, so it reproduces the identical C trajectory. With
DBG_PASSIVE=1 the C passive instrumentation ([Cjay]/[C012]/[Cqueue]) reveals
exactly when/what C decides. Step markers go to stderr to interleave with the
C fprintf logs.

  DBG_PASSIVE=1 python jax_env/tests/diag_c_only_pool.py <i> <j> <seed> <maxstep>
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
from test_l3_abilities_batch4 import DRIVER_TYPES_4  # noqa: E402


def main():
  i, j, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
  maxstep = int(sys.argv[4]) if len(sys.argv) > 4 else 100
  from training_deck_pool import load_training_deck_pool
  pool = [list(d) for d in load_training_deck_pool(
      str(REPO / ".codex/docs/azuki_tcg_decks_final.json"))]
  cref = CRef(seed, deck_pool=None)
  cref.reset_with_decks(seed, pool[i], pool[j])
  rng = np.random.default_rng(seed)
  for si in range(maxstep):
    cv = c_semantic_view(cref)
    active = cv["active"]
    sys.stderr.write(f"\n===== STEP {si} active={active} phase={cv.get('phase')} "
                     f"ab={cv.get('ab_phase')} =====\n")
    sys.stderr.write(f"   C garden0={cv.get('garden0')}\n   C garden1={cv.get('garden1')}\n")
    sys.stderr.flush()
    c_rows = cref.legal_actions(active)
    if not c_rows:
      sys.stderr.write(f"step {si}: no actions\n"); return
    driver = [r for r in c_rows if r[0] in DRIVER_TYPES_4]
    a = driver[int(rng.integers(0, len(driver)))]
    sys.stderr.write(f"  action={tuple(int(x) for x in a)}\n")
    cref.step(np.asarray(a, np.int32))
    if cref.dones()[0] or cref.dones()[1]:
      sys.stderr.write(f"step {si}: ended\n"); return
  sys.stderr.write("done\n")


if __name__ == "__main__":
  main()
