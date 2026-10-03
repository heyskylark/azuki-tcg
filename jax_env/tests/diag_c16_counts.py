"""Pinpoint the c16 STT02-012 latch divergence: replay pool 16v17 seed 7016
and, at each step near the divergence, dump both gardens' ENTITY counts and
STT02-012's atk/hp in C and JAX (plus JAX's latch). Shows whether the counts
ever differ or only the persisted latch does, and at which garden event."""
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
from test_l3_abilities_batch4 import DRIVER_TYPES_4  # noqa: E402

from azuki_jax import cards  # noqa: E402
from azuki_jax.engine.step import engine_step, stabilize  # noqa: E402
from azuki_jax.env import init_state_with_decks  # noqa: E402
from azuki_jax.setup import deck_tables_from_card_lists  # noqa: E402

D30 = cards.CODE_TO_ID["STT02-012"]


def gcount(view, p):
  # semantic garden tuples: (def, atk, hp, tapped, cooldown, keywords)
  g = view.get("garden" + str(p), [])
  return sum(1 for c in g if c and c[0] >= 0 and int(cards.TYPE[c[0]]) == 2)


def stt_stats(view, p):
  g = view.get("garden" + str(p), [])
  return [(c[1], c[2]) for c in g if c and c[0] == D30]


def main():
  i, j, seed = 16, 17, 7016
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
    for si in range(72):
      cv = c_semantic_view(cref)
      jv = jax_semantic_view(state)
      if 63 <= si <= 70:
        latch = np.asarray(state.stt02_012_latch)
        lz = [(p, k) for p in range(2) for k in range(latch.shape[1]) if latch[p, k]]
        print(f"[step {si}] active={cv['active']}  "
              f"Cgcount p0/p1={gcount(cv,0)}/{gcount(cv,1)}  "
              f"Jgcount p0/p1={gcount(jv,0)}/{gcount(jv,1)}", flush=True)
        print(f"    C  STT02-012 stats p0={stt_stats(cv,0)} p1={stt_stats(cv,1)}")
        print(f"    JAX STT02-012 stats p0={stt_stats(jv,0)} p1={stt_stats(jv,1)}  latch_on={lz}")
      active = cv["active"]
      c_rows = cref.legal_actions(active)
      if not c_rows:
        print(f"step {si}: no actions"); return
      driver = [r for r in c_rows if r[0] in DRIVER_TYPES_4]
      a = driver[int(rng.integers(0, len(driver)))]
      if 63 <= si <= 70:
        print(f"    -> action {tuple(int(x) for x in a)}")
      cref.step(np.asarray(a, np.int32))
      state = engine_step(state, np.asarray(a, np.int32))
      if cref.dones()[0] or cref.dones()[1]:
        print(f"step {si}: ended"); return
  print("done")


if __name__ == "__main__":
  main()
