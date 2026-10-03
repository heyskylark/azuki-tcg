"""Eager JAX trace of AZK01-019/010/012 armed/ever_played/passive state per step
for a pool pairing, to compare against C's [Cjay]/[C012] observer firings.
  python jax_env/tests/trace_jax_019.py <i> <j> <seed> <maxstep>
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
from test_l3_abilities_batch4 import DRIVER_TYPES_4  # noqa: E402
from training_deck_pool import load_training_deck_pool  # noqa: E402

from azuki_jax import cards  # noqa: E402
from azuki_jax.constants import Zone  # noqa: E402
from azuki_jax.engine.step import engine_step, stabilize  # noqa: E402
from azuki_jax.env import init_state_with_decks  # noqa: E402
from azuki_jax.setup import deck_tables_from_card_lists  # noqa: E402

POOL = [list(d) for d in load_training_deck_pool(
    str(REPO / ".codex/docs/azuki_tcg_decks_final.json"))]
WATCH = {cards.CODE_TO_ID[c]: c[-3:] for c in ("AZK01-019", "AZK01-010", "STT02-012")}
ZN = {int(Zone.GARDEN): "G", int(Zone.ALLEY): "A", int(Zone.HAND): "H",
      int(Zone.DISCARD): "D", int(Zone.DECK): "K"}


def snap(state):
  out = []
  z = np.asarray(state.zone); did = np.asarray(state.def_id)
  arm = np.asarray(state.passive_armed); ev = np.asarray(state.passive_ever_played)
  pa = np.asarray(state.passive_atk); ph = np.asarray(state.passive_hp)
  ca = np.asarray(state.cur_atk); ch = np.asarray(state.cur_hp)
  for p in (0, 1):
    for i in range(z.shape[1]):
      d = int(did[p, i])
      if d in WATCH and z[p, i] in (int(Zone.GARDEN), int(Zone.ALLEY)):
        out.append(f"p{p}.{ZN.get(int(z[p,i]),'?')}{WATCH[d]}"
                   f"=({int(ca[p,i])},{int(ch[p,i])})arm={int(arm[p,i])}ev={int(ev[p,i])}pa{int(pa[p,i])}ph{int(ph[p,i])}")
  return out


def main():
  i, j, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
  maxstep = int(sys.argv[4]) if len(sys.argv) > 4 else 100
  cref = CRef(seed, deck_pool=None)
  cref.reset_with_decks(seed, POOL[i], POOL[j])
  rng = np.random.default_rng(seed)
  tables = deck_tables_from_card_lists(POOL[i], POOL[j])
  prev = None
  with jax.disable_jit():
    state = stabilize(init_state_with_decks(seed, tables))
    for si in range(maxstep):
      cur = snap(state)
      if cur != prev:
        print(f">>> step {si}: {cur}", flush=True)
        prev = cur
      rows = cref.legal_actions(cref.active_player)
      if not rows:
        break
      drv = [r for r in rows if r[0] in DRIVER_TYPES_4]
      if not drv:
        break
      a = drv[int(rng.integers(0, len(drv)))]
      print(f"    step {si} action {tuple(int(x) for x in a)}", flush=True)
      cref.step(np.asarray(a, np.int32))
      state = engine_step(state, np.asarray(a, np.int32))
      if cref.dones()[0] or cref.dones()[1]:
        break
  cref.close()


if __name__ == "__main__":
  main()
