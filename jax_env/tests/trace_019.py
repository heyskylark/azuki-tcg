"""Step-correlated trace of AZK01-019 (and 010) cur_hp/cur_atk in the C engine
for a given pool pairing, to pin the exact passive apply/remove timing. Run with
DBG_PASSIVE=1 to also see the [Cjay] observer decisions, interleaved per step.
  DBG_PASSIVE=1 python jax_env/tests/trace_019.py <i> <j> <seed> <maxstep>
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
from test_l3_abilities_batch4 import DRIVER_TYPES_4  # noqa: E402
from training_deck_pool import load_training_deck_pool  # noqa: E402

POOL = [list(d) for d in load_training_deck_pool(
    str(REPO / ".codex/docs/azuki_tcg_decks_final.json"))]

# def ids of interest
WATCH = {53: "019", 45: "010", 73: "073", 30: "012"}


def board_hps(cref, player):
  """Return {(zone,slot): (def_id, atk, hp)} for garden+alley of player."""
  raw = cref.raw(player).my_observation_data
  out = []
  for zname, arr, n in (("G", raw.garden, 5), ("A", raw.alley, 5)):
    for i in range(n):
      c = arr[i]
      d = int(c.card_def_id)
      if d in WATCH:
        out.append(f"{zname}{i}:{WATCH[d]}=({int(c.cur_stats.cur_atk)},{int(c.cur_stats.cur_hp)})")
  return out


def main():
  i, j, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
  maxstep = int(sys.argv[4]) if len(sys.argv) > 4 else 100
  cref = CRef(seed, deck_pool=None)
  cref.reset_with_decks(seed, POOL[i], POOL[j])
  rng = np.random.default_rng(seed)
  prev = None
  for si in range(maxstep):
    cur = []
    for p in (0, 1):
      for s in board_hps(cref, p):
        cur.append(f"p{p}.{s}")
    if cur != prev:
      sys.stderr.write(f"\n>>> step {si}: {cur}\n")
      sys.stderr.flush()
      prev = cur
    rows = cref.legal_actions(cref.active_player)
    if not rows:
      break
    drv = [r for r in rows if r[0] in DRIVER_TYPES_4]
    if not drv:
      break
    a = drv[int(rng.integers(0, len(drv)))]
    sys.stderr.write(f"    step {si} action {tuple(int(x) for x in a)} active={cref.active_player}\n")
    cref.step(np.asarray(a, np.int32))
    if cref.dones()[0] or cref.dones()[1]:
      break
  cref.close()


if __name__ == "__main__":
  main()
