"""C-only probe of max selection.count across EVERY parity test case.

Imports each test module's deck groups / case lists / driver sets and replays the
exact deterministic C trajectory, recording peak ctx->selection.count. Drives the
choice of MAX_SELECTION_ROWS (mask compile bound) without risking parity.
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
from training_deck_pool import load_training_deck_pool  # noqa: E402

import test_l3_abilities_batch1 as b1  # noqa: E402
import test_l3_abilities_batch2 as b2  # noqa: E402
import test_l3_abilities_batch3 as b3  # noqa: E402
import test_l3_abilities_batch4 as b4  # noqa: E402
import test_l3_passives as pas  # noqa: E402
import test_l3_abilities as ab  # noqa: E402


def sel_count(cref, player):
  return int(cref.raw(player).my_observation_data.selection_count)


def replay(deck0, deck1, seed, driver, steps):
  cref = CRef(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck0, deck1)
  rng = np.random.default_rng(seed)
  peak, pstep = 0, -1
  for si in range(steps):
    active = cref.active_player
    c = max(sel_count(cref, 0), sel_count(cref, 1))
    if c > peak:
      peak, pstep = c, si
    rows = cref.legal_actions(active)
    if not rows:
      break
    drv = [r for r in rows if r[0] in driver]
    if not drv:
      break
    a = drv[int(rng.integers(0, len(drv)))]
    cref.step(np.asarray(a, np.int32))
    t, tr = cref.dones()
    if t or tr:
      break
  cref.close()
  return peak, pstep


def main():
  jobs = []  # (label, deck0, deck1, seed, driver, steps)

  # batch1: groups x seeds [7,1234], deck vs itself
  for g, deck in b1.DECK_GROUPS.items():
    for s in (7, 1234):
      jobs.append((f"b1-{g}-s{s}", deck, deck, s, b1.DRIVER_TYPES, 500))
  # batch2: CASES (group,seed)
  for g, s in b2.CASES:
    jobs.append((f"b2-{g}-s{s}", b2.DECK_GROUPS[g], b2.DECK_GROUPS[g], s,
                 b1.DRIVER_TYPES, 500))
  # batch3
  for g, s in b3.CASES:
    jobs.append((f"b3-{g}-s{s}", b3.DECK_GROUPS[g], b3.DECK_GROUPS[g], s,
                 b1.DRIVER_TYPES, 500))
  # batch4: CASES with DRIVER_TYPES_4
  for g, s in b4.CASES:
    jobs.append((f"b4-{g}-s{s}", b4.DECK_GROUPS[g], b4.DECK_GROUPS[g], s,
                 b4.DRIVER_TYPES_4, 500))
  # passives: groups x [7,1234]
  for g, deck in pas.DECK_GROUPS.items():
    for s in (7, 1234):
      jobs.append((f"pas-{g}-s{s}", deck, deck, s, pas.DRIVER_TYPES, 450))
  # test_l3_abilities: single deck x seeds [3,41,905]
  for s in (3, 41, 905):
    jobs.append((f"ab-s{s}", ab.ABILITY_DECK, ab.ABILITY_DECK, s,
                 ab.DRIVER_TYPES, 500))

  overall, worst = 0, None
  for label, d0, d1, s, drv, steps in jobs:
    peak, pstep = replay(d0, d1, s, drv, steps)
    flag = "  <==" if peak > 8 else ""
    print(f"{label:16s} peak_sel={peak} @step{pstep}{flag}", flush=True)
    if peak > overall:
      overall, worst = peak, label
  print(f"\n=== OVERALL MAX selection.count (batch/passive/ability) = {overall} at {worst} ===")


if __name__ == "__main__":
  main()
