"""C-only probe: max selection-zone count reached across all fullpool cases.

Replays the exact deterministic trajectories the parity suite drives (same
np.default_rng(seed) + DRIVER_TYPES_4 filter on C legal actions) and records the
peak ctx->selection.count, so we can pick a safe MAX_SELECTION_ROWS mask bound.
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
NUM = len(POOL)
MIRROR = [(i, i, 12345 + i) for i in range(NUM)]
CROSS = [(i, (i + 1) % NUM, 7000 + i) for i in range(NUM)]


def sel_count(cref, player):
  return int(cref.raw(player).my_observation_data.selection_count)


def run_case(tag, i, j, seed, steps=600):
  cref = CRef(seed, deck_pool=None)
  cref.reset_with_decks(seed, POOL[i], POOL[j])
  rng = np.random.default_rng(seed)
  peak = 0
  peak_step = -1
  for si in range(steps):
    active = cref.active_player
    c = max(sel_count(cref, 0), sel_count(cref, 1))
    if c > peak:
      peak, peak_step = c, si
    c_rows = cref.legal_actions(active)
    if not c_rows:
      break
    driver = [r for r in c_rows if r[0] in DRIVER_TYPES_4]
    if not driver:
      break
    a = driver[int(rng.integers(0, len(driver)))]
    cref.step(np.asarray(a, np.int32))
    term, trunc = cref.dones()
    if term or trunc:
      break
  cref.close()
  return peak, peak_step


def main():
  overall = 0
  worst = None
  for tag, cases in (("M", MIRROR), ("C", CROSS)):
    for (i, j, seed) in cases:
      peak, pstep = run_case(tag, i, j, seed)
      if peak > overall:
        overall = peak
        worst = (tag, i, j, seed, pstep)
      print(f"{tag} {i:2d}v{j:2d} s{seed} peak_sel={peak} @step{pstep}", flush=True)
  print(f"\n=== OVERALL MAX selection.count = {overall}  at {worst} ===")


if __name__ == "__main__":
  main()
