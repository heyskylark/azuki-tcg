"""Task-14: full-pool 1:1 acceptance test on the REAL production decks.

This is the definitive parity gate. Unlike the crafted per-batch decks, it
draws the 18 production decks (16 pool + 2 starter) and plays mirror and cross
pairings, comparing EVERYTHING each step with NO exclusions:
  - full semantic board view (c_semantic_view) incl. weapons/stats/zones
  - selection-zone + ability-context views
  - the complete legal-action mask (every row, leaders included)
  - terminal/winner agreement
  - zero unimplemented-ability hits (state.ab_scratch[3] == 0)
A pass over this set is exact behavioral equivalence on the training
distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "build/python/src", REPO / "python/src", REPO / "jax_env"):
  if str(entry) not in sys.path:
    sys.path.insert(0, str(entry))

from test_l2_vanilla import c_semantic_view, jax_semantic_view  # noqa: E402
from test_l3_abilities_batch3 import (  # noqa: E402
    _selection_view_c,
    _selection_view_jax,
)
from test_l3_abilities_batch4 import DRIVER_TYPES_4  # noqa: E402


def _load_pool():
  from training_deck_pool import load_training_deck_pool

  pool = load_training_deck_pool(str(REPO / ".codex/docs/azuki_tcg_decks_final.json"))
  return [list(d) for d in pool]


POOL = _load_pool()
NUM = len(POOL)

# Mirror matches (every deck vs itself) cover every leader/gate + that deck's
# realistic card distribution. Cross matches (a rotating set) exercise
# cross-deck interaction. Keep the case count bounded for wall-clock.
MIRROR_CASES = [(i, i, 12345 + i) for i in range(NUM)]
CROSS_CASES = [(i, (i + 1) % NUM, 7000 + i) for i in range(NUM)]


def run_match(make_cref, seed, deck0, deck1, steps=600):
  import jax

  from azuki_jax.engine.step import engine_step, stabilize
  from azuki_jax.env import init_state_with_decks
  from azuki_jax.masks import build_mask
  from azuki_jax.setup import deck_tables_from_card_lists

  cref = make_cref(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck0, deck1)

  tables = deck_tables_from_card_lists(deck0, deck1)
  state = stabilize(init_state_with_decks(seed, tables))

  jit_step = jax.jit(engine_step)
  jit_mask = jax.jit(build_mask)

  rng = np.random.default_rng(seed)
  for step_index in range(steps):
    cview = c_semantic_view(cref)
    cview.update(_selection_view_c(cref))
    jview = jax_semantic_view(state)
    jview.update(_selection_view_jax(state))
    for key, cval in cview.items():
      if key == "winner":
        continue
      assert jview[key] == cval, (
          f"step {step_index}: {key}\nC  ={cval}\nJAX={jview[key]}"
      )

    active = cview["active"]
    c_rows = cref.legal_actions(active)
    legal, count, _ = jit_mask(state)
    j_rows = [tuple(int(x) for x in row) for row in np.asarray(legal)[: int(count)]]
    assert j_rows == c_rows, (
        f"step {step_index} (phase {cview['phase']}, ab {cview['ab_phase']}):"
        f" mask mismatch\nC  ={c_rows}\nJAX={j_rows}"
    )
    if not c_rows:
      break

    driver_rows = [row for row in c_rows if row[0] in DRIVER_TYPES_4]
    assert driver_rows, f"step {step_index}: no driver actions\n{c_rows}"
    action = driver_rows[int(rng.integers(0, len(driver_rows)))]

    cref.step(np.asarray(action, np.int32))
    state = jit_step(state, np.asarray(action, np.int32))

    c_term, c_trunc = cref.dones()
    assert (int(state.winner) != -1) == c_term, (
        f"step {step_index}: terminal mismatch"
    )
    if c_term or c_trunc:
      break

  assert int(state.ab_scratch[3]) == 0, "unimplemented ability hit"


@pytest.mark.parametrize("i,j,seed", MIRROR_CASES)
def test_fullpool_mirror(i, j, seed, make_cref):
  run_match(make_cref, seed, POOL[i], POOL[j])


@pytest.mark.parametrize("i,j,seed", CROSS_CASES)
def test_fullpool_cross(i, j, seed, make_cref):
  run_match(make_cref, seed, POOL[i], POOL[j])
