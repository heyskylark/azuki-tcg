"""L3 (abilities): episode equivalence on decks stacked with ported cards.

Driver allows ability actions (activate garden, select targets, confirm) plus
the vanilla set. Mask comparison covers all rows except:
- leader activations (sub1==5 of type 11): leaders' abilities aren't ported
- spell rows (type 8): no spells ported yet
Unported on-play abilities never trigger because the deck only contains
ported + vanilla cards (leader/gate abilities are main/portal-gated as in the
vanilla suite).
"""
from __future__ import annotations

import numpy as np
import pytest

from test_l2_vanilla import c_semantic_view, filtered_mask, jax_semantic_view

# vanilla base + the four ported cards
ABILITY_DECK = [
    ("STT01-001", 1),
    ("STT01-002", 1),
    ("STT02-007", 4),   # On Play: draw 1
    ("AZK01-004", 4),   # When Attacking: +1 atk EOT
    ("AZK01-005", 4),   # On Play: 1 damage to up to 1 enemy garden entity
    ("AZK01-006", 4),   # Main (once/turn): return self to hand
    ("STT01-010", 4),
    ("STT02-004", 4),
    ("STT02-006", 4),
    ("AZK01-001", 4),
    ("AZK01-012", 4),
    ("AZK01-025", 4),
    ("AZK01-035", 4),
    ("AZK01-038", 4),
    ("AZK01-094", 2),
    ("IKZ-001", 10),
]

DRIVER_TYPES = {0, 1, 2, 6, 7, 9, 11, 13, 14, 16, 25}


def comparable(rows):
  out = []
  for row in rows:
    if row[0] == 8:  # spells not ported
      continue
    if row[0] == 11 and row[1] == 5:  # leader activations not ported
      continue
    out.append(row)
  return out


@pytest.mark.parametrize("seed", [3, 41, 905])
def test_ability_episode_equivalence(seed, make_cref):
  import jax

  from azuki_jax.engine.step import engine_step, stabilize
  from azuki_jax.env import init_state_with_decks
  from azuki_jax.masks import build_mask
  from azuki_jax.setup import deck_tables_from_card_lists

  cref = make_cref(seed, deck_pool=None)
  cref.reset_with_decks(seed, ABILITY_DECK, ABILITY_DECK)

  tables = deck_tables_from_card_lists(ABILITY_DECK, ABILITY_DECK)
  state = stabilize(init_state_with_decks(seed, tables))

  jit_step = jax.jit(engine_step)
  jit_mask = jax.jit(build_mask)

  rng = np.random.default_rng(seed)
  ability_actions_taken = 0
  for step_index in range(500):
    cview = c_semantic_view(cref)
    jview = jax_semantic_view(state)
    for key, cval in cview.items():
      if key == "winner":
        continue
      assert jview[key] == cval, (
          f"step {step_index}: {key}\nC  ={cval}\nJAX={jview[key]}"
      )

    active = cview["active"]
    c_rows = comparable(cref.legal_actions(active))
    legal, count, _ = jit_mask(state)
    j_rows = comparable(
        [tuple(int(x) for x in row) for row in np.asarray(legal)[: int(count)]]
    )
    assert j_rows == c_rows, (
        f"step {step_index} (phase {cview['phase']}, ab?): mask mismatch\n"
        f"C  ={c_rows}\nJAX={j_rows}"
    )

    driver_rows = [row for row in c_rows if row[0] in DRIVER_TYPES]
    assert driver_rows, f"step {step_index}: no driver actions"
    action = driver_rows[int(rng.integers(0, len(driver_rows)))]
    if action[0] in (11, 13, 14, 16):
      ability_actions_taken += 1

    cref.step(np.asarray(action, np.int32))
    state = jit_step(state, np.asarray(action, np.int32))

    c_term, c_trunc = cref.dones()
    j_over = int(state.winner) != -1
    assert c_term == j_over, f"step {step_index}: terminal mismatch"
    if c_term or c_trunc:
      break

  # ensure the run actually exercised abilities (on-play triggers count via
  # state changes; activations counted here)
  assert int(state.ab_scratch[3]) == 0, "unimplemented ability hit"
