"""L3 batch-1 ability ports: episode equivalence per ~5-card deck group.

Same pattern as test_l3_abilities.py. Decks contain only ported + vanilla
cards, so mask comparison covers spell rows (type 8) too; leader activations
(type 11 with sub1==5) stay excluded (leaders not ported).
"""
from __future__ import annotations

import numpy as np
import pytest

from test_l2_vanilla import c_semantic_view, jax_semantic_view

# Vanilla fillers (no abilities) usable to pad decks to 62.
FILLERS = [
    ("STT01-010", 4),
    ("STT02-004", 4),
    ("STT02-006", 4),
    ("STT02-008", 4),
    ("AZK01-001", 4),
    ("AZK01-012", 4),
    ("AZK01-025", 4),
    ("AZK01-035", 4),
    ("AZK01-037", 4),
    ("AZK01-038", 4),
    ("AZK01-049", 4),
    ("AZK01-054", 4),
    ("AZK01-094", 2),  # vanilla weapon
]


def build_deck(ported, leader="STT01-001", gate="STT01-002", copies=4,
               fillers=None):
  deck = [(leader, 1), (gate, 1)]
  count = 0
  for code in ported:
    deck.append((code, copies))
    count += copies
  need = 50 - count
  for code, qty in (fillers or FILLERS):
    if need <= 0:
      break
    take = min(qty, need)
    deck.append((code, take))
    need -= take
  assert need == 0, f"deck short by {need}"
  deck.append(("IKZ-001", 10))
  return deck


# earth-heavy fillers (AZK01-103 needs untapped earth entities in garden)
EARTH_FILLERS = [("AZK01-049", 4), ("AZK01-054", 4)] + FILLERS


DECK_GROUPS = {
    "A": build_deck(["STT02-005", "AZK01-116", "STT01-003", "AZK01-113",
                     "STT03-009"]),
    "B": build_deck(["STT01-012", "AZK01-036", "AZK01-047", "AZK01-060",
                     "STT03-013"]),
    "C": build_deck(["STT01-014", "AZK01-007", "AZK01-014", "AZK01-072",
                     "STT01-006"]),
    "D": build_deck(["AZK01-002", "STT02-014", "AZK01-127", "AZK01-040",
                     "AZK01-128"]),
    "E": build_deck(["STT01-017", "AZK01-042", "STT02-015", "STT03-016",
                     "AZK01-066"]),
    "F": build_deck(["STT01-007", "STT02-016", "AZK01-022", "AZK01-029",
                     "STT02-009"]),
    "G": build_deck(["STT03-004", "AZK01-105", "AZK01-011", "STT04-015",
                     "AZK01-065"]),
    "H": build_deck(["AZK01-068", "AZK01-070", "AZK01-058", "AZK01-008",
                     "STT03-011"]),
    "I": build_deck(["STT04-004", "AZK01-009", "AZK01-117", "STT02-011",
                     "STT01-005"]),
    "J": build_deck(["AZK01-103", "AZK01-032", "STT04-016", "STT04-017",
                     "STT04-014"], leader="STT04-001", gate="STT04-002",
                    fillers=EARTH_FILLERS),
    "K": build_deck(["AZK01-087", "AZK01-028", "STT01-013", "STT01-016",
                     "AZK01-020"]),
}

DRIVER_TYPES = {0, 1, 2, 6, 7, 8, 9, 11, 13, 14, 16, 25}


def comparable(rows):
  out = []
  for row in rows:
    if row[0] == 11 and row[1] == 5:  # leader activations not ported
      continue
    out.append(row)
  return out


def run_episode(make_cref, seed, deck, steps=500):
  import jax

  from azuki_jax.engine.step import engine_step, stabilize
  from azuki_jax.env import init_state_with_decks
  from azuki_jax.masks import build_mask
  from azuki_jax.setup import deck_tables_from_card_lists

  cref = make_cref(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck, deck)

  tables = deck_tables_from_card_lists(deck, deck)
  state = stabilize(init_state_with_decks(seed, tables))

  jit_step = jax.jit(engine_step)
  jit_mask = jax.jit(build_mask)

  rng = np.random.default_rng(seed)
  for step_index in range(steps):
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
        f"step {step_index} (phase {cview['phase']}): mask mismatch\n"
        f"C  ={c_rows}\nJAX={j_rows}"
    )

    if not c_rows:
      break  # both engines agree the game is stuck (cost-selection deadlock)

    driver_rows = [row for row in c_rows if row[0] in DRIVER_TYPES]
    assert driver_rows, f"step {step_index}: no driver actions\n{c_rows}"
    action = driver_rows[int(rng.integers(0, len(driver_rows)))]

    cref.step(np.asarray(action, np.int32))
    state = jit_step(state, np.asarray(action, np.int32))

    c_term, c_trunc = cref.dones()
    j_over = int(state.winner) != -1
    assert c_term == j_over, f"step {step_index}: terminal mismatch"
    if c_term or c_trunc:
      break

  assert int(state.ab_scratch[3]) == 0, "unimplemented ability hit"


@pytest.mark.parametrize("seed", [7, 1234])
@pytest.mark.parametrize("group", sorted(DECK_GROUPS))
def test_batch1_episode_equivalence(group, seed, make_cref):
  run_episode(make_cref, seed, DECK_GROUPS[group])
