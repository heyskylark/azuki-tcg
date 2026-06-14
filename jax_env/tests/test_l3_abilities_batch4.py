"""L3 batch-4 ability ports (FINAL tranche): episode equivalence per deck.

Same dual-engine pattern as batches 1-3 with:
- GATE_PORTAL (type 10) included in the driver, so the gate AOnGatePortal
  abilities (begun inline with the portal scratch) are exercised;
- NO comparable() filtering: every leader is now ported, so type-11 sub1==5
  rows must match exactly, as must portal-triggered ability rows.

Groups (one leader+gate pair each):
  W  STT04-001/STT04-002 + the takes-damage cards (AZK01-059/061/062 Pekiro
     redirect, STT04-007/009) over an effect-damage-heavy fire shell
  X  STT01-001/STT01-002 (charge grant; equip-from-discard portal) + weapons
  Y  STT02-001/STT02-002 (Shao response; IKZ untap portal) + STT02-017 bounce
     + STT02-012 latch verification deck (bounce/bottom-deck garden swings)
  Z  STT03-001/STT03-002 (Bobu latch + Defender-grant portal) + earth
     sacrifice/destroy shell
  V1 AZK01-119/AZK01-120 (discard-weapon buff; reequip portal)
  V2 AZK01-121/AZK01-122 (played-entities buff; play-from-hand portal)
  V3 AZK01-123/AZK01-124 (health buff; sacrifice-for-damage portal)
  V4 AZK01-125/AZK01-126 (cost reduction; spell-from-discard portal)
"""
from __future__ import annotations

import numpy as np
import pytest

from test_l2_vanilla import c_semantic_view, jax_semantic_view
from test_l3_abilities_batch1 import DRIVER_TYPES, build_deck
from test_l3_abilities_batch3 import _selection_view_c, _selection_view_jax

# + GATE_PORTAL and the ability/selection sub-phase actions
DRIVER_TYPES_4 = DRIVER_TYPES | {10, 12, 18, 19, 20, 21, 22, 23, 24}

DECK_GROUPS = {
    "W": build_deck(
        ["AZK01-059", "AZK01-061", "AZK01-062", "STT04-007", "STT04-009"],
        leader="STT04-001", gate="STT04-002",
        fillers=[("STT04-016", 4), ("STT04-015", 4), ("STT01-014", 4),
                 ("AZK01-005", 4), ("STT04-003", 4), ("STT04-004", 4),
                 ("AZK01-001", 4), ("STT02-004", 4)],
    ),
    "X": build_deck(
        [],
        leader="STT01-001", gate="STT01-002",
        fillers=[("STT01-012", 4), ("STT01-013", 4), ("STT01-016", 4),
                 ("AZK01-094", 2), ("STT01-003", 4), ("STT01-004", 4),
                 ("AZK01-097", 4), ("STT01-006", 4), ("STT01-014", 4),
                 ("STT01-007", 4), ("STT01-010", 4), ("AZK01-001", 4),
                 ("STT02-004", 4)],
    ),
    "Y": build_deck(
        ["STT02-017", "STT02-012", "AZK01-087", "STT02-015", "STT02-009"],
        leader="STT02-001", gate="STT02-002",
        fillers=[("STT02-005", 4), ("STT02-006", 4), ("STT02-008", 4),
                 ("STT02-007", 4), ("STT02-014", 4), ("STT02-016", 4),
                 ("STT02-004", 4), ("AZK01-001", 2)],
    ),
    "Z": build_deck(
        [],
        leader="STT03-001", gate="STT03-002",
        fillers=[("STT03-004", 4), ("STT03-013", 4), ("STT03-011", 4),
                 ("STT03-016", 4), ("STT03-009", 4), ("STT03-006", 4),
                 ("AZK01-105", 4), ("STT04-016", 4), ("AZK01-049", 4),
                 ("AZK01-054", 4), ("AZK01-128", 4), ("AZK01-002", 4),
                 ("STT02-004", 2)],
    ),
    "V1": build_deck(
        [],
        leader="AZK01-119", gate="AZK01-120",
        fillers=[("STT01-012", 4), ("STT01-013", 4), ("STT01-016", 4),
                 ("AZK01-094", 2), ("AZK01-041", 4), ("AZK01-086", 4),
                 ("AZK01-097", 4), ("AZK01-098", 4), ("AZK01-039", 4),
                 ("STT01-003", 4), ("AZK01-004", 4), ("AZK01-001", 4),
                 ("STT01-010", 4)],
    ),
    "V2": build_deck(
        [],
        leader="AZK01-121", gate="AZK01-122",
        fillers=[("AZK01-001", 4), ("AZK01-012", 4), ("AZK01-025", 4),
                 ("AZK01-035", 4), ("AZK01-037", 4), ("AZK01-038", 4),
                 ("STT02-004", 4), ("STT01-010", 4), ("AZK01-004", 4),
                 ("AZK01-005", 4), ("AZK01-007", 4), ("AZK01-113", 4),
                 ("AZK01-049", 2)],
    ),
    "V3": build_deck(
        [],
        leader="AZK01-123", gate="AZK01-124",
        fillers=[("AZK01-001", 4), ("AZK01-012", 4), ("AZK01-025", 4),
                 ("AZK01-035", 4), ("AZK01-049", 4), ("AZK01-054", 4),
                 ("STT02-004", 4), ("STT01-010", 4), ("AZK01-007", 4),
                 ("STT03-013", 4), ("AZK01-009", 4), ("STT01-014", 4),
                 ("AZK01-005", 2)],
    ),
    "V4": build_deck(
        [],
        leader="AZK01-125", gate="AZK01-126",
        fillers=[("STT01-007", 4), ("STT02-016", 4), ("AZK01-016", 4),
                 ("AZK01-068", 4), ("AZK01-029", 4), ("AZK01-002", 4),
                 ("STT01-014", 4), ("AZK01-005", 4), ("AZK01-007", 4),
                 ("AZK01-001", 4), ("STT02-004", 4), ("STT01-010", 4),
                 ("AZK01-012", 2)],
    ),
}


def _track_coverage(cov, state, cview, action, cards):
  defs = np.asarray(state.def_id)
  phase = int(state.ab_phase)
  if phase != 0:
    owner = max(int(state.ab_owner), 0)
    src = max(int(state.ab_source), 0)
    code = cards.CARD_CODES[int(defs[owner, src])]
    cov[f"ab{phase}:{code}"] += 1
  if action is not None and action[0] in (10, 11, 12):
    cov[f"act{action[0]}"] += 1


def run_episode(make_cref, seed, deck, steps=500, cov=None):
  import jax

  from azuki_jax import cards
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
    c_rows = cref.legal_actions(active)  # NO filtering: leaders are ported
    legal, count, _ = jit_mask(state)
    j_rows = [tuple(int(x) for x in row) for row in np.asarray(legal)[: int(count)]]
    assert j_rows == c_rows, (
        f"step {step_index} (phase {cview['phase']}, ab {cview['ab_phase']}):"
        f" mask mismatch\nC  ={c_rows}\nJAX={j_rows}"
    )

    if not c_rows:
      break  # both engines agree the game is stuck

    driver_rows = [row for row in c_rows if row[0] in DRIVER_TYPES_4]
    assert driver_rows, f"step {step_index}: no driver actions\n{c_rows}"
    action = driver_rows[int(rng.integers(0, len(driver_rows)))]

    if cov is not None:
      _track_coverage(cov, state, cview, action, cards)

    cref.step(np.asarray(action, np.int32))
    state = jit_step(state, np.asarray(action, np.int32))

    c_term, c_trunc = cref.dones()
    j_over = int(state.winner) != -1
    assert c_term == j_over, f"step {step_index}: terminal mismatch"
    if c_term or c_trunc:
      break

  assert int(state.ab_scratch[3]) == 0, "unimplemented ability hit"


# Seeds picked via a C-engine driver scan (jax_env/tests/scan_batch4_seeds.py):
#   W: 7/777 run the Pekiro redirect (ab3:AZK01-062) + STT04-009 confirm/
#      effect; 42 adds dense AZK01-059/STT04-002 coverage
#   X: both seeds activate STT01-001 (ab3) and run STT01-002's equip flow
#   Y: 2026 takes 36 portals (Hydromancy) + 18 Shao activations
#   Z: STT03-002 defender grants every seed; Bobu via (11,5) activations
#   V1: AZK01-120 reequip picks (act22) + AZK01-119 buffs
#   V2: 4242 runs 5 AZK01-122 selection flows
#   V3/V4: default seeds already cover 123/124 (confirm+cost+effect) and 126
SEEDS = {
    "W": [7, 42, 777],
    "X": [42, 13],
    "Y": [7, 2026],
    "Z": [42, 31337],
    "V1": [13, 321],
    "V2": [99, 4242],
    "V3": [7, 1234],
    "V4": [7, 1234],
}

CASES = [(g, s) for g in sorted(DECK_GROUPS) for s in SEEDS[g]]


@pytest.mark.parametrize("group,seed", CASES)
def test_batch4_episode_equivalence(group, seed, make_cref):
  from collections import Counter

  cov = Counter()
  run_episode(make_cref, seed, DECK_GROUPS[group], cov=cov)
  print(f"\n[coverage {group} seed {seed}] {dict(cov)}")
