"""L3 batch-2 ability ports: episode equivalence per deck group.

Same dual-engine pattern as test_l3_abilities_batch1 (identical seeds, random
driver over the C legal rows, semantic views + masks compared every step).
Groups:
  L: when-equipped — AZK01-039 (host gains Charge), STT01-015 (Tenraku +1/+1
     at 15+ discard) with mill support to grow the discard pile
  M: when-returned-to-hand — STT02-010 plus ported bouncers (AZK01-006 self-
     return, AZK01-022 / STT02-015 / AZK01-032)
  N: start-of-each-turn — STT04-003 (1 self-damage per turn)
  O: AZK01-044 lightning kanabo (hardcoded combat shock, once/turn/weapon)
     with AZK01-039 hosts (charge -> attack the turn the weapon lands)
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import cached_jit_mask, cached_static_jit_step

from test_l2_vanilla import c_semantic_view, jax_semantic_view
from test_l3_abilities_batch1 import DRIVER_TYPES, build_deck, comparable

DECK_GROUPS = {
    "L": build_deck(["AZK01-039", "STT01-015", "STT01-003", "STT01-012"]),
    "M": build_deck(["STT02-010", "AZK01-006", "AZK01-022", "STT02-015",
                     "AZK01-032"]),
    "N": build_deck(["STT04-003"]),
    "O": build_deck(["AZK01-044", "AZK01-039"]),
}


def _track_coverage(cov, state, cards, Zone):
  """Cheap per-step activation signals (report-only, no assertions)."""
  defs = np.asarray(state.def_id)
  zone = np.asarray(state.zone)

  is_039 = defs == cards.CODE_TO_ID["AZK01-039"]
  if bool((is_039 & np.asarray(state.grant_charge)).any()):
    cov["azk01_039_charge"] += 1

  is_015 = defs == cards.CODE_TO_ID["STT01-015"]
  attached = zone == int(Zone.ATTACHED)
  if bool((is_015 & attached & (np.asarray(state.cur_atk) >= 4)).any()):
    cov["stt01_015_bonus"] += 1

  is_443 = defs == cards.CODE_TO_ID["STT04-003"]
  in_play = (zone == int(Zone.GARDEN)) | (zone == int(Zone.ALLEY))
  hurt = in_play & (np.asarray(state.cur_hp) < 2)
  dead = zone == int(Zone.DISCARD)
  if bool((is_443 & (hurt | dead)).any()):
    cov["stt04_003_selfdmg"] += 1

  if bool((np.asarray(state.shocked_dur) != 0).any()):
    cov["shock_active"] += 1

  for p in (0, 1):
    cov["max_discard"] = max(
        cov["max_discard"], int((zone[p] == int(Zone.DISCARD)).sum())
    )

  if int(state.ab_phase) == 1:  # confirmation pending at this decision point
    owner = max(int(state.ab_owner), 0)
    src = max(int(state.ab_source), 0)
    if int(defs[owner, src]) == cards.CODE_TO_ID["STT02-010"]:
      cov["stt02_010_confirm"] += 1


def run_episode(make_cref, seed, deck, steps=500, cov=None):
  

  from azuki_jax import cards
  from azuki_jax.constants import Zone
  from azuki_jax.engine.step import stabilize
  from azuki_jax.env import init_state_with_decks
  
  from azuki_jax.setup import deck_tables_from_card_lists

  cref = make_cref(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck, deck)

  tables = deck_tables_from_card_lists(deck, deck)
  state = stabilize(init_state_with_decks(seed, tables))

  jit_step = cached_static_jit_step
  jit_mask = cached_jit_mask()

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

    if cov is not None:
      _track_coverage(cov, state, cards, Zone)

    if not c_rows:
      break  # both engines agree the game is stuck (cost-selection deadlock)

    driver_rows = [row for row in c_rows if row[0] in DRIVER_TYPES]
    assert driver_rows, f"step {step_index}: no driver actions\n{c_rows}"
    action = driver_rows[int(rng.integers(0, len(driver_rows)))]

    if cov is not None:
      if action[0] == 7:
        cov["attach"] += 1
      if action[0] == 16 and int(state.ab_phase) == 1:
        cov["confirm_taken"] += 1

    cref.step(np.asarray(action, np.int32))
    state = jit_step(state, np.asarray(action, np.int32))

    c_term, c_trunc = cref.dones()
    j_over = int(state.winner) != -1
    assert c_term == j_over, f"step {step_index}: terminal mismatch"
    if c_term or c_trunc:
      break

  assert int(state.ab_scratch[3]) == 0, "unimplemented ability hit"


SEEDS = {
    # seeds picked via a C-engine driver scan so the rare branches activate:
    # L: 1234/250 reach 15+ discard (Tenraku bonus; 7 = attach without bonus)
    # M: 42/421/777 hit STT02-010 confirm points with both accepts + declines
    # O: 7/42/888 attach AZK01-044 repeatedly (3-6 attaches per episode);
    #    77/1234 land an attach ON an AZK01-039 host (its charge trigger)
    "L": [7, 1234, 250],
    "M": [42, 421, 777],
    "N": [7, 1234],
    "O": [7, 42, 888, 77, 1234],
}

CASES = [(g, s) for g in sorted(DECK_GROUPS) for s in SEEDS[g]]


@pytest.mark.parametrize("group,seed", CASES)
def test_batch2_episode_equivalence(group, seed, make_cref):
  from collections import Counter

  cov = Counter(max_discard=0)
  run_episode(make_cref, seed, DECK_GROUPS[group], cov=cov)
  print(f"\n[coverage {group} seed {seed}] {dict(cov)}")
