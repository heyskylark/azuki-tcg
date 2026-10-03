"""L3 batch-3 ability ports: selection-zone flows, episode equivalence.

Same dual-engine pattern as batches 1/2, with the driver extended to the
selection-phase actions (18/19/20/21/22/23/24) and alley activations (12),
plus per-step comparison of the selection ZONE contents and the live
ability-context observation fields (phase, source, selection counters).

Groups (leader picked to exercise AZK01-015's element branches):
  P (lightning): reveal-to-hand — AZK01-003 (BlackJade), STT01-004 (weapon
     cost + weapon reveal), AZK01-033 (Steelborn), AZK01-069 (Beanz),
     STT04-005 (Pyreskin), AZK01-015 (lightning: self Charge)
  Q (water): AZK01-021 (Driftward), AZK01-031 (water, top-deck allowed),
     STT02-003 (Watercrafting), STT02-013 (water<=2, alley placement),
     AZK01-092 (water<=2, garden/alley/equip/hand), AZK01-015 (water: untap)
  R (fire): effect-hook cards — AZK01-015 (fire: 2 dmg to a leader),
     AZK01-016 (draw 2 discard 2), AZK01-017 (1 dmg per target class),
     AZK01-111 (sacrifice: 2 dmg + play from hand), STT03-006 (destroyed:
     draw 1 discard 1)
  S (earth): AZK01-045 (Obsidian), AZK01-056 (Scorchweaver), AZK01-015
     (earth: heal 2), STT03-006
  T (lightning): discard/hand-to-selection — AZK01-024 (bounce + replay),
     AZK01-041 (equip from discard), AZK01-084 (entity from discard),
     AZK01-086 (weapons from discard + leader buff), AZK01-097 (mill 5 pick
     weapon), AZK01-098 (tap: play weapon from hand equipped)
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import cached_jit_mask, cached_static_jit_step

from test_l2_vanilla import c_semantic_view, jax_semantic_view
from test_l3_abilities_batch1 import DRIVER_TYPES, build_deck, comparable

# selection-phase actions + ACTIVATE_ALLEY_ABILITY for AZK01-111
DRIVER_TYPES_3 = DRIVER_TYPES | {12, 18, 19, 20, 21, 22, 23, 24}

DECK_GROUPS = {
    "P": build_deck(
        ["AZK01-003", "STT01-004", "AZK01-033", "AZK01-069", "STT04-005",
         "AZK01-015"],
        fillers=[("STT01-003", 4), ("AZK01-001", 4), ("STT04-016", 4),
                 ("STT01-012", 4), ("STT01-010", 4), ("STT02-004", 4),
                 ("AZK01-094", 2)],
    ),
    "Q": build_deck(
        ["AZK01-021", "AZK01-031", "STT02-003", "STT02-013", "AZK01-092",
         "AZK01-015"],
        leader="STT02-001", gate="STT02-002",
        fillers=[("STT02-006", 4), ("STT02-008", 4), ("STT02-014", 4),
                 ("STT02-016", 4), ("AZK01-022", 4), ("STT02-005", 4),
                 ("AZK01-094", 2)],
    ),
    "R": build_deck(
        ["AZK01-015", "AZK01-016", "AZK01-017", "AZK01-111", "STT03-006"],
        leader="STT04-001", gate="STT04-002",
        fillers=[("STT02-004", 4), ("AZK01-001", 4), ("AZK01-068", 4),
                 ("AZK01-070", 4), ("STT01-007", 4), ("AZK01-037", 4),
                 ("AZK01-038", 4), ("AZK01-094", 2)],
    ),
    "S": build_deck(
        ["AZK01-045", "AZK01-056", "AZK01-015", "STT03-006"],
        leader="STT03-001", gate="STT03-002",
        fillers=[("STT03-013", 4), ("STT03-011", 4), ("AZK01-128", 4),
                 ("STT04-016", 4), ("STT04-015", 4), ("STT03-004", 4),
                 ("AZK01-049", 4), ("AZK01-054", 4), ("STT03-009", 2)],
    ),
    "T": build_deck(
        ["AZK01-024", "AZK01-041", "AZK01-084", "AZK01-086", "AZK01-097",
         "AZK01-098"],
        fillers=[("STT01-003", 4), ("STT01-012", 4), ("STT01-014", 4),
                 ("STT01-013", 4), ("STT02-004", 4), ("AZK01-001", 4),
                 ("AZK01-094", 2)],
    ),
}


def _selection_view_jax(state) -> dict:
  from azuki_jax.constants import Zone

  view = {}
  zone = np.asarray(state.zone)
  zpos = np.asarray(state.zpos)
  ids = np.asarray(state.def_id)
  # C obs rule: while the ctx holds a live selection (phase 4/5, count > 0)
  # BOTH players' selection blocks list ctx->selection.cards (holes = -1);
  # otherwise the player's selection zone in list order.
  use_ctx = int(state.ab_phase) in (4, 5) and int(state.ab_sel_count) > 0
  ctx_owner = max(int(state.ab_owner), 0)
  sel_cards = np.asarray(state.ab_sel_cards)
  for p in (0, 1):
    if use_ctx:
      view[f"selection{p}"] = [
          int(ids[ctx_owner][inst]) if inst >= 0 else -1
          for inst in sel_cards[: int(state.ab_sel_count)]
      ]
    else:
      in_sel = zone[p] == int(Zone.SELECTION)
      order = np.argsort(zpos[p][in_sel])
      view[f"selection{p}"] = [int(x) for x in ids[p][in_sel][order]]
  view["ab_phase"] = int(state.ab_phase)
  active = int(state.ab_phase) != 0
  owner = max(int(state.ab_owner), 0)
  src = max(int(state.ab_source), 0)
  src_def = int(np.asarray(state.def_id)[owner, src]) if active else -1
  view["ab_source_def"] = src_def if active else -1
  view["ab_sel_count"] = int(state.ab_sel_count) if active else 0
  view["ab_sel_picked"] = int(state.ab_sel_picked_count) if active else 0
  view["ab_sel_pick_max"] = int(state.ab_sel_pick_max) if active else 0
  return view


def _selection_view_c(cref) -> dict:
  view = {}
  for p in (0, 1):
    my = cref.raw(p).my_observation_data
    view[f"selection{p}"] = [
        int(my.selection[i].card_def_id)
        for i in range(int(my.selection_count))
    ]
  ctx = cref.raw(0).ability_context
  view["ab_phase"] = int(ctx.phase)
  view["ab_source_def"] = (
      int(ctx.source_card_def_id) if bool(ctx.has_source_card_def_id) else -1
  )
  view["ab_sel_count"] = int(ctx.selection_count)
  view["ab_sel_picked"] = int(ctx.selection_picked)
  view["ab_sel_pick_max"] = int(ctx.selection_pick_max)
  return view


def _track_coverage(cov, state, action, cards):
  defs = np.asarray(state.def_id)
  phase = int(state.ab_phase)
  if phase in (4, 5):
    owner = max(int(state.ab_owner), 0)
    src = max(int(state.ab_source), 0)
    code = cards.CARD_CODES[int(defs[owner, src])]
    cov[f"{'pick' if phase == 4 else 'bottom'}:{code}"] += 1
  if phase == 3:
    owner = max(int(state.ab_owner), 0)
    src = max(int(state.ab_source), 0)
    code = cards.CARD_CODES[int(defs[owner, src])]
    cov[f"effect:{code}"] += 1
  if action is not None and action[0] in (18, 19, 20, 21, 22, 23, 24):
    cov[f"act{action[0]}"] += 1


def run_episode(make_cref, seed, deck, steps=500, cov=None):
  

  from azuki_jax import cards
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
    c_rows = comparable(cref.legal_actions(active))
    legal, count, _ = jit_mask(state)
    j_rows = comparable(
        [tuple(int(x) for x in row) for row in np.asarray(legal)[: int(count)]]
    )
    assert j_rows == c_rows, (
        f"step {step_index} (phase {cview['phase']}, ab {cview['ab_phase']}):"
        f" mask mismatch\nC  ={c_rows}\nJAX={j_rows}"
    )

    if not c_rows:
      break  # both engines agree the game is stuck

    driver_rows = [row for row in c_rows if row[0] in DRIVER_TYPES_3]
    assert driver_rows, f"step {step_index}: no driver actions\n{c_rows}"
    action = driver_rows[int(rng.integers(0, len(driver_rows)))]

    if cov is not None:
      _track_coverage(cov, state, action, cards)

    cref.step(np.asarray(action, np.int32))
    state = jit_step(state, np.asarray(action, np.int32))

    c_term, c_trunc = cref.dones()
    j_over = int(state.winner) != -1
    assert c_term == j_over, f"step {step_index}: terminal mismatch"
    if c_term or c_trunc:
      break

  assert int(state.ab_scratch[3]) == 0, "unimplemented ability hit"


# Seeds picked via a C-engine driver scan (selection/effect phase entries per
# batch3 card + selection actions taken):
#   P: 99/888 run STT01-004's full confirm+cost+reveal+bottom flow
#   Q: act21/23/24 (alley/garden placements, top-deck) + all reveals
#   R: 42/2026 activate AZK01-111 (effect + to-garden); 7 covers 015-fire/
#      016/017/STT03-006 densely
#   T: 99 runs AZK01-024 placements + AZK01-041 equips; 2026 runs AZK01-098's
#      SELECT_TO_EQUIP; both run 084/086/097 selection flows
SEEDS = {
    "P": [7, 99, 888],
    "Q": [7, 1234],
    "R": [7, 42, 2026],
    "S": [7, 1234],
    "T": [99, 2026],
}

CASES = [(g, s) for g in sorted(DECK_GROUPS) for s in SEEDS[g]]


@pytest.mark.parametrize("group,seed", CASES)
def test_batch3_episode_equivalence(group, seed, make_cref):
  from collections import Counter

  cov = Counter()
  run_episode(make_cref, seed, DECK_GROUPS[group], cov=cov)
  print(f"\n[coverage {group} seed {seed}] {dict(cov)}")
