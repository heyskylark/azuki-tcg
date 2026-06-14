"""L2: composed gameplay equivalence on vanilla decks (no card abilities).

Drives the C engine and the JAX engine with identical action sequences chosen
from the C mask (restricted to ability-free action types) and compares the
full semantic state every step.
"""
from __future__ import annotations

import numpy as np
import pytest

# Vanilla deck: leader STT01-001 (AMain-only ability), gate STT01-002
# (portal-only ability), 50 ability-free mains, 10 IKZ.
VANILLA_DECK = [
    ("STT01-001", 1),
    ("STT01-002", 1),
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
    ("IKZ-001", 10),
]

# Action types both engines fully implement in the vanilla milestone.
# GATE_PORTAL(10) is excluded from the DRIVER (it would fire unimplemented
# gate abilities) but included in MASK comparison (its legality is core).
DRIVER_TYPES = {0, 1, 2, 6, 7, 9, 25}
MASK_COMPARE_TYPES = DRIVER_TYPES | {10}


def jax_semantic_view(state) -> dict:
  from azuki_jax.constants import Zone

  view = {}
  view["phase"] = int(state.phase)
  view["active"] = int(state.active_player)
  view["winner"] = int(state.winner)
  zone = np.asarray(state.zone)
  zpos = np.asarray(state.zpos)
  ids = np.asarray(state.def_id)
  atk = np.asarray(state.cur_atk)
  hp = np.asarray(state.cur_hp)
  tapped = np.asarray(state.tapped)
  cooldown = np.asarray(state.cooldown)
  for p in (0, 1):
    in_hand = zone[p] == int(Zone.HAND)
    order = np.argsort(zpos[p][in_hand])
    view[f"hand{p}"] = [int(x) for x in ids[p][in_hand][order]]
    in_deck = zone[p] == int(Zone.DECK)
    order = np.argsort(-zpos[p][in_deck])
    view[f"deck{p}"] = [int(x) for x in ids[p][in_deck][order]]
    in_discard = zone[p] == int(Zone.DISCARD)
    order = np.argsort(zpos[p][in_discard])
    view[f"discard{p}"] = [int(x) for x in ids[p][in_discard][order]]
    attached_to = np.asarray(state.attached_to)

    def weapon_ids(inst: int) -> tuple:
      mask = (zone[p] == int(Zone.ATTACHED)) & (attached_to[p] == inst)
      order = np.argsort(zpos[p][mask])
      return tuple(int(x) for x in ids[p][mask][order])

    for zone_name, z in (("garden", int(Zone.GARDEN)), ("alley", int(Zone.ALLEY))):
      slots = []
      for slot in range(5):
        match = np.flatnonzero((zone[p] == z) & (zpos[p] == slot))
        if len(match) == 0:
          slots.append(None)
        else:
          inst = int(match[0])
          slots.append(
              (
                  int(ids[p][inst]),
                  int(atk[p][inst]),
                  int(hp[p][inst]),
                  bool(tapped[p][inst]),
                  bool(cooldown[p][inst]),
                  weapon_ids(inst),
              )
          )
      view[f"{zone_name}{p}"] = slots
    leader = np.flatnonzero(zone[p] == int(Zone.LEADER))
    inst = int(leader[0])
    view[f"leader{p}"] = (
        int(ids[p][inst]), int(atk[p][inst]), int(hp[p][inst]),
        bool(tapped[p][inst]),
    )
    gate = np.flatnonzero(zone[p] == int(Zone.GATE))
    view[f"gate{p}"] = bool(tapped[p][int(gate[0])])
    in_area = zone[p] == int(Zone.IKZ_AREA)
    view[f"ikz_area{p}"] = (int(in_area.sum()), int((in_area & tapped[p]).sum()))
    view[f"ikz_pile{p}"] = int((zone[p] == int(Zone.IKZ_PILE)).sum())
    # C obs reports the READY token (exists && untapped)
    view[f"token{p}"] = int(
        (zone[p][62] == int(Zone.TOKEN)) and not tapped[p][62]
    )
  return view


def c_semantic_view(cref) -> dict:
  view = {}
  raw0 = cref.raw(0)
  view["phase"] = int(raw0.phase)
  view["active"] = cref.active_player
  # winner is not directly exposed; terminals + rewards capture it
  view["winner"] = -1
  for p in (0, 1):
    raw = cref.raw(p)
    my = raw.my_observation_data
    view[f"hand{p}"] = [
        int(my.hand[i].card_def_id) for i in range(int(my.hand_count))
    ]
    priv = raw.critic_privileged
    view[f"deck{p}"] = [
        int(priv.self_deck[i].card_def_id) for i in range(int(my.deck_count))
    ]
    view[f"discard{p}"] = [
        int(my.discard[i].card_def_id)
        for i in range(50)
        if int(my.discard[i].card_def_id) >= 0
    ]
    for zone_name, arr, size in (("garden", my.garden, 5), ("alley", my.alley, 5)):
      slots = []
      for slot in range(size):
        card = arr[slot]
        if int(card.card_def_id) < 0:
          slots.append(None)
        else:
          slots.append(
              (
                  int(card.card_def_id),
                  int(card.cur_stats.cur_atk),
                  int(card.cur_stats.cur_hp),
                  bool(card.tap_state.tapped),
                  bool(card.tap_state.cooldown),
                  tuple(
                      int(card.weapons[w].card_def_id)
                      for w in range(int(card.weapon_count))
                  ),
              )
          )
      view[f"{zone_name}{p}"] = slots
    leader = my.leader
    view[f"leader{p}"] = (
        int(leader.card_def_id),
        int(leader.cur_stats.cur_atk),
        int(leader.cur_stats.cur_hp),
        bool(leader.tap_state.tapped),
    )
    view[f"gate{p}"] = bool(my.gate.tap_state.tapped)
    area_total = 0
    area_tapped = 0
    for i in range(10):
      card = my.ikz_area[i]
      if int(card.card_def_id) >= 0:
        area_total += 1
        if card.tap_state.tapped:
          area_tapped += 1
    view[f"ikz_area{p}"] = (area_total, area_tapped)
    view[f"ikz_pile{p}"] = int(my.ikz_pile_count)
    view[f"token{p}"] = int(bool(my.has_ikz_token))
  return view


def filtered_mask(rows, types):
  return [row for row in rows if row[0] in types]


@pytest.mark.parametrize("seed", [11, 222, 3333])
def test_vanilla_episode_equivalence(seed, make_cref):
  import jax

  from azuki_jax.engine.step import engine_step, stabilize
  from azuki_jax.env import init_state_with_decks
  from azuki_jax.masks import build_mask
  from azuki_jax.setup import deck_tables_from_card_lists

  cref = make_cref(seed, deck_pool=None)
  cref.reset_with_decks(seed, VANILLA_DECK, VANILLA_DECK)

  tables = deck_tables_from_card_lists(VANILLA_DECK, VANILLA_DECK)
  state = init_state_with_decks(seed, tables)
  state = stabilize(state)

  jit_step = jax.jit(engine_step)
  jit_mask = jax.jit(build_mask)

  rng = np.random.default_rng(seed)
  for step_index in range(400):
    # --- compare semantic views ---
    cview = c_semantic_view(cref)
    jview = jax_semantic_view(state)
    for key, cval in cview.items():
      if key == "winner":
        continue
      assert jview[key] == cval, (
          f"step {step_index}: mismatch at {key}:\nC  ={cval}\nJAX={jview[key]}"
      )

    # --- compare masks on implemented action types ---
    active = cview["active"]
    c_rows = filtered_mask(cref.legal_actions(active), MASK_COMPARE_TYPES)
    legal, count, _ = jit_mask(state)
    legal = np.asarray(legal)[: int(count)]
    j_rows = filtered_mask([tuple(int(x) for x in row) for row in legal],
                           MASK_COMPARE_TYPES)
    assert j_rows == c_rows, (
        f"step {step_index} (phase {cview['phase']}, active {active}): mask"
        f" mismatch\nC  ={c_rows}\nJAX={j_rows}"
    )

    # --- choose an action among driver-supported types, step both ---
    driver_rows = filtered_mask(c_rows, DRIVER_TYPES)
    assert driver_rows, f"step {step_index}: no driver-supported actions"
    action = driver_rows[int(rng.integers(0, len(driver_rows)))]

    cref.step(np.asarray(action, np.int32))
    state = jit_step(state, np.asarray(action, np.int32))

    c_terminal, c_trunc = cref.dones()
    j_over = int(state.winner) != -1
    assert c_terminal == j_over, (
        f"step {step_index}: terminal mismatch C={c_terminal} JAX={j_over}"
    )
    if c_terminal or c_trunc:
      break
