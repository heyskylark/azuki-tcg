"""Selection-zone runtime: ABILITY_PHASE_SELECTION_PICK / BOTTOM_DECK.

Mirrors src/abilities/selection/{ability_selection,ability_selection_helpers}.c,
src/abilities/cards/common/reveal_selection.c, ability_system.c process_* and
deck_utils look_at_top_n_cards / add_card_to_bottom_of_deck.

Conventions: ab_sel_cards holds instance ids (-1 = empty/consumed slot) in the
C ctx->selection.cards order (reveal order = deck top first). Cards physically
sit in zone SELECTION with zpos = reveal order. The selection always belongs
to ab_owner.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from azuki_jax import cards
from azuki_jax.abilities import tables
from azuki_jax.constants import (
    GARDEN_SIZE,
    MAX_ABILITY_SELECTION,
    MAX_SELECTION_ZONE_SIZE,
    AbilityPhase,
    CardType,
    Zone,
)
from azuki_jax.state import State
from azuki_jax.zones import zone_count


def _np(table):
  return jnp.asarray(table)


def _where_state(pred, a: State, b: State) -> State:
  return jax.tree.map(lambda x, y: jnp.where(pred, x, y), a, b)


def _owner_src(state: State):
  return (
      jnp.maximum(state.ab_owner.astype(jnp.int32), 0),
      jnp.maximum(state.ab_source.astype(jnp.int32), 0),
  )


# ---------------------------------------------------------------------------
# Setup: reveal top N of deck into the selection zone
# ---------------------------------------------------------------------------

def reveal_top_into_selection(state: State, reveal_count, pick_max, do) -> State:
  """look_at_top_n_cards + azk_init_selection_state (phase set separately by
  callers via matching count). Moves up to reveal_count cards from the deck
  top into zone SELECTION (zpos = reveal order, top first) and records them
  in ab_sel_cards. reveal_count is bounded by 8 unrolled moves (max used by
  any card is 5)."""
  do = jnp.asarray(do)
  owner, _ = _owner_src(state)
  deck_count = zone_count(state.zone[owner], Zone.DECK)
  n_take = jnp.minimum(jnp.asarray(reveal_count, jnp.int32), deck_count)
  # append after any cards already in the zone (stranded by the CLEAR-mode
  # quirk); C reparents as ordered children = append (deck_utils.c:288)
  sel_base = zone_count(state.zone[owner], Zone.SELECTION)

  sel_cards = jnp.full((MAX_SELECTION_ZONE_SIZE,), -1, jnp.int8)
  zone_row = state.zone[owner]
  zpos_row = state.zpos[owner]
  for i in range(8):
    take = do & (i < n_take)
    # current deck top
    in_deck = zone_row == Zone.DECK
    key = jnp.where(in_deck, zpos_row.astype(jnp.int32), -1)
    inst = jnp.argmax(key)
    has = in_deck.any()
    take = take & has
    zone_row = jnp.where(take & (jnp.arange(zone_row.shape[0]) == inst),
                         jnp.int8(Zone.SELECTION), zone_row)
    zpos_row = jnp.where(take & (jnp.arange(zpos_row.shape[0]) == inst),
                         (sel_base + i).astype(jnp.int8), zpos_row)
    sel_cards = sel_cards.at[i].set(
        jnp.where(take, inst.astype(jnp.int8), sel_cards[i])
    )

  state = state._replace(
      zone=state.zone.at[owner].set(jnp.where(do, zone_row, state.zone[owner])),
      zpos=state.zpos.at[owner].set(jnp.where(do, zpos_row, state.zpos[owner])),
      ab_sel_cards=jnp.where(do, sel_cards, state.ab_sel_cards),
      ab_sel_count=jnp.where(do, n_take, state.ab_sel_count).astype(jnp.int8),
      ab_sel_picked=jnp.where(
          do, jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8), state.ab_sel_picked
      ),
      ab_sel_picked_count=jnp.where(do, 0, state.ab_sel_picked_count).astype(jnp.int8),
      ab_sel_pick_max=jnp.where(do, jnp.asarray(pick_max, jnp.int8), state.ab_sel_pick_max),
  )
  return state


# Selection processing bound for move-matching flows (discard/hand -> selection)
# and return-remaining. C caps at MAX_SELECTION_ZONE_SIZE (the whole discard
# pile can be surfaced, e.g. AZK01-126 returns any eligible discard spell), and
# the action heads (subaction = MAX_DECK_SIZE) address every index, so match it.
# reveal_top_into_selection / process_bottom_deck_all stay bounded by their own
# reveal_count (<=5).
MAX_MOVED = MAX_SELECTION_ZONE_SIZE


def move_matching_zone_to_selection(state: State, src_zone, match_row,
                                    pick_max, do) -> State:
  """azk_move_matching_hand_cards_to_selection (and the AZK01-041/084/086
  discard movers): move every card of ab_owner matching `match_row` (a
  (NUM_INSTANCES,) bool over the owner's row) from list zone `src_zone` into
  SELECTION (zone order preserved), init the selection state, and set
  phase = SELECTION_PICK if any moved else NONE. Bounded at MAX_MOVED."""
  do = jnp.asarray(do)
  owner, _ = _owner_src(state)
  n = state.zone.shape[1]
  idx = jnp.arange(n)
  in_src = state.zone[owner] == src_zone
  match = in_src & match_row
  zpos = state.zpos[owner].astype(jnp.int32)

  # rank within matches by zpos (= flecs ordered-children iteration order)
  rank = jnp.sum((zpos[None, :] < zpos[:, None]) & match[None, :], axis=1)
  take = match & (rank < MAX_MOVED)
  count = jnp.sum(take, dtype=jnp.int32)

  sel_base = zone_count(state.zone[owner], Zone.SELECTION)
  removed_below = jnp.sum((zpos[None, :] < zpos[:, None]) & take[None, :], axis=1)
  new_zpos = jnp.where(in_src & ~take, zpos - removed_below, zpos)
  new_zpos = jnp.where(take, sel_base + rank, new_zpos)
  new_zone = jnp.where(take, jnp.int8(Zone.SELECTION), state.zone[owner])

  sel_cards = jnp.full((MAX_SELECTION_ZONE_SIZE,), -1, jnp.int8)
  sel_cards = sel_cards.at[jnp.where(take, rank, MAX_SELECTION_ZONE_SIZE)].set(
      idx.astype(jnp.int8), mode="drop"
  )

  phase = jnp.where(
      count > 0, jnp.int8(AbilityPhase.SELECTION_PICK), jnp.int8(AbilityPhase.NONE)
  )
  return state._replace(
      zone=state.zone.at[owner].set(jnp.where(do, new_zone, state.zone[owner])),
      zpos=state.zpos.at[owner].set(
          jnp.where(do, new_zpos.astype(jnp.int8), state.zpos[owner])
      ),
      ab_sel_cards=jnp.where(do, sel_cards, state.ab_sel_cards),
      ab_sel_count=jnp.where(do, count, state.ab_sel_count).astype(jnp.int8),
      ab_sel_picked=jnp.where(
          do, jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8), state.ab_sel_picked
      ),
      ab_sel_picked_count=jnp.where(do, 0, state.ab_sel_picked_count).astype(jnp.int8),
      ab_sel_pick_max=jnp.where(
          do, jnp.asarray(pick_max, jnp.int32), state.ab_sel_pick_max
      ).astype(jnp.int8),
      ab_phase=jnp.where(do, phase, state.ab_phase),
  )


def begin_bottom_deck_for_remaining(state: State, do) -> State:
  """azk_begin_bottom_deck_for_remaining_selection: phase = BOTTOM_DECK iff
  any selection card remains (phase untouched otherwise)."""
  remaining = remaining_count(state) > 0
  return state._replace(
      ab_phase=jnp.where(
          jnp.asarray(do) & remaining,
          jnp.int8(AbilityPhase.BOTTOM_DECK),
          state.ab_phase,
      )
  )


def return_remaining_to_hand(state: State, do) -> State:
  """azk_return_remaining_selection_cards_to_hand: each remaining ab_sel_cards
  entry still in the selection zone is appended to hand; entries zeroed."""
  do = jnp.asarray(do)
  owner, _ = _owner_src(state)
  for k in range(MAX_MOVED):
    inst = state.ab_sel_cards[k]
    safe = jnp.maximum(inst.astype(jnp.int32), 0)
    has = do & (inst >= 0)
    in_sel = state.zone[owner, safe] == Zone.SELECTION
    state = _to_hand(state, owner, safe, has & in_sel)
    state = state._replace(
        ab_sel_cards=state.ab_sel_cards.at[k].set(
            jnp.where(has, jnp.int8(-1), state.ab_sel_cards[k])
        )
    )
  return state


def _to_discard_raw(state: State, owner, inst, do) -> State:
  """Selection -> discard append (raw ecs reparent in C: NO stat reset, no
  discard counters, no triggers)."""
  state = _selection_remove(state, owner, inst, do)
  count = zone_count(state.zone[owner], Zone.DISCARD)
  zone_row = state.zone[owner].at[inst].set(
      jnp.where(do, jnp.int8(Zone.DISCARD), state.zone[owner][inst])
  )
  zpos_row = state.zpos[owner].at[inst].set(
      jnp.where(do, count.astype(jnp.int8), state.zpos[owner][inst])
  )
  return state._replace(
      zone=state.zone.at[owner].set(zone_row),
      zpos=state.zpos.at[owner].set(zpos_row),
  )


def return_remaining_to_discard(state: State, do) -> State:
  """azk_return_remaining_selection_cards_to_discard (entries are in-selection
  by construction; guarded here for safety)."""
  do = jnp.asarray(do)
  owner, _ = _owner_src(state)
  for k in range(MAX_MOVED):
    inst = state.ab_sel_cards[k]
    safe = jnp.maximum(inst.astype(jnp.int32), 0)
    has = do & (inst >= 0)
    in_sel = state.zone[owner, safe] == Zone.SELECTION
    state = _to_discard_raw(state, owner, safe, has & in_sel)
    state = state._replace(
        ab_sel_cards=state.ab_sel_cards.at[k].set(
            jnp.where(has, jnp.int8(-1), state.ab_sel_cards[k])
        )
    )
  return state


def selection_matching_count(state: State) -> jax.Array:
  """Count of selection cards passing the card's selection-target validator."""
  from azuki_jax.abilities import cards_impl

  owner, src = _owner_src(state)
  def_id = state.def_id[owner, src]

  def check(k):
    inst = state.ab_sel_cards[k]
    ok = inst >= 0
    return ok & cards_impl.selection_target_validator(
        state, def_id, owner, jnp.maximum(inst.astype(jnp.int32), 0)
    )

  return jnp.sum(jax.vmap(check)(jnp.arange(MAX_SELECTION_ZONE_SIZE)), dtype=jnp.int32)


def enter_selection_phase(state: State, do) -> State:
  """Phase = SELECTION_PICK if any matching else BOTTOM_DECK (reveal helper)."""
  matching = selection_matching_count(state)
  phase = jnp.where(
      matching > 0,
      jnp.int8(AbilityPhase.SELECTION_PICK),
      jnp.int8(AbilityPhase.BOTTOM_DECK),
  )
  return state._replace(
      ab_phase=jnp.where(jnp.asarray(do), phase, state.ab_phase)
  )


# ---------------------------------------------------------------------------
# Internal: deck placement + hand move for selection cards
# ---------------------------------------------------------------------------

def _selection_remove(state: State, owner, inst, do) -> State:
  """Remove inst from the SELECTION list zone (compact zpos)."""
  from azuki_jax.zones import remove_from_list_zone

  zone_row, zpos_row = state.zone[owner], state.zpos[owner]
  nzr, npr = remove_from_list_zone(zone_row, zpos_row, inst)
  return state._replace(
      zone=state.zone.at[owner].set(jnp.where(do, nzr, zone_row)),
      zpos=state.zpos.at[owner].set(jnp.where(do, npr, zpos_row)),
  )


def _to_deck_bottom(state: State, owner, inst, do) -> State:
  """add_card_to_bottom_of_deck: deck zpos += 1 (all), card zpos = 0."""
  state = _selection_remove(state, owner, inst, do)
  in_deck = state.zone[owner] == Zone.DECK
  zpos_row = jnp.where(do & in_deck, state.zpos[owner] + 1, state.zpos[owner])
  zone_row = state.zone[owner].at[inst].set(
      jnp.where(do, jnp.int8(Zone.DECK), state.zone[owner][inst])
  )
  zpos_row = zpos_row.at[inst].set(jnp.where(do, 0, zpos_row[inst]).astype(jnp.int8))
  return state._replace(
      zone=state.zone.at[owner].set(zone_row),
      zpos=state.zpos.at[owner].set(zpos_row),
  )


def _to_deck_top(state: State, owner, inst, do) -> State:
  state = _selection_remove(state, owner, inst, do)
  count = zone_count(state.zone[owner], Zone.DECK)
  zone_row = state.zone[owner].at[inst].set(
      jnp.where(do, jnp.int8(Zone.DECK), state.zone[owner][inst])
  )
  zpos_row = state.zpos[owner].at[inst].set(
      jnp.where(do, count.astype(jnp.int8), state.zpos[owner][inst])
  )
  return state._replace(
      zone=state.zone.at[owner].set(zone_row),
      zpos=state.zpos.at[owner].set(zpos_row),
  )


def _to_hand(state: State, owner, inst, do) -> State:
  state = _selection_remove(state, owner, inst, do)
  count = zone_count(state.zone[owner], Zone.HAND)
  zone_row = state.zone[owner].at[inst].set(
      jnp.where(do, jnp.int8(Zone.HAND), state.zone[owner][inst])
  )
  zpos_row = state.zpos[owner].at[inst].set(
      jnp.where(do, count.astype(jnp.int8), state.zpos[owner][inst])
  )
  return state._replace(
      zone=state.zone.at[owner].set(zone_row),
      zpos=state.zpos.at[owner].set(zpos_row),
  )


def move_picked_to_hand(state: State, do) -> State:
  """azk_move_picked_selection_cards_to_hand (still-in-selection variant is
  equivalent here since picks only leave via these helpers)."""
  owner, _ = _owner_src(state)
  for k in range(MAX_ABILITY_SELECTION):
    inst = state.ab_sel_picked[k]
    ok = jnp.asarray(do) & (inst >= 0) & (
        state.zone[owner, jnp.maximum(inst.astype(jnp.int32), 0)] == Zone.SELECTION
    )
    state = _to_hand(state, owner, jnp.maximum(inst.astype(jnp.int32), 0), ok)
  return state


def remaining_count(state: State) -> jax.Array:
  return jnp.sum(state.ab_sel_cards >= 0, dtype=jnp.int32)


def _bounce_pick_to_hand(state: State, owner, inst, do) -> State:
  """C deferred-ops quirk (ability_resolution runs in the readonly stage):
  when a TO_GARDEN/TO_ALLEY/TO_EQUIP pick completes the selection, the
  placement reparent is still deferred while on_selection_complete's
  *_if_still_in_selection check runs — the committed parent is still the
  selection zone, so the hook re-moves the picked card to hand and the hand
  reparent wins at flush. The placement's side effects (counters, queued
  triggers, displaced-card discard, tap writes, equip attack bonus) all
  stand. Cards flagged selection_complete_if_still get this bounce."""
  do = jnp.asarray(do)
  count = zone_count(state.zone[owner], Zone.HAND)
  return state._replace(
      zone=state.zone.at[owner, inst].set(
          jnp.where(do, jnp.int8(Zone.HAND), state.zone[owner, inst])
      ),
      zpos=state.zpos.at[owner, inst].set(
          jnp.where(do, count.astype(jnp.int8), state.zpos[owner, inst])
      ),
      attached_to=state.attached_to.at[owner, inst].set(
          jnp.where(do, jnp.int8(-1), state.attached_to[owner, inst])
      ),
  )


def _record_pick(state: State, sel_idx, inst, do) -> State:
  pk = jnp.clip(state.ab_sel_picked_count.astype(jnp.int32), 0,
                MAX_ABILITY_SELECTION - 1)
  return state._replace(
      ab_sel_picked=state.ab_sel_picked.at[pk].set(
          jnp.where(do, inst.astype(jnp.int8), state.ab_sel_picked[pk])
      ),
      ab_sel_picked_count=jnp.where(
          do, state.ab_sel_picked_count + 1, state.ab_sel_picked_count
      ).astype(jnp.int8),
      ab_sel_cards=state.ab_sel_cards.at[sel_idx].set(
          jnp.where(do, jnp.int8(-1), state.ab_sel_cards[sel_idx])
      ),
  )


# ---------------------------------------------------------------------------
# Finish (azk_finish_selection_resolution)
# ---------------------------------------------------------------------------

def finish_selection(state: State, do) -> State:
  from azuki_jax.abilities import cards_impl, runtime

  do = jnp.asarray(do)
  owner, src = _owner_src(state)
  def_id = state.def_id[owner, src]

  hooked = cards_impl.dispatch_on_selection_complete(state)
  state = _where_state(do, hooked, state)

  clear_mode = jnp.where(
      def_id >= 0, _np(tables.IMPLEMENTED)[jnp.maximum(def_id, 0)]
      & _np(_CLEAR_IF_ACTIVE)[jnp.maximum(def_id, 0)], False
  )

  # ALLOW_BOTTOM_DECK mode
  allow = do & ~clear_mode
  not_terminal = (state.ab_phase != AbilityPhase.BOTTOM_DECK) & (
      state.ab_phase != AbilityPhase.NONE
  )
  remaining = remaining_count(state)
  to_bottom = allow & not_terminal & (remaining > 0)
  done_now = allow & not_terminal & (remaining == 0)
  state = state._replace(
      ab_phase=jnp.where(
          to_bottom, jnp.int8(AbilityPhase.BOTTOM_DECK), state.ab_phase
      )
  )

  # CLEAR_IF_STILL_ACTIVE mode
  clear_now = (do & clear_mode & (state.ab_phase != AbilityPhase.NONE)) | done_now
  cleared = runtime._clear_context(state)
  return _where_state(clear_now, cleared, state)


_CLEAR_IF_ACTIVE = np.zeros(cards.CARD_DEF_COUNT, np.bool_)
from azuki_jax.abilities import registry_generated as _rg  # noqa: E402

for _entry in _rg.ENTRIES:
  _CLEAR_IF_ACTIVE[
      cards.CODE_TO_ID[_entry["code"].replace("_", "-")]
  ] = _entry["clear_selection_if_still_active"]


# ---------------------------------------------------------------------------
# User-action processors
# ---------------------------------------------------------------------------

def _pick_common_ok(state: State, sel_idx):
  from azuki_jax.abilities import cards_impl

  owner, src = _owner_src(state)
  def_id = state.def_id[owner, src]
  idx = jnp.clip(sel_idx, 0, MAX_SELECTION_ZONE_SIZE - 1)
  in_range = (sel_idx >= 0) & (sel_idx < state.ab_sel_count)
  inst = state.ab_sel_cards[idx]
  has = inst >= 0
  target = jnp.maximum(inst.astype(jnp.int32), 0)
  validator_ok = cards_impl.selection_target_validator(state, def_id, owner, target)
  return owner, def_id, idx, target, in_range & has & validator_ok


def process_selection_pick(state: State, sel_idx, do) -> State:
  """ACT_SELECT_FROM_SELECTION: record pick (move-to-hand happens in the
  card's on_selection_complete)."""
  do = jnp.asarray(do) & (state.ab_phase == AbilityPhase.SELECTION_PICK)
  owner, def_id, idx, target, ok = _pick_common_ok(state, sel_idx)
  safe_def = jnp.maximum(def_id, 0)
  special = (
      _np(tables.SEL_TO_GARDEN)[safe_def]
      | _np(tables.SEL_TO_ALLEY)[safe_def]
      | _np(tables.SEL_TO_EQUIP)[safe_def]
  )
  to_hand_ok = ~special | _np(tables.SEL_TO_HAND)[safe_def]
  do = do & ok & to_hand_ok

  state = _record_pick(state, idx, target, do)
  finish = do & (state.ab_sel_picked_count >= state.ab_sel_pick_max)
  return finish_selection(state, finish)


def process_selection_to_garden(state: State, sel_idx, slot, do) -> State:
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.triggers import queue_enter_garden, queue_on_play

  do = jnp.asarray(do) & (state.ab_phase == AbilityPhase.SELECTION_PICK)
  owner, def_id, idx, target, ok = _pick_common_ok(state, sel_idx)
  safe_def = jnp.maximum(def_id, 0)
  do = do & ok & _np(tables.SEL_TO_GARDEN)[safe_def]
  target_type = jnp.where(
      state.def_id[owner, target] >= 0,
      _np(cards.TYPE)[jnp.maximum(state.def_id[owner, target], 0)],
      -1,
  )
  do = do & (target_type == CardType.ENTITY) & (slot >= 0) & (slot < GARDEN_SIZE)

  # slot legality: empty, or full garden (forced replacement)
  occupied = jnp.any(
      (state.zone[owner] == Zone.GARDEN)
      & (state.zpos[owner] == jnp.asarray(slot, jnp.int8))
  )
  full = zone_count(state.zone[owner], Zone.GARDEN) >= GARDEN_SIZE
  do = do & (~occupied | full)

  state = _enter_board_slot(state, owner, target, Zone.GARDEN, slot, do)
  state = state._replace(
      entities_played_garden_turn=state.entities_played_garden_turn.at[owner].add(
          do.astype(jnp.uint8)
      ),
      cards_played_turn=state.cards_played_turn.at[owner].add(do.astype(jnp.uint8)),
      next_play_cost_reduction=state.next_play_cost_reduction.at[owner].set(
          jnp.where(do, 0, state.next_play_cost_reduction[owner])
      ),
  )
  state = queue_enter_garden(state, owner, target, do=do)
  state = queue_on_play(state, owner, target, do=do)

  state = _record_pick(state, idx, target, do)
  finish = do & (state.ab_sel_picked_count >= state.ab_sel_pick_max)
  from azuki_jax.abilities import cards_impl

  bounce = finish & _np(cards_impl.SEL_COMPLETE_IF_STILL)[safe_def]
  state = finish_selection(state, finish)
  return _bounce_pick_to_hand(state, owner, target, bounce)


def process_selection_to_alley(state: State, sel_idx, slot, do) -> State:
  from azuki_jax.engine.helpers import discard
  from azuki_jax.engine.triggers import queue_on_play

  do = jnp.asarray(do) & (state.ab_phase == AbilityPhase.SELECTION_PICK)
  owner, def_id, idx, target, ok = _pick_common_ok(state, sel_idx)
  safe_def = jnp.maximum(def_id, 0)
  do = do & ok & _np(tables.SEL_TO_ALLEY)[safe_def]
  target_type = jnp.where(
      state.def_id[owner, target] >= 0,
      _np(cards.TYPE)[jnp.maximum(state.def_id[owner, target], 0)],
      -1,
  )
  do = do & (target_type == CardType.ENTITY) & (slot >= 0) & (slot < GARDEN_SIZE)

  occupied_inst = jnp.where(
      ((state.zone[owner] == Zone.ALLEY)
       & (state.zpos[owner] == jnp.asarray(slot, jnp.int8))).any(),
      jnp.argmax(
          (state.zone[owner] == Zone.ALLEY)
          & (state.zpos[owner] == jnp.asarray(slot, jnp.int8))
      ),
      -1,
  )
  full = zone_count(state.zone[owner], Zone.ALLEY) >= GARDEN_SIZE
  do = do & ((occupied_inst < 0) | full)
  state = discard(
      state, owner, jnp.maximum(occupied_inst, 0),
      reason_replacement=True, ignore_godmode=True,
      do=do & (occupied_inst >= 0) & full,
  )

  # move: selection -> alley slot; tap reset; board_seq stamp
  state = _selection_remove(state, owner, target, do)
  state = state._replace(
      zone=state.zone.at[owner, target].set(
          jnp.where(do, jnp.int8(Zone.ALLEY), state.zone[owner, target])
      ),
      zpos=state.zpos.at[owner, target].set(
          jnp.where(do, jnp.asarray(slot, jnp.int8), state.zpos[owner, target])
      ),
      board_seq=state.board_seq.at[owner, target].set(
          jnp.where(do, state.seq_counter, state.board_seq[owner, target])
      ),
      seq_counter=(state.seq_counter + jnp.where(do, 1, 0)).astype(jnp.int16),
      tapped=state.tapped.at[owner, target].set(
          jnp.where(do, False, state.tapped[owner, target])
      ),
      cooldown=state.cooldown.at[owner, target].set(
          jnp.where(do, 0, state.cooldown[owner, target])
      ),
      entities_played_alley_turn=state.entities_played_alley_turn.at[owner].add(
          do.astype(jnp.uint8)
      ),
      cards_played_turn=state.cards_played_turn.at[owner].add(do.astype(jnp.uint8)),
      next_play_cost_reduction=state.next_play_cost_reduction.at[owner].set(
          jnp.where(do, 0, state.next_play_cost_reduction[owner])
      ),
  )
  state = queue_on_play(state, owner, target, do=do)

  state = _record_pick(state, idx, target, do)
  finish = do & (state.ab_sel_picked_count >= state.ab_sel_pick_max)
  from azuki_jax.abilities import cards_impl

  bounce = finish & _np(cards_impl.SEL_COMPLETE_IF_STILL)[safe_def]
  state = finish_selection(state, finish)
  return _bounce_pick_to_hand(state, owner, target, bounce)


def can_select_to_equip(state: State, sel_idx, entity_idx) -> jax.Array:
  """azk_can_select_to_equip (mask + action validation)."""
  from azuki_jax.abilities import cards_impl
  from azuki_jax.engine.helpers import card_at_slot, leader_instance

  in_phase = state.ab_phase == AbilityPhase.SELECTION_PICK
  owner, def_id, idx, weapon, ok = _pick_common_ok(state, sel_idx)
  safe_def = jnp.maximum(def_id, 0)
  ok = ok & _np(tables.SEL_TO_EQUIP)[safe_def]
  weapon_type = jnp.where(
      state.def_id[owner, weapon] >= 0,
      _np(cards.TYPE)[jnp.maximum(state.def_id[owner, weapon], 0)],
      -1,
  )
  ok = ok & (weapon_type == CardType.WEAPON)
  host = jnp.where(
      entity_idx == GARDEN_SIZE,
      leader_instance(state, owner),
      card_at_slot(state, owner, Zone.GARDEN, entity_idx),
  )
  ok = ok & (host >= 0) & (entity_idx >= 0) & (entity_idx <= GARDEN_SIZE)
  # reequip-origin restriction (azk_can_select_to_equip: with
  # selection_to_equip_is_reequip the new host must differ from the weapon's
  # ReequipOrigin.previous_host)
  reequip = _np(tables.SEL_EQUIP_REEQUIP)[safe_def]
  prev = state.reequip_prev_host[owner, weapon].astype(jnp.int32)
  ok = ok & ~(reequip & (prev >= 0) & (host == prev))
  del cards_impl
  return in_phase & ok


def process_selection_to_equip(state: State, sel_idx, entity_idx, do) -> State:
  from azuki_jax.engine.helpers import card_at_slot, leader_instance
  from azuki_jax.engine.triggers import queue_on_play, queue_when_equipped

  do = jnp.asarray(do) & can_select_to_equip(state, sel_idx, entity_idx)
  owner, def_id, idx, weapon, _ = _pick_common_ok(state, sel_idx)
  host = jnp.where(
      entity_idx == GARDEN_SIZE,
      leader_instance(state, owner),
      card_at_slot(state, owner, Zone.GARDEN, entity_idx),
  )
  safe_host = jnp.maximum(host, 0)

  # attach (mirror apply_attach_weapon's core)
  from azuki_jax.engine.helpers import weapons_of

  weapon_count = jnp.sum(weapons_of(state, owner, safe_host), dtype=jnp.int32)
  state = _selection_remove(state, owner, weapon, do)
  state = state._replace(
      zone=state.zone.at[owner, weapon].set(
          jnp.where(do, jnp.int8(Zone.ATTACHED), state.zone[owner, weapon])
      ),
      zpos=state.zpos.at[owner, weapon].set(
          jnp.where(do, weapon_count.astype(jnp.int8), state.zpos[owner, weapon])
      ),
      attached_to=state.attached_to.at[owner, weapon].set(
          jnp.where(do, safe_host.astype(jnp.int8), state.attached_to[owner, weapon])
      ),
  )
  weapon_atk = state.cur_atk[owner, weapon].astype(jnp.int16)
  new_atk = jnp.maximum(
      state.cur_atk[owner, safe_host].astype(jnp.int16) + weapon_atk, 0
  ).astype(jnp.int8)
  state = state._replace(
      cur_atk=state.cur_atk.at[owner, safe_host].set(
          jnp.where(do, new_atk, state.cur_atk[owner, safe_host])
      )
  )
  is_018 = state.def_id[owner, weapon] == cards.CODE_TO_ID["AZK01-018"]
  host_is_leader = entity_idx == GARDEN_SIZE
  state = state._replace(
      cmb_in_perm=state.cmb_in_perm.at[owner, safe_host].add(
          jnp.where(do & is_018 & host_is_leader, -1, 0).astype(jnp.int8)
      )
  )
  # azk_process_selection_to_equip: reequips skip the on-play trigger and the
  # play counters, and consume the ReequipOrigin; new equips get both
  reequip = _np(tables.SEL_EQUIP_REEQUIP)[jnp.maximum(def_id, 0)]
  state = queue_on_play(state, owner, weapon, do=do & ~reequip)
  state = queue_when_equipped(state, owner, weapon, do=do)
  state = queue_when_equipped(state, owner, safe_host, do=do)
  state = state._replace(
      cards_played_turn=state.cards_played_turn.at[owner].add(
          (do & ~reequip).astype(jnp.uint8)
      ),
      next_play_cost_reduction=state.next_play_cost_reduction.at[owner].set(
          jnp.where(do & ~reequip, 0, state.next_play_cost_reduction[owner])
      ),
      reequip_prev_host=state.reequip_prev_host.at[owner, weapon].set(
          jnp.where(do & reequip, jnp.int8(-1),
                    state.reequip_prev_host[owner, weapon])
      ),
  )

  state = _record_pick(state, idx, weapon, do)
  # equips finish with CLEAR_IF_STILL_ACTIVE regardless of registry flag
  from azuki_jax.abilities import cards_impl, runtime

  finish = do & (state.ab_sel_picked_count >= state.ab_sel_pick_max)
  bounce = finish & _np(cards_impl.SEL_COMPLETE_IF_STILL)[jnp.maximum(def_id, 0)]
  hooked = cards_impl.dispatch_on_selection_complete(state)
  state = _where_state(finish, hooked, state)
  clear_now = finish & (state.ab_phase != AbilityPhase.NONE)
  cleared = runtime._clear_context(state)
  state = _where_state(clear_now, cleared, state)
  return _bounce_pick_to_hand(state, owner, weapon, bounce)


def process_skip_selection(state: State, do) -> State:
  owner, src = _owner_src(state)
  def_id = state.def_id[owner, src]
  optional = jnp.where(
      def_id >= 0, _np(tables.SEL_PICK_OPTIONAL)[jnp.maximum(def_id, 0)], False
  )
  do = jnp.asarray(do) & (state.ab_phase == AbilityPhase.SELECTION_PICK) & optional
  return finish_selection(state, do)


def process_bottom_deck(state: State, sel_idx, do) -> State:
  from azuki_jax.abilities import runtime

  do = jnp.asarray(do) & (state.ab_phase == AbilityPhase.BOTTOM_DECK)
  owner, _ = _owner_src(state)
  idx = jnp.clip(sel_idx, 0, MAX_SELECTION_ZONE_SIZE - 1)
  inst = state.ab_sel_cards[idx]
  do = do & (sel_idx >= 0) & (sel_idx < state.ab_sel_count) & (inst >= 0)
  target = jnp.maximum(inst.astype(jnp.int32), 0)
  state = _to_deck_bottom(state, owner, target, do)
  state = state._replace(
      ab_sel_cards=state.ab_sel_cards.at[idx].set(
          jnp.where(do, jnp.int8(-1), state.ab_sel_cards[idx])
      )
  )
  done = do & (remaining_count(state) == 0)
  cleared = runtime._clear_context(state)
  return _where_state(done, cleared, state)


def process_top_deck(state: State, sel_idx, do) -> State:
  from azuki_jax.abilities import runtime

  owner, src = _owner_src(state)
  def_id = state.def_id[owner, src]
  can_top = jnp.where(
      def_id >= 0, _np(tables.CAN_TOPDECK)[jnp.maximum(def_id, 0)], False
  )
  do = jnp.asarray(do) & (state.ab_phase == AbilityPhase.BOTTOM_DECK) & can_top
  idx = jnp.clip(sel_idx, 0, MAX_SELECTION_ZONE_SIZE - 1)
  inst = state.ab_sel_cards[idx]
  do = do & (sel_idx >= 0) & (sel_idx < state.ab_sel_count) & (inst >= 0)
  target = jnp.maximum(inst.astype(jnp.int32), 0)
  state = _to_deck_top(state, owner, target, do)
  state = state._replace(
      ab_sel_cards=state.ab_sel_cards.at[idx].set(
          jnp.where(do, jnp.int8(-1), state.ab_sel_cards[idx])
      )
  )
  done = do & (remaining_count(state) == 0)
  cleared = runtime._clear_context(state)
  return _where_state(done, cleared, state)


def process_bottom_deck_all(state: State, do) -> State:
  from azuki_jax.abilities import runtime

  do = jnp.asarray(do) & (state.ab_phase == AbilityPhase.BOTTOM_DECK)
  owner, _ = _owner_src(state)
  # reveal flows are bounded at 8 cards (max any card uses is 5)
  for k in range(8):
    inst = state.ab_sel_cards[k]
    ok = do & (inst >= 0)
    state = _to_deck_bottom(state, owner, jnp.maximum(inst.astype(jnp.int32), 0), ok)
    state = state._replace(
        ab_sel_cards=state.ab_sel_cards.at[k].set(
            jnp.where(ok, jnp.int8(-1), state.ab_sel_cards[k])
        )
    )
  cleared = runtime._clear_context(state)
  return _where_state(do, cleared, state)
