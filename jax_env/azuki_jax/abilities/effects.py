"""Per-card ability implementations + the ability resolution runtime.

Vanilla milestone: no card abilities are implemented yet (IMPLEMENTED table is
all-False). Triggered effects pop as no-ops for unimplemented cards, and a
coverage counter records that it happened so verification can exclude/flag
those games. Cards get ported here one by one, gated by tests vs the C engine.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from azuki_jax.abilities import tables
from azuki_jax.abilities import cards_impl as _cards_impl  # populates IMPLEMENTED
from azuki_jax.state import State


def _np(table):
  return jnp.asarray(table)


def response_ability_available(state: State, p, available_ikz) -> jax.Array:
  """Any usable AResponse ability on the player's board cards.

  C player_util.c defender_can_respond (board-ability section, lines 158-194)
  scans EVERY registered ability (frozen / once-per-turn / ikz cost /
  def->validate) — it is NOT gated on our porting status. Every AResponse
  card in the card set is now ported, so the validate dispatch below is
  always exact; a future unported card would count optimistically (validate
  treated as true)."""
  from azuki_jax.abilities import cards_impl
  from azuki_jax.constants import Zone
  from azuki_jax.engine.helpers import is_frozen

  n = state.zone.shape[1]
  idx = jnp.arange(n)
  z = state.zone[p]
  on_board = (z == Zone.GARDEN) | (z == Zone.ALLEY) | (z == Zone.LEADER)
  def_ids = state.def_id[p]
  valid = def_ids >= 0
  has_ability = jnp.where(valid, _np(tables.HAS_ABILITY)[def_ids], False)
  resp = jnp.where(valid, _np(tables.TIMING_IS_RESPONSE)[def_ids], False)
  implemented = jnp.where(valid, _np(tables.IMPLEMENTED)[def_ids], False)
  not_frozen = ~jax.vmap(lambda i: is_frozen(state, p, i))(idx)
  once_blocked = jnp.where(
      valid & _np(tables.ONCE_PER_TURN)[def_ids],
      (state.once_per_turn_used[p] & 1) != 0,
      False,
  )
  cost_ok = jnp.where(
      valid, _np(tables.ABILITY_IKZ_COST)[def_ids] <= available_ikz, False
  )
  candidate = on_board & resp & has_ability & not_frozen & ~once_blocked & cost_ok
  # def->validate consultation (C player_util defender_can_respond); ported
  # cards dispatch their validate, unported count optimistically
  card_ok = jax.vmap(
      lambda i, c: jnp.where(
          c, cards_impl.validate_card(state, def_ids[i], p, i), False
      )
  )(idx, candidate & implemented)
  return jnp.any(candidate & (card_ok | ~implemented))


def resolve_triggered_effect(state: State, src, owner, timing) -> State:
  """azk_process_triggered_effect_queue head processing.

  Implemented cards run the begin flow (validate / once-per-turn / cost
  availability preconditions). Unimplemented cards no-op and bump the
  coverage counter ab_scratch[3]."""
  from azuki_jax.abilities import cards_impl, runtime

  owner_idx = jnp.maximum(owner.astype(jnp.int32), 0)
  src_idx = jnp.maximum(src.astype(jnp.int32), 0)
  def_id = state.def_id[owner_idx, src_idx]
  valid = (src >= 0) & (def_id >= 0)
  safe_def = jnp.maximum(def_id, 0)
  has_ability = jnp.where(valid, _np(tables.HAS_ABILITY)[safe_def], False)
  implemented = jnp.where(valid, _np(tables.IMPLEMENTED)[safe_def], False)

  unimplemented = valid & has_ability & ~implemented
  state = state._replace(
      ab_scratch=state.ab_scratch.at[3].add(
          jnp.where(unimplemented, 1, 0).astype(jnp.int16)
      )
  )

  run = valid & has_ability & implemented
  # preconditions (azk_process_triggered_effect_queue). Targets are counted
  # with source/owner pre-set so per-card target validators see the right
  # source (C passes the source card explicitly).
  probe = state._replace(
      ab_source=src_idx.astype(jnp.int8), ab_owner=owner_idx.astype(jnp.int8)
  )
  once = jnp.where(valid, _np(tables.ONCE_PER_TURN)[safe_def], False)
  once_ok = ~once | ((state.once_per_turn_used[owner_idx, src_idx] & 1) == 0)
  cost_min = jnp.where(valid, _np(tables.COST_MIN)[safe_def], 0)
  cost_avail = runtime.count_targets(probe, def_id, jnp.bool_(True), owner_idx)
  cost_ok = (cost_min == 0) | (cost_avail >= cost_min)
  card_ok = cards_impl.validate_card(state, def_id, owner_idx, src_idx)
  eff_min = jnp.where(valid, _np(tables.EFFECT_MIN)[safe_def], 0)
  eff_avail = runtime.count_targets(probe, def_id, jnp.bool_(False), owner_idx)
  eff_ok = (eff_min == 0) | (eff_avail >= eff_min)

  run = run & once_ok & cost_ok & card_ok & eff_ok
  del timing  # timing tags are enforced at queue time (queue_effect)
  return runtime.begin_ability(
      state, owner_idx, src_idx, runtime.BEGIN_TRIGGERED, run
  )


def queue_end_of_turn_abilities(state: State) -> State:
  """azk_trigger_end_of_turn_abilities: active player's garden, leader, alley
  (slot zones in flecs insertion order = board_seq)."""
  from azuki_jax.constants import Zone
  from azuki_jax.engine.helpers import leader_instance
  from azuki_jax.engine.triggers import (
      TIMING_END_OF_TURN,
      queue_effect,
      queue_zone_by_seq,
  )

  p = state.active_player.astype(jnp.int32)
  state = queue_zone_by_seq(state, p, Zone.GARDEN, TIMING_END_OF_TURN)
  leader = leader_instance(state, p)
  state = queue_effect(
      state, p, jnp.maximum(leader, 0), TIMING_END_OF_TURN, leader >= 0
  )
  return queue_zone_by_seq(state, p, Zone.ALLEY, TIMING_END_OF_TURN)
