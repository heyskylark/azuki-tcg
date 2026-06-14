"""Action validators mirroring src/validation/action_validation.c.

Each returns a bool scalar for one candidate action under the CURRENT state
(active player implied). Vanilla scope: ability-activation/spell validators
consult the registry tables; per-card validate() hooks come from the ability
layer (azuki_jax.abilities.effects.VALIDATE_OK placeholder until translated).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from azuki_jax import cards
from azuki_jax.constants import (
    ALLEY_SIZE,
    GARDEN_SIZE,
    MAX_HAND_SIZE,
    CardType,
    Phase,
    Zone,
)
from azuki_jax.engine import ikz
from azuki_jax.engine.helpers import (
    attr_attack_alley,
    attr_leaders_only,
    can_tap,
    card_at_slot,
    has_defender_kw,
    has_infiltrate,
    has_rooted,
    has_taunt,
    hand_instance,
    is_frozen,
    leader_instance,
    gate_instance,
)
from azuki_jax.state import State
from azuki_jax.zones import zone_count


def _np(table) -> jnp.ndarray:
  return jnp.asarray(table)


def effective_play_cost(state: State, p, inst) -> jax.Array:
  """azk_get_effective_card_play_cost (AZK01-106 defender discount +
  next_card_play_cost_reduction)."""
  def_id = state.def_id[p, inst]
  cost = jnp.where(def_id >= 0, _np(cards.IKZ_COST)[def_id], 0).astype(jnp.int16)
  is_106 = def_id == cards.CODE_TO_ID["AZK01-106"]
  n = state.zone.shape[1]
  defenders = jnp.sum(
      (state.zone[p] == Zone.GARDEN)
      & has_defender_kw(state, p, jnp.arange(n))
  ).astype(jnp.int16)
  cost = jnp.where(is_106, cost - defenders, cost)
  cost = cost - state.next_play_cost_reduction[p].astype(jnp.int16)
  return jnp.maximum(cost, 0)


def can_play_from_hand_in_response(state: State, p, inst) -> jax.Array:
  """azk_can_play_card_from_hand_during_response_window (vanilla: spells with
  AResponse timing; non-spells with registry can_play_as_response_from_hand)."""
  from azuki_jax.abilities.tables import (
      RESPONSE_PLAY_FROM_HAND,
      TIMING_IS_RESPONSE,
      HAS_ABILITY,
  )

  def_id = state.def_id[p, inst]
  valid = def_id >= 0
  is_spell = jnp.where(valid, _np(cards.TYPE)[def_id] == CardType.SPELL, False)
  spell_ok = jnp.where(valid, _np(TIMING_IS_RESPONSE)[def_id], False) & jnp.where(
      valid, _np(HAS_ABILITY)[def_id], False
  )
  other_ok = jnp.where(valid, _np(RESPONSE_PLAY_FROM_HAND)[def_id], False)
  return jnp.where(is_spell, spell_ok, other_ok)


def validate_play_entity(state: State, placement_zone, hand_index, slot,
                         use_token) -> jax.Array:
  """azk_validate_play_entity_action (active player implied)."""
  p = state.active_player
  phase_ok = (state.phase == Phase.MAIN) | (state.phase == Phase.RESPONSE_WINDOW)

  inst = hand_instance(state, p, hand_index)
  exists = inst >= 0
  safe = jnp.maximum(inst, 0)
  def_id = state.def_id[p, safe]
  is_entity = jnp.where(exists, _np(cards.TYPE)[def_id] == CardType.ENTITY, False)

  response_ok = jnp.where(
      state.phase == Phase.RESPONSE_WINDOW,
      can_play_from_hand_in_response(state, p, safe),
      True,
  )

  occupied = card_at_slot(state, p, placement_zone, slot) >= 0
  count = zone_count(state.zone[p], placement_zone)
  full = count >= GARDEN_SIZE
  slot_ok = ~occupied | full

  cost = effective_play_cost(state, p, safe)
  pay_ok = ikz.can_pay(state, p, cost, use_token)

  return phase_ok & exists & is_entity & response_ok & slot_ok & pay_ok


def validate_gate_portal(state: State, alley_index, garden_index) -> jax.Array:
  p = state.active_player
  gate = gate_instance(state, p)
  gate_untapped = ~state.tapped[p, jnp.maximum(gate, 0)] & (gate >= 0)

  alley_card = card_at_slot(state, p, Zone.ALLEY, alley_index)
  alley_ok = (alley_card >= 0) & ~state.tapped[p, jnp.maximum(alley_card, 0)]

  occupied = card_at_slot(state, p, Zone.GARDEN, garden_index) >= 0
  full = zone_count(state.zone[p], Zone.GARDEN) >= GARDEN_SIZE
  slot_ok = ~occupied | full

  return gate_untapped & alley_ok & slot_ok


def validate_attack(state: State, attacker_index, defender_index) -> jax.Array:
  p = state.active_player
  opp = (p + 1) % 2

  attacker_is_leader = attacker_index == GARDEN_SIZE
  leader = leader_instance(state, p)
  garden_attacker = card_at_slot(state, p, Zone.GARDEN, attacker_index)
  attacker = jnp.where(attacker_is_leader, leader, garden_attacker)
  attacker_ok = attacker >= 0
  safe_attacker = jnp.maximum(attacker, 0)

  leader_atk_ok = jnp.where(
      attacker_is_leader, state.cur_atk[p, jnp.maximum(leader, 0)] > 0, True
  )
  tap_ok = can_tap(state, p, safe_attacker)
  not_frozen = ~is_frozen(state, p, safe_attacker)
  not_rooted = ~has_rooted(state, p, safe_attacker)

  defender_is_leader = defender_index == GARDEN_SIZE
  is_garden_target = defender_index < GARDEN_SIZE
  opp_leader = leader_instance(state, opp)
  garden_target = card_at_slot(state, opp, Zone.GARDEN, defender_index)
  alley_target = card_at_slot(
      state, opp, Zone.ALLEY, defender_index - (GARDEN_SIZE + 1)
  )
  defender = jnp.where(
      defender_is_leader,
      opp_leader,
      jnp.where(is_garden_target, garden_target, alley_target),
  )
  defender_ok = defender >= 0
  safe_defender = jnp.maximum(defender, 0)

  garden_target_tapped_ok = jnp.where(
      is_garden_target & ~defender_is_leader,
      state.tapped[opp, safe_defender],
      True,
  )
  alley_ok = jnp.where(
      ~defender_is_leader & ~is_garden_target,
      attr_attack_alley(state, p, safe_attacker),
      True,
  )

  # tapped-Taunt-first rule
  n = state.zone.shape[1]
  any_tapped_taunt = jnp.any(
      (state.zone[opp] == Zone.GARDEN)
      & state.tapped[opp]
      & has_taunt(state, opp, jnp.arange(n))
  )
  taunt_ok = ~any_tapped_taunt | has_taunt(state, opp, safe_defender)

  leaders_only_ok = jnp.where(
      attr_leaders_only(state, p, safe_attacker), defender_is_leader, True
  )

  return (
      attacker_ok & leader_atk_ok & tap_ok & not_frozen & not_rooted
      & defender_ok & garden_target_tapped_ok & alley_ok & taunt_ok
      & leaders_only_ok
  )


def validate_attach_weapon(state: State, hand_index, entity_index,
                           use_token) -> jax.Array:
  p = state.active_player
  phase_ok = (state.phase == Phase.MAIN) | (state.phase == Phase.RESPONSE_WINDOW)

  inst = hand_instance(state, p, hand_index)
  exists = inst >= 0
  safe = jnp.maximum(inst, 0)
  is_weapon = jnp.where(
      exists, _np(cards.TYPE)[state.def_id[p, safe]] == CardType.WEAPON, False
  )
  response_ok = jnp.where(
      state.phase == Phase.RESPONSE_WINDOW,
      can_play_from_hand_in_response(state, p, safe),
      True,
  )

  target_is_leader = entity_index == GARDEN_SIZE
  leader = leader_instance(state, p)
  garden_target = card_at_slot(state, p, Zone.GARDEN, entity_index)
  target = jnp.where(target_is_leader, leader, garden_target)
  target_ok = target >= 0

  cost = effective_play_cost(state, p, safe)
  pay_ok = ikz.can_pay(state, p, cost, use_token)

  return phase_ok & exists & is_weapon & response_ok & target_ok & pay_ok


def validate_declare_defender(state: State, garden_index) -> jax.Array:
  p = state.active_player
  phase_ok = state.phase == Phase.RESPONSE_WINDOW
  not_intercepted = ~state.combat_intercepted

  atk_p = state.combat_defender_player  # attacker is the other player
  attacker_p = (state.combat_defender_player + 1) % 2
  attacker = state.combat_attacker
  attacker_ok = attacker >= 0
  no_infiltrate = ~has_infiltrate(
      state, attacker_p, jnp.maximum(attacker.astype(jnp.int32), 0)
  )

  card = card_at_slot(state, p, Zone.GARDEN, garden_index)
  card_ok = card >= 0
  safe = jnp.maximum(card, 0)
  defender_kw = has_defender_kw(state, p, safe)
  untapped = ~state.tapped[p, safe]

  del atk_p
  return (
      phase_ok & not_intercepted & attacker_ok & no_infiltrate & card_ok
      & defender_kw & untapped
  )


def validate_play_spell(state: State, hand_index, use_token) -> jax.Array:
  """azk_validate_play_spell_action (IMPLEMENTED cards only)."""
  from azuki_jax.abilities import cards_impl, tables as ab_tables

  p = state.active_player
  inst = hand_instance(state, p, hand_index)
  exists = inst >= 0
  safe = jnp.maximum(inst, 0)
  def_id = state.def_id[p, safe]
  valid = exists & (def_id >= 0)
  safe_def = jnp.maximum(def_id, 0)

  is_spell = jnp.where(valid, _np(cards.TYPE)[safe_def] == CardType.SPELL, False)
  implemented = jnp.where(valid, _np(ab_tables.IMPLEMENTED)[safe_def], False)
  has_ability = jnp.where(valid, _np(ab_tables.HAS_ABILITY)[safe_def], False)
  is_main = jnp.where(valid, _np(ab_tables.TIMING_IS_MAIN)[safe_def], False)
  is_resp = jnp.where(valid, _np(ab_tables.TIMING_IS_RESPONSE)[safe_def], False)
  timing_ok = jnp.where(
      state.phase == Phase.RESPONSE_WINDOW, is_resp,
      jnp.where(state.phase == Phase.MAIN, is_main, False),
  )

  cost = effective_play_cost(state, p, safe)
  pay_ok = ikz.can_pay(state, p, cost, use_token)
  card_ok = cards_impl.validate_card(state, def_id, p, safe)

  return (
      exists & is_spell & implemented & has_ability & timing_ok & pay_ok & card_ok
  )


def _validate_activate_common(state: State, p, inst, in_response) -> jax.Array:
  from azuki_jax.abilities import cards_impl, tables as ab_tables
  from azuki_jax.engine.helpers import is_frozen

  exists = inst >= 0
  safe = jnp.maximum(inst, 0)
  def_id = state.def_id[p, safe]
  valid = exists & (def_id >= 0)
  safe_def = jnp.maximum(def_id, 0)

  implemented = jnp.where(valid, _np(ab_tables.IMPLEMENTED)[safe_def], False)
  has_ability = jnp.where(valid, _np(ab_tables.HAS_ABILITY)[safe_def], False)
  not_frozen = ~is_frozen(state, p, safe)
  once = jnp.where(valid, _np(ab_tables.ONCE_PER_TURN)[safe_def], False)
  once_ok = ~once | ((state.once_per_turn_used[p, safe] & 1) == 0)
  is_main = jnp.where(valid, _np(ab_tables.TIMING_IS_MAIN)[safe_def], False)
  is_resp = jnp.where(valid, _np(ab_tables.TIMING_IS_RESPONSE)[safe_def], False)
  timing_ok = jnp.where(in_response, is_resp, is_main)
  card_ok = cards_impl.validate_card(state, def_id, p, safe)
  return exists & implemented & has_ability & not_frozen & once_ok & timing_ok & card_ok


def validate_activate_garden_or_leader(state: State, slot, ability_index,
                                       use_token) -> jax.Array:
  from azuki_jax.abilities import tables as ab_tables

  p = state.active_player
  inst = jnp.where(
      slot == GARDEN_SIZE,
      leader_instance(state, p),
      card_at_slot(state, p, Zone.GARDEN, slot),
  )
  ok = _validate_activate_common(
      state, p, inst, state.phase == Phase.RESPONSE_WINDOW
  ) & (ability_index == 0)
  safe = jnp.maximum(inst, 0)
  def_id = jnp.maximum(state.def_id[p, safe], 0)
  cost = _np(ab_tables.ABILITY_IKZ_COST)[def_id]
  pay_ok = ikz.can_pay(state, p, cost, use_token)
  return ok & pay_ok


def validate_activate_alley(state: State, ability_index, slot) -> jax.Array:
  p = state.active_player
  inst = card_at_slot(state, p, Zone.ALLEY, slot)
  return _validate_activate_common(
      state, p, inst, state.phase == Phase.RESPONSE_WINDOW
  ) & (ability_index == 0)


def validate_noop(state: State) -> jax.Array:
  return (
      (state.phase == Phase.MAIN)
      | (state.phase == Phase.PREGAME_MULLIGAN)
      | (state.phase == Phase.RESPONSE_WINDOW)
  )


def validate_mulligan(state: State) -> jax.Array:
  return state.phase == Phase.PREGAME_MULLIGAN
