"""Legal-action enumeration in exactly the C enumerator's order.

Mirrors src/validation/action_enumerator.c + action_schema.c. The output is a
fixed-size (MAX_LEGAL_ACTIONS, 4) uint8 table + count + head0 mask, built by
generating candidate actions in the C iteration order with validity flags and
compacting via argsort (stable, preserves order).

Spec order (filtered by phase): NOOP, PLAY_ENTITY_TO_GARDEN,
PLAY_ENTITY_TO_ALLEY, ATTACH_WEAPON_FROM_HAND, GATE_PORTAL, ATTACK,
MULLIGAN_SHUFFLE, PLAY_SPELL_FROM_HAND, ACTIVATE_ALLEY_ABILITY,
ACTIVATE_GARDEN_OR_LEADER_ABILITY, DECLARE_DEFENDER.

Vanilla scope: spell/ability activations enumerate nothing (no implemented
abilities); ability sub-phase masks are produced by the ability layer.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from azuki_jax import cards
from azuki_jax.constants import (
    ACTION_TYPE_COUNT,
    ALLEY_SIZE,
    GARDEN_SIZE,
    MAX_HAND_SIZE,
    MAX_LEGAL_ACTIONS,
    MAX_SELECTION_ZONE_SIZE,
    Act,
    Phase,
    Zone,
)
from azuki_jax.engine import ikz
from azuki_jax.engine.validate import (
    validate_attach_weapon,
    validate_attack,
    validate_declare_defender,
    validate_gate_portal,
    validate_mulligan,
    validate_noop,
    validate_play_entity,
)
from azuki_jax.state import State
from azuki_jax.zones import zone_count

# Candidate layout (precomputed on host): every (type, s1, s2, s3) the C
# enumerator can ever try, in its exact iteration order.
_CANDS: list[tuple[int, int, int, int, int]] = []  # (kind tag for validity fn)
KIND_NOOP = 0
KIND_PLAY_G = 1
KIND_PLAY_A = 2
KIND_WEAPON = 3
KIND_PORTAL = 4
KIND_ATTACK = 5
KIND_MULLIGAN = 6
KIND_DEFEND = 7
KIND_SPELL = 8
KIND_ACT_ALLEY = 9
KIND_ACT_GARDEN = 10
# ability-phase rows (enumerate_ability_actions)
KIND_AB_CONFIRM = 11
KIND_AB_DECLINE = 12
KIND_AB_COST_NOOP = 13
KIND_AB_COST_SEL = 14
KIND_AB_EFF_NOOP = 15
KIND_AB_EFF_SEL = 16
KIND_AB_EFF_NOOP2 = 17  # duplicate NOOP the C emits when zero choices remain
# selection-zone phases
KIND_SEL_NOOP = 18
KIND_SEL_PICK = 19
KIND_SEL_GARDEN = 20
KIND_SEL_ALLEY = 21
KIND_SEL_EQUIP = 22
KIND_BD_TOP = 23
KIND_BD_CARD = 24
KIND_BD_ALL = 25

MAX_TARGET_CHOICES = 12
MAX_SELECTION_ROWS = MAX_SELECTION_ZONE_SIZE  # C enumerates a pick action per
# revealed card (for i < selection.count); the whole discard pile can be
# surfaced, so enumerate every index. Only the valid subset is compacted into
# the (MAX_LEGAL_ACTIONS, 4) output, so a larger universe is fine.


def _build_candidates() -> np.ndarray:
  rows = []
  # --- ability-phase segment (replaces everything when ab_phase != NONE) ---
  rows.append((KIND_AB_CONFIRM, Act.CONFIRM_ABILITY, 0, 0, 0))
  rows.append((KIND_AB_DECLINE, Act.NOOP, 0, 0, 0))
  rows.append((KIND_AB_COST_NOOP, Act.NOOP, 0, 0, 0))
  for k in range(MAX_TARGET_CHOICES):
    rows.append((KIND_AB_COST_SEL, Act.SELECT_COST_TARGET, k, 0, 0))
  rows.append((KIND_AB_EFF_NOOP, Act.NOOP, 0, 0, 0))
  for k in range(MAX_TARGET_CHOICES):
    rows.append((KIND_AB_EFF_SEL, Act.SELECT_EFFECT_TARGET, k, 0, 0))
  rows.append((KIND_AB_EFF_NOOP2, Act.NOOP, 0, 0, 0))
  # SELECTION_PICK: NOOP (if optional), then per-index variants in C order
  rows.append((KIND_SEL_NOOP, Act.NOOP, 0, 0, 0))
  for i in range(MAX_SELECTION_ROWS):
    rows.append((KIND_SEL_PICK, Act.SELECT_FROM_SELECTION, i, 0, 0))
    for slot in range(GARDEN_SIZE):
      rows.append((KIND_SEL_GARDEN, Act.SELECT_TO_GARDEN, i, slot, 0))
    for slot in range(ALLEY_SIZE):
      rows.append((KIND_SEL_ALLEY, Act.SELECT_TO_ALLEY, i, slot, 0))
    for target in range(GARDEN_SIZE + 1):
      rows.append((KIND_SEL_EQUIP, Act.SELECT_TO_EQUIP, i, target, 0))
  # BOTTOM_DECK: top-deck rows (when allowed), bottom-deck rows, then ALL
  for i in range(MAX_SELECTION_ROWS):
    rows.append((KIND_BD_TOP, Act.TOP_DECK_CARD, i, 0, 0))
  for i in range(MAX_SELECTION_ROWS):
    rows.append((KIND_BD_CARD, Act.BOTTOM_DECK_CARD, i, 0, 0))
  rows.append((KIND_BD_ALL, Act.BOTTOM_DECK_ALL, 0, 0, 0))

  # --- normal segment, C action_schema spec order ---
  rows.append((KIND_NOOP, Act.NOOP, 0, 0, 0))
  for hand in range(MAX_HAND_SIZE):
    for slot in range(GARDEN_SIZE):
      for token in range(2):
        rows.append((KIND_PLAY_G, Act.PLAY_ENTITY_TO_GARDEN, hand, slot, token))
  for hand in range(MAX_HAND_SIZE):
    for slot in range(ALLEY_SIZE):
      for token in range(2):
        rows.append((KIND_PLAY_A, Act.PLAY_ENTITY_TO_ALLEY, hand, slot, token))
  for hand in range(MAX_HAND_SIZE):
    for target in range(GARDEN_SIZE + 1):
      for token in range(2):
        rows.append((KIND_WEAPON, Act.ATTACH_WEAPON_FROM_HAND, hand, target, token))
  for alley in range(ALLEY_SIZE):
    for slot in range(GARDEN_SIZE):
      rows.append((KIND_PORTAL, Act.GATE_PORTAL, alley, slot, 0))
  for attacker in range(GARDEN_SIZE + 1):
    for defender in range(GARDEN_SIZE + ALLEY_SIZE + 1):
      rows.append((KIND_ATTACK, Act.ATTACK, attacker, defender, 0))
  rows.append((KIND_MULLIGAN, Act.MULLIGAN_SHUFFLE, 0, 0, 0))
  # PLAY_SPELL_FROM_HAND: hand x ability(0) x token (single primary ability)
  for hand in range(MAX_HAND_SIZE):
    for token in range(2):
      rows.append((KIND_SPELL, Act.PLAY_SPELL_FROM_HAND, hand, 0, token))
  # ACTIVATE_ALLEY_ABILITY: ability(0) x alley slot
  for slot in range(ALLEY_SIZE):
    rows.append((KIND_ACT_ALLEY, Act.ACTIVATE_ALLEY_ABILITY, 0, slot, 0))
  # ACTIVATE_GARDEN_OR_LEADER: slot (garden 0-4, leader 5) x ability(0) x token
  for slot in range(GARDEN_SIZE + 1):
    for token in range(2):
      rows.append((KIND_ACT_GARDEN, Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY, slot, 0, token))
  for slot in range(GARDEN_SIZE):
    rows.append((KIND_DEFEND, Act.DECLARE_DEFENDER, slot, 0, 0))
  return np.asarray(rows, np.int32)


_CAND_TABLE = _build_candidates()
NUM_CANDIDATES = len(_CAND_TABLE)


def _candidate_valid(state: State, kind, act, s1, s2, s3) -> jax.Array:
  """Validity of one candidate in the current state (phase checks included)."""
  p = state.active_player
  phase = state.phase
  in_mull = phase == Phase.PREGAME_MULLIGAN
  in_main = phase == Phase.MAIN
  in_resp = phase == Phase.RESPONSE_WINDOW

  hand_count = zone_count(state.zone[p], Zone.HAND)
  has_token = ikz.token_ready(state, p)

  # enumerate bounds: hand index < hand_count; token loop only if ready token
  token_ok = (s3 == 0) | has_token
  hand_ok = s1 < hand_count

  result = jnp.asarray(False)

  is_noop = kind == KIND_NOOP
  result = jnp.where(is_noop & (in_mull | in_main | in_resp),
                     validate_noop(state), result)

  is_play_g = kind == KIND_PLAY_G
  ok = (
      (in_main | in_resp) & hand_ok & token_ok
      & validate_play_entity(state, Zone.GARDEN, s1, s2, s3 != 0)
      & _slot_enumerable(state, p, Zone.GARDEN, s2)
  )
  result = jnp.where(is_play_g, ok, result)

  is_play_a = kind == KIND_PLAY_A
  ok = (
      (in_main | in_resp) & hand_ok & token_ok
      & validate_play_entity(state, Zone.ALLEY, s1, s2, s3 != 0)
      & _slot_enumerable(state, p, Zone.ALLEY, s2)
  )
  result = jnp.where(is_play_a, ok, result)

  is_weapon = kind == KIND_WEAPON
  target_enumerable = jnp.where(
      s2 < GARDEN_SIZE,
      _slot_occupied(state, p, Zone.GARDEN, s2),
      True,  # leader always present
  )
  ok = (
      (in_main | in_resp) & hand_ok & token_ok & target_enumerable
      & validate_attach_weapon(state, s1, s2, s3 != 0)
  )
  result = jnp.where(is_weapon, ok, result)

  is_portal = kind == KIND_PORTAL
  ok = (
      in_main
      & _slot_occupied(state, p, Zone.ALLEY, s1)
      & _slot_enumerable(state, p, Zone.GARDEN, s2)
      & validate_gate_portal(state, s1, s2)
  )
  result = jnp.where(is_portal, ok, result)

  is_attack = kind == KIND_ATTACK
  ok = in_main & validate_attack(state, s1, s2)
  result = jnp.where(is_attack, ok, result)

  is_mull_act = kind == KIND_MULLIGAN
  result = jnp.where(is_mull_act & in_mull, validate_mulligan(state), result)

  is_defend = kind == KIND_DEFEND
  ok = in_resp & validate_declare_defender(state, s1)
  result = jnp.where(is_defend, ok, result)

  # spells / ability activations (IMPLEMENTED cards only — C enumerates all
  # registered abilities; comparisons exclude unported cards)
  from azuki_jax.engine.validate import (
      validate_activate_alley,
      validate_activate_garden_or_leader,
      validate_play_spell,
  )

  is_spell = kind == KIND_SPELL
  ok = (in_main | in_resp) & hand_ok & validate_play_spell(state, s1, s3 != 0)
  spell_token_ok = (s3 == 0) | has_token  # bool param bound, no cost gate
  result = jnp.where(is_spell, ok & spell_token_ok, result)

  is_act_alley = kind == KIND_ACT_ALLEY
  ok = (in_main | in_resp) & validate_activate_alley(state, s1, s2)
  result = jnp.where(is_act_alley, ok, result)

  is_act_garden = kind == KIND_ACT_GARDEN
  ok = (in_main | in_resp) & validate_activate_garden_or_leader(
      state, s1, s2, s3 != 0
  )
  # token row enumerated only when ability has ikz cost and token is ready
  garden_token_ok = (s3 == 0) | (
      has_token & _activate_uses_ikz(state, s1)
  )
  result = jnp.where(is_act_garden, ok & garden_token_ok, result)

  return result


def _activate_uses_ikz(state: State, slot) -> jax.Array:
  from azuki_jax.abilities import tables as ab_tables
  from azuki_jax.engine.helpers import card_at_slot, leader_instance

  p = state.active_player
  inst = jnp.where(
      slot == GARDEN_SIZE,
      leader_instance(state, p),
      card_at_slot(state, p, Zone.GARDEN, slot),
  )
  def_id = state.def_id[p, jnp.maximum(inst, 0)]
  return (inst >= 0) & (def_id >= 0) & (
      jnp.asarray(ab_tables.ABILITY_IKZ_COST)[jnp.maximum(def_id, 0)] > 0
  )


def _ability_phase_valid(state: State, kind, s1) -> jax.Array:
  """Validity of ability-phase rows (enumerate_ability_actions order)."""
  from azuki_jax.abilities import runtime, tables as ab_tables

  phase = state.ab_phase
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  def_id = state.def_id[owner, src]

  in_confirm = phase == 1
  in_cost = phase == 2
  in_effect = phase == 3

  result = jnp.asarray(False)
  result = jnp.where(kind == KIND_AB_CONFIRM, in_confirm, result)
  result = jnp.where(kind == KIND_AB_DECLINE, in_confirm, result)

  cost_min = jnp.where(def_id >= 0, jnp.asarray(ab_tables.COST_MIN)[jnp.maximum(def_id, 0)], 0)
  cost_max = runtime._cost_max_allowed(state, def_id, owner)
  cost_skip_ok = in_cost & (state.ab_cost_selected >= cost_min) & (
      state.ab_cost_selected < cost_max
  )
  result = jnp.where(kind == KIND_AB_COST_NOOP, cost_skip_ok, result)

  cost_ok, _, _ = runtime.collect_targets(state, def_id, jnp.bool_(True), owner)
  ti = jnp.clip(s1, 0, runtime.MAX_TARGET_CHOICES - 1)
  result = jnp.where(kind == KIND_AB_COST_SEL, in_cost & cost_ok[ti], result)

  # ctx->effect.min_required/max_allowed (hooks may override the table values)
  eff_min = state.ab_eff_min.astype(jnp.int32)
  eff_max = jnp.maximum(state.ab_eff_max.astype(jnp.int32), 0)
  eff_skip_ok = in_effect & (state.ab_eff_selected >= eff_min) & (
      state.ab_eff_selected < eff_max
  )
  result = jnp.where(kind == KIND_AB_EFF_NOOP, eff_skip_ok, result)

  eff_ok, _, _ = runtime.collect_targets(state, def_id, jnp.bool_(False), owner)
  result = jnp.where(kind == KIND_AB_EFF_SEL, in_effect & eff_ok[ti], result)

  none_left = in_effect & (jnp.sum(eff_ok) == 0)
  result = jnp.where(kind == KIND_AB_EFF_NOOP2, none_left, result)
  return result


def _selection_phase_valid(state: State, kind, s1, s2) -> jax.Array:
  """Validity of SELECTION_PICK / BOTTOM_DECK rows (C enumerate order)."""
  from azuki_jax.abilities import cards_impl, selection as sel_mod
  from azuki_jax.abilities import tables as ab_tables

  in_pick = state.ab_phase == 4
  in_bottom = state.ab_phase == 5
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  def_id = state.def_id[owner, src]
  safe_def = jnp.maximum(def_id, 0)

  idx = jnp.clip(s1, 0, MAX_SELECTION_ROWS - 1)
  inst = state.ab_sel_cards[idx]
  slot_live = (s1 < state.ab_sel_count) & (inst >= 0)
  target = jnp.maximum(inst.astype(jnp.int32), 0)
  validator_ok = cards_impl.selection_target_validator(state, def_id, owner, target)

  result = jnp.asarray(False)
  optional = jnp.asarray(ab_tables.SEL_PICK_OPTIONAL)[safe_def]
  result = jnp.where(kind == KIND_SEL_NOOP, in_pick & optional, result)

  special = (
      jnp.asarray(ab_tables.SEL_TO_GARDEN)[safe_def]
      | jnp.asarray(ab_tables.SEL_TO_ALLEY)[safe_def]
      | jnp.asarray(ab_tables.SEL_TO_EQUIP)[safe_def]
  )
  to_hand = ~special | jnp.asarray(ab_tables.SEL_TO_HAND)[safe_def]
  result = jnp.where(
      kind == KIND_SEL_PICK,
      in_pick & slot_live & validator_ok & to_hand,
      result,
  )

  target_def = state.def_id[owner, target]
  is_entity = jnp.where(
      target_def >= 0, jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)] == 2, False
  )
  garden_slot_ok = _slot_enumerable(state, owner, Zone.GARDEN, s2)
  result = jnp.where(
      kind == KIND_SEL_GARDEN,
      in_pick & slot_live & validator_ok & is_entity
      & jnp.asarray(ab_tables.SEL_TO_GARDEN)[safe_def] & garden_slot_ok,
      result,
  )
  alley_slot_ok = _slot_enumerable(state, owner, Zone.ALLEY, s2)
  result = jnp.where(
      kind == KIND_SEL_ALLEY,
      in_pick & slot_live & validator_ok & is_entity
      & jnp.asarray(ab_tables.SEL_TO_ALLEY)[safe_def] & alley_slot_ok,
      result,
  )
  result = jnp.where(
      kind == KIND_SEL_EQUIP,
      sel_mod.can_select_to_equip(state, s1, s2),
      result,
  )

  can_top = jnp.asarray(ab_tables.CAN_TOPDECK)[safe_def]
  result = jnp.where(
      kind == KIND_BD_TOP, in_bottom & slot_live & can_top, result
  )
  result = jnp.where(kind == KIND_BD_CARD, in_bottom & slot_live, result)
  result = jnp.where(kind == KIND_BD_ALL, in_bottom, result)
  return result


def _slot_occupied(state: State, p, zone, slot) -> jax.Array:
  return jnp.any((state.zone[p] == zone) & (state.zpos[p] == jnp.asarray(slot, jnp.int8)))


def _slot_enumerable(state: State, p, zone, slot) -> jax.Array:
  """collect_placement_slots: empty slots, or all slots when the zone is full."""
  occupied = _slot_occupied(state, p, zone, slot)
  full = zone_count(state.zone[p], zone) >= GARDEN_SIZE
  return ~occupied | full


def build_mask(state: State):
  """Returns (legal (MAX_LEGAL_ACTIONS, 4) uint8, count uint16, head0 (26,) bool).

  Mask is for the ACTIVE player; the other player's mask is empty. Ability
  sub-phase masks (CONFIRM/SELECT_*) come from the ability layer when active.
  """
  cands = jnp.asarray(_CAND_TABLE)  # (C, 5)
  kinds, acts, s1s, s2s, s3s = (cands[:, k] for k in range(5))

  game_over = state.winner != -1
  in_ability = state.ab_phase != 0
  is_ability_row = kinds >= KIND_AB_CONFIRM

  is_selection_row = kinds >= KIND_SEL_NOOP

  valid = jax.vmap(
      lambda kind, act, s1, s2, s3: _candidate_valid(state, kind, act, s1, s2, s3)
  )(kinds, acts, s1s, s2s, s3s)
  ability_valid = jax.vmap(
      lambda kind, s1: _ability_phase_valid(state, kind, s1)
  )(kinds, s1s)
  selection_valid = jax.vmap(
      lambda kind, s1, s2: _selection_phase_valid(state, kind, s1, s2)
  )(kinds, s1s, s2s)
  valid = jnp.where(
      is_selection_row,
      selection_valid & in_ability,
      jnp.where(is_ability_row, ability_valid & in_ability, valid & ~in_ability),
  )
  valid = valid & ~game_over

  # dynamic enumeration order for SELECT_*_TARGET rows: C iterates targets in
  # zone-insertion order, not action_index order
  from azuki_jax.abilities import runtime as ab_runtime

  ab_owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  ab_src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  ab_def = state.def_id[ab_owner, ab_src]
  priority = jnp.arange(NUM_CANDIDATES)
  for kind, scope_is_cost in ((KIND_AB_COST_SEL, True), (KIND_AB_EFF_SEL, False)):
    block = np.flatnonzero(_CAND_TABLE[:, 0] == kind)
    base = int(block[0])
    ok, inst_v, player_v = ab_runtime.collect_targets(
        state, ab_def, jnp.bool_(scope_is_cost), ab_owner
    )
    keys = ab_runtime.collect_target_order_keys(
        state, ab_def, jnp.bool_(scope_is_cost), ab_owner, ok, inst_v, player_v
    )
    rank = jnp.sum(
        (keys[None, :] < keys[:, None])
        | ((keys[None, :] == keys[:, None])
           & (jnp.arange(12)[None, :] < jnp.arange(12)[:, None])),
        axis=1,
    )
    priority = priority.at[base : base + 12].set(base + rank)

  # stable compaction: order by (invalid, priority)
  order = jnp.argsort(jnp.where(valid, priority,
                                NUM_CANDIDATES + jnp.arange(NUM_CANDIDATES)))
  count = jnp.sum(valid, dtype=jnp.int32)
  take = jnp.minimum(count, MAX_LEGAL_ACTIONS)

  rows = cands[order][:MAX_LEGAL_ACTIONS, 1:5].astype(jnp.int32)
  slot_idx = jnp.arange(MAX_LEGAL_ACTIONS)
  legal = jnp.where(
      (slot_idx < take)[:, None], rows, 0
  ).astype(jnp.uint8)

  head0 = jnp.zeros((ACTION_TYPE_COUNT,), jnp.bool_)
  act_ids = jnp.where(valid, acts, 0)
  head0 = head0.at[act_ids].max(valid, mode="drop")

  return legal, take.astype(jnp.uint16), head0
