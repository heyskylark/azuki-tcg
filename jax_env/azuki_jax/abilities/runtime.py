"""Ability resolution runtime: begin flow, target collection, FSM handlers.

Mirrors src/abilities/core/{ability_runtime,ability_flow,ability_context}.c,
src/abilities/ability_system.c (process_* handlers, queue processing) and
src/abilities/targeting/*. Card behavior hooks dispatch via
azuki_jax.abilities.cards_impl tables keyed by card_def_id.

Selection-zone phases (SELECTION_PICK / BOTTOM_DECK) are not implemented yet;
cards using them stay out of tables.IMPLEMENTED.
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
    AbilityPhase,
    Zone,
)
from azuki_jax.state import State

# Target action_index spaces per target type (max indices enumerable)
MAX_TARGET_CHOICES = 12  # ANY_LEADER_OR_GARDEN: 0..11

# begin variants (which AbilityBeginOptions bundle)
BEGIN_TRIGGERED = 0  # queued: confirm-if-optional, transfer control, clamp, costs-before
BEGIN_MAIN = 1       # costs-before; no confirm entry, no transfer, no clamp
BEGIN_SPELL = 2      # clamp; costs NOT before effect selection
BEGIN_RESPONSE = 3   # no clamp; costs NOT before
BEGIN_GATE_PORTAL = 4  # immediate (not queued): confirm-if-optional, no transfer, clamp, costs-before

_ENTER_CONFIRM = np.array([1, 0, 0, 0, 1], np.bool_)
_TRANSFER = np.array([1, 0, 0, 0, 0], np.bool_)
_CLAMP_EFFECT = np.array([1, 0, 1, 0, 1], np.bool_)
_COSTS_BEFORE = np.array([1, 1, 0, 0, 1], np.bool_)


def _np(table):
  return jnp.asarray(table)


def _scope_tables(scope_is_cost):
  ttype = jnp.where(
      scope_is_cost, _np(tables.COST_TARGET_TYPE), _np(tables.EFFECT_TARGET_TYPE)
  )
  tmin = jnp.where(scope_is_cost, _np(tables.COST_MIN), _np(tables.EFFECT_MIN))
  tmax = jnp.where(scope_is_cost, _np(tables.COST_MAX), _np(tables.EFFECT_MAX))
  return ttype, tmin, tmax


_T = {name: i for i, name in enumerate(tables.TARGET_TYPE_NAMES)}


def collect_targets(state: State, def_id, scope_is_cost, owner):
  """azk_collect_ability_target_choices.

  Returns (valid (MAX_TARGET_CHOICES,), inst (.,), player (.,)) where slot k
  corresponds to action_index k of the card's target type encoding. Per-card
  target validators are applied via cards_impl.target_validator()."""
  from azuki_jax.abilities import cards_impl

  ttype_tab, _, _ = _scope_tables(scope_is_cost)
  ttype = jnp.where(def_id >= 0, ttype_tab[jnp.maximum(def_id, 0)], 0)
  opp = (owner + 1) % 2

  k = jnp.arange(MAX_TARGET_CHOICES)
  inst = jnp.full((MAX_TARGET_CHOICES,), -1, jnp.int32)
  player = jnp.full((MAX_TARGET_CHOICES,), -1, jnp.int32)
  valid = jnp.zeros((MAX_TARGET_CHOICES,), jnp.bool_)

  def at(p, zone, slot):
    match = (state.zone[p] == zone) & (state.zpos[p] == jnp.asarray(slot, jnp.int8))
    return jnp.where(match.any(), jnp.argmax(match), -1).astype(jnp.int32)

  def leader_of(p):
    match = state.zone[p] == Zone.LEADER
    return jnp.where(match.any(), jnp.argmax(match), -1).astype(jnp.int32)

  def hand_at(p, pos):
    match = (state.zone[p] == Zone.HAND) & (state.zpos[p] == jnp.asarray(pos, jnp.int8))
    return jnp.where(match.any(), jnp.argmax(match), -1).astype(jnp.int32)

  # FRIENDLY_HAND / FRIENDLY_HAND_WEAPON: action_index = hand index
  is_hand = (ttype == _T["FRIENDLY_HAND"]) | (ttype == _T["FRIENDLY_HAND_WEAPON"])
  hand_inst = jax.vmap(lambda i: hand_at(owner, i))(k)
  inst = jnp.where(is_hand, hand_inst, inst)
  player = jnp.where(is_hand, owner, player)
  valid = jnp.where(is_hand & (hand_inst >= 0), True, valid)

  # FRIENDLY_GARDEN_ENTITY: slots 0..4, plus the pending gate-portal card at
  # its saved slot when it has LEFT the owner's garden (C
  # collect_pending_gate_portal_target)
  is_fg = ttype == _T["FRIENDLY_GARDEN_ENTITY"]
  fg_inst = jax.vmap(lambda i: at(owner, Zone.GARDEN, i))(jnp.minimum(k, 4))
  cond = is_fg & (k < GARDEN_SIZE)
  inst = jnp.where(cond, fg_inst, inst)
  player = jnp.where(cond, owner, player)
  valid = jnp.where(cond & (fg_inst >= 0), True, valid)

  portal_active = state.ab_scratch[2] == 1
  portaled = jnp.clip(state.ab_scratch[0].astype(jnp.int32), 0,
                      state.zone.shape[1] - 1)
  portal_slot = state.ab_scratch[1].astype(jnp.int32)
  portaled_in_garden = state.zone[owner, portaled] == Zone.GARDEN
  portal_case = (
      is_fg & portal_active & ~portaled_in_garden & (k == portal_slot)
  )
  inst = jnp.where(portal_case, portaled, inst)
  player = jnp.where(portal_case, owner, player)
  valid = jnp.where(portal_case, True, valid)

  # FRIENDLY_ALLEY_ENTITY
  is_fa = ttype == _T["FRIENDLY_ALLEY_ENTITY"]
  fa_inst = jax.vmap(lambda i: at(owner, Zone.ALLEY, i))(jnp.minimum(k, 4))
  cond = is_fa & (k < GARDEN_SIZE)
  inst = jnp.where(cond, fa_inst, inst)
  player = jnp.where(cond, owner, player)
  valid = jnp.where(cond & (fa_inst >= 0), True, valid)

  # ENEMY_GARDEN_ENTITY
  is_eg = ttype == _T["ENEMY_GARDEN_ENTITY"]
  eg_inst = jax.vmap(lambda i: at(opp, Zone.GARDEN, i))(jnp.minimum(k, 4))
  cond = is_eg & (k < GARDEN_SIZE)
  inst = jnp.where(cond, eg_inst, inst)
  player = jnp.where(cond, opp, player)
  valid = jnp.where(cond & (eg_inst >= 0), True, valid)

  # ENEMY_LEADER_OR_GARDEN_ENTITY: 0..4 enemy garden, 5 enemy leader
  is_elg = ttype == _T["ENEMY_LEADER_OR_GARDEN_ENTITY"]
  elg_inst = jnp.where(
      k < GARDEN_SIZE,
      jax.vmap(lambda i: at(opp, Zone.GARDEN, i))(jnp.minimum(k, 4)),
      jnp.where(k == GARDEN_SIZE, leader_of(opp), -1),
  )
  cond = is_elg & (k <= GARDEN_SIZE)
  inst = jnp.where(cond, elg_inst, inst)
  player = jnp.where(cond, opp, player)
  valid = jnp.where(cond & (elg_inst >= 0), True, valid)

  # FRIENDLY_GARDEN_OR_ALLEY_ENTITY: 0..4 garden, 5..9 alley
  is_fga = ttype == _T["FRIENDLY_GARDEN_OR_ALLEY_ENTITY"]
  fga_inst = jnp.where(
      k < GARDEN_SIZE,
      jax.vmap(lambda i: at(owner, Zone.GARDEN, i))(jnp.minimum(k, 4)),
      jax.vmap(lambda i: at(owner, Zone.ALLEY, i))(
          jnp.clip(k - GARDEN_SIZE, 0, 4)
      ),
  )
  cond = is_fga & (k < 2 * GARDEN_SIZE)
  inst = jnp.where(cond, fga_inst, inst)
  player = jnp.where(cond, owner, player)
  valid = jnp.where(cond & (fga_inst >= 0), True, valid)

  # ANY_GARDEN_ENTITY: 0..4 friendly, 5..9 enemy
  is_ag = ttype == _T["ANY_GARDEN_ENTITY"]
  ag_inst = jnp.where(
      k < GARDEN_SIZE,
      jax.vmap(lambda i: at(owner, Zone.GARDEN, i))(jnp.minimum(k, 4)),
      jax.vmap(lambda i: at(opp, Zone.GARDEN, i))(jnp.clip(k - GARDEN_SIZE, 0, 4)),
  )
  ag_player = jnp.where(k < GARDEN_SIZE, owner, opp)
  cond = is_ag & (k < 2 * GARDEN_SIZE)
  inst = jnp.where(cond, ag_inst, inst)
  player = jnp.where(cond, ag_player, player)
  valid = jnp.where(cond & (ag_inst >= 0), True, valid)

  # ANY_LEADER: 0 friendly, 1 enemy
  is_al = ttype == _T["ANY_LEADER"]
  al_inst = jnp.where(k == 0, leader_of(owner), jnp.where(k == 1, leader_of(opp), -1))
  al_player = jnp.where(k == 0, owner, opp)
  cond = is_al & (k <= 1)
  inst = jnp.where(cond, al_inst, inst)
  player = jnp.where(cond, al_player, player)
  valid = jnp.where(cond & (al_inst >= 0), True, valid)

  # ANY_LEADER_OR_GARDEN_ENTITY: garden encoding 0..9, friendly leader 10, enemy leader 11
  is_alg = ttype == _T["ANY_LEADER_OR_GARDEN_ENTITY"]
  alg_inst = jnp.where(
      k < 2 * GARDEN_SIZE,
      ag_inst,
      jnp.where(k == 10, leader_of(owner), leader_of(opp)),
  )
  alg_player = jnp.where(k < 2 * GARDEN_SIZE, ag_player,
                         jnp.where(k == 10, owner, opp))
  inst = jnp.where(is_alg, alg_inst, inst)
  player = jnp.where(is_alg, alg_player, player)
  valid = jnp.where(is_alg & (alg_inst >= 0), True, valid)

  # per-card target validator hook
  ok = jax.vmap(
      lambda target_inst, target_player, target_valid: jnp.where(
          target_valid,
          cards_impl.target_validator(
              state, def_id, scope_is_cost, owner,
              jnp.maximum(target_player, 0), jnp.maximum(target_inst, 0),
          ),
          False,
      )
  )(inst, player, valid)
  return ok, inst, player


def collect_target_order_keys(state: State, def_id, scope_is_cost, owner,
                              ok, inst, player):
  """Enumeration-order key per action_index (C iterates flecs ordered
  children = insertion order). Lower key = enumerated earlier. Hand targets
  keep list order; board targets use board_seq; zone groups (friendly-then-
  enemy / garden-then-alley / leaders-last) get segment offsets."""
  ttype_tab, _, _ = _scope_tables(scope_is_cost)
  ttype = jnp.where(def_id >= 0, ttype_tab[jnp.maximum(def_id, 0)], 0)
  k = jnp.arange(MAX_TARGET_CHOICES)
  opp = (owner + 1) % 2

  seq = state.board_seq[
      jnp.maximum(player, 0), jnp.maximum(inst, 0)
  ].astype(jnp.int32)

  is_hand = (ttype == _T["FRIENDLY_HAND"]) | (ttype == _T["FRIENDLY_HAND_WEAPON"])
  key = jnp.where(is_hand, k, seq)

  # second-zone segment offsets
  second_seg = jnp.zeros((MAX_TARGET_CHOICES,), jnp.int32)
  is_fga = ttype == _T["FRIENDLY_GARDEN_OR_ALLEY_ENTITY"]
  second_seg = jnp.where(is_fga & (k >= GARDEN_SIZE), 1 << 14, second_seg)
  is_ag = ttype == _T["ANY_GARDEN_ENTITY"]
  second_seg = jnp.where(is_ag & (player == opp), 1 << 14, second_seg)
  is_alg = ttype == _T["ANY_LEADER_OR_GARDEN_ENTITY"]
  second_seg = jnp.where(
      is_alg & (player == opp) & (k < 2 * GARDEN_SIZE), 1 << 14, second_seg
  )
  second_seg = jnp.where(is_alg & (k == 10), 1 << 15, second_seg)
  second_seg = jnp.where(is_alg & (k == 11), (1 << 15) + 1, second_seg)
  is_elg = ttype == _T["ENEMY_LEADER_OR_GARDEN_ENTITY"]
  second_seg = jnp.where(is_elg & (k == GARDEN_SIZE), 1 << 15, second_seg)
  is_al = ttype == _T["ANY_LEADER"]
  key = jnp.where(is_al, k, key)

  key = key + second_seg
  return jnp.where(ok, key, 1 << 20)


def count_targets(state: State, def_id, scope_is_cost, owner):
  ok, _, _ = collect_targets(state, def_id, scope_is_cost, owner)
  return jnp.sum(ok, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# begin / finish
# ---------------------------------------------------------------------------

def _clear_context(state: State) -> State:
  """azk_clear_ability_context: once-per-turn mark + restore control."""
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  has_src = state.ab_source >= 0
  def_id = state.def_id[owner, src]
  once = jnp.where(
      has_src & (def_id >= 0), _np(tables.ONCE_PER_TURN)[jnp.maximum(def_id, 0)], False
  )
  used = state.once_per_turn_used.at[owner, src].set(
      jnp.where(once, state.once_per_turn_used[owner, src] | 1,
                state.once_per_turn_used[owner, src])
  )
  restore = state.ab_restores_active & (state.ab_saved_active >= 0)
  active = jnp.where(restore, state.ab_saved_active, state.active_player)
  return state._replace(
      once_per_turn_used=used,
      active_player=active.astype(jnp.int8),
      ab_phase=jnp.int8(AbilityPhase.NONE),
      ab_source=jnp.int8(-1),
      ab_owner=jnp.int8(-1),
      ab_slot=jnp.int8(0),
      ab_is_optional=jnp.bool_(False),
      ab_costs_applied=jnp.bool_(False),
      ab_saved_active=jnp.int8(-1),
      ab_restores_active=jnp.bool_(False),
      ab_cost_selected=jnp.int8(0),
      ab_cost_max=jnp.int8(0),
      ab_cost_targets=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_cost_target_players=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_eff_selected=jnp.int8(0),
      ab_eff_min=jnp.int8(0),
      ab_eff_max=jnp.int8(0),
      ab_eff_targets=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_eff_target_players=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      # azk_reset_ability_context_state zeroes the WHOLE context, including
      # the selection block (ability_context.c azk_clear_ability_context ->
      # azk_reset_ability_context_state)
      ab_sel_cards=jnp.full(state.ab_sel_cards.shape, -1, jnp.int8),
      ab_sel_count=jnp.int8(0),
      ab_sel_picked=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_sel_picked_count=jnp.int8(0),
      ab_sel_pick_max=jnp.int8(0),
      ab_scratch=state.ab_scratch.at[0:3].set(0),
  )


def _where_state(pred, a: State, b: State) -> State:
  return jax.tree.map(lambda x, y: jnp.where(pred, x, y), a, b)


def _apply_costs(state: State, do) -> State:
  from azuki_jax.abilities import cards_impl

  out = cards_impl.dispatch_apply_costs(state)
  out = out._replace(ab_costs_applied=jnp.bool_(True))
  return _where_state(do, out, state)


def _apply_effects(state: State, do) -> State:
  from azuki_jax.abilities import cards_impl

  out = cards_impl.dispatch_apply_effects(state)
  return _where_state(do, out, state)


def _enter_initial_phase(state: State, def_id, owner, begin_kind, do) -> State:
  """azk_enter_initial_phase + finish for the immediate-resolve path."""
  from azuki_jax.abilities import cards_impl

  cost_min = jnp.where(def_id >= 0, _np(tables.COST_MIN)[jnp.maximum(def_id, 0)], 0)
  costs_before = _np(_COSTS_BEFORE)[begin_kind]

  # cost selection needed?
  to_cost = do & (cost_min > 0)
  state = state._replace(
      ab_phase=jnp.where(to_cost, jnp.int8(AbilityPhase.COST_SELECTION), state.ab_phase)
  )

  # on_cost_paid hook (C: apply costs, call hook — which usually reveals into
  # the selection zone and sets SELECTION_PICK/BOTTOM_DECK, or overrides the
  # ctx effect min/max and enters EFFECT_SELECTION; active iff the hook left
  # a phase set)
  has_ocp = jnp.where(
      def_id >= 0, _np(cards_impl.HAS_ON_COST_PAID)[jnp.maximum(def_id, 0)], False
  )
  run_ocp = do & ~to_cost & has_ocp
  state = _apply_costs(state, run_ocp)
  hooked = cards_impl.dispatch_on_cost_paid(state)
  state = _where_state(run_ocp, hooked, state)
  ocp_done = run_ocp & (state.ab_phase == AbilityPhase.NONE)
  cleared_ocp = _clear_context(state)
  state = _where_state(ocp_done, cleared_ocp, state)

  # effect selection? (select_effects_when_max_positive is TRUE in all four
  # begin variants, so: enter when ctx effect.max_allowed > 0). The ctx value
  # was initialized at begin (azk_init_ability_context) — C should_enter reads
  # ctx->effect.max_allowed, not the def table.
  max_allowed = state.ab_eff_max.astype(jnp.int32)
  to_effect_pre = do & ~to_cost & ~run_ocp & (max_allowed > 0)
  # immediate resolve: no cost selection, no effect selection, no hook
  immediate = do & ~to_cost & ~run_ocp & (max_allowed == 0)

  # single costs dispatch covers both the costs-before-effect-selection path
  # and the immediate-resolve path (entry always has costs_applied == False;
  # the two predicates are disjoint and immediate ignores `remaining`)
  apply_now = to_effect_pre & costs_before
  state = _apply_costs(state, apply_now | immediate)
  remaining = count_targets(state, def_id, jnp.bool_(False), owner)
  eff_max_table = jnp.where(
      def_id >= 0, _np(tables.EFFECT_MAX)[jnp.maximum(def_id, 0)], 0
  ).astype(jnp.int32)
  none_left = apply_now & (remaining == 0)
  to_effect = to_effect_pre & ~none_left

  # context max_allowed: re-clamped after costs when costs ran
  # (azk_prepare_effect_selection_after_costs: max = clamp(avail, def max),
  # min lowered to max), else the begin-variant value
  ctx_max = jnp.where(
      apply_now, jnp.minimum(eff_max_table, remaining), max_allowed
  )
  ctx_min = jnp.where(
      apply_now,
      jnp.minimum(state.ab_eff_min.astype(jnp.int32), ctx_max),
      state.ab_eff_min.astype(jnp.int32),
  )
  state = state._replace(
      ab_phase=jnp.where(to_effect, jnp.int8(AbilityPhase.EFFECT_SELECTION), state.ab_phase),
      ab_eff_max=jnp.where(to_effect, ctx_max, state.ab_eff_max).astype(jnp.int8),
      ab_eff_min=jnp.where(to_effect, ctx_min, state.ab_eff_min).astype(jnp.int8),
  )

  # single effects dispatch: none-left-after-costs and immediate-resolve
  state = _apply_effects(state, none_left | immediate)

  finished = immediate | none_left
  cleared = _clear_context(state)
  return _where_state(finished, cleared, state)


def _effect_max_allowed(state: State, def_id, owner, begin_kind):
  eff_max = jnp.where(def_id >= 0, _np(tables.EFFECT_MAX)[jnp.maximum(def_id, 0)], 0)
  clamp = _np(_CLAMP_EFFECT)[begin_kind]
  available = count_targets(state, def_id, jnp.bool_(False), owner)
  eff_min = jnp.where(def_id >= 0, _np(tables.EFFECT_MIN)[jnp.maximum(def_id, 0)], 0)
  clamped = jnp.minimum(available, eff_max)
  clamped = jnp.maximum(clamped, eff_min)  # context-init keeps max >= min
  return jnp.where(clamp, clamped, eff_max)


def begin_ability(state: State, owner, src_inst, begin_kind, do) -> State:
  """azk_trigger_* + azk_begin_ability for an IMPLEMENTED card.

  Precondition checks done by callers (timing, frozen, once-per-turn, cost
  availability, validate())."""
  def_id = state.def_id[
      jnp.maximum(owner.astype(jnp.int32), 0), jnp.maximum(src_inst, 0)
  ]
  optional = jnp.where(
      def_id >= 0, _np(tables.IS_OPTIONAL)[jnp.maximum(def_id, 0)], False
  )

  init = state._replace(
      ab_source=src_inst.astype(jnp.int8),
      ab_owner=owner.astype(jnp.int8),
      ab_is_optional=optional,
      ab_costs_applied=jnp.bool_(False),
      ab_saved_active=jnp.int8(-1),
      ab_restores_active=jnp.bool_(False),
      ab_cost_selected=jnp.int8(0),
      ab_eff_selected=jnp.int8(0),
  )
  state = _where_state(do, init, state)

  # azk_init_ability_context: cost.max_allowed = min(available at begin,
  # def->cost_req.max); FIXED for the rest of the resolution (the dynamic
  # availability may shrink as validators exclude already-selected targets).
  cost_max_table = jnp.where(
      def_id >= 0, _np(tables.COST_MAX)[jnp.maximum(def_id, 0)], 0
  ).astype(jnp.int32)
  cost_avail = count_targets(
      state, def_id, jnp.bool_(True), owner.astype(jnp.int32)
  )
  state = state._replace(
      ab_cost_max=jnp.where(
          do, jnp.minimum(cost_max_table, cost_avail), state.ab_cost_max
      ).astype(jnp.int8)
  )

  # azk_init_ability_target_state(effect): min_required = min(def min, def
  # max), max_allowed = def max (clamped to availability for the clamp begin
  # variants, raised back to min). Hooks (on_cost_paid) may override both.
  eff_init_max = _effect_max_allowed(state, def_id, owner, begin_kind)
  eff_min_table = jnp.where(
      def_id >= 0, _np(tables.EFFECT_MIN)[jnp.maximum(def_id, 0)], 0
  ).astype(jnp.int32)
  eff_max_table = jnp.where(
      def_id >= 0, _np(tables.EFFECT_MAX)[jnp.maximum(def_id, 0)], 0
  ).astype(jnp.int32)
  state = state._replace(
      ab_eff_max=jnp.where(do, eff_init_max, state.ab_eff_max).astype(jnp.int8),
      ab_eff_min=jnp.where(
          do, jnp.minimum(eff_min_table, eff_max_table), state.ab_eff_min
      ).astype(jnp.int8),
  )

  # confirmation entry (triggered abilities only)
  enter_confirm = (
      do & optional & _np(_ENTER_CONFIRM)[begin_kind]
  )
  transfer = _np(_TRANSFER)[begin_kind]
  needs_transfer = enter_confirm & transfer & (state.active_player != owner.astype(jnp.int8))
  confirmed = state._replace(
      ab_phase=jnp.int8(AbilityPhase.CONFIRMATION),
      ab_restores_active=needs_transfer,
      ab_saved_active=jnp.where(needs_transfer, state.active_player, jnp.int8(-1)),
      active_player=jnp.where(
          needs_transfer, owner.astype(jnp.int8), state.active_player
      ),
  )
  state = _where_state(enter_confirm, confirmed, state)

  # straight to initial phase otherwise
  run_initial = do & ~enter_confirm
  state = _enter_initial_phase(state, def_id, owner, begin_kind, run_initial)

  # transfer control if a phase is now active and variant transfers
  active_phase = state.ab_phase != AbilityPhase.NONE
  needs_transfer2 = (
      run_initial & active_phase & transfer
      & (state.active_player != owner.astype(jnp.int8))
  )
  transferred = state._replace(
      ab_restores_active=jnp.bool_(True),
      ab_saved_active=state.active_player,
      active_player=owner.astype(jnp.int8),
  )
  return _where_state(needs_transfer2, transferred, state)


# ---------------------------------------------------------------------------
# user-action handlers (ability phases)
# ---------------------------------------------------------------------------

def process_confirm(state: State, do) -> State:
  """ACT_CONFIRM_ABILITY in CONFIRMATION."""
  do = do & (state.ab_phase == AbilityPhase.CONFIRMATION)
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  def_id = state.def_id[owner, src]

  cost_min = jnp.where(def_id >= 0, _np(tables.COST_MIN)[jnp.maximum(def_id, 0)], 0)
  cost_avail = count_targets(state, def_id, jnp.bool_(True), owner)
  no_cost_targets = do & (cost_min > 0) & (cost_avail < cost_min)
  cleared = _clear_context(state)
  state = _where_state(no_cost_targets, cleared, state)

  proceed = do & ~no_cost_targets
  # C confirm path re-clamps cost.max_allowed down to fresh availability
  reclamp = proceed & (cost_min > 0) & (
      state.ab_cost_max.astype(jnp.int32) > cost_avail
  )
  state = state._replace(
      ab_cost_max=jnp.where(reclamp, cost_avail, state.ab_cost_max).astype(
          jnp.int8
      )
  )
  # confirmation always uses costs-before semantics (matches C confirm path)
  return _enter_initial_phase(state, def_id, owner, BEGIN_TRIGGERED, proceed)


def process_decline(state: State, do) -> State:
  """ACT_NOOP in CONFIRMATION (optional only)."""
  do = do & (state.ab_phase == AbilityPhase.CONFIRMATION) & state.ab_is_optional
  cleared = _clear_context(state)
  return _where_state(do, cleared, state)


def _finish_cost_selection(state: State, do) -> State:
  """finish_cost_selection: apply costs, on_cost_paid hook (may enter the
  selection phases), then effect selection or resolve."""
  from azuki_jax.abilities import cards_impl

  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  def_id = state.def_id[owner, src]

  state = _apply_costs(state, do)

  has_ocp = jnp.where(
      def_id >= 0, _np(cards_impl.HAS_ON_COST_PAID)[jnp.maximum(def_id, 0)], False
  )
  run_ocp = do & has_ocp
  hooked = cards_impl.dispatch_on_cost_paid(state)
  state = _where_state(run_ocp, hooked, state)
  in_selection = (state.ab_phase == AbilityPhase.SELECTION_PICK) | (
      state.ab_phase == AbilityPhase.BOTTOM_DECK
  )
  hook_pauses = run_ocp & in_selection
  do = do & ~hook_pauses

  # gate on the ctx value (C reads ctx->effect.max_allowed, which the hook
  # may have just modified); the reclamp uses the def table
  # (azk_prepare_effect_selection_after_costs)
  eff_max = jnp.where(def_id >= 0, _np(tables.EFFECT_MAX)[jnp.maximum(def_id, 0)], 0)
  has_effect_targets = state.ab_eff_max.astype(jnp.int32) > 0
  remaining = count_targets(state, def_id, jnp.bool_(False), owner)
  to_effect = do & has_effect_targets & (remaining > 0)
  exhausted = do & has_effect_targets & (remaining == 0)
  resolve_now = do & ~has_effect_targets

  new_max = jnp.minimum(eff_max.astype(jnp.int32), remaining)
  state = state._replace(
      ab_phase=jnp.where(
          to_effect, jnp.int8(AbilityPhase.EFFECT_SELECTION), state.ab_phase
      ),
      ab_eff_max=jnp.where(to_effect, new_max, state.ab_eff_max).astype(jnp.int8),
      ab_eff_min=jnp.where(
          to_effect,
          jnp.minimum(state.ab_eff_min.astype(jnp.int32), new_max),
          state.ab_eff_min,
      ).astype(jnp.int8),
  )
  state = _apply_effects(state, exhausted | resolve_now)
  cleared = _clear_context(state)
  return _where_state(exhausted | resolve_now, cleared, state)


def process_cost_action(state: State, target_index, do_select, do_skip) -> State:
  """ACT_SELECT_COST_TARGET / ACT_NOOP in COST_SELECTION (shared finish tail
  so the cost/effect hook dispatch is instantiated once)."""
  in_cost = state.ab_phase == AbilityPhase.COST_SELECTION
  do_select = do_select & in_cost
  do_skip = do_skip & in_cost
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  def_id = state.def_id[owner, src]

  ok, inst, player = collect_targets(state, def_id, jnp.bool_(True), owner)
  ti = jnp.clip(target_index, 0, MAX_TARGET_CHOICES - 1)
  do_select = do_select & ok[ti]

  sel = jnp.clip(state.ab_cost_selected.astype(jnp.int32), 0, MAX_ABILITY_SELECTION - 1)
  state = state._replace(
      ab_cost_targets=state.ab_cost_targets.at[sel].set(
          jnp.where(do_select, inst[ti].astype(jnp.int8), state.ab_cost_targets[sel])
      ),
      ab_cost_target_players=state.ab_cost_target_players.at[sel].set(
          jnp.where(do_select, player[ti].astype(jnp.int8), state.ab_cost_target_players[sel])
      ),
      ab_cost_selected=jnp.where(
          do_select, state.ab_cost_selected + 1, state.ab_cost_selected
      ).astype(jnp.int8),
  )

  cost_max = _cost_max_allowed(state, def_id, owner)
  select_finish = do_select & (state.ab_cost_selected >= cost_max)
  cost_min = jnp.where(def_id >= 0, _np(tables.COST_MIN)[jnp.maximum(def_id, 0)], 0)
  skip_finish = do_skip & (state.ab_cost_selected >= cost_min)
  return _finish_cost_selection(state, select_finish | skip_finish)


def _cost_max_allowed(state: State, def_id, owner):
  """The context's cost.max_allowed (begin-clamped, fixed thereafter)."""
  del def_id, owner
  return state.ab_cost_max.astype(jnp.int32)


def process_effect_action(state: State, target_index, do_select, do_skip) -> State:
  """ACT_SELECT_EFFECT_TARGET / ACT_NOOP in EFFECT_SELECTION (shared tail)."""
  in_effect = state.ab_phase == AbilityPhase.EFFECT_SELECTION
  do_select = do_select & in_effect
  do_skip = do_skip & in_effect
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  def_id = state.def_id[owner, src]

  ok, inst, player = collect_targets(state, def_id, jnp.bool_(False), owner)
  ti = jnp.clip(target_index, 0, MAX_TARGET_CHOICES - 1)
  do_select = do_select & ok[ti]
  remaining = jnp.sum(ok, dtype=jnp.int32)

  sel = jnp.clip(state.ab_eff_selected.astype(jnp.int32), 0, MAX_ABILITY_SELECTION - 1)
  state = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[sel].set(
          jnp.where(do_select, inst[ti].astype(jnp.int8), state.ab_eff_targets[sel])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[sel].set(
          jnp.where(do_select, player[ti].astype(jnp.int8), state.ab_eff_target_players[sel])
      ),
      ab_eff_selected=jnp.where(
          do_select, state.ab_eff_selected + 1, state.ab_eff_selected
      ).astype(jnp.int8),
  )

  # selection done when selected >= context max_allowed (set at entry)
  max_allowed = _ctx_effect_max(state, def_id, owner)
  select_finish = do_select & (state.ab_eff_selected >= max_allowed)
  # NOOP allowed when ctx min reached or no targets remain (C reads
  # ctx->effect.min_required, which hooks may have raised)
  eff_min = state.ab_eff_min.astype(jnp.int32)
  skip_finish = do_skip & (
      (state.ab_eff_selected >= eff_min) | (remaining == 0)
  )

  finish = select_finish | skip_finish
  state = _apply_costs(state, finish & ~state.ab_costs_applied)
  state = _apply_effects(state, finish)
  # azk_process_effect_selection/skip: apply_effects may set a NEW phase
  # (e.g. AZK01-111 moves hand cards into the selection zone); when it did
  # (phase changed and != NONE) the FSM pauses instead of clearing
  paused = finish & (state.ab_phase != AbilityPhase.EFFECT_SELECTION) & (
      state.ab_phase != AbilityPhase.NONE
  )
  cleared = _clear_context(state)
  return _where_state(finish & ~paused, cleared, state)


def _ctx_effect_max(state: State, def_id, owner):
  """The context's effect.max_allowed, persisted at EFFECT_SELECTION entry."""
  del def_id, owner
  return jnp.maximum(state.ab_eff_max.astype(jnp.int32), 1)
