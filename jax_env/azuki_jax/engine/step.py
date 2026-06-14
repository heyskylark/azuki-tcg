"""Engine step: action dispatch + auto-resolve loop (azk_engine_tick ladder).

`engine_step(state, action)` = submit one user action at a decision point,
then auto-advance until the next decision point or game over. This matches
the observable behavior of tcg.h c_step's engine portion (rewards/obs are
layered on in azuki_jax.step).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from azuki_jax.abilities.effects import (
    queue_end_of_turn_abilities,
    resolve_triggered_effect,
)
from azuki_jax.constants import Act, Phase, Zone
from azuki_jax.engine import apply as apply_mod
from azuki_jax.engine.phases import (
    combat_resolve,
    end_turn,
    phase_gate,
    requires_action,
    start_of_turn,
)
from azuki_jax.engine.triggers import has_queued, pop_effect
from azuki_jax.state import State

MAX_AUTO_TICKS = 64


def _where_state(pred, a: State, b: State) -> State:
  return jax.tree.map(lambda x, y: jnp.where(pred, x, y), a, b)


def apply_user_action(state: State, action: jax.Array) -> State:
  """Dispatch one (assumed legal) user action [type, s1, s2, s3]."""
  act = action[0]
  s1, s2, s3 = action[1], action[2], action[3]
  use_token = s3 != 0
  in_mulligan = state.phase == Phase.PREGAME_MULLIGAN
  in_main = state.phase == Phase.MAIN
  in_response = state.phase == Phase.RESPONSE_WINDOW

  out = state

  # Mulligan phase consumes NOOP/MULLIGAN_SHUFFLE
  mull = apply_mod.apply_mulligan(state, act)
  out = _where_state(in_mulligan, mull, out)

  battle = in_main | in_response

  played_g = apply_mod.apply_play_entity(
      state, Zone.GARDEN, s1, s2, use_token,
      do=battle & (act == Act.PLAY_ENTITY_TO_GARDEN),
  )
  out = _where_state(battle & (act == Act.PLAY_ENTITY_TO_GARDEN), played_g, out)

  played_a = apply_mod.apply_play_entity(
      state, Zone.ALLEY, s1, s2, use_token,
      do=battle & (act == Act.PLAY_ENTITY_TO_ALLEY),
  )
  out = _where_state(battle & (act == Act.PLAY_ENTITY_TO_ALLEY), played_a, out)

  attached = apply_mod.apply_attach_weapon(
      state, s1, s2, use_token,
      do=battle & (act == Act.ATTACH_WEAPON_FROM_HAND),
  )
  out = _where_state(battle & (act == Act.ATTACH_WEAPON_FROM_HAND), attached, out)

  portal = apply_mod.apply_gate_portal(
      state, s1, s2, do=in_main & (act == Act.GATE_PORTAL)
  )
  out = _where_state(in_main & (act == Act.GATE_PORTAL), portal, out)

  attacked = apply_mod.apply_attack(
      state, s1, s2, do=in_main & (act == Act.ATTACK)
  )
  out = _where_state(in_main & (act == Act.ATTACK), attacked, out)

  defended = apply_mod.apply_declare_defender(
      state, s1, do=in_response & (act == Act.DECLARE_DEFENDER)
  )
  out = _where_state(in_response & (act == Act.DECLARE_DEFENDER), defended, out)

  in_ability = state.ab_phase != 0
  battle_noop = ~in_ability

  noop_main = apply_mod.apply_noop_main(
      state, do=in_main & battle_noop & (act == Act.NOOP)
  )
  out = _where_state(in_main & battle_noop & (act == Act.NOOP), noop_main, out)

  noop_resp = apply_mod.apply_noop_response(
      state, do=in_response & battle_noop & (act == Act.NOOP)
  )
  out = _where_state(in_response & battle_noop & (act == Act.NOOP), noop_resp, out)

  # --- ability-phase actions (AbilityResolution system) ---
  from azuki_jax.abilities import runtime

  ab = state.ab_phase
  confirm = in_ability & (act == Act.CONFIRM_ABILITY)
  out = _where_state(confirm, runtime.process_confirm(state, confirm), out)

  ab_noop = in_ability & (act == Act.NOOP)
  decline = ab_noop & (ab == 1)
  out = _where_state(decline, runtime.process_decline(state, decline), out)

  cost_skip = ab_noop & (ab == 2)
  cost_sel = in_ability & (act == Act.SELECT_COST_TARGET)
  out = _where_state(
      cost_sel | cost_skip,
      runtime.process_cost_action(state, s1, cost_sel, cost_skip),
      out,
  )
  # selection-zone phases (4 = SELECTION_PICK, 5 = BOTTOM_DECK)
  from azuki_jax.abilities import selection as sel_mod

  pick_skip = ab_noop & (ab == 4)
  out = _where_state(
      pick_skip, sel_mod.process_skip_selection(state, pick_skip), out
  )
  pick = in_ability & (act == Act.SELECT_FROM_SELECTION)
  out = _where_state(pick, sel_mod.process_selection_pick(state, s1, pick), out)
  to_garden = in_ability & (act == Act.SELECT_TO_GARDEN)
  out = _where_state(
      to_garden, sel_mod.process_selection_to_garden(state, s1, s2, to_garden), out
  )
  to_alley = in_ability & (act == Act.SELECT_TO_ALLEY)
  out = _where_state(
      to_alley, sel_mod.process_selection_to_alley(state, s1, s2, to_alley), out
  )
  to_equip = in_ability & (act == Act.SELECT_TO_EQUIP)
  out = _where_state(
      to_equip, sel_mod.process_selection_to_equip(state, s1, s2, to_equip), out
  )
  bottom = in_ability & (act == Act.BOTTOM_DECK_CARD)
  out = _where_state(bottom, sel_mod.process_bottom_deck(state, s1, bottom), out)
  top = in_ability & (act == Act.TOP_DECK_CARD)
  out = _where_state(top, sel_mod.process_top_deck(state, s1, top), out)
  bottom_all = in_ability & (act == Act.BOTTOM_DECK_ALL)
  out = _where_state(
      bottom_all, sel_mod.process_bottom_deck_all(state, bottom_all), out
  )

  eff_skip = ab_noop & (ab == 3)
  eff_sel = in_ability & (act == Act.SELECT_EFFECT_TARGET)
  out = _where_state(
      eff_sel | eff_skip,
      runtime.process_effect_action(state, s1, eff_sel, eff_skip),
      out,
  )

  # --- spell / ability activations (main + response phases), merged so the
  # begin flow (and card hook dispatch) is instantiated once ---
  battle_act = battle_noop & battle
  spell = battle_act & (act == Act.PLAY_SPELL_FROM_HAND)
  act_garden = battle_act & (act == Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
  act_alley = battle_act & (act == Act.ACTIVATE_ALLEY_ABILITY)
  any_ability_act = spell | act_garden | act_alley
  out = _where_state(
      any_ability_act,
      apply_mod.apply_ability_action(
          state, s1, s2, use_token, spell, act_garden, act_alley
      ),
      out,
  )

  # passive auras: C observers fire during the action and the binding ticks
  # until the passive queue drains before exposing the next decision point
  # (azk_engine_requires_action returns false while buffs are pending,
  # azuki_engine.c:280). recompute_passives self-gates on ab_phase == 0
  # (drain deferred while an ability resolves) and winner == -1.
  from azuki_jax.abilities.passives import recompute_passives

  out = recompute_passives(out)
  return out


def micro_tick(state: State) -> State:
  """One azk_engine_tick: queues first, then phase gate, then phase systems."""
  in_ability = state.ab_phase != 0
  queued = has_queued(state) & ~in_ability

  # 1) process triggered queue head
  popped, src, owner, timing = pop_effect(state)
  popped = resolve_triggered_effect(popped, src, owner, timing)
  state = _where_state(queued, popped, state)

  # 2) phase gate (only when nothing was popped)
  gated = phase_gate(state)
  state = _where_state(~queued, gated, state)

  # 2b) response auto-close (HandleResponseAction head): if the defender has
  # no options left, transition to combat resolve without consuming an action
  from azuki_jax.engine.phases import defender_can_respond, transition_to_combat_resolve

  in_response = (state.phase == Phase.RESPONSE_WINDOW) & ~queued
  dead_response = in_response & ~has_queued(state) & (state.ab_phase == 0) & (
      ~defender_can_respond(state, state.active_player)
  )
  closed = transition_to_combat_resolve(state, do=dead_response)
  state = _where_state(dead_response, closed, state)

  # 3) phase systems for auto phases (suppressed while an ability FSM is
  # active — the C phase gate routes exclusively to the ability pipeline)
  in_ab = state.ab_phase != 0
  is_start = (state.phase == Phase.START_OF_TURN) & ~queued & ~in_ab
  started = start_of_turn(state, do=is_start)
  state = _where_state(is_start, started, state)

  is_combat = (
      (state.phase == Phase.COMBAT_RESOLVE) & ~queued & ~has_queued(state) & ~in_ab
  )
  resolved = combat_resolve(state, do=is_combat)
  state = _where_state(is_combat, resolved, state)

  is_end = (state.phase == Phase.END_TURN) & ~queued & ~has_queued(state) & ~in_ab
  # EOT abilities first (queue then process), then the cleanup pass
  needs_eot_q = is_end & ~state.eot_abilities_queued
  eot_q = queue_end_of_turn_abilities(state)._replace(
      eot_abilities_queued=jnp.bool_(True)
  )
  state = _where_state(needs_eot_q, eot_q, state)
  run_end = is_end & state.eot_abilities_queued & ~has_queued(state)
  ended = end_turn(state, do=run_end)
  state = _where_state(run_end, ended, state)

  # passive auras (C tick-ladder step 2). Placed at the END of the tick
  # rather than the top: end-of-tick(k) is top-of-tick(k+1) for the loop
  # interior, and additionally covers the loop EXIT — C never exposes a
  # decision point with pending passive buffs (azk_engine_requires_action is
  # false until the queue drains), so the final tick's aura changes must be
  # applied before auto_resolve's requires_action check can stop the loop.
  from azuki_jax.abilities.passives import recompute_passives

  state = recompute_passives(state)
  return state


def auto_resolve(state: State) -> State:
  """Tick until requires_action or game over (bounded)."""

  def cond(st):
    return (
        ~requires_action(st)
        & (st.winner == -1)
        & (st.tick_guard < MAX_AUTO_TICKS)
    )

  def body(st):
    st = micro_tick(st)
    return st._replace(tick_guard=st.tick_guard + 1)

  state = state._replace(tick_guard=jnp.int32(0))
  state = jax.lax.while_loop(cond, body, state)
  return state


def engine_step(state: State, action: jax.Array) -> State:
  state = apply_user_action(state, action)
  return auto_resolve(state)


def stabilize(state: State) -> State:
  """stabilize_new_engine: auto-tick a fresh game to its first decision."""
  return auto_resolve(state)
