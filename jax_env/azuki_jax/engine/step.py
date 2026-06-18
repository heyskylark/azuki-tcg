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
from azuki_jax.constants import ACTION_TYPE_COUNT, Act, AbilityPhase, Phase, Zone
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

  def identity(st: State) -> State:
    return st

  def action_index():
    return jnp.clip(
        act.astype(jnp.int32), jnp.int32(0), jnp.int32(ACTION_TYPE_COUNT - 1)
    )

  def battle_phase(st: State):
    return (st.phase == Phase.MAIN) | (st.phase == Phase.RESPONSE_WINDOW)

  def normal_noop(st: State) -> State:
    return jax.lax.cond(
        st.phase == Phase.MAIN,
        lambda x: apply_mod.apply_noop_main(x, do=True),
        lambda x: jax.lax.cond(
            x.phase == Phase.RESPONSE_WINDOW,
            lambda y: apply_mod.apply_noop_response(y, do=True),
            identity,
            x,
        ),
        st,
    )

  def normal_action(st: State) -> State:
    """Action dispatcher when no ability FSM is active."""

    def play_garden(x):
      return apply_mod.apply_play_entity(
          x, Zone.GARDEN, s1, s2, use_token, do=battle_phase(x)
      )

    def play_alley(x):
      return apply_mod.apply_play_entity(
          x, Zone.ALLEY, s1, s2, use_token, do=battle_phase(x)
      )

    def attach_weapon(x):
      return apply_mod.apply_attach_weapon(
          x, s1, s2, use_token, do=battle_phase(x)
      )

    def gate_portal(x):
      return apply_mod.apply_gate_portal(
          x, s1, s2, do=x.phase == Phase.MAIN
      )

    def attack(x):
      return apply_mod.apply_attack(x, s1, s2, do=x.phase == Phase.MAIN)

    def declare_defender(x):
      return apply_mod.apply_declare_defender(
          x, s1, do=x.phase == Phase.RESPONSE_WINDOW
      )

    def play_spell(x):
      b = battle_phase(x)
      return apply_mod.apply_ability_action(
          x, s1, s2, use_token, b, jnp.bool_(False), jnp.bool_(False)
      )

    def activate_garden(x):
      b = battle_phase(x)
      return apply_mod.apply_ability_action(
          x, s1, s2, use_token, jnp.bool_(False), b, jnp.bool_(False)
      )

    def activate_alley(x):
      b = battle_phase(x)
      return apply_mod.apply_ability_action(
          x, s1, s2, use_token, jnp.bool_(False), jnp.bool_(False), b
      )

    branches = [identity for _ in range(ACTION_TYPE_COUNT)]
    branches[int(Act.NOOP)] = normal_noop
    branches[int(Act.PLAY_ENTITY_TO_GARDEN)] = play_garden
    branches[int(Act.PLAY_ENTITY_TO_ALLEY)] = play_alley
    branches[int(Act.ATTACH_WEAPON_FROM_HAND)] = attach_weapon
    branches[int(Act.GATE_PORTAL)] = gate_portal
    branches[int(Act.ATTACK)] = attack
    branches[int(Act.DECLARE_DEFENDER)] = declare_defender
    branches[int(Act.PLAY_SPELL_FROM_HAND)] = play_spell
    branches[int(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)] = activate_garden
    branches[int(Act.ACTIVATE_ALLEY_ABILITY)] = activate_alley
    return jax.lax.switch(action_index(), branches, st)

  def ability_noop(st: State) -> State:
    from azuki_jax.abilities import runtime
    from azuki_jax.abilities import selection as sel_mod

    def decline(x):
      return runtime.process_decline(x, jnp.bool_(True))

    def cost_skip(x):
      return runtime.process_cost_action(
          x, s1, jnp.bool_(False), jnp.bool_(True)
      )

    def effect_skip(x):
      return runtime.process_effect_action(
          x, s1, jnp.bool_(False), jnp.bool_(True)
      )

    def pick_skip(x):
      return sel_mod.process_skip_selection(x, jnp.bool_(True))

    branches = [identity for _ in range(6)]
    branches[int(AbilityPhase.CONFIRMATION)] = decline
    branches[int(AbilityPhase.COST_SELECTION)] = cost_skip
    branches[int(AbilityPhase.EFFECT_SELECTION)] = effect_skip
    branches[int(AbilityPhase.SELECTION_PICK)] = pick_skip
    phase = jnp.clip(st.ab_phase.astype(jnp.int32), 0, 5)
    return jax.lax.switch(phase, branches, st)

  def ability_action(st: State) -> State:
    """Action dispatcher while the ability FSM owns input."""
    from azuki_jax.abilities import runtime
    from azuki_jax.abilities import selection as sel_mod

    def confirm(x):
      return runtime.process_confirm(x, jnp.bool_(True))

    def cost_select(x):
      return runtime.process_cost_action(
          x, s1, jnp.bool_(True), jnp.bool_(False)
      )

    def effect_select(x):
      return runtime.process_effect_action(
          x, s1, jnp.bool_(True), jnp.bool_(False)
      )

    def selection_pick(x):
      return sel_mod.process_selection_pick(x, s1, jnp.bool_(True))

    def selection_to_garden(x):
      return sel_mod.process_selection_to_garden(
          x, s1, s2, jnp.bool_(True)
      )

    def selection_to_alley(x):
      return sel_mod.process_selection_to_alley(x, s1, s2, jnp.bool_(True))

    def selection_to_equip(x):
      return sel_mod.process_selection_to_equip(x, s1, s2, jnp.bool_(True))

    def bottom_deck(x):
      return sel_mod.process_bottom_deck(x, s1, jnp.bool_(True))

    def top_deck(x):
      return sel_mod.process_top_deck(x, s1, jnp.bool_(True))

    def bottom_deck_all(x):
      return sel_mod.process_bottom_deck_all(x, jnp.bool_(True))

    branches = [identity for _ in range(ACTION_TYPE_COUNT)]
    branches[int(Act.NOOP)] = ability_noop
    branches[int(Act.CONFIRM_ABILITY)] = confirm
    branches[int(Act.SELECT_COST_TARGET)] = cost_select
    branches[int(Act.SELECT_EFFECT_TARGET)] = effect_select
    branches[int(Act.SELECT_FROM_SELECTION)] = selection_pick
    branches[int(Act.SELECT_TO_GARDEN)] = selection_to_garden
    branches[int(Act.SELECT_TO_ALLEY)] = selection_to_alley
    branches[int(Act.SELECT_TO_EQUIP)] = selection_to_equip
    branches[int(Act.BOTTOM_DECK_CARD)] = bottom_deck
    branches[int(Act.TOP_DECK_CARD)] = top_deck
    branches[int(Act.BOTTOM_DECK_ALL)] = bottom_deck_all
    return jax.lax.switch(action_index(), branches, st)

  def non_ability_action(st: State) -> State:
    return jax.lax.cond(
        st.phase == Phase.PREGAME_MULLIGAN,
        lambda x: apply_mod.apply_mulligan(x, act),
        normal_action,
        st,
    )

  out = jax.lax.cond(state.ab_phase != 0, ability_action, non_ability_action, state)

  # passive auras: C observers fire during the action and the binding ticks
  # until the passive queue drains before exposing the next decision point
  # (azk_engine_requires_action returns false while buffs are pending,
  # azuki_engine.c:280). recompute_passives self-gates on ab_phase == 0
  # (drain deferred while an ability resolves) and winner == -1.
  from azuki_jax.abilities.passives import recompute_passives

  out = recompute_passives(out)
  return out


def apply_user_action_static(state: State, action: jax.Array, act_type: int) -> State:
  """Static-action variant of apply_user_action.

  `act_type` must be a Python int known at trace time. This lets vector
  backends compile one action-family kernel at a time instead of forcing XLA to
  lower the whole dynamic action dispatcher into every step executable.
  """
  act_type = int(act_type)
  s1, s2, s3 = action[1], action[2], action[3]
  use_token = s3 != 0

  def identity(st: State) -> State:
    return st

  def battle_phase(st: State):
    return (st.phase == Phase.MAIN) | (st.phase == Phase.RESPONSE_WINDOW)

  def normal_noop(st: State) -> State:
    return jax.lax.cond(
        st.phase == Phase.MAIN,
        lambda x: apply_mod.apply_noop_main(x, do=True),
        lambda x: jax.lax.cond(
            x.phase == Phase.RESPONSE_WINDOW,
            lambda y: apply_mod.apply_noop_response(y, do=True),
            identity,
            x,
        ),
        st,
    )

  def normal_known_action(st: State) -> State:
    if act_type == int(Act.NOOP):
      return normal_noop(st)
    if act_type == int(Act.PLAY_ENTITY_TO_GARDEN):
      return apply_mod.apply_play_entity(
          st, Zone.GARDEN, s1, s2, use_token, do=battle_phase(st)
      )
    if act_type == int(Act.PLAY_ENTITY_TO_ALLEY):
      return apply_mod.apply_play_entity(
          st, Zone.ALLEY, s1, s2, use_token, do=battle_phase(st)
      )
    if act_type == int(Act.ATTACH_WEAPON_FROM_HAND):
      return apply_mod.apply_attach_weapon(
          st, s1, s2, use_token, do=battle_phase(st)
      )
    if act_type == int(Act.GATE_PORTAL):
      return apply_mod.apply_gate_portal(
          st, s1, s2, do=st.phase == Phase.MAIN
      )
    if act_type == int(Act.ATTACK):
      return apply_mod.apply_attack(st, s1, s2, do=st.phase == Phase.MAIN)
    if act_type == int(Act.DECLARE_DEFENDER):
      return apply_mod.apply_declare_defender(
          st, s1, do=st.phase == Phase.RESPONSE_WINDOW
      )
    if act_type == int(Act.PLAY_SPELL_FROM_HAND):
      b = battle_phase(st)
      return apply_mod.apply_ability_action(
          st, s1, s2, use_token, b, jnp.bool_(False), jnp.bool_(False)
      )
    if act_type == int(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY):
      b = battle_phase(st)
      return apply_mod.apply_ability_action(
          st, s1, s2, use_token, jnp.bool_(False), b, jnp.bool_(False)
      )
    if act_type == int(Act.ACTIVATE_ALLEY_ABILITY):
      b = battle_phase(st)
      return apply_mod.apply_ability_action(
          st, s1, s2, use_token, jnp.bool_(False), jnp.bool_(False), b
      )
    return st

  def ability_noop(st: State) -> State:
    from azuki_jax.abilities import runtime
    from azuki_jax.abilities import selection as sel_mod

    def decline(x):
      return runtime.process_decline(x, jnp.bool_(True))

    def cost_skip(x):
      return runtime.process_cost_action(
          x, s1, jnp.bool_(False), jnp.bool_(True)
      )

    def effect_skip(x):
      return runtime.process_effect_action(
          x, s1, jnp.bool_(False), jnp.bool_(True)
      )

    def pick_skip(x):
      return sel_mod.process_skip_selection(x, jnp.bool_(True))

    branches = [identity for _ in range(6)]
    branches[int(AbilityPhase.CONFIRMATION)] = decline
    branches[int(AbilityPhase.COST_SELECTION)] = cost_skip
    branches[int(AbilityPhase.EFFECT_SELECTION)] = effect_skip
    branches[int(AbilityPhase.SELECTION_PICK)] = pick_skip
    phase = jnp.clip(st.ab_phase.astype(jnp.int32), 0, 5)
    return jax.lax.switch(phase, branches, st)

  def ability_known_action(st: State) -> State:
    from azuki_jax.abilities import runtime
    from azuki_jax.abilities import selection as sel_mod

    if act_type == int(Act.NOOP):
      return ability_noop(st)
    if act_type == int(Act.CONFIRM_ABILITY):
      return runtime.process_confirm(st, jnp.bool_(True))
    if act_type == int(Act.SELECT_COST_TARGET):
      return runtime.process_cost_action(
          st, s1, jnp.bool_(True), jnp.bool_(False)
      )
    if act_type == int(Act.SELECT_EFFECT_TARGET):
      return runtime.process_effect_action(
          st, s1, jnp.bool_(True), jnp.bool_(False)
      )
    if act_type == int(Act.SELECT_FROM_SELECTION):
      return sel_mod.process_selection_pick(st, s1, jnp.bool_(True))
    if act_type == int(Act.SELECT_TO_GARDEN):
      return sel_mod.process_selection_to_garden(st, s1, s2, jnp.bool_(True))
    if act_type == int(Act.SELECT_TO_ALLEY):
      return sel_mod.process_selection_to_alley(st, s1, s2, jnp.bool_(True))
    if act_type == int(Act.SELECT_TO_EQUIP):
      return sel_mod.process_selection_to_equip(st, s1, s2, jnp.bool_(True))
    if act_type == int(Act.BOTTOM_DECK_CARD):
      return sel_mod.process_bottom_deck(st, s1, jnp.bool_(True))
    if act_type == int(Act.TOP_DECK_CARD):
      return sel_mod.process_top_deck(st, s1, jnp.bool_(True))
    if act_type == int(Act.BOTTOM_DECK_ALL):
      return sel_mod.process_bottom_deck_all(st, jnp.bool_(True))
    return st

  def non_ability_action(st: State) -> State:
    return jax.lax.cond(
        st.phase == Phase.PREGAME_MULLIGAN,
        lambda x: apply_mod.apply_mulligan(x, jnp.asarray(act_type, action.dtype)),
        normal_known_action,
        st,
    )

  out = jax.lax.cond(
      state.ab_phase != 0, ability_known_action, non_ability_action, state
  )

  from azuki_jax.abilities.passives import recompute_passives

  return recompute_passives(out)


def engine_step_static_action(
    state: State, action: jax.Array, act_type: int
) -> State:
  state = apply_user_action_static(state, action, act_type)
  return auto_resolve(state)


def micro_tick(state: State) -> State:
  """One azk_engine_tick: queues first, then phase gate, then phase systems."""
  in_ability = state.ab_phase != 0
  queued = has_queued(state) & ~in_ability

  def identity(st: State) -> State:
    return st

  def process_queue(st: State) -> State:
    popped, src, owner, timing = pop_effect(st)
    return resolve_triggered_effect(popped, src, owner, timing)

  def run_phase_systems(st: State) -> State:
    # 2) phase gate (only when nothing was popped)
    st = phase_gate(st)

    # 2b) response auto-close (HandleResponseAction head): if the defender has
    # no options left, transition to combat resolve without consuming an action
    from azuki_jax.engine.phases import (
        defender_can_respond,
        transition_to_combat_resolve,
    )

    dead_response = (
        (st.phase == Phase.RESPONSE_WINDOW)
        & ~has_queued(st)
        & (st.ab_phase == 0)
        & (~defender_can_respond(st, st.active_player))
    )
    st = jax.lax.cond(
        dead_response,
        lambda x: transition_to_combat_resolve(x, do=True),
        identity,
        st,
    )

    def start_branch(x: State) -> State:
      return start_of_turn(x, do=True)

    def combat_branch(x: State) -> State:
      return jax.lax.cond(
          ~has_queued(x),
          lambda y: combat_resolve(y, do=True),
          identity,
          x,
      )

    def end_branch(x: State) -> State:
      def queue_eot(y: State) -> State:
        return queue_end_of_turn_abilities(y)._replace(
            eot_abilities_queued=jnp.bool_(True)
        )

      x = jax.lax.cond(
          ~x.eot_abilities_queued,
          queue_eot,
          identity,
          x,
      )
      return jax.lax.cond(
          x.eot_abilities_queued & ~has_queued(x),
          lambda y: end_turn(y, do=True),
          identity,
          x,
      )

    def no_ability_auto_phase(x: State) -> State:
      branches = [identity for _ in range(8)]
      branches[int(Phase.START_OF_TURN)] = start_branch
      branches[int(Phase.COMBAT_RESOLVE)] = combat_branch
      branches[int(Phase.END_TURN)] = end_branch
      phase = jnp.clip(x.phase.astype(jnp.int32), 0, 7)
      return jax.lax.switch(phase, branches, x)

    # 3) phase systems for auto phases (suppressed while an ability FSM is
    # active — the C phase gate routes exclusively to the ability pipeline)
    return jax.lax.cond(st.ab_phase != 0, identity, no_ability_auto_phase, st)

  state = jax.lax.cond(queued, process_queue, run_phase_systems, state)

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
