"""Env-level step: tcg.h c_step (rewards, terminals, truncations, auto-reset).

step(state, actions, pool) consumes the ACTIVE player's action row from
actions (2, 4) int32, mirrors vec_step semantics:
- if both terminals were set by the previous step -> reset instead of stepping
- zero-legal-action -> truncation
- engine step -> terminal / shaped rewards
Returns (state, rewards (2,), terminals (2,), truncations (2,)).

AZK_MAX_TICKS_PER_EPISODE defaults to 0 (disabled) in training, so the
timeout-truncation branch is keyed off `episode_cap` (0 = disabled).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from azuki_jax.constants import (
    Act,
    ALLEY_SIZE,
    AbilityPhase,
    CardType,
    GARDEN_SIZE,
    MAX_ABILITY_SELECTION,
    Phase,
    PBRS_GARDEN_ATTACK_CAP,
    SHAPED_BOARD_DELTA_WEIGHT,
    SHAPED_LEADER_DELTA_WEIGHT,
    SHAPED_NOOP_PENALTY,
    TERMINAL_REWARD,
    TRUNCATION_BOARD_EDGE_WEIGHT,
    TRUNCATION_LEADER_EDGE_WEIGHT,
    TRUNCATION_TIMEOUT_PENALTY,
    PBRS_TIME_DECAY,
    Zone,
)
from azuki_jax.engine.step import engine_step, engine_step_static_action
from azuki_jax.env import reset_state
from azuki_jax.masks import build_mask
from azuki_jax.rewards import (
    compute_phi_pair,
    leader_health_transform,
    reward_snapshot,
)
from azuki_jax.setup import DeckPoolTables
from azuki_jax.state import State


def _safe_delta(numerator, denominator):
  return jnp.where(jnp.abs(denominator) <= 1e-6, 0.0, numerator / denominator)


def _board_edge(state: State) -> jax.Array:
  leader_ratio, garden_attack, untapped_garden, untapped_ikz = reward_snapshot(state)
  attack_edge = _safe_delta(garden_attack[0] - garden_attack[1], 10.0)
  untapped_edge = _safe_delta(untapped_garden[0] - untapped_garden[1], 5.0)
  ikz_edge = _safe_delta(untapped_ikz[0] - untapped_ikz[1], 10.0)
  return 0.6 * attack_edge + 0.3 * untapped_edge + 0.1 * ikz_edge


def _truncation_rewards(state: State) -> jax.Array:
  leader_ratio, *_ = reward_snapshot(state)
  p0 = leader_health_transform(leader_ratio[0])
  p1 = leader_health_transform(leader_ratio[1])
  leader_edge = TRUNCATION_LEADER_EDGE_WEIGHT * (p0 - p1)
  board_edge = TRUNCATION_BOARD_EDGE_WEIGHT * _board_edge(state)
  r0 = leader_edge + board_edge - TRUNCATION_TIMEOUT_PENALTY
  r1 = -leader_edge - board_edge - TRUNCATION_TIMEOUT_PENALTY
  return jnp.stack([r0, r1]).astype(jnp.float32)


def _terminal_rewards(state: State) -> jax.Array:
  return jnp.where(
      state.winner == 0,
      jnp.asarray([TERMINAL_REWARD, -TERMINAL_REWARD], jnp.float32),
      jnp.where(
          state.winner == 1,
          jnp.asarray([-TERMINAL_REWARD, TERMINAL_REWARD], jnp.float32),
          jnp.zeros(2, jnp.float32),
      ),
  )



def _begin_queued_stt03_006_destroy_trigger(state: State, do) -> State:
  """Begin queued STT03-006 When Destroyed trigger if it is at the queue head."""
  from azuki_jax import cards
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.engine.triggers import TIMING_WHEN_DESTROYED, pop_effect

  popped, trig_src, trig_owner, trig_timing = pop_effect(state)
  trig_src_i = jnp.maximum(trig_src.astype(jnp.int32), 0)
  trig_owner_i = jnp.maximum(trig_owner.astype(jnp.int32), 0)
  begin = (
      jnp.asarray(do)
      & (state.ab_phase == AbilityPhase.NONE)
      & (state.trig_count > 0)
      & (trig_timing == jnp.int8(TIMING_WHEN_DESTROYED))
      & (state.def_id[trig_owner_i, trig_src_i] == cards.CODE_TO_ID["STT03-006"])
  )
  begun = resolve_triggered_effect(popped, trig_src, trig_owner, trig_timing)
  return jax.tree.map(lambda a, b: jnp.where(begin, a, b), begun, state)


def _close_response_combat_if_idle(state: State) -> State:
  """Auto-advance an idle response/combat window after a response effect."""
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.phases import (
      combat_resolve,
      defender_can_respond,
      phase_gate,
      transition_to_combat_resolve,
  )
  from azuki_jax.engine.triggers import has_queued

  close_response = (
      (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == AbilityPhase.NONE)
      & ~has_queued(state)
      & ~defender_can_respond(state, state.active_player)
  )
  state = jax.lax.cond(
      close_response,
      lambda st: transition_to_combat_resolve(st, do=True),
      lambda st: st,
      state,
  )
  pending_main_combat = (
      (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (state.combat_attacker >= 0)
      & ~has_queued(state)
  )
  state = jax.lax.cond(
      pending_main_combat,
      lambda st: phase_gate(st),
      lambda st: st,
      state,
  )
  auto_combat = (state.phase == Phase.COMBAT_RESOLVE) & ~has_queued(state)
  state = jax.lax.cond(
      auto_combat,
      lambda st: combat_resolve(st, do=True),
      lambda st: st,
      state,
  )
  # C ticks the passive buff queue (combat deaths fire garden observers)
  # before the triggered-effect queue begins the next ability
  # (azuki_engine.c: passive processing returns before triggered work).
  state = recompute_passives(state)
  return _begin_queued_stt03_006_destroy_trigger(state, auto_combat)


def _shaped_rewards(state: State, prev: State, acting, action_type,
                    noop_had_alternatives) -> tuple[State, jax.Array]:
  phi = compute_phi_pair(state)
  opp = (acting + 1) % 2
  phi_delta = phi[acting] - prev.last_phi[acting]

  leader_ratio, garden_attack, *_ = reward_snapshot(state)
  prev_leader_edge = (
      prev.last_leader_ratio[acting] - prev.last_leader_ratio[opp]
  )
  curr_leader_edge = leader_ratio[acting] - leader_ratio[opp]
  leader_delta = SHAPED_LEADER_DELTA_WEIGHT * (curr_leader_edge - prev_leader_edge)

  prev_board_edge = _safe_delta(
      prev.last_garden_attack[acting] - prev.last_garden_attack[opp],
      PBRS_GARDEN_ATTACK_CAP,
  )
  curr_board_edge = _safe_delta(
      garden_attack[acting] - garden_attack[opp], PBRS_GARDEN_ATTACK_CAP
  )
  board_delta = SHAPED_BOARD_DELTA_WEIGHT * (curr_board_edge - prev_board_edge)

  noop_penalty = jnp.where(
      (action_type == Act.NOOP) & noop_had_alternatives, SHAPED_NOOP_PENALTY, 0.0
  )

  shaped = (
      state.time_weight * phi_delta + leader_delta + board_delta - noop_penalty
  )
  rewards = jnp.zeros(2, jnp.float32).at[acting].set(shaped).at[opp].set(-shaped)

  state = state._replace(
      last_phi=phi,
      last_leader_ratio=leader_ratio,
      last_garden_attack=garden_attack,
      time_weight=state.time_weight * PBRS_TIME_DECAY,
  )
  return state, rewards


def _begin_queued_stt03_013_enter_trigger(state: State, do) -> State:
  """Begin a queued STT03-013 enters-garden trigger after an active ability ends."""
  from azuki_jax import cards
  from azuki_jax.engine.triggers import TIMING_WHEN_ENTERS_GARDEN, pop_effect

  popped, src, owner, timing = pop_effect(state)
  src_i = jnp.maximum(src.astype(jnp.int32), 0)
  owner_i = jnp.maximum(owner.astype(jnp.int32), 0)
  begin = (
      jnp.asarray(do)
      & (state.ab_phase == AbilityPhase.NONE)
      & (state.trig_count > 0)
      & (timing == jnp.int8(TIMING_WHEN_ENTERS_GARDEN))
      & (state.def_id[owner_i, src_i] == cards.CODE_TO_ID["STT03-013"])
  )
  needs_transfer = popped.active_player != owner.astype(jnp.int8)
  begun = popped._replace(
      ab_source=src.astype(jnp.int8),
      ab_owner=owner.astype(jnp.int8),
      ab_slot=jnp.int8(0),
      ab_is_optional=jnp.bool_(True),
      ab_costs_applied=jnp.bool_(False),
      ab_saved_active=jnp.where(needs_transfer, popped.active_player, jnp.int8(-1)),
      ab_restores_active=needs_transfer,
      ab_cost_selected=jnp.int8(0),
      ab_cost_max=jnp.int8(0),
      ab_cost_targets=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_cost_target_players=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_eff_selected=jnp.int8(0),
      ab_eff_min=jnp.int8(0),
      ab_eff_max=jnp.int8(0),
      ab_eff_targets=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_eff_target_players=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_phase=jnp.int8(AbilityPhase.CONFIRMATION),
      active_player=jnp.where(
          needs_transfer, owner_i.astype(jnp.int8), popped.active_player
      ),
  )
  return jax.tree.map(lambda a, b: jnp.where(begin, a, b), begun, state)


def step(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    episode_cap: int = 0,
):
  """One vec_step-style transition. Returns (state, rewards, terms, truncs)."""
  legal, count, _ = build_mask(state)
  del legal
  return step_with_legal_count(
      state, actions, prev_terminals, prev_truncations, pool, count, episode_cap
  )


def step_with_legal_count(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """One vec_step-style transition using a precomputed current legal count."""
  # auto-reset when the previous step ended the episode
  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  # A fresh game always rests at the first mulligan decision point, so no
  # auto-resolve (stabilize) is needed here.
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action[0] == Act.NOOP) & (legal_count > 1)

  prev = state
  stepped = engine_step(state, action)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action[0], noop_had_alternatives
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_with_legal_count_static_action(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    action_type: int,
    episode_cap: int = 0,
):
  """Static-action version of step_with_legal_count.

  `action_type` is a Python int known at trace time. Callers that split a
  batch by action type can compile smaller kernels and merge the selected rows
  on the host/device side.
  """
  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (
      (jnp.asarray(action_type, jnp.int32) == Act.NOOP) & (legal_count > 1)
  )

  prev = state
  stepped = engine_step_static_action(state, action, action_type)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, jnp.asarray(action_type, jnp.int32),
      noop_had_alternatives
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_zero_legal_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast wrapper path for rows with no legal action.

  The generic step wrapper still calls the engine before merging the original
  state back on `zero_legal`. This path keeps the same reset/tick/truncation
  bookkeeping while skipping action application entirely.
  """
  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_main_noop_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast subset for MAIN-phase NOOP with no queued trigger work.

  This is intentionally narrower than `step_with_legal_count_static_action`.
  Callers must guard that the row is a plain MAIN NOOP with no ability FSM,
  combat, existing trigger queue, or EOT/start trigger candidates. Under those
  conditions the generic auto-resolve sequence is:
    apply_noop_main -> recompute_passives -> end_turn -> recompute_passives
    -> start_of_turn -> recompute_passives -> next decision.
  """
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.apply import apply_noop_main
  from azuki_jax.engine.phases import end_turn, start_of_turn

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = legal_count > 1

  prev = state
  stepped = apply_noop_main(state, do=True)
  stepped = recompute_passives(stepped)
  stepped = end_turn(stepped, do=True)
  stepped = recompute_passives(stepped)
  stepped = start_of_turn(stepped, do=True)
  stepped = recompute_passives(stepped)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, jnp.asarray(Act.NOOP, jnp.int32),
      noop_had_alternatives
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def _clear_turn_damage_trackers(state: State) -> State:
  n2 = state.took_damage_turn.shape
  return state._replace(
      took_damage_turn=jnp.zeros(n2, jnp.bool_),
      dealt_damage_turn=jnp.zeros(n2, jnp.bool_),
      last_dmg_taken=jnp.zeros(n2, jnp.int8),
      last_dmg_src_player=jnp.full(n2, -1, jnp.int8),
      last_dmg_src_inst=jnp.full(n2, -1, jnp.int8),
      last_dmg_from_effect=jnp.zeros(n2, jnp.bool_),
      dmg_src_keys=jnp.full(state.dmg_src_keys.shape, -1, jnp.int16),
      dmg_src_count=jnp.zeros(n2, jnp.int8),
  )


def _simple_end_turn_no_cleanup(state: State) -> State:
  """End-turn subset when host guards prove cleanup hooks are absent."""
  from azuki_jax import cards

  p = state.active_player
  nxt = (p + 1) % 2

  for player in (p, nxt):
    z = state.zone[player]
    in_play = (z == Zone.GARDEN) | (z == Zone.ALLEY)
    def_id = state.def_id[player]
    base_hp = jnp.where(
        def_id >= 0, jnp.asarray(cards.BASE_HP)[def_id], 0
    ).astype(jnp.int16)
    healed = jnp.clip(
        base_hp
        + state.hp_buff_perm[player].astype(jnp.int16)
        + state.hp_buff_eot[player].astype(jnp.int16)
        + state.passive_hp[player].astype(jnp.int16),
        -128,
        127,
    ).astype(jnp.int8)
    state = state._replace(
        cur_hp=state.cur_hp.at[player].set(
            jnp.where(in_play, healed, state.cur_hp[player])
        )
    )

  return state._replace(
      eot_abilities_queued=jnp.bool_(False),
      phase=jnp.int8(Phase.START_OF_TURN),
      active_player=nxt.astype(jnp.int8),
  )


def _simple_start_turn_no_triggers(state: State) -> State:
  """Start-turn subset when host guards prove status/trigger hooks are absent."""
  from azuki_jax.zones import move_top_n, zone_count
  from azuki_jax.engine.phases import _tick_start_statuses, _untap_all_for

  p = state.active_player
  state = state._replace(
      turn_number=(state.turn_number + 1).astype(jnp.int16)
  )
  state = _clear_turn_damage_trackers(state)
  for player in (0, 1):
    state = _tick_start_statuses(state, player, True)
  zeros = jnp.zeros(2, jnp.uint8)
  state = state._replace(
      entities_played_garden_turn=zeros,
      entities_played_alley_turn=zeros,
      cards_played_turn=zeros,
      discarded_cards_turn=zeros,
      returned_to_hand_turn=zeros,
      next_play_cost_reduction=jnp.zeros(2, jnp.int8),
      once_per_turn_used=jnp.zeros_like(state.once_per_turn_used),
  )

  state = _untap_all_for(state, p, True)

  should_draw = state.turn_number > 1
  deck_count = zone_count(state.zone[p], Zone.DECK)
  deck_out = should_draw & (deck_count == 0)
  zone_row, zpos_row = state.zone[p], state.zpos[p]
  nzr, npr = move_top_n(zone_row, zpos_row, Zone.DECK, Zone.HAND, jnp.int32(1), 1)
  draw = should_draw & ~deck_out
  state = state._replace(
      zone=state.zone.at[p].set(jnp.where(draw, nzr, zone_row)),
      zpos=state.zpos.at[p].set(jnp.where(draw, npr, zpos_row)),
      winner=jnp.where(deck_out, (p + 1) % 2, state.winner).astype(jnp.int8),
  )

  zone_row, zpos_row = state.zone[p], state.zpos[p]
  nzr, npr = move_top_n(
      zone_row, zpos_row, Zone.IKZ_PILE, Zone.IKZ_AREA, jnp.int32(1), 1
  )
  grant = ~deck_out
  state = state._replace(
      zone=state.zone.at[p].set(jnp.where(grant, nzr, zone_row)),
      zpos=state.zpos.at[p].set(jnp.where(grant, npr, zpos_row)),
  )

  return state._replace(
      phase=jnp.where(
          state.winner != -1, jnp.int8(Phase.END_MATCH), jnp.int8(Phase.MAIN)
      )
  )


def step_main_noop_simple_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Very narrow MAIN NOOP step for empty/simple end-turn cleanup."""
  del actions
  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = legal_count > 1

  prev = state
  stepped = state._replace(phase=jnp.int8(Phase.END_TURN))
  stepped = _simple_end_turn_no_cleanup(stepped)
  stepped = _simple_start_turn_no_triggers(stepped)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, jnp.asarray(Act.NOOP, jnp.int32),
      noop_had_alternatives
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_main_noop_stt03_006_trigger_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast MAIN NOOP that begins a queued STT03-006 death trigger."""
  del actions
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import draw_with_deckout
  from azuki_jax.engine.triggers import TIMING_WHEN_DESTROYED, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = legal_count > 1

  prev = state
  popped, trig_src, trig_owner, trig_timing = pop_effect(state)
  trig_owner_i = jnp.maximum(trig_owner.astype(jnp.int32), 0)
  trig_src_i = jnp.maximum(trig_src.astype(jnp.int32), 0)
  begin = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (state.trig_count > 0)
      & (trig_timing == jnp.int8(TIMING_WHEN_DESTROYED))
      & (state.def_id[trig_owner_i, trig_src_i] == cards.CODE_TO_ID["STT03-006"])
  )
  drawn = draw_with_deckout(popped, trig_owner_i, 1, begin)
  to_effect = (
      begin
      & (drawn.winner == -1)
      & (jnp.sum(drawn.zone[trig_owner_i] == Zone.HAND) > 0)
  )
  transfer = to_effect & (drawn.active_player != trig_owner.astype(jnp.int8))
  begun = drawn._replace(
      ab_source=trig_src.astype(jnp.int8),
      ab_owner=trig_owner.astype(jnp.int8),
      ab_is_optional=jnp.bool_(False),
      ab_costs_applied=jnp.bool_(True),
      ab_saved_active=jnp.where(transfer, drawn.active_player, jnp.int8(-1)),
      ab_restores_active=transfer,
      ab_cost_selected=jnp.int8(0),
      ab_cost_max=jnp.int8(0),
      ab_eff_selected=jnp.int8(0),
      ab_eff_min=jnp.int8(1),
      ab_eff_max=jnp.int8(1),
      ab_phase=jnp.int8(AbilityPhase.EFFECT_SELECTION),
      active_player=trig_owner.astype(jnp.int8),
  )
  stepped = jax.tree.map(lambda a, b: jnp.where(begin, a, b), drawn, state)
  stepped = jax.tree.map(lambda a, b: jnp.where(to_effect, a, b), begun, stepped)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, jnp.asarray(Act.NOOP, jnp.int32),
      noop_had_alternatives
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def _apply_stt04_003_start_each_damage(state: State, do) -> State:
  from azuki_jax import cards
  from azuki_jax.engine.helpers import discard
  from azuki_jax.engine.triggers import record_damage_event

  seer_id = cards.CODE_TO_ID["STT04-003"]
  in_play = (state.zone == Zone.GARDEN) | (state.zone == Zone.ALLEY)
  candidates = in_play & (state.def_id == seer_id)
  flat = jnp.argmax(candidates.reshape(-1))
  n = state.zone.shape[1]
  owner = (flat // n).astype(jnp.int32)
  inst = (flat % n).astype(jnp.int32)
  exists = candidates.reshape(-1)[flat]
  hp = state.cur_hp[owner, inst].astype(jnp.int16)
  do_damage = jnp.asarray(do) & exists & (hp > 0) & (state.winner == -1)
  next_hp = hp - 1
  state = state._replace(
      cur_hp=state.cur_hp.at[owner, inst].set(
          jnp.where(do_damage, next_hp.astype(jnp.int8), state.cur_hp[owner, inst])
      )
  )
  state = record_damage_event(
      state, owner, inst, owner, inst, jnp.int16(1), do_damage,
      from_effect=True
  )
  return discard(state, owner, inst, do=do_damage & (next_hp <= 0))


def _apply_stt04_003_start_each_damage_at(state: State, owner, inst, do) -> State:
  from azuki_jax import cards
  from azuki_jax.engine.helpers import discard
  from azuki_jax.engine.triggers import record_damage_event

  seer_id = cards.CODE_TO_ID["STT04-003"]
  owner = owner.astype(jnp.int32)
  inst = inst.astype(jnp.int32)
  in_play = (state.zone[owner, inst] == Zone.GARDEN) | (
      state.zone[owner, inst] == Zone.ALLEY
  )
  is_seer = state.def_id[owner, inst] == seer_id
  hp = state.cur_hp[owner, inst].astype(jnp.int16)
  do_damage = jnp.asarray(do) & in_play & is_seer & (hp > 0) & (state.winner == -1)
  next_hp = hp - 1
  state = state._replace(
      cur_hp=state.cur_hp.at[owner, inst].set(
          jnp.where(do_damage, next_hp.astype(jnp.int8), state.cur_hp[owner, inst])
      )
  )
  state = record_damage_event(
      state, owner, inst, owner, inst, jnp.int16(1), do_damage,
      from_effect=True
  )
  return discard(state, owner, inst, do=do_damage & (next_hp <= 0))


def _apply_ordered_stt04_003_start_each_damage(
    state: State, do, max_count: int = 2
) -> State:
  """Apply already-validated STT04-003 start-each triggers in C queue order."""
  from azuki_jax import cards

  seer_id = cards.CODE_TO_ID["STT04-003"]
  z = state.zone
  candidates = (
      ((z == Zone.GARDEN) | (z == Zone.ALLEY))
      & (state.def_id == seer_id)
  )
  player_key = jnp.arange(2, dtype=jnp.int32)[:, None] * (1 << 18)
  zone_key = jnp.where(
      z == Zone.GARDEN,
      0,
      jnp.where(z == Zone.LEADER, 1 << 16, 2 << 16),
  )
  key = player_key + zone_key + state.board_seq.astype(jnp.int32)
  key = jnp.where(candidates, key, 1 << 30)
  flat_key = key.reshape(-1)
  order = jnp.argsort(flat_key)
  n = state.zone.shape[1]
  for k in range(max_count):
    flat = order[k]
    has = flat_key[flat] < (1 << 30)
    owner = (flat // n).astype(jnp.int32)
    inst = (flat % n).astype(jnp.int32)
    state = _apply_stt04_003_start_each_damage_at(state, owner, inst, do & has)
  return state


def step_main_noop_stt04_003_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """MAIN NOOP fast path with one or two clean STT04-003 start-each triggers."""
  from azuki_jax import cards
  from azuki_jax.engine.phases import end_turn
  from azuki_jax.engine.triggers import TIMING_WHEN_TAKES_DAMAGE, pop_effect
  del actions
  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = legal_count > 1

  prev = state
  trig_src = jnp.maximum(state.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(state.trig_owner[0].astype(jnp.int32), 0)
  trig_zone = state.zone[trig_owner, trig_src]
  queued_dead_azk01_059 = (
      ~(did_reset | zero_legal)
      & (state.trig_count == 1)
      & (state.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (state.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-059"])
      & (trig_zone != Zone.GARDEN)
      & (trig_zone != Zone.ALLEY)
  )
  popped, _, _, _ = pop_effect(state)
  state = jax.tree.map(
      lambda a, b: jnp.where(queued_dead_azk01_059, a, b), popped, state
  )
  stepped = state._replace(phase=jnp.int8(Phase.END_TURN))
  stepped = end_turn(stepped, do=True)
  stepped = _simple_start_turn_no_triggers(stepped)
  stepped = _apply_ordered_stt04_003_start_each_damage(
      stepped, ~(did_reset | zero_legal), max_count=2
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, jnp.asarray(Act.NOOP, jnp.int32),
      noop_had_alternatives
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def _apply_single_azk01_011_eot(state: State, do) -> State:
  """Apply the single AZK01-011 EOT trigger admitted by the host mask."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import destroy_card
  from azuki_jax.engine.helpers import discard_equipped_weapons

  p = state.active_player.astype(jnp.int32)
  azk01_011 = cards.CODE_TO_ID["AZK01-011"]
  candidates = (state.zone[p] == Zone.GARDEN) & (state.def_id[p] == azk01_011)
  key = jnp.where(candidates, state.board_seq[p].astype(jnp.int32), 1 << 30)
  inst = jnp.argmin(key).astype(jnp.int32)
  exists = candidates[inst]
  do_effect = jnp.asarray(do) & exists & ~state.tapped[p, inst]
  state = discard_equipped_weapons(state, p, inst, do=do_effect)
  return destroy_card(state, p, inst, do=do_effect)


def step_main_noop_azk01_011_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """MAIN NOOP fast path with one clean AZK01-011 EOT trigger."""
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.phases import end_turn, start_of_turn

  del actions
  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = legal_count > 1
  do_action = ~(did_reset | zero_legal)

  prev = state
  stepped = state._replace(phase=jnp.int8(Phase.END_TURN))
  stepped = _apply_single_azk01_011_eot(stepped, do_action)
  stepped = recompute_passives(stepped)
  stepped = end_turn(stepped, do=True)
  stepped = recompute_passives(stepped)
  stepped = start_of_turn(stepped, do=True)
  stepped = recompute_passives(stepped)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, jnp.asarray(Act.NOOP, jnp.int32),
      noop_had_alternatives
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def _clear_simple_unimplemented_play_triggers(
    state: State, p, inst, placement_zone, do
) -> State:
  """Account for queued unimplemented play triggers without ability runtime."""
  from azuki_jax.abilities import tables as ab_tables

  def_id = state.def_id[p, inst]
  safe_def = jnp.maximum(def_id, 0)
  valid = def_id >= 0
  has_ability = jnp.where(valid, jnp.asarray(ab_tables.HAS_ABILITY)[safe_def], False)
  implemented = jnp.where(valid, jnp.asarray(ab_tables.IMPLEMENTED)[safe_def], False)
  on_play = jnp.where(valid, jnp.asarray(ab_tables.TIMING_ON_PLAY)[safe_def], False)
  enter_garden = (
      jnp.asarray(placement_zone, jnp.int8) == jnp.int8(Zone.GARDEN)
  ) & jnp.where(
      valid, jnp.asarray(ab_tables.TIMING_WHEN_ENTERS_GARDEN)[safe_def], False
  )
  unimplemented = (on_play.astype(jnp.int16) + enter_garden.astype(jnp.int16)) * (
      has_ability & ~implemented
  ).astype(jnp.int16)
  return state._replace(
      ab_scratch=state.ab_scratch.at[3].add(
          jnp.where(jnp.asarray(do), unimplemented, 0).astype(jnp.int16)
      )
  )


def _apply_simple_implemented_play_trigger(state: State, p, inst, do) -> State:
  """Small implemented on-play effects supported by simple play fast path."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import (
      apply_charge_grant,
      deal_effect_damage,
      draw_with_deckout,
      heal_leader,
      ikz_grant_tapped,
      mill_with_deckout,
  )
  from azuki_jax.constants import TOKEN_INSTANCE
  from azuki_jax.engine.helpers import GRANT_PHASE_END, is_effect_immune, leader_instance

  def_id = state.def_id[p, inst]
  is_azk01_005 = def_id == cards.CODE_TO_ID["AZK01-005"]
  is_stt01_003 = def_id == cards.CODE_TO_ID["STT01-003"]
  is_stt01_004 = def_id == cards.CODE_TO_ID["STT01-004"]
  is_stt02_005 = def_id == cards.CODE_TO_ID["STT02-005"]
  is_stt02_007 = def_id == cards.CODE_TO_ID["STT02-007"]
  is_stt03_009 = def_id == cards.CODE_TO_ID["STT03-009"]
  is_stt03_013 = def_id == cards.CODE_TO_ID["STT03-013"]
  is_azk01_098 = def_id == cards.CODE_TO_ID["AZK01-098"]
  is_stt04_004 = def_id == cards.CODE_TO_ID["STT04-004"]
  is_azk01_116 = def_id == cards.CODE_TO_ID["AZK01-116"]
  is_azk01_015 = def_id == cards.CODE_TO_ID["AZK01-015"]
  def_ids = state.def_id[p]
  valid = def_ids >= 0
  is_weapon = jnp.where(
      valid,
      jnp.asarray(cards.TYPE)[jnp.maximum(def_ids, 0)] == CardType.WEAPON,
      False,
  )
  has_ikz_cost = jnp.where(
      valid, jnp.asarray(cards.HAS_IKZ_COST)[jnp.maximum(def_ids, 0)], False
  )
  ikz_cost = jnp.where(
      valid, jnp.asarray(cards.IKZ_COST)[jnp.maximum(def_ids, 0)], 0
  )
  weapons = jnp.sum((state.zone[p] == Zone.DISCARD) & is_weapon)
  n = jnp.where(weapons == 0, 5, 3)
  total_entities_played = (
      state.entities_played_garden_turn[p].astype(jnp.int32)
      + state.entities_played_alley_turn[p].astype(jnp.int32)
  )
  state = mill_with_deckout(state, p, n, jnp.asarray(do) & is_stt01_003, max_n=5)
  state = draw_with_deckout(
      state, p, 1, jnp.asarray(do) & is_stt02_005 & (total_entities_played >= 3)
  )
  state = draw_with_deckout(state, p, 1, jnp.asarray(do) & is_stt02_007)
  state = ikz_grant_tapped(state, p, jnp.asarray(do) & is_stt03_009)
  leader = leader_instance(state, p)
  safe_leader = jnp.maximum(leader, 0)
  state = deal_effect_damage(
      state,
      p,
      safe_leader,
      3,
      jnp.asarray(do) & is_azk01_116 & (leader >= 0),
      src_player=p,
      src_inst=inst,
  )

  azk01_015_leader_def = state.def_id[p, safe_leader]
  azk01_015_leader_element = jnp.where(
      azk01_015_leader_def >= 0,
      jnp.asarray(cards.ELEMENT)[jnp.maximum(azk01_015_leader_def, 0)],
      jnp.int8(-1),
  )
  azk01_015_water = is_azk01_015 & (leader >= 0) & (
      azk01_015_leader_element == jnp.int8(2)
  )
  azk01_015_earth = is_azk01_015 & (leader >= 0) & (
      azk01_015_leader_element == jnp.int8(3)
  )
  azk01_015_lightning = is_azk01_015 & (leader >= 0) & (
      azk01_015_leader_element == jnp.int8(1)
  )
  azk01_015_fire = is_azk01_015 & (leader >= 0) & (
      azk01_015_leader_element == jnp.int8(4)
  )
  token_tapped = (
      (state.zone[p, TOKEN_INSTANCE] == Zone.TOKEN) & state.tapped[p, TOKEN_INSTANCE]
  )
  tapped_ikz_area = (state.zone[p] == Zone.IKZ_AREA) & state.tapped[p]
  tapped_ikz_key = jnp.where(
      tapped_ikz_area, state.zpos[p].astype(jnp.int32), jnp.int32(1 << 20)
  )
  tapped_ikz_area_inst = jnp.where(tapped_ikz_area.any(), jnp.argmin(tapped_ikz_key), -1)
  tapped_ikz = jnp.where(
      token_tapped, jnp.asarray(TOKEN_INSTANCE, jnp.int32), tapped_ikz_area_inst
  )
  safe_ikz = jnp.maximum(tapped_ikz, 0)
  azk01_015_water_do = jnp.asarray(do) & azk01_015_water & (tapped_ikz >= 0)
  state = state._replace(
      tapped=state.tapped.at[p, safe_ikz].set(
          jnp.where(azk01_015_water_do, False, state.tapped[p, safe_ikz])
      ),
      cooldown=state.cooldown.at[p, safe_ikz].set(
          jnp.where(azk01_015_water_do, 0, state.cooldown[p, safe_ikz])
      ),
  )
  state = heal_leader(state, p, 2, jnp.asarray(do) & azk01_015_earth)
  state = apply_charge_grant(
      state, p, inst, GRANT_PHASE_END, 1, jnp.asarray(do) & azk01_015_lightning
  )
  idx = jnp.arange(state.zone.shape[1])
  opp = (p + 1) % 2
  azk01_005_target_available = jnp.any(
      (state.zone[opp] == Zone.GARDEN)
      & ~jax.vmap(lambda i: is_effect_immune(state, opp, i))(idx)
  )
  azk01_005_effect = is_azk01_005 & azk01_005_target_available
  hand_weapons = jnp.sum((state.zone[p] == Zone.HAND) & is_weapon)
  hand_weapons_le3 = jnp.sum(
      (state.zone[p] == Zone.HAND)
      & is_weapon
      & has_ikz_cost
      & (ikz_cost <= 3)
  )
  azk01_098_valid = (
      is_azk01_098
      & (state.zone[p, inst] == Zone.ALLEY)
      & ~state.tapped[p, inst]
      & (state.cooldown[p, inst] == 0)
      & (hand_weapons_le3 > 0)
  )
  stt03_013_confirm = is_stt03_013 & (state.zone[p, inst] == Zone.GARDEN)
  enter_confirm = jnp.asarray(do) & (
      is_stt04_004
      | stt03_013_confirm
      | (is_stt01_004 & (hand_weapons > 0))
      | azk01_098_valid
  )
  enter_effect = jnp.asarray(do) & (azk01_015_fire | azk01_005_effect)
  enter_any = enter_confirm | enter_effect
  stt01_004_confirm = enter_confirm & is_stt01_004
  stt04_004_confirm = enter_confirm & is_stt04_004
  return state._replace(
      ab_phase=jnp.where(
          enter_effect,
          jnp.int8(AbilityPhase.EFFECT_SELECTION),
          jnp.where(
              enter_confirm, jnp.int8(AbilityPhase.CONFIRMATION), state.ab_phase
          ),
      ),
      ab_source=jnp.where(enter_any, inst.astype(jnp.int8), state.ab_source),
      ab_owner=jnp.where(enter_any, p.astype(jnp.int8), state.ab_owner),
      ab_slot=jnp.where(enter_any, jnp.int8(0), state.ab_slot),
      ab_is_optional=jnp.where(
          enter_any, jnp.where(enter_effect, False, True), state.ab_is_optional
      ),
      ab_costs_applied=jnp.where(
          enter_any, jnp.where(enter_effect, True, False), state.ab_costs_applied
      ),
      ab_saved_active=jnp.where(enter_any, jnp.int8(-1), state.ab_saved_active),
      ab_restores_active=jnp.where(enter_any, False, state.ab_restores_active),
      ab_cost_selected=jnp.where(enter_any, jnp.int8(0), state.ab_cost_selected),
      ab_cost_max=jnp.where(
          enter_any,
          jnp.where(stt01_004_confirm, jnp.int8(1), jnp.int8(0)),
          state.ab_cost_max,
      ),
      ab_cost_targets=jnp.where(
          enter_any,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          enter_any,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(enter_any, jnp.int8(0), state.ab_eff_selected),
      ab_eff_min=jnp.where(
          enter_any,
          jnp.where(azk01_015_fire | stt04_004_confirm, jnp.int8(1), jnp.int8(0)),
          state.ab_eff_min,
      ),
      ab_eff_max=jnp.where(
          enter_any,
          jnp.where(enter_effect | stt04_004_confirm, jnp.int8(1), jnp.int8(0)),
          state.ab_eff_max,
      ),
      ab_eff_targets=jnp.where(
          enter_any,
          jnp.full_like(state.ab_eff_targets, -1),
          state.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          enter_any,
          jnp.full_like(state.ab_eff_target_players, -1),
          state.ab_eff_target_players,
      ),
  )


def step_play_entity_simple_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Narrow fast path for simple hand entity plays into a board slot.

  Callers must guard that displacement, trigger work, and passive observers are
  simple enough to avoid the full apply+auto-resolve pipeline.
  """
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine import ikz
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)
  place = do_action & (inst >= 0)

  cost = effective_play_cost(state, acting, safe)
  stepped = ikz.pay(state, acting, cost, use_token, do=place)

  pz = jnp.asarray(placement_zone, jnp.int8)
  stepped = _enter_board_slot(stepped, acting, safe, placement_zone, slot, place)

  is_garden = pz == jnp.int8(Zone.GARDEN)
  stepped = stepped._replace(
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((place & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((place & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          place.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(place, 0, stepped.next_play_cost_reduction[acting])
      ),
  )
  stepped = _apply_simple_implemented_play_trigger(
      stepped, acting, safe, place
  )
  stepped = _clear_simple_unimplemented_play_triggers(
      stepped, acting, safe, placement_zone, place
  )
  stepped = recompute_passives(stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      pz == jnp.int8(Zone.GARDEN),
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_play_azk01_028_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-028 play: discard hand, bounce other garden entities."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import garden_seq_order, return_to_hand
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine import ikz
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import batch_discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)
  is_azk01_028 = state.def_id[acting, safe] == cards.CODE_TO_ID["AZK01-028"]
  play_garden = action[0] == Act.PLAY_ENTITY_TO_GARDEN
  play_alley = action[0] == Act.PLAY_ENTITY_TO_ALLEY
  do_play = (
      do_action
      & (play_garden | play_alley)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (inst >= 0)
      & is_azk01_028
  )

  cost = effective_play_cost(state, acting, safe)
  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = _enter_board_slot(
      stepped, acting, safe, Zone.GARDEN, slot, do_play & play_garden
  )
  stepped = _enter_board_slot(
      stepped, acting, safe, Zone.ALLEY, slot, do_play & play_alley
  )
  stepped = stepped._replace(
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((do_play & play_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((do_play & play_alley).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
  )

  in_hand = stepped.zone[acting] == Zone.HAND
  discard_count = jnp.sum(in_hand, dtype=jnp.uint8)
  stepped = batch_discard(
      stepped,
      acting,
      in_hand,
      stepped.zpos[acting].astype(jnp.int32),
      do=do_play,
  )
  stepped = stepped._replace(
      discarded_cards_turn=stepped.discarded_cards_turn.at[acting].add(
          jnp.where(do_play, discard_count, jnp.uint8(0))
      )
  )

  for p in (0, 1):
    order, in_garden = garden_seq_order(stepped, p)
    row_defs = stepped.def_id[p]
    row_safe_defs = jnp.maximum(row_defs, 0)
    marked = (
        in_garden
        & (jnp.asarray(cards.TYPE)[row_safe_defs] == CardType.ENTITY)
    )
    for k in range(GARDEN_SIZE):
      target = order[k]
      not_self = ~((p == acting) & (target == safe))
      stepped = return_to_hand(
          stepped, p, target, do_play & marked[target] & not_self
      )

  stepped = recompute_passives(stepped)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      play_garden,
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type,
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_play_stt01_007_confirm_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast STT01-007 play setup into optional discard/draw confirmation."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine import ikz
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)
  source_def = state.def_id[acting, safe]
  place = (
      do_action
      & (inst >= 0)
      & (source_def == cards.CODE_TO_ID["STT01-007"])
  )

  cost = effective_play_cost(state, acting, safe)
  stepped = ikz.pay(state, acting, cost, use_token, do=place)

  pz = jnp.asarray(placement_zone, jnp.int8)
  stepped = _enter_board_slot(stepped, acting, safe, placement_zone, slot, place)
  stepped = recompute_passives(stepped)

  is_garden = pz == jnp.int8(Zone.GARDEN)
  hand_available = jnp.any(stepped.zone[acting] == Zone.HAND)
  deck_available = jnp.any(stepped.zone[acting] == Zone.DECK)
  enter_confirm = place & hand_available & deck_available
  needs_transfer = enter_confirm & (
      stepped.active_player != acting.astype(jnp.int8)
  )
  stepped = stepped._replace(
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((place & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((place & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          place.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(place, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_phase=jnp.where(
          enter_confirm,
          jnp.int8(AbilityPhase.CONFIRMATION),
          stepped.ab_phase,
      ),
      ab_source=jnp.where(enter_confirm, safe.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(enter_confirm, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(enter_confirm, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(enter_confirm, True, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(enter_confirm, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(
          enter_confirm,
          jnp.where(needs_transfer, stepped.active_player, jnp.int8(-1)),
          stepped.ab_saved_active,
      ),
      ab_restores_active=jnp.where(
          enter_confirm, needs_transfer, stepped.ab_restores_active
      ),
      active_player=jnp.where(
          needs_transfer, acting.astype(jnp.int8), stepped.active_player
      ),
      ab_cost_selected=jnp.where(
          enter_confirm, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(enter_confirm, jnp.int8(1), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          enter_confirm, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(enter_confirm, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(enter_confirm, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      pz == jnp.int8(Zone.GARDEN),
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_azk01_007_effect_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast AZK01-007 play setup into mandatory friendly-garden effect pick."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine import ikz
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)
  source_def = state.def_id[acting, safe]
  place = (
      do_action
      & (inst >= 0)
      & (source_def == cards.CODE_TO_ID["AZK01-007"])
  )

  cost = effective_play_cost(state, acting, safe)
  stepped = ikz.pay(state, acting, cost, use_token, do=place)

  pz = jnp.asarray(placement_zone, jnp.int8)
  stepped = _enter_board_slot(stepped, acting, safe, placement_zone, slot, place)

  is_garden = pz == jnp.int8(Zone.GARDEN)
  target_available = jnp.sum(stepped.zone[acting] == Zone.GARDEN) > 0
  enter_effect = place & target_available
  stepped = stepped._replace(
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((place & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((place & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          place.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(place, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(enter_effect, safe.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(enter_effect, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(enter_effect, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(enter_effect, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(enter_effect, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(
          enter_effect, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          enter_effect, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          enter_effect, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(enter_effect, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          enter_effect,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          enter_effect,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          enter_effect, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(enter_effect, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(enter_effect, jnp.int8(1), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          enter_effect,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          enter_effect,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_phase=jnp.where(
          enter_effect, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  stepped = recompute_passives(stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      pz == jnp.int8(Zone.GARDEN),
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_stt03_011_effect_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT03-011 garden play setup into optional enemy low-base-HP destroy."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine import ikz
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)
  source_def = state.def_id[acting, safe]
  cost = effective_play_cost(state, acting, safe)
  can_pay = ikz.can_pay(state, acting, cost, use_token)

  opp = (acting + 1) % 2
  opp_defs = state.def_id[opp]
  opp_safe_defs = jnp.maximum(opp_defs, 0)
  target_available = jnp.any(
      (state.zone[opp] == Zone.GARDEN)
      & (opp_defs >= 0)
      & (jnp.asarray(cards.TYPE)[opp_safe_defs] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_BASE_STATS)[opp_safe_defs]
      & (jnp.asarray(cards.BASE_HP)[opp_safe_defs].astype(jnp.int32) <= 2)
  )
  place = (
      do_action
      & (action[0] == Act.PLAY_ENTITY_TO_GARDEN)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (inst >= 0)
      & (source_def == cards.CODE_TO_ID["STT03-011"])
      & can_pay
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=place)
  stepped = _enter_board_slot(stepped, acting, safe, Zone.GARDEN, slot, place)
  enter_effect = place & target_available
  stepped = stepped._replace(
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add(place.astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          place.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(place, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(enter_effect, safe.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(enter_effect, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(enter_effect, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(enter_effect, True, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(enter_effect, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(
          enter_effect, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          enter_effect, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          enter_effect, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(enter_effect, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          enter_effect,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          enter_effect,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          enter_effect, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(enter_effect, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(enter_effect, jnp.int8(1), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          enter_effect,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          enter_effect,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_phase=jnp.where(
          enter_effect, jnp.int8(AbilityPhase.CONFIRMATION), stepped.ab_phase
      ),
  )
  stepped = recompute_passives(stepped)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_azk01_003_reveal_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast AZK01-003/AZK01-021 play/reveal path."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.engine import ikz
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import (
      card_at_slot,
      discard_equipped_weapons,
      hand_instance,
  )
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)
  source_def = state.def_id[acting, safe]
  is_azk01_003 = source_def == cards.CODE_TO_ID["AZK01-003"]
  is_azk01_021 = source_def == cards.CODE_TO_ID["AZK01-021"]
  valid_card = is_azk01_003 | is_azk01_021
  place = do_action & (inst >= 0) & valid_card

  cost = effective_play_cost(state, acting, safe)
  stepped = ikz.pay(state, acting, cost, use_token, do=place)
  displaced = card_at_slot(stepped, acting, placement_zone, slot)
  displaced_safe = jnp.maximum(displaced, 0)
  zone_full = jnp.sum(
      stepped.zone[acting] == jnp.asarray(placement_zone, jnp.int8),
      dtype=jnp.int32,
  ) >= GARDEN_SIZE
  displacing = place & (displaced >= 0) & zone_full
  stepped = discard_equipped_weapons(
      stepped, acting, displaced_safe, do=displacing
  )
  stepped = _enter_board_slot(stepped, acting, safe, placement_zone, slot, place)
  stepped = recompute_passives(stepped)

  pz = jnp.asarray(placement_zone, jnp.int8)
  is_garden = pz == jnp.int8(Zone.GARDEN)
  stepped = stepped._replace(
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((place & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((place & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          place.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(place, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(place, safe.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(place, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(place, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(place, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(place, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(place, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(place, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(place, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(place, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          place,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          place,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(place, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(place, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(place, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          place,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          place,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )
  stepped = sel_mod.reveal_top_into_selection(stepped, 5, 1, place)
  revealed = stepped.ab_sel_count > 0
  black_jade = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("BlackJade")], jnp.bool_
  )
  driftward = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Driftward")], jnp.bool_
  )
  sel_cards = stepped.ab_sel_cards
  sel_safe = jnp.maximum(sel_cards.astype(jnp.int32), 0)
  sel_defs = stepped.def_id[acting, sel_safe]
  sel_valid = sel_cards >= 0
  matching = jnp.any(
      sel_valid
      & (
          (
              is_azk01_003
              & black_jade[jnp.maximum(sel_defs, 0)]
              & (sel_defs != cards.CODE_TO_ID["AZK01-003"])
          )
          | (is_azk01_021 & driftward[jnp.maximum(sel_defs, 0)])
      )
  )
  next_phase = jnp.where(
      matching,
      jnp.int8(AbilityPhase.SELECTION_PICK),
      jnp.int8(AbilityPhase.BOTTOM_DECK),
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          place & revealed,
          next_phase,
          stepped.ab_phase,
      )
  )
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(place & ~revealed, a, b),
      cleared,
      stepped,
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      pz == jnp.int8(Zone.GARDEN),
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_azk01_097_reveal_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast AZK01-097 play/reveal path into optional weapon selection."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.engine import ikz
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)
  valid_card = state.def_id[acting, safe] == cards.CODE_TO_ID["AZK01-097"]
  place = do_action & (inst >= 0) & valid_card

  cost = effective_play_cost(state, acting, safe)
  stepped = ikz.pay(state, acting, cost, use_token, do=place)
  stepped = _enter_board_slot(stepped, acting, safe, placement_zone, slot, place)
  stepped = recompute_passives(stepped)

  pz = jnp.asarray(placement_zone, jnp.int8)
  is_garden = pz == jnp.int8(Zone.GARDEN)
  stepped = stepped._replace(
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((place & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((place & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          place.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(place, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_phase=jnp.where(place, jnp.int8(AbilityPhase.NONE), stepped.ab_phase),
      ab_source=jnp.where(place, safe.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(place, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(place, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(place, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(place, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(place, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(place, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(place, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(place, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          place,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          place,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(place, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(place, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(place, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          place,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          place,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_scratch=jnp.where(
          place, jnp.zeros_like(stepped.ab_scratch), stepped.ab_scratch
      ),
  )
  stepped = sel_mod.reveal_top_into_selection(stepped, 5, 1, place)
  sel_cards = stepped.ab_sel_cards
  sel_safe = jnp.maximum(sel_cards.astype(jnp.int32), 0)
  sel_defs = stepped.def_id[acting, sel_safe]
  sel_valid = sel_cards >= 0
  is_weapon = jnp.where(
      sel_valid,
      jnp.asarray(cards.TYPE)[jnp.maximum(sel_defs, 0)] == CardType.WEAPON,
      False,
  )
  matching = jnp.any(is_weapon)
  revealed = stepped.ab_sel_count > 0
  stepped = stepped._replace(
      ab_phase=jnp.where(
          place & revealed & matching,
          jnp.int8(AbilityPhase.SELECTION_PICK),
          stepped.ab_phase,
      )
  )
  no_pick = place & revealed & ~matching
  stepped = sel_mod.return_remaining_to_discard(stepped, no_pick)
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(no_pick | (place & ~revealed), a, b),
      cleared,
      stepped,
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      pz == jnp.int8(Zone.GARDEN),
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_azk01_097_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-097 selection pick or optional decline."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  sel_idx = action[1].astype(jnp.int32)
  idx = jnp.clip(sel_idx, 0, state.ab_sel_cards.shape[0] - 1)
  inst = state.ab_sel_cards[idx]
  target = jnp.maximum(inst.astype(jnp.int32), 0)
  source_def = state.def_id[owner, src]
  target_def = state.def_id[owner, target]
  target_is_weapon = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)] == CardType.WEAPON,
      False,
  )
  source_ok = (
      (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_def == cards.CODE_TO_ID["AZK01-097"])
      & (state.ab_sel_pick_max == 1)
      & (state.ab_sel_picked_count == 0)
  )
  pick_ok = (
      (action_type == Act.SELECT_FROM_SELECTION)
      & (sel_idx >= 0)
      & (sel_idx < state.ab_sel_count.astype(jnp.int32))
      & (inst >= 0)
      & target_is_weapon
  )
  decline = action_type == Act.NOOP
  do_action = ~(did_reset | zero_legal) & source_ok & (pick_ok | decline)

  stepped = state._replace(
      ab_sel_picked=state.ab_sel_picked.at[0].set(
          jnp.where(do_action & pick_ok, inst, state.ab_sel_picked[0])
      ),
      ab_sel_picked_count=jnp.where(
          do_action & pick_ok, jnp.int8(1), state.ab_sel_picked_count
      ),
      ab_sel_cards=state.ab_sel_cards.at[idx].set(
          jnp.where(do_action & pick_ok, jnp.int8(-1), state.ab_sel_cards[idx])
      ),
  )
  stepped = sel_mod.move_picked_to_hand(stepped, do_action & pick_ok)
  stepped = sel_mod.return_remaining_to_discard(stepped, do_action)
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(do_action, a, b), cleared, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, noop_had_alternatives
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_spell_stt04_016_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast play setup for STT04-016 into mandatory cost selection."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["STT04-016"]
  do_play = do_action & (spell >= 0) & is_spell
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(1), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.COST_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_play_spell_azk01_016_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-016 play: pay, discard spell, draw 2, select 2 discards."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import draw_with_deckout
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost
  from azuki_jax.zones import zone_count

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["AZK01-016"]
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)
  hand_count = zone_count(state.zone[acting], Zone.HAND)
  spell_in_hand = state.zone[acting, safe_spell] == Zone.HAND
  deck_count = zone_count(state.zone[acting], Zone.DECK)
  available_after_draw = (
      hand_count - spell_in_hand.astype(jnp.int32) + jnp.minimum(deck_count, 2)
  )
  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
      & (deck_count >= 1)
      & (available_after_draw >= 2)
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
  )
  stepped = draw_with_deckout(stepped, acting, 2, do_play)
  to_select = do_play & (stepped.winner == -1)
  stepped = stepped._replace(
      ab_source=jnp.where(to_select, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(to_select, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(to_select, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(to_select, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(to_select, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(to_select, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(to_select, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(to_select, jnp.int8(0), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(to_select, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(to_select, jnp.int8(2), stepped.ab_eff_min),
      ab_eff_max=jnp.where(to_select, jnp.int8(2), stepped.ab_eff_max),
      ab_phase=jnp.where(
          to_select, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_016_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast hand-discard selections for AZK01-016 and AZK01-068."""
  from azuki_jax import cards
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import discard, hand_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  selected = jnp.clip(
      state.ab_eff_selected.astype(jnp.int32), 0, MAX_ABILITY_SELECTION - 1
  )
  target = hand_instance(state, owner, action[1])
  safe_target = jnp.maximum(target, 0)
  already = jnp.any(
      (jnp.arange(MAX_ABILITY_SELECTION) < state.ab_eff_selected.astype(jnp.int32))
      & (state.ab_eff_targets.astype(jnp.int32) == safe_target)
  )
  source_def = state.def_id[owner, src]
  source_azk01_016 = source_def == cards.CODE_TO_ID["AZK01-016"]
  source_azk01_068 = source_def == cards.CODE_TO_ID["AZK01-068"]
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_azk01_016 | source_azk01_068)
      & state.ab_costs_applied
      & (state.ab_eff_selected < state.ab_eff_max)
      & (
          (
              source_azk01_016
              & (state.ab_eff_min == 2)
              & (state.ab_eff_max == 2)
          )
          | (
              source_azk01_068
              & (state.ab_eff_min == 1)
              & (state.ab_eff_max == 1)
          )
      )
  )
  target_ok = (
      (target >= 0)
      & (target != src)
      & (state.zone[owner, safe_target] == Zone.HAND)
      & ~already
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[selected].set(
          jnp.where(
              do_select,
              safe_target.astype(jnp.int8),
              state.ab_eff_targets[selected],
          )
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[selected].set(
          jnp.where(
              do_select,
              owner.astype(jnp.int8),
              state.ab_eff_target_players[selected],
          )
      ),
      ab_eff_selected=jnp.where(
          do_select, state.ab_eff_selected + 1, state.ab_eff_selected
      ).astype(jnp.int8),
  )

  finish = do_select & (stepped.ab_eff_selected >= stepped.ab_eff_max)
  first = jnp.maximum(stepped.ab_eff_targets[0].astype(jnp.int32), 0)
  second = jnp.maximum(stepped.ab_eff_targets[1].astype(jnp.int32), 0)
  first_has = finish & (stepped.ab_eff_targets[0] >= 0)
  second_has = finish & (stepped.ab_eff_targets[1] >= 0)
  stepped = discard(stepped, owner, first, do=first_has)
  second_still_hand = stepped.zone[owner, second] == Zone.HAND
  stepped = discard(stepped, owner, second, do=second_has & second_still_hand)
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), cleared, stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_play_spell_azk01_017_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-017 play: enter optional two-class damage selection."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  spell_def = state.def_id[acting, safe_spell]
  is_azk01_017 = spell_def == cards.CODE_TO_ID["AZK01-017"]
  is_azk01_087 = spell_def == cards.CODE_TO_ID["AZK01-087"]
  is_spell = is_azk01_017 | is_azk01_087
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)
  safe_defs = jnp.maximum(state.def_id, 0)
  any_garden = jnp.any(
      (state.zone == Zone.GARDEN)
      & (state.def_id >= 0)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
  )
  any_leader = jnp.any((state.zone == Zone.LEADER) & (state.def_id >= 0))
  classes = any_garden.astype(jnp.int32) + any_leader.astype(jnp.int32)
  opp = (acting + 1) % 2
  enemy_garden = jnp.any(
      (state.zone[opp] == Zone.GARDEN)
      & (state.def_id[opp] >= 0)
      & (jnp.asarray(cards.TYPE)[jnp.maximum(state.def_id[opp], 0)] == CardType.ENTITY)
  )
  max_targets = jnp.where(is_azk01_087, 2, classes).astype(jnp.int32)
  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
      & ((is_azk01_017 & (classes > 0)) | (is_azk01_087 & enemy_garden))
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, max_targets.astype(jnp.int8), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_017_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-017 effect selection/skip for clean nonlethal damage."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import MAX_TARGET_CHOICES, _clear_context
  from azuki_jax.engine.helpers import card_at_slot, leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  selected = jnp.clip(
      state.ab_eff_selected.astype(jnp.int32), 0, MAX_ABILITY_SELECTION - 1
  )
  target_index = action[1].astype(jnp.int32)
  opp = (owner + 1) % 2
  target_player = jnp.where(
      target_index < GARDEN_SIZE,
      owner,
      jnp.where(target_index < 2 * GARDEN_SIZE, opp, jnp.where(target_index == 10, owner, opp)),
  )
  garden_slot = jnp.where(
      target_index < GARDEN_SIZE,
      target_index,
      target_index - GARDEN_SIZE,
  )
  garden_slot = jnp.clip(garden_slot, 0, GARDEN_SIZE - 1)
  target_is_leader = target_index >= 2 * GARDEN_SIZE
  garden_target = card_at_slot(state, target_player, Zone.GARDEN, garden_slot)
  leader = leader_instance(state, target_player)
  target = jnp.where(target_is_leader, leader, garden_target)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[target_player, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)

  prior_k = jnp.arange(MAX_ABILITY_SELECTION)
  prior_active = prior_k < state.ab_eff_selected.astype(jnp.int32)
  prior_target = state.ab_eff_targets.astype(jnp.int32)
  prior_player = state.ab_eff_target_players.astype(jnp.int32)
  prior_same = (
      prior_active
      & (prior_target == safe_target)
      & (prior_player == target_player)
  )
  prior_is_leader = (
      state.zone[jnp.maximum(prior_player, 0), jnp.maximum(prior_target, 0)]
      == Zone.LEADER
  )
  prior_class = prior_active & (prior_is_leader == target_is_leader)
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-017"])
      & state.ab_costs_applied
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max > 0)
      & (state.ab_eff_selected < state.ab_eff_max)
  )
  target_is_entity = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY,
      False,
  )
  target_ok = (
      (target_index >= 0)
      & (target_index < MAX_TARGET_CHOICES)
      & (target >= 0)
      & (
          (target_is_leader & (state.zone[target_player, safe_target] == Zone.LEADER))
          | (
              ~target_is_leader
              & (state.zone[target_player, safe_target] == Zone.GARDEN)
              & target_is_entity
          )
      )
      & ~jnp.any(prior_same | prior_class)
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  do_skip = do_action & (action_type == Act.NOOP) & source_ok

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[selected].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[selected])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[selected].set(
          jnp.where(
              do_select,
              target_player.astype(jnp.int8),
              state.ab_eff_target_players[selected],
          )
      ),
      ab_eff_selected=jnp.where(
          do_select, state.ab_eff_selected + 1, state.ab_eff_selected
      ).astype(jnp.int8),
  )
  finish = do_skip | (do_select & (stepped.ab_eff_selected >= stepped.ab_eff_max))
  for k in range(2):
    tp = jnp.maximum(stepped.ab_eff_target_players[k].astype(jnp.int32), 0)
    ti = jnp.maximum(stepped.ab_eff_targets[k].astype(jnp.int32), 0)
    has = finish & (k < stepped.ab_eff_selected.astype(jnp.int32)) & (
        stepped.ab_eff_targets[k] >= 0
    )
    stepped = deal_effect_damage(stepped, tp, ti, 1, do=has)
  cleared = recompute_passives(_clear_context(stepped))
  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), cleared, stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )


  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_087_fast(
    state: State,
    actions: jax.Array,
    prev_terminals: jax.Array,
    prev_truncations: jax.Array,
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-087 target selection: bottom-deck up to two enemy entities."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import bottom_deck_from_play
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-087"])
      & state.ab_costs_applied
      & (state.ab_eff_selected < state.ab_eff_max)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 2)
  )

  target_slot = action[1].astype(jnp.int32)
  target = card_at_slot(state, opp, Zone.GARDEN, target_slot)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  selected_count = jnp.clip(
      state.ab_eff_selected.astype(jnp.int32), 0, MAX_ABILITY_SELECTION
  )
  target_axis = jnp.arange(MAX_ABILITY_SELECTION)
  already = jnp.any(
      (target_axis < selected_count)
      & (state.ab_eff_targets.astype(jnp.int32) == safe_target)
      & (state.ab_eff_target_players.astype(jnp.int32) == opp)
  )
  picked0 = jnp.maximum(state.ab_eff_targets[0].astype(jnp.int32), 0)
  picked1 = jnp.maximum(state.ab_eff_targets[1].astype(jnp.int32), 0)
  picked0_def = state.def_id[opp, picked0]
  picked1_def = state.def_id[opp, picked1]
  picked0_cost = jnp.where(
      state.ab_eff_selected > 0,
      jnp.asarray(cards.IKZ_COST)[jnp.maximum(picked0_def, 0)].astype(jnp.int32),
      0,
  )
  picked1_cost = jnp.where(
      state.ab_eff_selected > 1,
      jnp.asarray(cards.IKZ_COST)[jnp.maximum(picked1_def, 0)].astype(jnp.int32),
      0,
  )
  current_cost = picked0_cost + picked1_cost
  target_cost = jnp.asarray(cards.IKZ_COST)[safe_target_def].astype(jnp.int32)
  target_ok = (
      (target_slot >= 0)
      & (target_slot < GARDEN_SIZE)
      & (target >= 0)
      & (target_def >= 0)
      & (state.zone[opp, safe_target] == Zone.GARDEN)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_target_def]
      & (target_cost <= 5)
      & ((current_cost + target_cost) <= 5)
      & ~already
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  do_skip = do_action & (action_type == Act.NOOP) & source_ok

  selected_index = jnp.clip(
      state.ab_eff_selected.astype(jnp.int32), 0, MAX_ABILITY_SELECTION - 1
  )
  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[selected_index].set(
          jnp.where(
              do_select,
              safe_target.astype(jnp.int8),
              state.ab_eff_targets[selected_index],
          )
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[selected_index].set(
          jnp.where(
              do_select,
              opp.astype(jnp.int8),
              state.ab_eff_target_players[selected_index],
          )
      ),
      ab_eff_selected=jnp.where(
          do_select, state.ab_eff_selected + jnp.int8(1), state.ab_eff_selected
      ).astype(jnp.int8),
  )
  finish = do_skip | (do_select & (stepped.ab_eff_selected >= stepped.ab_eff_max))
  for idx in range(2):
    target_i = jnp.maximum(stepped.ab_eff_targets[idx].astype(jnp.int32), 0)
    player_i = jnp.clip(stepped.ab_eff_target_players[idx].astype(jnp.int32), 0, 1)
    has_i = stepped.ab_eff_targets[idx] >= 0
    stepped = bottom_deck_from_play(stepped, player_i, target_i, finish & has_i)
  cleared = recompute_passives(_clear_context(stepped))
  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), cleared, stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_select | do_skip, a, b), stepped, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )
  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )
  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations


def step_play_spell_stt02_014_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-014 play: pay/discard, then select an enemy low-cost garden entity."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["STT02-014"]
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)

  opp = (acting + 1) % 2
  opp_defs = state.def_id[opp]
  opp_safe_defs = jnp.maximum(opp_defs, 0)
  target_available = jnp.any(
      (state.zone[opp] == Zone.GARDEN)
      & (opp_defs >= 0)
      & (jnp.asarray(cards.TYPE)[opp_safe_defs] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[opp_safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[opp_safe_defs].astype(jnp.int32) <= 2)
  )
  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
      & target_available
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_play_spell_stt02_015_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-015 response spell setup into any-garden bounce selection."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["STT02-015"]
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)

  defs = state.def_id
  safe_defs = jnp.maximum(defs, 0)
  target_available = jnp.any(
      (state.zone == Zone.GARDEN)
      & (defs >= 0)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[safe_defs].astype(jnp.int32) <= 3)
  )
  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
      & target_available
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_spell_stt02_016_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast response spell setup for STT02-016."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance, leader_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["STT02-016"]
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)
  opp = (acting + 1) % 2
  target_available = (
      leader_instance(state, opp) >= 0
  ) | jnp.any(state.zone[opp] == Zone.GARDEN)
  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
      & target_available
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(1), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.COST_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_spell_azk01_029_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast response spell setup for AZK01-029 into discard-2 cost selection."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["AZK01-029"]
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)
  other_hand_count = jnp.sum(
      (state.zone[acting] == Zone.HAND)
      & (jnp.arange(state.zone.shape[1]) != safe_spell),
      dtype=jnp.int32,
  )
  has_effect_target = (
      jnp.any(state.zone[acting] == Zone.LEADER)
      | jnp.any(state.zone[(acting + 1) % 2] == Zone.LEADER)
      | jnp.any(state.zone[acting] == Zone.GARDEN)
      | jnp.any(state.zone[(acting + 1) % 2] == Zone.GARDEN)
  )
  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
      & (other_hand_count >= 2)
      & has_effect_target
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(2), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.COST_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_play_spell_stt02_017_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-017 play: bounce low-cost opponent garden entities."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import garden_seq_order, return_to_hand
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance, leader_instance
  from azuki_jax.engine.triggers import (
      TIMING_WHEN_RETURNED_TO_HAND,
      pop_effect,
  )
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["STT02-017"]
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)
  leader = leader_instance(state, acting)
  safe_leader = jnp.maximum(leader, 0)
  leader_def = state.def_id[acting, safe_leader]
  leader_shao = jnp.where(
      leader_def >= 0,
      jnp.asarray(cards.SUBTYPE_MATRIX)[
          jnp.maximum(leader_def, 0), cards.subtype_index("Shao")
      ],
      False,
  )
  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
      & leader_shao
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
  )

  opp = (acting + 1) % 2
  order, in_garden = garden_seq_order(stepped, opp)
  opp_defs = stepped.def_id[opp]
  opp_safe_defs = jnp.maximum(opp_defs, 0)
  marked = (
      in_garden
      & (jnp.asarray(cards.TYPE)[opp_safe_defs] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[opp_safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[opp_safe_defs].astype(jnp.int32) <= 4)
  )
  for k in range(GARDEN_SIZE):
    inst = order[k]
    stepped = return_to_hand(stepped, opp, inst, do_play & marked[inst])

  stepped = recompute_passives(stepped)
  popped, src, owner, timing = pop_effect(stepped)
  src_i = jnp.maximum(src.astype(jnp.int32), 0)
  owner_i = jnp.maximum(owner.astype(jnp.int32), 0)
  begin_stt02_010 = (
      do_play
      & (stepped.trig_count > 0)
      & (timing == jnp.int8(TIMING_WHEN_RETURNED_TO_HAND))
      & (stepped.def_id[owner_i, src_i] == cards.CODE_TO_ID["STT02-010"])
  )
  begun = resolve_triggered_effect(popped, src, owner, timing)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(begin_stt02_010, a, b), begun, stepped
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_play_spell_stt01_017_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast response play setup for STT01-017 into 1-2 target selection."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["STT01-017"]
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)
  opp = (acting + 1) % 2
  target_available = jnp.any(state.zone[opp] == Zone.GARDEN)
  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
      & target_available
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(2), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_play_spell_azk01_031_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast spell setup for AZK01-031/AZK01-092 reveal selections."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  source_def = state.def_id[acting, safe_spell]
  is_azk01_031 = source_def == cards.CODE_TO_ID["AZK01-031"]
  is_azk01_092 = source_def == cards.CODE_TO_ID["AZK01-092"]
  is_spell = is_azk01_031 | is_azk01_092
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)
  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(do_play, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_play,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_play,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_play,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_play,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )
  reveal_count = jnp.where(is_azk01_092, jnp.int32(5), jnp.int32(3))
  stepped = sel_mod.reveal_top_into_selection(stepped, reveal_count, 1, do_play)
  revealed = stepped.ab_sel_count > 0
  sel_cards = stepped.ab_sel_cards
  sel_safe = jnp.maximum(sel_cards.astype(jnp.int32), 0)
  sel_defs = stepped.def_id[acting, sel_safe]
  sel_safe_defs = jnp.maximum(sel_defs, 0)
  sel_valid = sel_cards >= 0
  element = jnp.asarray(cards.ELEMENT)
  has_cost = jnp.asarray(cards.HAS_IKZ_COST)
  ikz_cost = jnp.asarray(cards.IKZ_COST)
  water = element[sel_safe_defs] == 2
  water_le2 = water & has_cost[sel_safe_defs] & (ikz_cost[sel_safe_defs] <= 2)
  matching = jnp.any(
      sel_valid
      & jnp.where(is_azk01_092, water_le2, water)
  )
  next_phase = jnp.where(
      matching,
      jnp.int8(AbilityPhase.SELECTION_PICK),
      jnp.int8(AbilityPhase.BOTTOM_DECK),
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(do_play & revealed, next_phase, stepped.ab_phase)
  )
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_play & ~revealed, a, b),
      cleared,
      stepped,
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations
def step_play_spell_azk01_032_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast spell setup for AZK01-032 into mandatory cost selection."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["AZK01-032"]
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)
  row_defs = state.def_id[acting]
  row_safe_defs = jnp.maximum(row_defs, 0)
  cost_targets = (
      (state.zone[acting] == Zone.GARDEN)
      & (row_defs >= 0)
      & (jnp.asarray(cards.TYPE)[row_safe_defs] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[row_safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[row_safe_defs].astype(jnp.int32) >= 2)
  )
  cost_target_available = jnp.any(cost_targets)
  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
      & cost_target_available
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(1), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.COST_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_spell_azk01_002_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast immediate spell path for AZK01-002 healing the owner leader."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import heal_leader
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["AZK01-002"]
  do_play = do_action & (spell >= 0) & is_spell
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
  )
  stepped = heal_leader(stepped, acting, 2, do_play)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_spell_stt03_016_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast immediate STT03-016 spell: destroy enemy garden entities at HP <= 2."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import destroy_card, garden_seq_order
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["STT03-016"]
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)
  opp = (acting + 1) % 2
  order, in_garden = garden_seq_order(state, opp)
  opp_defs = state.def_id[opp]
  opp_safe_defs = jnp.maximum(opp_defs, 0)
  marked = (
      in_garden
      & (opp_defs >= 0)
      & (jnp.asarray(cards.TYPE)[opp_safe_defs] == CardType.ENTITY)
      & (state.cur_hp[opp] <= 2)
  )
  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
      & jnp.any(marked)
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
  )
  for k in range(GARDEN_SIZE):
    inst = order[k]
    stepped = destroy_card(stepped, opp, inst, do=do_play & marked[inst])
  stepped = recompute_passives(stepped)
  # C's tick then pops the queued when-destroyed trigger: a destroyed
  # STT03-006 begins its draw/discard ability with control transferred to
  # its owner. The host mask only admits rows whose sole destroy trigger is
  # a single STT03-006.
  stepped = _begin_queued_stt03_006_destroy_trigger(stepped, do_play)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_spell_azk01_065_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast play setup for AZK01-065 into required effect selection."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["AZK01-065"]
  do_play = do_action & (spell >= 0) & is_spell
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_spell_stt04_015_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT04-015 play: ping own leader, then deal 2 to enemy leader."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance, leader_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["STT04-015"]
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)
  own_leader = leader_instance(state, acting)
  opp = (acting + 1) % 2
  enemy_leader = leader_instance(state, opp)
  safe_own_leader = jnp.maximum(own_leader, 0)
  safe_enemy_leader = jnp.maximum(enemy_leader, 0)
  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
      & (own_leader >= 0)
      & (enemy_leader >= 0)
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
  )
  stepped = deal_effect_damage(stepped, acting, safe_own_leader, 1, do_play)
  stepped = deal_effect_damage(stepped, opp, safe_enemy_leader, 2, do_play)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_spell_azk01_066_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-066 Firestorm: deal 2 to all leaders and garden entities."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage, garden_seq_order
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance, leader_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["AZK01-066"]
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)
  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
  )

  for p in (0, 1):
    leader = leader_instance(stepped, p)
    stepped = deal_effect_damage(
        stepped,
        p,
        jnp.maximum(leader, 0),
        2,
        do_play & (leader >= 0),
        src_player=acting,
        src_inst=safe_spell,
    )
    order, in_garden = garden_seq_order(stepped, p)
    safe_defs = jnp.maximum(stepped.def_id[p], 0)
    marked = (
        in_garden
        & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
    )
    for k in range(GARDEN_SIZE):
      inst = order[k]
      stepped = deal_effect_damage(
          stepped,
          p,
          inst,
          2,
          do_play & marked[inst],
          src_player=acting,
          src_inst=safe_spell,
      )
  stepped = recompute_passives(stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_play_spell_azk01_086_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-084/AZK01-086 discard-to-selection spell play."""
  from azuki_jax import cards
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  spell_def = state.def_id[acting, safe_spell]
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)
  safe_defs = jnp.maximum(state.def_id[acting], 0)
  has_cost = jnp.asarray(cards.HAS_IKZ_COST)[safe_defs]
  normal_entity_row = (
      (state.def_id[acting] >= 0)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
      & (jnp.asarray(cards.ELEMENT)[safe_defs] == 0)
      & has_cost
      & (jnp.asarray(cards.IKZ_COST)[safe_defs] <= 6)
  )
  weapon_row = (
      (state.def_id[acting] >= 0)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.WEAPON)
  )
  is_azk01_084 = spell_def == cards.CODE_TO_ID["AZK01-084"]
  is_azk01_086 = spell_def == cards.CODE_TO_ID["AZK01-086"]
  eligible_row = (
      (is_azk01_084 & normal_entity_row)
      | (is_azk01_086 & weapon_row)
  )
  eligible_discard = (state.zone[acting] == Zone.DISCARD) & eligible_row
  eligible_count = jnp.sum(eligible_discard, dtype=jnp.int32)
  pick_max = jnp.where(
      is_azk01_084,
      jnp.int32(1),
      jnp.minimum(eligible_count, jnp.int32(5)),
  )
  do_play = (
      do_action
      & (action_type == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & (is_azk01_084 | is_azk01_086)
      & can_pay
      & (eligible_count > 0)
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.SELECTION_PICK), stepped.ab_phase
      ),
  )
  stepped = sel_mod.move_matching_zone_to_selection(
      stepped,
      Zone.DISCARD,
      eligible_row,
      pick_max,
      do_play,
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_play_spell_azk01_009_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast main-phase AZK01-009 spell setup into required effect selection."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["AZK01-009"]
  do_play = do_action & (spell >= 0) & is_spell
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_play_spell_azk01_117_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-117 play: self-damage cost into Charge target selection."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance, leader_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["AZK01-117"]
  leader = leader_instance(state, acting)
  safe_leader = jnp.maximum(leader, 0)
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)

  all_defs = state.def_id
  safe_defs = jnp.maximum(all_defs, 0)
  target_available = jnp.any(
      (state.zone == Zone.GARDEN)
      & (all_defs >= 0)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[safe_defs].astype(jnp.int32) <= 5)
  )

  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
      & (leader >= 0)
      & target_available
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_play_spell_azk01_042_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast main-phase AZK01-042 setup into three enemy-garden selections."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["AZK01-042"]
  do_play = do_action & (spell >= 0) & is_spell
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(3), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(3), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_play_spell_azk01_127_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast response play setup for AZK01-127 into required effect selection."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["AZK01-127"]
  do_play = do_action & (spell >= 0) & is_spell
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_spell_azk01_128_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast response play setup for AZK01-128 into required effect selection."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["AZK01-128"]
  do_play = do_action & (spell >= 0) & is_spell
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(1), stepped.ab_eff_max),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_play_stt02_003_reveal_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast STT02-003 play/reveal path."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.engine import ikz
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import (
      attr_force_tapped,
      hand_instance,
      has_charge,
  )
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)

  place = do_action & (inst >= 0)
  cost = effective_play_cost(state, acting, safe)
  stepped = ikz.pay(state, acting, cost, use_token, do=place)

  pz = jnp.asarray(placement_zone, jnp.int8)
  stepped = _enter_board_slot(stepped, acting, safe, placement_zone, slot, place)
  stepped = recompute_passives(stepped)

  is_garden = pz == jnp.int8(Zone.GARDEN)
  enters_tapped = attr_force_tapped(stepped, acting, safe)
  stepped = stepped._replace(
      tapped=stepped.tapped.at[acting, safe].set(
          jnp.where(
              do_action & is_garden,
              enters_tapped | stepped.tapped[acting, safe],
              stepped.tapped[acting, safe],
          )
      ),
      cooldown=stepped.cooldown.at[acting, safe].set(
          jnp.where(
              do_action & is_garden,
              (~has_charge(stepped, acting, safe)).astype(jnp.uint8),
              stepped.cooldown[acting, safe],
          )
      ),
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((do_action & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((do_action & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_action.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_action, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_action, safe.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(do_action, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )
  stepped = sel_mod.reveal_top_into_selection(stepped, 5, 1, do_action)
  revealed = stepped.ab_sel_count > 0
  watercrafting = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Watercrafting")], jnp.bool_
  )
  sel_cards = stepped.ab_sel_cards
  sel_safe = jnp.maximum(sel_cards.astype(jnp.int32), 0)
  sel_defs = stepped.def_id[acting, sel_safe]
  sel_valid = sel_cards >= 0
  matching = jnp.any(sel_valid & watercrafting[jnp.maximum(sel_defs, 0)])
  next_phase = jnp.where(
      matching,
      jnp.int8(AbilityPhase.SELECTION_PICK),
      jnp.int8(AbilityPhase.BOTTOM_DECK),
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action & revealed,
          next_phase,
          stepped.ab_phase,
      )
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      pz == jnp.int8(Zone.GARDEN),
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_stt02_013_reveal_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast STT02-013 play/reveal path."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.engine import ikz
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)

  cost = effective_play_cost(state, acting, safe)
  place = do_action & (inst >= 0)
  stepped = ikz.pay(state, acting, cost, use_token, do=place)

  pz = jnp.asarray(placement_zone, jnp.int8)
  stepped = _enter_board_slot(stepped, acting, safe, placement_zone, slot, place)
  stepped = recompute_passives(stepped)
  is_garden = pz == jnp.int8(Zone.GARDEN)
  stepped = stepped._replace(
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((place & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((place & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          place.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(place, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(place, safe.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(place, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(place, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(place, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(place, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(place, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(place, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(place, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(place, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          place,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          place,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(place, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(place, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(place, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          place,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          place,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )
  stepped = sel_mod.reveal_top_into_selection(stepped, 3, 1, do_action)
  revealed = stepped.ab_sel_count > 0
  sel_cards = stepped.ab_sel_cards
  sel_safe = jnp.maximum(sel_cards.astype(jnp.int32), 0)
  sel_defs = stepped.def_id[acting, sel_safe]
  sel_safe_defs = jnp.maximum(sel_defs, 0)
  sel_valid = sel_cards >= 0
  element = jnp.asarray(cards.ELEMENT, jnp.int8)
  has_cost = jnp.asarray(cards.HAS_IKZ_COST, jnp.bool_)
  ikz_cost = jnp.asarray(cards.IKZ_COST, jnp.int8)
  matching = jnp.any(
      sel_valid
      & has_cost[sel_safe_defs]
      & (ikz_cost[sel_safe_defs] <= 2)
      & (element[sel_safe_defs] == 2)
  )
  next_phase = jnp.where(
      matching,
      jnp.int8(AbilityPhase.SELECTION_PICK),
      jnp.int8(AbilityPhase.BOTTOM_DECK),
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action & revealed,
          next_phase,
          stepped.ab_phase,
      )
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      pz == jnp.int8(Zone.GARDEN),
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_azk01_033_reveal_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast AZK01-033/AZK01-069 play/reveal path."""
  from azuki_jax import cards
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import (
      _detach_from_location,
      attr_force_tapped,
      hand_instance,
      has_charge,
  )
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)
  source_def = state.def_id[acting, safe]

  cost = effective_play_cost(state, acting, safe)
  stepped = ikz.pay(state, acting, cost, use_token, do=do_action & (inst >= 0))
  stepped = _detach_from_location(stepped, acting, safe, do_action & (inst >= 0))

  pz = jnp.asarray(placement_zone, jnp.int8)
  stepped = stepped._replace(
      zone=stepped.zone.at[acting, safe].set(
          jnp.where(do_action, pz, stepped.zone[acting, safe])
      ),
      zpos=stepped.zpos.at[acting, safe].set(
          jnp.where(do_action, slot.astype(jnp.int8), stepped.zpos[acting, safe])
      ),
      board_seq=stepped.board_seq.at[acting, safe].set(
          jnp.where(do_action, stepped.seq_counter, stepped.board_seq[acting, safe])
      ),
      seq_counter=(stepped.seq_counter + do_action.astype(jnp.int16)).astype(jnp.int16),
  )

  is_garden = pz == jnp.int8(Zone.GARDEN)
  enters_tapped = attr_force_tapped(stepped, acting, safe)
  stepped = stepped._replace(
      tapped=stepped.tapped.at[acting, safe].set(
          jnp.where(
              do_action & is_garden,
              enters_tapped | stepped.tapped[acting, safe],
              stepped.tapped[acting, safe],
          )
      ),
      cooldown=stepped.cooldown.at[acting, safe].set(
          jnp.where(
              do_action & is_garden,
              (~has_charge(stepped, acting, safe)).astype(jnp.uint8),
              stepped.cooldown[acting, safe],
          )
      ),
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((do_action & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((do_action & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_action.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_action, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_action, safe.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(do_action, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )
  stepped = sel_mod.reveal_top_into_selection(stepped, 5, 1, do_action)
  revealed = stepped.ab_sel_count > 0
  steelborn = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Steelborn")], jnp.bool_
  )
  beanz = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Beanz")], jnp.bool_
  )
  source_azk01_033 = source_def == cards.CODE_TO_ID["AZK01-033"]
  source_azk01_069 = source_def == cards.CODE_TO_ID["AZK01-069"]
  sel_cards = stepped.ab_sel_cards
  sel_safe = jnp.maximum(sel_cards.astype(jnp.int32), 0)
  sel_defs = stepped.def_id[acting, sel_safe]
  sel_valid = sel_cards >= 0
  safe_sel_defs = jnp.maximum(sel_defs, 0)
  matching = jnp.any(
      sel_valid
      & (
          (source_azk01_033 & steelborn[safe_sel_defs])
          | (source_azk01_069 & beanz[safe_sel_defs])
      )
  )
  next_phase = jnp.where(
      matching,
      jnp.int8(AbilityPhase.SELECTION_PICK),
      jnp.int8(AbilityPhase.BOTTOM_DECK),
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action & revealed,
          next_phase,
          stepped.ab_phase,
      )
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      pz == jnp.int8(Zone.GARDEN),
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_azk01_045_reveal_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast AZK01-045 play/reveal path."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import (
      _detach_from_location,
      attr_force_tapped,
      hand_instance,
      has_charge,
  )
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)

  cost = effective_play_cost(state, acting, safe)
  stepped = ikz.pay(state, acting, cost, use_token, do=do_action & (inst >= 0))
  stepped = _detach_from_location(stepped, acting, safe, do_action & (inst >= 0))

  pz = jnp.asarray(placement_zone, jnp.int8)
  stepped = stepped._replace(
      zone=stepped.zone.at[acting, safe].set(
          jnp.where(do_action, pz, stepped.zone[acting, safe])
      ),
      zpos=stepped.zpos.at[acting, safe].set(
          jnp.where(do_action, slot.astype(jnp.int8), stepped.zpos[acting, safe])
      ),
      board_seq=stepped.board_seq.at[acting, safe].set(
          jnp.where(do_action, stepped.seq_counter, stepped.board_seq[acting, safe])
      ),
      seq_counter=(stepped.seq_counter + do_action.astype(jnp.int16)).astype(jnp.int16),
  )

  is_garden = pz == jnp.int8(Zone.GARDEN)
  enters_tapped = attr_force_tapped(stepped, acting, safe)
  stepped = stepped._replace(
      tapped=stepped.tapped.at[acting, safe].set(
          jnp.where(
              do_action & is_garden,
              enters_tapped | stepped.tapped[acting, safe],
              stepped.tapped[acting, safe],
          )
      ),
      cooldown=stepped.cooldown.at[acting, safe].set(
          jnp.where(
              do_action & is_garden,
              (~has_charge(stepped, acting, safe)).astype(jnp.uint8),
              stepped.cooldown[acting, safe],
          )
      ),
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((do_action & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((do_action & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_action.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_action, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_action, safe.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(do_action, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )
  stepped = recompute_passives(stepped)
  stepped = sel_mod.reveal_top_into_selection(stepped, 5, 1, do_action)
  revealed = stepped.ab_sel_count > 0
  obsidian = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Obsidian")], jnp.bool_
  )
  sel_cards = stepped.ab_sel_cards
  sel_safe = jnp.maximum(sel_cards.astype(jnp.int32), 0)
  sel_defs = stepped.def_id[acting, sel_safe]
  sel_valid = sel_cards >= 0
  matching = jnp.any(sel_valid & obsidian[jnp.maximum(sel_defs, 0)])
  next_phase = jnp.where(
      matching,
      jnp.int8(AbilityPhase.SELECTION_PICK),
      jnp.int8(AbilityPhase.BOTTOM_DECK),
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action & revealed,
          next_phase,
          stepped.ab_phase,
      )
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      pz == jnp.int8(Zone.GARDEN),
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_azk01_056_reveal_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast AZK01-056 play/reveal path."""
  from azuki_jax import cards
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import (
      _detach_from_location,
      attr_force_tapped,
      hand_instance,
      has_charge,
  )
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)

  cost = effective_play_cost(state, acting, safe)
  stepped = ikz.pay(state, acting, cost, use_token, do=do_action & (inst >= 0))
  stepped = _detach_from_location(stepped, acting, safe, do_action & (inst >= 0))

  pz = jnp.asarray(placement_zone, jnp.int8)
  stepped = stepped._replace(
      zone=stepped.zone.at[acting, safe].set(
          jnp.where(do_action, pz, stepped.zone[acting, safe])
      ),
      zpos=stepped.zpos.at[acting, safe].set(
          jnp.where(do_action, slot.astype(jnp.int8), stepped.zpos[acting, safe])
      ),
      board_seq=stepped.board_seq.at[acting, safe].set(
          jnp.where(do_action, stepped.seq_counter, stepped.board_seq[acting, safe])
      ),
      seq_counter=(stepped.seq_counter + do_action.astype(jnp.int16)).astype(jnp.int16),
  )

  is_garden = pz == jnp.int8(Zone.GARDEN)
  enters_tapped = attr_force_tapped(stepped, acting, safe)
  stepped = stepped._replace(
      tapped=stepped.tapped.at[acting, safe].set(
          jnp.where(
              do_action & is_garden,
              enters_tapped | stepped.tapped[acting, safe],
              stepped.tapped[acting, safe],
          )
      ),
      cooldown=stepped.cooldown.at[acting, safe].set(
          jnp.where(
              do_action & is_garden,
              (~has_charge(stepped, acting, safe)).astype(jnp.uint8),
              stepped.cooldown[acting, safe],
          )
      ),
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((do_action & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((do_action & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_action.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_action, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_action, safe.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(do_action, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )
  stepped = sel_mod.reveal_top_into_selection(stepped, 5, 1, do_action)
  revealed = stepped.ab_sel_count > 0
  scorchweaver = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Scorchweaver")], jnp.bool_
  )
  sel_cards = stepped.ab_sel_cards
  sel_safe = jnp.maximum(sel_cards.astype(jnp.int32), 0)
  sel_defs = stepped.def_id[acting, sel_safe]
  sel_valid = sel_cards >= 0
  matching = jnp.any(sel_valid & scorchweaver[jnp.maximum(sel_defs, 0)])
  next_phase = jnp.where(
      matching,
      jnp.int8(AbilityPhase.SELECTION_PICK),
      jnp.int8(AbilityPhase.BOTTOM_DECK),
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action & revealed,
          next_phase,
          stepped.ab_phase,
      )
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      pz == jnp.int8(Zone.GARDEN),
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_stt04_005_reveal_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast STT04-005 play/reveal path."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.engine import ikz
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import (
      card_at_slot,
      discard_equipped_weapons,
      hand_instance,
  )
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)
  valid_card = state.def_id[acting, safe] == cards.CODE_TO_ID["STT04-005"]
  place = do_action & (inst >= 0) & valid_card

  cost = effective_play_cost(state, acting, safe)
  stepped = ikz.pay(state, acting, cost, use_token, do=place)
  displaced = card_at_slot(stepped, acting, placement_zone, slot)
  displaced_safe = jnp.maximum(displaced, 0)
  zone_full = jnp.sum(
      stepped.zone[acting] == jnp.asarray(placement_zone, jnp.int8),
      dtype=jnp.int32,
  ) >= GARDEN_SIZE
  displacing = place & (displaced >= 0) & zone_full
  stepped = discard_equipped_weapons(
      stepped, acting, displaced_safe, do=displacing
  )
  stepped = _enter_board_slot(stepped, acting, safe, placement_zone, slot, place)
  stepped = recompute_passives(stepped)

  pz = jnp.asarray(placement_zone, jnp.int8)
  is_garden = pz == jnp.int8(Zone.GARDEN)
  stepped = stepped._replace(
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((place & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((place & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          place.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(place, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(place, safe.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(place, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(place, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(place, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(place, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(place, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(place, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(place, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(place, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          place,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          place,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(place, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(place, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(place, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          place,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          place,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )
  stepped = sel_mod.reveal_top_into_selection(stepped, 5, 1, place)
  revealed = stepped.ab_sel_count > 0
  pyreskin = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Pyreskin")], jnp.bool_
  )
  sel_cards = stepped.ab_sel_cards
  sel_safe = jnp.maximum(sel_cards.astype(jnp.int32), 0)
  sel_defs = stepped.def_id[acting, sel_safe]
  sel_valid = sel_cards >= 0
  matching = jnp.any(sel_valid & pyreskin[jnp.maximum(sel_defs, 0)])
  next_phase = jnp.where(
      matching,
      jnp.int8(AbilityPhase.SELECTION_PICK),
      jnp.int8(AbilityPhase.BOTTOM_DECK),
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          place & revealed,
          next_phase,
          stepped.ab_phase,
      )
  )
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(place & ~revealed, a, b),
      cleared,
      stepped,
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      pz == jnp.int8(Zone.GARDEN),
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_cost_stt04_016_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast cost selection for STT04-016 into effect selection."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.engine.helpers import card_at_slot, discard
  from azuki_jax.engine.triggers import TIMING_WHEN_TAKES_DAMAGE, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = state.def_id[owner, src] == cards.CODE_TO_ID["STT04-016"]
  target = card_at_slot(state, owner, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  target_ok = (
      (target_def >= 0)
      & (jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)] == CardType.ENTITY)
  )
  inst_cols = jnp.arange(state.def_id.shape[1])
  owner_defs = state.def_id[owner]
  owner_safe_defs = jnp.maximum(owner_defs, 0)
  has_other_garden_entity = jnp.any(
      (state.zone[owner] == Zone.GARDEN)
      & (owner_defs >= 0)
      & (jnp.asarray(cards.TYPE)[owner_safe_defs] == CardType.ENTITY)
      & (inst_cols != safe_target)
  )
  azk01_059_spent_lethal = (
      (target_def == cards.CODE_TO_ID["AZK01-059"])
      & ((state.once_per_turn_used[owner, safe_target] & 1) != 0)
      & (state.cur_hp[owner, safe_target] <= 1)
  )
  do_select = (
      do_action
      & (state.ab_phase == AbilityPhase.COST_SELECTION)
      & source_ok
      & (target >= 0)
      & target_ok
  )
  stepped = state._replace(
      ab_cost_targets=state.ab_cost_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_cost_targets[0])
      ),
      ab_cost_target_players=state.ab_cost_target_players.at[0].set(
          jnp.where(do_select, owner.astype(jnp.int8), state.ab_cost_target_players[0])
      ),
      ab_cost_selected=jnp.where(
          do_select, jnp.int8(1), state.ab_cost_selected
      ),
  )
  stepped = deal_effect_damage(
      stepped, owner, safe_target, 1, do_select & ~azk01_059_spent_lethal
  )
  stepped = discard(
      stepped, owner, safe_target, do=do_select & azk01_059_spent_lethal
  )
  azk01_059_no_target = (
      do_select
      & (target_def == cards.CODE_TO_ID["AZK01-059"])
      & ~has_other_garden_entity
  )
  trig_src0 = jnp.maximum(stepped.trig_source[0].astype(jnp.int32), 0)
  trig_owner0 = jnp.maximum(stepped.trig_owner[0].astype(jnp.int32), 0)
  azk01_059_fizzle = (
      azk01_059_no_target
      & (stepped.trig_count > 0)
      & (stepped.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (stepped.trig_owner[0] == owner.astype(jnp.int8))
      & (stepped.trig_source[0] == safe_target.astype(jnp.int8))
      & (stepped.def_id[trig_owner0, trig_src0] == cards.CODE_TO_ID["AZK01-059"])
  )
  popped, _, _, _ = pop_effect(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(azk01_059_fizzle, a, b), popped, stepped
  )
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(do_select, True, stepped.ab_costs_applied),
      ab_phase=jnp.where(
          do_select, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
      ab_eff_min=jnp.where(do_select, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_select, jnp.int8(1), stepped.ab_eff_max),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_COST_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_cost_stt02_016_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-016 discard-cost selection into effect selection."""
  from azuki_jax import cards
  from azuki_jax.engine.helpers import discard, hand_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.COST_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT02-016"])
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 1)
  )
  target = hand_instance(state, owner, action[1])
  safe_target = jnp.maximum(target, 0)
  target_ok = (target >= 0) & (safe_target != src)
  do_select = (
      do_action
      & (action[0] == Act.SELECT_COST_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_cost_targets=state.ab_cost_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_cost_targets[0])
      ),
      ab_cost_target_players=state.ab_cost_target_players.at[0].set(
          jnp.where(do_select, owner.astype(jnp.int8), state.ab_cost_target_players[0])
      ),
      ab_cost_selected=jnp.where(
          do_select, jnp.int8(1), state.ab_cost_selected
      ),
  )
  stepped = discard(stepped, owner, safe_target, do=do_select)
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(do_select, True, stepped.ab_costs_applied),
      ab_phase=jnp.where(
          do_select, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
      ab_eff_min=jnp.where(do_select, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_select, jnp.int8(1), stepped.ab_eff_max),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_COST_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_cost_azk01_029_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-029 discard-cost selection into effect selection."""
  from azuki_jax import cards
  from azuki_jax.engine.helpers import discard, hand_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  selected = jnp.minimum(state.ab_cost_selected.astype(jnp.int32), MAX_ABILITY_SELECTION - 1)
  source_ok = (
      (state.ab_phase == AbilityPhase.COST_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-029"])
      & ~state.ab_costs_applied
      & (state.ab_cost_selected < 2)
      & (state.ab_cost_max == 2)
  )
  target = hand_instance(state, owner, action[1])
  safe_target = jnp.maximum(target, 0)
  already = jnp.any(
      (jnp.arange(MAX_ABILITY_SELECTION) < state.ab_cost_selected.astype(jnp.int32))
      & (state.ab_cost_targets.astype(jnp.int32) == safe_target)
      & (state.ab_cost_target_players == owner.astype(jnp.int8))
  )
  target_ok = (
      (target >= 0)
      & (safe_target != src)
      & (state.zone[owner, safe_target] == Zone.HAND)
      & ~already
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_COST_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_cost_targets=state.ab_cost_targets.at[selected].set(
          jnp.where(
              do_select,
              safe_target.astype(jnp.int8),
              state.ab_cost_targets[selected],
          )
      ),
      ab_cost_target_players=state.ab_cost_target_players.at[selected].set(
          jnp.where(
              do_select,
              owner.astype(jnp.int8),
              state.ab_cost_target_players[selected],
          )
      ),
      ab_cost_selected=jnp.where(
          do_select, state.ab_cost_selected + 1, state.ab_cost_selected
      ).astype(jnp.int8),
  )
  finish = do_select & (stepped.ab_cost_selected >= stepped.ab_cost_max)
  first = jnp.maximum(stepped.ab_cost_targets[0].astype(jnp.int32), 0)
  second = jnp.maximum(stepped.ab_cost_targets[1].astype(jnp.int32), 0)
  first_has = finish & (stepped.ab_cost_targets[0] >= 0)
  second_has = finish & (stepped.ab_cost_targets[1] >= 0)
  stepped = discard(stepped, owner, first, do=first_has)
  second_still_hand = stepped.zone[owner, second] == Zone.HAND
  stepped = discard(stepped, owner, second, do=second_has & second_still_hand)
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(finish, True, stepped.ab_costs_applied),
      ab_phase=jnp.where(
          finish, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
      ab_eff_min=jnp.where(finish, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(finish, jnp.int8(1), stepped.ab_eff_max),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_COST_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_effect_stt02_014_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-014 effect selection: freeze one low-cost enemy garden entity."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_frozen
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT02-014"])
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )

  target_index = action[1].astype(jnp.int32)
  target = card_at_slot(state, opp, Zone.GARDEN, target_index)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_ok = (
      (target_index >= 0)
      & (target_index < GARDEN_SIZE)
      & (target >= 0)
      & (state.zone[opp, safe_target] == Zone.GARDEN)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_target_def]
      & (jnp.asarray(cards.IKZ_COST)[safe_target_def].astype(jnp.int32) <= 2)
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = apply_frozen(stepped, opp, safe_target, 2, do_select)
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(do_select, a, b), cleared, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_spell_azk01_020_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-020 play: pay/discard, then select two friendly garden entities."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import discard, hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  spell = hand_instance(state, acting, action[1])
  safe_spell = jnp.maximum(spell, 0)
  is_spell = state.def_id[acting, safe_spell] == cards.CODE_TO_ID["AZK01-020"]
  use_token = action[3] != 0
  cost = effective_play_cost(state, acting, safe_spell)
  can_pay = ikz.can_pay(state, acting, cost, use_token)
  row_defs = state.def_id[acting]
  row_safe_defs = jnp.maximum(row_defs, 0)
  friendly_garden_entities = (
      (state.zone[acting] == Zone.GARDEN)
      & (row_defs >= 0)
      & (jnp.asarray(cards.TYPE)[row_safe_defs] == CardType.ENTITY)
  )
  target_available = jnp.sum(friendly_garden_entities.astype(jnp.int32)) >= 2
  phase_ok = (state.phase == Phase.MAIN) | (state.phase == Phase.RESPONSE_WINDOW)
  do_play = (
      do_action
      & (action[0] == Act.PLAY_SPELL_FROM_HAND)
      & phase_ok
      & (state.ab_phase == AbilityPhase.NONE)
      & (spell >= 0)
      & is_spell
      & can_pay
      & target_available
  )

  stepped = ikz.pay(state, acting, cost, use_token, do=do_play)
  stepped = discard(stepped, acting, safe_spell, do=do_play)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_play.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_play, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(do_play, safe_spell.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_play, acting.astype(jnp.int8), stepped.ab_owner),
      ab_is_optional=jnp.where(do_play, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_play, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_play, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_play, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_play, jnp.int8(0), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(do_play, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_play, jnp.int8(2), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_play, jnp.int8(2), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_play,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_play,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_phase=jnp.where(
          do_play, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.PLAY_SPELL_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_effect_azk01_005_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-005 optional effect: deal 1 damage to an enemy garden entity."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot, is_effect_immune

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-005"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 1)
  )

  target_index = action[1].astype(jnp.int32)
  target = card_at_slot(state, opp, Zone.GARDEN, target_index)
  safe_target = jnp.maximum(target, 0)
  target_ok = (
      (target_index >= 0)
      & (target_index < GARDEN_SIZE)
      & (target >= 0)
      & (state.zone[opp, safe_target] == Zone.GARDEN)
      & ~is_effect_immune(state, opp, safe_target)
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  do_skip = do_action & (action_type == Act.NOOP) & source_ok
  finish = do_select | do_skip

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = deal_effect_damage(stepped, opp, safe_target, 1, do_select)
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), cleared, state)
  stepped = recompute_passives(stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_015_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-015 fire effect: deal 2 effect damage to a leader."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  target_index = action[1].astype(jnp.int32)
  opp = (owner + 1) % 2
  target_player = jnp.where(target_index == 0, owner, opp).astype(jnp.int32)
  target = leader_instance(state, target_player)
  safe_target = jnp.maximum(target, 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-015"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )
  target_ok = (
      (target_index >= 0)
      & (target_index <= 1)
      & (target >= 0)
      & (state.zone[target_player, safe_target] == Zone.LEADER)
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(
              do_select,
              target_player.astype(jnp.int8),
              state.ab_eff_target_players[0],
          )
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = deal_effect_damage(stepped, target_player, safe_target, 2, do_select)
  stepped = _clear_context(stepped)
  stepped = recompute_passives(stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_effect_azk01_020_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-020 effect: buff two friendly garden entities."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import (
      apply_attack_modifier,
      apply_health_modifier,
  )
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  selected = jnp.clip(
      state.ab_eff_selected.astype(jnp.int32), 0, MAX_ABILITY_SELECTION - 1
  )
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-020"])
      & ~state.ab_costs_applied
      & (state.ab_eff_min == 2)
      & (state.ab_eff_max == 2)
      & (state.ab_eff_selected < 2)
  )

  target_index = action[1].astype(jnp.int32)
  target = card_at_slot(state, owner, Zone.GARDEN, target_index)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  already_selected = (
      (state.ab_eff_selected > 0)
      & (state.ab_eff_target_players[0] == owner.astype(jnp.int8))
      & (state.ab_eff_targets[0] == safe_target.astype(jnp.int8))
  )
  target_ok = (
      (target_index >= 0)
      & (target_index < GARDEN_SIZE)
      & (target >= 0)
      & (state.zone[owner, safe_target] == Zone.GARDEN)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & ~already_selected
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[selected].set(
          jnp.where(
              do_select,
              safe_target.astype(jnp.int8),
              state.ab_eff_targets[selected],
          )
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[selected].set(
          jnp.where(
              do_select,
              owner.astype(jnp.int8),
              state.ab_eff_target_players[selected],
          )
      ),
      ab_eff_selected=jnp.where(
          do_select, state.ab_eff_selected + 1, state.ab_eff_selected
      ).astype(jnp.int8),
  )
  finish = do_select & (stepped.ab_eff_selected >= stepped.ab_eff_max)
  is_response = state.phase == Phase.RESPONSE_WINDOW
  for k in range(2):
    target_player = jnp.maximum(stepped.ab_eff_target_players[k].astype(jnp.int32), 0)
    target_inst = jnp.maximum(stepped.ab_eff_targets[k].astype(jnp.int32), 0)
    has_target = (
        (stepped.ab_eff_target_players[k] >= 0)
        & (stepped.ab_eff_targets[k] >= 0)
        & (jnp.asarray(k, jnp.int32) < stepped.ab_eff_selected.astype(jnp.int32))
    )
    stepped = apply_health_modifier(
        stepped,
        target_player,
        target_inst,
        1,
        expires_eot=True,
        do=finish & has_target & is_response,
    )
    stepped = apply_attack_modifier(
        stepped,
        target_player,
        target_inst,
        1,
        expires_eot=True,
        do=finish & has_target & ~is_response,
    )

  cleared = _close_response_combat_if_idle(_clear_context(stepped))
  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), cleared, stepped)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_effect_stt02_015_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-015 effect: return one clean any-garden low-cost entity."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import return_to_hand
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT02-015"])
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )

  target_index = action[1].astype(jnp.int32)
  target_player = jnp.where(target_index < GARDEN_SIZE, owner, opp)
  target_slot = jnp.where(target_index < GARDEN_SIZE, target_index, target_index - GARDEN_SIZE)
  target = card_at_slot(state, target_player, Zone.GARDEN, target_slot)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[target_player, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_ok = (
      (target_index >= 0)
      & (target_index < (2 * GARDEN_SIZE))
      & (target >= 0)
      & (state.zone[target_player, safe_target] == Zone.GARDEN)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_target_def]
      & (jnp.asarray(cards.IKZ_COST)[safe_target_def].astype(jnp.int32) <= 3)
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(
              do_select,
              target_player.astype(jnp.int8),
              state.ab_eff_target_players[0],
          )
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = return_to_hand(stepped, target_player, safe_target, do_select)
  cleared = recompute_passives(_clear_context(stepped))
  cleared = _close_response_combat_if_idle(cleared)
  stepped = jax.tree.map(lambda a, b: jnp.where(do_select, a, b), cleared, state)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_stt02_016_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-016 effect: enemy leader/garden target gets -2 ATK EOT."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_attack_modifier
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot, leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT02-016"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )

  target_index = action[1].astype(jnp.int32)
  target_is_leader = target_index == GARDEN_SIZE
  garden_slot = jnp.clip(target_index, 0, GARDEN_SIZE - 1)
  garden_target = card_at_slot(state, opp, Zone.GARDEN, garden_slot)
  leader = leader_instance(state, opp)
  target = jnp.where(target_is_leader, leader, garden_target)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[safe_target_def],
      jnp.int8(-1),
  )
  target_ok = (
      (target_index >= 0)
      & (target_index <= GARDEN_SIZE)
      & (target >= 0)
      & (
          (
              target_is_leader
              & (state.zone[opp, safe_target] == Zone.LEADER)
          )
          | (
              ~target_is_leader
              & (state.zone[opp, safe_target] == Zone.GARDEN)
              & (target_type == CardType.ENTITY)
          )
      )
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = apply_attack_modifier(
      stepped, opp, safe_target, -2, expires_eot=True, do=do_select
  )
  cleared = _close_response_combat_if_idle(_clear_context(stepped))
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_select, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_029_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-029 effect: any leader/garden target gets -3 ATK EOT."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_attack_modifier
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot, leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-029"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )

  target_index = action[1].astype(jnp.int32)
  friendly_garden = target_index < GARDEN_SIZE
  enemy_garden = (target_index >= GARDEN_SIZE) & (target_index < 2 * GARDEN_SIZE)
  own_leader = target_index == 2 * GARDEN_SIZE
  enemy_leader = target_index == (2 * GARDEN_SIZE + 1)
  garden_player = jnp.where(friendly_garden, owner, opp)
  garden_slot = jnp.where(
      friendly_garden,
      target_index,
      target_index - GARDEN_SIZE,
  )
  garden_inst = card_at_slot(state, garden_player, Zone.GARDEN, garden_slot)
  target_player = jnp.where(friendly_garden | own_leader, owner, opp).astype(jnp.int32)
  target_inst = jnp.where(
      friendly_garden | enemy_garden,
      garden_inst,
      jnp.where(own_leader, leader_instance(state, owner), leader_instance(state, opp)),
  ).astype(jnp.int32)
  safe_target = jnp.maximum(target_inst, 0)
  safe_player = jnp.clip(target_player, 0, 1)
  target_def = state.def_id[safe_player, safe_target]
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)],
      jnp.int8(-1),
  )
  valid_target = (
      ((own_leader | enemy_leader) & (target_inst >= 0))
      | (
          (friendly_garden | enemy_garden)
          & (target_inst >= 0)
          & (target_type == CardType.ENTITY)
      )
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & valid_target
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, safe_player.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = apply_attack_modifier(
      stepped, safe_player, safe_target, -3, expires_eot=True, do=do_select
  )
  cleared = _close_response_combat_if_idle(_clear_context(stepped))
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_select, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_stt01_017_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-017 effect selection; damage applies when selection finishes."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  selected = jnp.clip(
      state.ab_eff_selected.astype(jnp.int32),
      0,
      MAX_ABILITY_SELECTION - 1,
  )
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT01-017"])
      & ~state.ab_costs_applied
      & (state.ab_eff_selected < 2)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 2)
  )

  target_index = action[1].astype(jnp.int32)
  target = card_at_slot(state, opp, Zone.GARDEN, target_index)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[safe_target_def],
      jnp.int8(-1),
  )
  already_selected = (
      (state.ab_eff_selected > 0)
      & (state.ab_eff_target_players[0] == opp.astype(jnp.int8))
      & (state.ab_eff_targets[0] == safe_target.astype(jnp.int8))
  )
  target_ok = (
      (target_index >= 0)
      & (target_index < GARDEN_SIZE)
      & (target >= 0)
      & (state.zone[opp, safe_target] == Zone.GARDEN)
      & (target_type == CardType.ENTITY)
      & ~already_selected
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  do_skip = (
      do_action
      & (action_type == Act.NOOP)
      & source_ok
      & (state.ab_eff_selected >= state.ab_eff_min)
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[selected].set(
          jnp.where(
              do_select,
              safe_target.astype(jnp.int8),
              state.ab_eff_targets[selected],
          )
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[selected].set(
          jnp.where(
              do_select,
              opp.astype(jnp.int8),
              state.ab_eff_target_players[selected],
          )
      ),
      ab_eff_selected=jnp.where(
          do_select, state.ab_eff_selected + 1, state.ab_eff_selected
      ).astype(jnp.int8),
  )

  finish = do_skip | (do_select & (stepped.ab_eff_selected >= stepped.ab_eff_max))
  for k in range(2):
    tp = jnp.maximum(stepped.ab_eff_target_players[k].astype(jnp.int32), 0)
    ti = jnp.maximum(stepped.ab_eff_targets[k].astype(jnp.int32), 0)
    has = (
        (stepped.ab_eff_target_players[k] >= 0)
        & (stepped.ab_eff_targets[k] >= 0)
        & (jnp.asarray(k, jnp.int32) < stepped.ab_eff_selected.astype(jnp.int32))
    )
    stepped = deal_effect_damage(stepped, tp, ti, 1, finish & has)

  cleared = recompute_passives(_clear_context(stepped))
  cleared = _close_response_combat_if_idle(cleared)
  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), cleared, stepped)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_stt04_016_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast effect selection for STT04-016 into context clear."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot, leader_instance
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.engine.triggers import TIMING_WHEN_TAKES_DAMAGE, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = state.def_id[owner, src] == cards.CODE_TO_ID["STT04-016"]
  target_index = action[1].astype(jnp.int32)
  target_is_leader = target_index == GARDEN_SIZE
  garden_slot = jnp.clip(target_index, 0, GARDEN_SIZE - 1)
  garden_target = card_at_slot(state, opp, Zone.GARDEN, garden_slot)
  leader = leader_instance(state, opp)
  target = jnp.where(target_is_leader, leader, garden_target)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[safe_target_def],
      jnp.int8(-1),
  )
  target_ok = (
      (target_index >= 0)
      & (target_index <= GARDEN_SIZE)
      & (target >= 0)
      & (
          (
              target_is_leader
              & (state.zone[opp, safe_target] == Zone.LEADER)
          )
          | (
              ~target_is_leader
              & (state.zone[opp, safe_target] == Zone.GARDEN)
              & (target_type == CardType.ENTITY)
          )
      )
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & source_ok
      & state.ab_costs_applied
      & target_ok
  )
  do_skip = (
      do_action
      & (action_type == Act.NOOP)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & source_ok
      & state.ab_costs_applied
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 1)
  )
  finish = do_select | do_skip
  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = deal_effect_damage(stepped, opp, safe_target, 2, do_select)
  cleared = recompute_passives(_clear_context(stepped))

  trig_src = jnp.maximum(cleared.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(cleared.trig_owner[0].astype(jnp.int32), 0)
  azk01_059_trigger = (
      finish
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == TIMING_WHEN_TAKES_DAMAGE)
      & (cleared.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-059"])
      & ((cleared.once_per_turn_used[trig_owner, trig_src] & 1) == 0)
  )
  azk01_062_trigger = (
      finish
      & (cleared.redirect_count > 0)
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == TIMING_WHEN_TAKES_DAMAGE)
      & (cleared.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-062"])
  )
  stt04_009_trigger = (
      finish
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == TIMING_WHEN_TAKES_DAMAGE)
      & (cleared.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["STT04-009"])
      & cleared.last_dmg_from_effect[trig_owner, trig_src]
      & ((cleared.once_per_turn_used[trig_owner, trig_src] & 1) == 0)
  )
  popped, src2, owner2, timing2 = pop_effect(cleared)
  owner2_i32 = owner2.astype(jnp.int32)
  needs_transfer = popped.active_player != owner2.astype(jnp.int8)
  begun = popped._replace(
      ab_source=src2.astype(jnp.int8),
      ab_owner=owner2.astype(jnp.int8),
      ab_is_optional=jnp.bool_(False),
      ab_costs_applied=jnp.bool_(False),
      ab_saved_active=jnp.where(needs_transfer, popped.active_player, jnp.int8(-1)),
      ab_restores_active=needs_transfer,
      ab_cost_selected=jnp.int8(0),
      ab_cost_max=jnp.int8(0),
      ab_eff_selected=jnp.int8(0),
      ab_eff_min=jnp.int8(1),
      ab_eff_max=jnp.int8(1),
      ab_phase=jnp.int8(AbilityPhase.EFFECT_SELECTION),
      active_player=jnp.where(
          needs_transfer, owner2_i32.astype(jnp.int8), popped.active_player
      ),
  )
  begun_generic = resolve_triggered_effect(popped, src2, owner2, timing2)
  begun = jax.tree.map(
      lambda a, b: jnp.where(azk01_062_trigger | stt04_009_trigger, a, b),
      begun_generic,
      begun,
  )
  begin_trigger = azk01_059_trigger | azk01_062_trigger | stt04_009_trigger
  stepped = jax.tree.map(
      lambda a, b: jnp.where(begin_trigger, a, b), begun, cleared
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(finish, a, b), stepped, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_effect_azk01_062_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-062 redirect resolution."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_batch4 import _azk01_062_consume
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot
  from azuki_jax.engine.phases import (
      phase_gate,
      combat_resolve,
      defender_can_respond,
      transition_to_combat_resolve,
  )
  from azuki_jax.engine.triggers import (
      TIMING_WHEN_TAKES_DAMAGE,
      has_queued,
      pop_effect,
  )

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-062"])
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 1)
  )

  target_index = action[1].astype(jnp.int32)
  friendly = (target_index >= 0) & (target_index < GARDEN_SIZE)
  enemy = (target_index >= GARDEN_SIZE) & (target_index < 2 * GARDEN_SIZE)
  target_player = jnp.where(friendly, owner, opp)
  target_slot = jnp.where(friendly, target_index, target_index - GARDEN_SIZE)
  target = card_at_slot(state, target_player, Zone.GARDEN, target_slot)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[target_player, safe_target]
  target_ok = (
      (friendly | enemy)
      & (target >= 0)
      & (target_def >= 0)
      & (jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)] == CardType.ENTITY)
      & ((target_player != owner) | (safe_target != src))
  )
  do_select = (
      ~(did_reset | zero_legal)
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  do_skip = (
      ~(did_reset | zero_legal)
      & (action_type == Act.NOOP)
      & source_ok
  )
  finish = do_select | do_skip
  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, target_player.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped, has_redirect, src_player, src_inst, damage = _azk01_062_consume(stepped)
  resolved_player = jnp.where(do_select, target_player, owner)
  resolved_inst = jnp.where(do_select, safe_target, src)
  same_target = (resolved_player == owner) & (resolved_inst == src)
  stepped = deal_effect_damage(
      stepped,
      resolved_player,
      resolved_inst,
      damage,
      do=finish & has_redirect,
      allow_redirect=~same_target,
      src_player=src_player,
      src_inst=src_inst,
  )
  cleared = _clear_context(stepped)
  cleared = recompute_passives(cleared)

  trig_src = jnp.maximum(cleared.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(cleared.trig_owner[0].astype(jnp.int32), 0)
  redirect_slots = jnp.arange(8, dtype=jnp.int32)
  head_has_redirect = jnp.any(
      (redirect_slots < cleared.redirect_count.astype(jnp.int32))
      & (cleared.redirect_tgt_player.astype(jnp.int32) == trig_owner)
      & (cleared.redirect_tgt_inst.astype(jnp.int32) == trig_src)
  )
  azk01_062_fizzle = (
      finish
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (cleared.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-062"])
      & ~head_has_redirect
  )
  popped_fizzle, _, _, _ = pop_effect(cleared)
  cleared = jax.tree.map(
      lambda a, b: jnp.where(azk01_062_fizzle, a, b),
      popped_fizzle,
      cleared,
  )

  trig_src = jnp.maximum(cleared.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(cleared.trig_owner[0].astype(jnp.int32), 0)
  begin_next_azk01_062 = (
      finish
      & (cleared.redirect_count > 0)
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (cleared.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-062"])
  )
  popped, src2, owner2, timing2 = pop_effect(cleared)
  begun = resolve_triggered_effect(popped, src2, owner2, timing2)
  cleared = jax.tree.map(
      lambda a, b: jnp.where(begin_next_azk01_062, a, b),
      begun,
      cleared,
  )

  close_response = (
      (cleared.phase == Phase.RESPONSE_WINDOW)
      & (cleared.ab_phase == AbilityPhase.NONE)
      & ~has_queued(cleared)
      & ~defender_can_respond(cleared, cleared.active_player)
  )
  cleared = jax.lax.cond(
      close_response,
      lambda st: transition_to_combat_resolve(st, do=True),
      lambda st: st,
      cleared,
  )
  pending_main_combat = (
      (cleared.phase == Phase.MAIN)
      & (cleared.ab_phase == AbilityPhase.NONE)
      & (cleared.combat_attacker >= 0)
      & ~has_queued(cleared)
  )
  cleared = jax.lax.cond(
      pending_main_combat,
      lambda st: phase_gate(st),
      lambda st: st,
      cleared,
  )
  auto_combat = (cleared.phase == Phase.COMBAT_RESOLVE) & ~has_queued(cleared)
  cleared = jax.lax.cond(
      auto_combat,
      lambda st: combat_resolve(st, do=True),
      lambda st: st,
      cleared,
  )

  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), cleared, state)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )
  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type,
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )
  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations


def step_confirm_stt04_009_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast confirm for STT04-009 into effect selection."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.CONFIRMATION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT04-009"])
      & state.ab_is_optional
      & (state.ab_cost_max == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )
  do_confirm = do_action & (action[0] == Act.CONFIRM_ABILITY) & source_ok

  stepped = runtime.process_confirm(state, do_confirm)
  stepped = jax.tree.map(lambda a, b: jnp.where(do_confirm, a, b), stepped, state)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.CONFIRM_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )
  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations


def step_effect_stt04_009_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT04-009 effect: reflect capped effect damage to another target."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot, leader_instance
  from azuki_jax.engine.triggers import TIMING_WHEN_DESTROYED, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  target_index = jnp.clip(action[1].astype(jnp.int32), 0, 2 * GARDEN_SIZE + 1)
  friendly_garden = target_index < GARDEN_SIZE
  enemy_garden = (target_index >= GARDEN_SIZE) & (target_index < 2 * GARDEN_SIZE)
  own_leader = target_index == 2 * GARDEN_SIZE
  enemy_leader = target_index == (2 * GARDEN_SIZE + 1)
  target_player = jnp.where(
      friendly_garden | own_leader, owner, opp
  ).astype(jnp.int32)
  garden_slot = jnp.where(
      friendly_garden, target_index, target_index - GARDEN_SIZE
  )
  garden_inst = card_at_slot(state, target_player, Zone.GARDEN, garden_slot)
  leader_inst = jnp.where(
      own_leader,
      leader_instance(state, owner),
      leader_instance(state, opp),
  )
  target = jnp.where(friendly_garden | enemy_garden, garden_inst, leader_inst)
  safe_target = jnp.maximum(target, 0)
  safe_player = jnp.clip(target_player, 0, 1)
  target_def = state.def_id[safe_player, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_type = jnp.where(
      target_def >= 0, jnp.asarray(cards.TYPE)[safe_target_def], jnp.int8(-1)
  )
  target_is_leader = own_leader | enemy_leader
  target_is_garden = friendly_garden | enemy_garden
  same_source = (safe_player == owner) & (safe_target == src)
  valid_target = (
      (target >= 0)
      & ~same_source
      & (
          (target_is_leader & (target_type == CardType.LEADER))
          | (target_is_garden & (target_type == CardType.ENTITY))
      )
  )
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT04-009"])
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )
  damage = jnp.minimum(state.last_dmg_taken[owner, src].astype(jnp.int16), 2)
  do_select = (
      do_action
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & valid_target
      & (damage > 0)
  )
  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, safe_player.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = deal_effect_damage(stepped, safe_player, safe_target, damage, do_select)
  stepped = _clear_context(stepped)
  stepped = recompute_passives(stepped)
  trig_src = jnp.maximum(stepped.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(stepped.trig_owner[0].astype(jnp.int32), 0)
  begin_stt03_006 = (
      do_select
      & (stepped.trig_count > 0)
      & (stepped.trig_timing[0] == TIMING_WHEN_DESTROYED)
      & (stepped.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["STT03-006"])
  )
  popped, src2, owner2, timing2 = pop_effect(stepped)
  resolved = resolve_triggered_effect(popped, src2, owner2, timing2)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(begin_stt03_006, a, b), resolved, stepped
  )
  stepped = jax.tree.map(lambda a, b: jnp.where(do_select, a, b), stepped, state)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )
  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations



def step_play_azk01_024_confirm_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast AZK01-024 play setup into optional cost confirmation."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine import ikz
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)
  source_def = state.def_id[acting, safe]
  place = (
      do_action
      & (inst >= 0)
      & (source_def == cards.CODE_TO_ID["AZK01-024"])
  )

  cost = effective_play_cost(state, acting, safe)
  stepped = ikz.pay(state, acting, cost, use_token, do=place)
  pz = jnp.asarray(placement_zone, jnp.int8)
  stepped = _enter_board_slot(stepped, acting, safe, placement_zone, slot, place)
  stepped = recompute_passives(stepped)

  defs = stepped.def_id[acting]
  safe_defs = jnp.maximum(defs, 0)
  valid_cost_target = (
      (stepped.zone[acting] == Zone.GARDEN)
      & (defs >= 0)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
  )
  enter_confirm = place & jnp.any(valid_cost_target)
  is_garden = pz == jnp.int8(Zone.GARDEN)
  stepped = stepped._replace(
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((place & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((place & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          place.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(place, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(enter_confirm, safe.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(enter_confirm, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(enter_confirm, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(enter_confirm, True, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          enter_confirm, False, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          enter_confirm, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          enter_confirm, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          enter_confirm, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(enter_confirm, jnp.int8(1), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          enter_confirm, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(enter_confirm, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(enter_confirm, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_phase=jnp.where(
          enter_confirm, jnp.int8(AbilityPhase.CONFIRMATION), stepped.ab_phase
      ),
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )
  action_type = jnp.where(
      pz == jnp.int8(Zone.GARDEN),
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations


def step_confirm_azk01_024_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-024 optional confirmation into cost selection."""
  from azuki_jax import cards
  from azuki_jax.abilities.runtime import _clear_context

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_def = state.def_id[owner, src]
  source_ok = (
      (state.ab_phase == AbilityPhase.CONFIRMATION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_def == cards.CODE_TO_ID["AZK01-024"])
      & state.ab_is_optional
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 1)
  )
  do_confirm = (
      do_action
      & (action_type == Act.CONFIRM_ABILITY)
      & source_ok
  )
  do_decline = do_action & (action_type == Act.NOOP) & source_ok

  stepped = state._replace(
      ab_phase=jnp.where(
          do_confirm, jnp.int8(AbilityPhase.COST_SELECTION), state.ab_phase
      ),
      ab_costs_applied=jnp.where(do_confirm, False, state.ab_costs_applied),
      ab_cost_selected=jnp.where(do_confirm, jnp.int8(0), state.ab_cost_selected),
      ab_cost_max=jnp.where(do_confirm, jnp.int8(1), state.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
  )
  declined = _clear_context(state)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_decline, a, b), declined, stepped
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )
  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )
  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations


def step_select_cost_azk01_024_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-024 cost bounce, then move eligible hand entity to selection."""
  from azuki_jax import cards
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.abilities.cards_impl import return_to_hand
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_def = state.def_id[owner, src]
  source_ok = (
      (state.ab_phase == AbilityPhase.COST_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_def == cards.CODE_TO_ID["AZK01-024"])
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 1)
  )

  target = card_at_slot(state, owner, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  target_ok = (
      (target >= 0)
      & (jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)] == CardType.ENTITY)
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_COST_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_cost_targets=state.ab_cost_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_cost_targets[0])
      ),
      ab_cost_target_players=state.ab_cost_target_players.at[0].set(
          jnp.where(do_select, owner.astype(jnp.int8), state.ab_cost_target_players[0])
      ),
      ab_cost_selected=jnp.where(
          do_select, jnp.int8(1), state.ab_cost_selected
      ),
  )
  stepped = return_to_hand(stepped, owner, safe_target, do_select)
  defs = stepped.def_id[owner]
  safe_defs = jnp.maximum(defs, 0)
  idx = jnp.arange(stepped.zone.shape[1])
  eligible = (
      (stepped.zone[owner] == Zone.HAND)
      & (defs >= 0)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[safe_defs].astype(jnp.int32) <= 2)
      & ~(idx == safe_target)
  )
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(do_select, True, stepped.ab_costs_applied)
  )
  stepped = sel_mod.move_matching_zone_to_selection(
      stepped, Zone.HAND, eligible, 1, do_select
  )
  exhausted = do_select & (stepped.ab_sel_count == 0)
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(exhausted, a, b), cleared, stepped
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )
  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )
  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_COST_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )
  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations


def step_select_azk01_024_place_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast AZK01-024 selected entity placement to garden/alley."""
  from azuki_jax import cards
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.apply import _enter_board_slot

  placement_zone = int(placement_zone)
  action_id = (
      Act.SELECT_TO_GARDEN
      if placement_zone == int(Zone.GARDEN)
      else Act.SELECT_TO_ALLEY
  )

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  sel_idx = action[1].astype(jnp.int32)
  slot = action[2].astype(jnp.int32)
  idx = jnp.clip(sel_idx, 0, state.ab_sel_cards.shape[0] - 1)
  inst = state.ab_sel_cards[idx]
  target = jnp.maximum(inst.astype(jnp.int32), 0)
  source_def = state.def_id[owner, src]
  target_def = state.def_id[owner, target]
  target_ok = (
      (target_def >= 0)
      & (jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[jnp.maximum(target_def, 0)]
      & (jnp.asarray(cards.IKZ_COST)[jnp.maximum(target_def, 0)].astype(jnp.int32) <= 2)
  )
  source_ok = (
      (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_def == cards.CODE_TO_ID["AZK01-024"])
      & state.ab_costs_applied
      & (state.ab_sel_pick_max == 1)
      & (state.ab_sel_picked_count == 0)
  )
  pick_ok = (
      (action[0] == jnp.asarray(action_id, jnp.int32))
      & (sel_idx >= 0)
      & (sel_idx < state.ab_sel_count.astype(jnp.int32))
      & (inst >= 0)
      & (state.zone[owner, target] == Zone.SELECTION)
      & target_ok
  )
  do_place = do_action & source_ok & pick_ok
  pz = jnp.asarray(placement_zone, jnp.int8)
  is_garden = pz == jnp.int8(Zone.GARDEN)
  stepped = _enter_board_slot(state, owner, target, placement_zone, slot, do_place)
  stepped = stepped._replace(
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          owner
      ].add((do_place & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          owner
      ].add((do_place & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[owner].add(
          do_place.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[owner].set(
          jnp.where(do_place, 0, stepped.next_play_cost_reduction[owner])
      ),
  )
  stepped = _apply_simple_implemented_play_trigger(stepped, owner, target, do_place)
  stepped = sel_mod.return_remaining_to_hand(stepped, do_place)
  cleared = _clear_context(stepped)

  defs = cleared.def_id[owner]
  safe_defs = jnp.maximum(defs, 0)
  hand_available = jnp.any((cleared.zone[owner] == Zone.HAND) & (defs >= 0))
  any_bounce_target = jnp.zeros((), jnp.bool_)
  for player in (0, 1):
    target_defs = cleared.def_id[player]
    safe_target_defs = jnp.maximum(target_defs, 0)
    any_bounce_target = any_bounce_target | jnp.any(
        (cleared.zone[player] == Zone.GARDEN)
        & (target_defs >= 0)
        & (jnp.asarray(cards.TYPE)[safe_target_defs] == CardType.ENTITY)
        & jnp.asarray(cards.HAS_IKZ_COST)[safe_target_defs]
        & (jnp.asarray(cards.IKZ_COST)[safe_target_defs].astype(jnp.int32) <= 2)
    )
  target_is_azk01_022 = do_place & (
      target_def == cards.CODE_TO_ID["AZK01-022"]
  )
  enter_confirm = target_is_azk01_022 & hand_available & any_bounce_target
  azk01_022 = cleared._replace(
      ab_source=jnp.where(enter_confirm, target.astype(jnp.int8), cleared.ab_source),
      ab_owner=jnp.where(enter_confirm, owner.astype(jnp.int8), cleared.ab_owner),
      ab_slot=jnp.where(enter_confirm, jnp.int8(0), cleared.ab_slot),
      ab_is_optional=jnp.where(enter_confirm, True, cleared.ab_is_optional),
      ab_costs_applied=jnp.where(enter_confirm, False, cleared.ab_costs_applied),
      ab_saved_active=jnp.where(enter_confirm, jnp.int8(-1), cleared.ab_saved_active),
      ab_restores_active=jnp.where(enter_confirm, False, cleared.ab_restores_active),
      ab_cost_selected=jnp.where(enter_confirm, jnp.int8(0), cleared.ab_cost_selected),
      ab_cost_max=jnp.where(enter_confirm, jnp.int8(1), cleared.ab_cost_max),
      ab_cost_targets=jnp.where(
          enter_confirm,
          jnp.full_like(cleared.ab_cost_targets, -1),
          cleared.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          enter_confirm,
          jnp.full_like(cleared.ab_cost_target_players, -1),
          cleared.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(enter_confirm, jnp.int8(0), cleared.ab_eff_selected),
      ab_eff_min=jnp.where(enter_confirm, jnp.int8(1), cleared.ab_eff_min),
      ab_eff_max=jnp.where(enter_confirm, jnp.int8(1), cleared.ab_eff_max),
      ab_eff_targets=jnp.where(
          enter_confirm,
          jnp.full_like(cleared.ab_eff_targets, -1),
          cleared.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          enter_confirm,
          jnp.full_like(cleared.ab_eff_target_players, -1),
          cleared.ab_eff_target_players,
      ),
      ab_phase=jnp.where(
          enter_confirm, jnp.int8(AbilityPhase.CONFIRMATION), cleared.ab_phase
      ),
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(enter_confirm, a, b), azk01_022, cleared
  )
  target_is_stt02_003 = do_place & (
      target_def == cards.CODE_TO_ID["STT02-003"]
  )
  stt02_base = stepped._replace(
      ab_source=jnp.where(target_is_stt02_003, target.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(target_is_stt02_003, owner.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(target_is_stt02_003, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(target_is_stt02_003, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(target_is_stt02_003, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(target_is_stt02_003, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(target_is_stt02_003, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(target_is_stt02_003, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(target_is_stt02_003, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          target_is_stt02_003,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          target_is_stt02_003,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(target_is_stt02_003, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(target_is_stt02_003, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(target_is_stt02_003, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          target_is_stt02_003,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          target_is_stt02_003,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )
  stt02_reveal = sel_mod.reveal_top_into_selection(stt02_base, 5, 1, target_is_stt02_003)
  stt02_revealed = stt02_reveal.ab_sel_count > 0
  watercrafting = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Watercrafting")], jnp.bool_
  )
  stt02_sel_cards = stt02_reveal.ab_sel_cards
  stt02_sel_safe = jnp.maximum(stt02_sel_cards.astype(jnp.int32), 0)
  stt02_sel_defs = stt02_reveal.def_id[owner, stt02_sel_safe]
  stt02_sel_valid = stt02_sel_cards >= 0
  stt02_matching = jnp.any(
      stt02_sel_valid & watercrafting[jnp.maximum(stt02_sel_defs, 0)]
  )
  stt02_next_phase = jnp.where(
      stt02_matching,
      jnp.int8(AbilityPhase.SELECTION_PICK),
      jnp.int8(AbilityPhase.BOTTOM_DECK),
  )
  stt02_reveal = stt02_reveal._replace(
      ab_phase=jnp.where(
          target_is_stt02_003 & stt02_revealed,
          stt02_next_phase,
          stt02_reveal.ab_phase,
      )
  )
  stt02_done = _clear_context(stt02_reveal)
  stt02_reveal = jax.tree.map(
      lambda a, b: jnp.where(target_is_stt02_003 & ~stt02_revealed, a, b),
      stt02_done,
      stt02_reveal,
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(target_is_stt02_003, a, b),
      stt02_reveal,
      stepped,
  )
  stepped = recompute_passives(stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(do_place, a, b), stepped, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )
  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action[0].astype(jnp.int32), jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )
  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations

def step_play_stt02_009_confirm_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast STT02-009 play setup into optional confirmation."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine import ikz
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)
  source_def = state.def_id[acting, safe]
  place = (
      do_action
      & (inst >= 0)
      & (source_def == cards.CODE_TO_ID["STT02-009"])
  )

  cost = effective_play_cost(state, acting, safe)
  stepped = ikz.pay(state, acting, cost, use_token, do=place)
  pz = jnp.asarray(placement_zone, jnp.int8)
  stepped = _enter_board_slot(stepped, acting, safe, placement_zone, slot, place)
  stepped = recompute_passives(stepped)

  is_garden = pz == jnp.int8(Zone.GARDEN)
  def_ids = stepped.def_id[acting]
  safe_defs = jnp.maximum(def_ids, 0)
  valid_cost_target = (
      (stepped.zone[acting] == Zone.GARDEN)
      & (def_ids >= 0)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[safe_defs].astype(jnp.int32) >= 2)
  )
  enter_confirm = place & jnp.any(valid_cost_target)
  stepped = stepped._replace(
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((place & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((place & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          place.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(place, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(enter_confirm, safe.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(enter_confirm, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(enter_confirm, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(enter_confirm, True, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          enter_confirm, False, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          enter_confirm, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          enter_confirm, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          enter_confirm, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(enter_confirm, jnp.int8(1), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          enter_confirm, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(enter_confirm, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(enter_confirm, jnp.int8(1), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_phase=jnp.where(
          enter_confirm, jnp.int8(AbilityPhase.CONFIRMATION), stepped.ab_phase
      ),
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      pz == jnp.int8(Zone.GARDEN),
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_confirm_stt02_009_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-009 optional confirmation into cost selection."""
  from azuki_jax import cards

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.CONFIRMATION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT02-009"])
      & state.ab_is_optional
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 1)
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 1)
  )
  do_confirm = (
      do_action
      & (action[0] == Act.CONFIRM_ABILITY)
      & source_ok
  )

  stepped = state._replace(
      ab_phase=jnp.where(
          do_confirm, jnp.int8(AbilityPhase.COST_SELECTION), state.ab_phase
      ),
      ab_costs_applied=jnp.where(do_confirm, False, state.ab_costs_applied),
      ab_cost_selected=jnp.where(do_confirm, jnp.int8(0), state.ab_cost_selected),
      ab_cost_max=jnp.where(do_confirm, jnp.int8(1), state.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_confirm, jnp.int8(0), state.ab_eff_selected),
      ab_eff_min=jnp.where(do_confirm, jnp.int8(0), state.ab_eff_min),
      ab_eff_max=jnp.where(do_confirm, jnp.int8(1), state.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_eff_targets, -1),
          state.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_eff_target_players, -1),
          state.ab_eff_target_players,
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.CONFIRM_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_cost_stt02_009_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-009 cost bounce into optional effect selection."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import return_to_hand
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.COST_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT02-009"])
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 1)
  )
  target = card_at_slot(state, owner, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_ok = (
      (target >= 0)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_target_def]
      & (jnp.asarray(cards.IKZ_COST)[safe_target_def].astype(jnp.int32) >= 2)
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_COST_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_cost_targets=state.ab_cost_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_cost_targets[0])
      ),
      ab_cost_target_players=state.ab_cost_target_players.at[0].set(
          jnp.where(do_select, owner.astype(jnp.int8), state.ab_cost_target_players[0])
      ),
      ab_cost_selected=jnp.where(
          do_select, jnp.int8(1), state.ab_cost_selected
      ),
  )
  stepped = return_to_hand(stepped, owner, safe_target, do_select)
  stepped = recompute_passives(stepped)

  opp_defs = stepped.def_id[opp]
  opp_safe_defs = jnp.maximum(opp_defs, 0)
  effect_targets = (
      (stepped.zone[opp] == Zone.GARDEN)
      & (opp_defs >= 0)
      & (jnp.asarray(cards.TYPE)[opp_safe_defs] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[opp_safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[opp_safe_defs].astype(jnp.int32) <= 2)
  )
  remaining = jnp.sum(effect_targets, dtype=jnp.int32)
  to_effect = do_select & (remaining > 0)
  exhausted = do_select & (remaining == 0)
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(do_select, True, stepped.ab_costs_applied),
      ab_phase=jnp.where(
          to_effect, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
      ab_eff_min=jnp.where(to_effect, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(
          to_effect, jnp.minimum(remaining, 1).astype(jnp.int8), stepped.ab_eff_max
      ),
  )
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(exhausted, a, b), cleared, stepped
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_COST_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_stt02_009_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-009 effect target or skip, then clear context."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import return_to_hand
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot
  from azuki_jax.engine.triggers import TIMING_WHEN_RETURNED_TO_HAND, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT02-009"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 1)
  )
  target = card_at_slot(state, opp, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_ok = (
      (target >= 0)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_target_def]
      & (jnp.asarray(cards.IKZ_COST)[safe_target_def].astype(jnp.int32) <= 2)
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  do_skip = do_action & (action_type == Act.NOOP) & source_ok

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = return_to_hand(stepped, opp, safe_target, do_select)
  finish = do_select | do_skip
  cleared = _clear_context(stepped)
  cleared = recompute_passives(cleared)
  popped, trig_src, trig_owner, trig_timing = pop_effect(cleared)
  trig_src_i = jnp.maximum(trig_src.astype(jnp.int32), 0)
  trig_owner_i = jnp.maximum(trig_owner.astype(jnp.int32), 0)
  head_stt02_010 = (
      finish
      & (cleared.trig_count > 0)
      & (trig_timing == jnp.int8(TIMING_WHEN_RETURNED_TO_HAND))
      & (cleared.def_id[trig_owner_i, trig_src_i] == cards.CODE_TO_ID["STT02-010"])
  )
  first_stt02_010_can_begin = (
      head_stt02_010
      & (cleared.zone[trig_owner_i, trig_src_i] == Zone.GARDEN)
  )
  drop_invalid_first = head_stt02_010 & ~first_stt02_010_can_begin
  popped2, trig_src2, trig_owner2, trig_timing2 = pop_effect(popped)
  trig_src2_i = jnp.maximum(trig_src2.astype(jnp.int32), 0)
  trig_owner2_i = jnp.maximum(trig_owner2.astype(jnp.int32), 0)
  begin_second_stt02_010 = (
      drop_invalid_first
      & (popped.trig_count > 0)
      & (trig_timing2 == jnp.int8(TIMING_WHEN_RETURNED_TO_HAND))
      & (
          popped.def_id[trig_owner2_i, trig_src2_i]
          == cards.CODE_TO_ID["STT02-010"]
      )
  )
  begin_stt02_010 = first_stt02_010_can_begin | begin_second_stt02_010
  begin_base = jax.tree.map(
      lambda a, b: jnp.where(begin_second_stt02_010, a, b), popped2, popped
  )
  begin_src = jnp.where(begin_second_stt02_010, trig_src2, trig_src)
  begin_owner = jnp.where(begin_second_stt02_010, trig_owner2, trig_owner)
  begin_timing = jnp.where(begin_second_stt02_010, trig_timing2, trig_timing)
  begun = resolve_triggered_effect(
      begin_base, begin_src, begin_owner, begin_timing
  )
  dropped_invalid = jax.tree.map(
      lambda a, b: jnp.where(drop_invalid_first, a, b), popped, cleared
  )
  cleared = jax.tree.map(
      lambda a, b: jnp.where(begin_stt02_010, a, b), begun, dropped_invalid
  )
  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), cleared, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_play_azk01_022_confirm_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast AZK01-022 play setup into optional discard/bounce confirmation."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine import ikz
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import hand_instance
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  slot = action[2]
  use_token = action[3] != 0
  inst = hand_instance(state, acting, hand_index)
  safe = jnp.maximum(inst, 0)
  source_def = state.def_id[acting, safe]
  place = (
      do_action
      & (inst >= 0)
      & (source_def == cards.CODE_TO_ID["AZK01-022"])
  )

  cost = effective_play_cost(state, acting, safe)
  stepped = ikz.pay(state, acting, cost, use_token, do=place)
  pz = jnp.asarray(placement_zone, jnp.int8)
  stepped = _enter_board_slot(stepped, acting, safe, placement_zone, slot, place)
  stepped = recompute_passives(stepped)

  owner_defs = stepped.def_id[acting]
  safe_owner_defs = jnp.maximum(owner_defs, 0)
  hand_available = jnp.any(
      (stepped.zone[acting] == Zone.HAND) & (owner_defs >= 0)
  )
  any_bounce_target = jnp.zeros((), jnp.bool_)
  for player in (0, 1):
    target_defs = stepped.def_id[player]
    safe_target_defs = jnp.maximum(target_defs, 0)
    any_bounce_target = any_bounce_target | jnp.any(
        (stepped.zone[player] == Zone.GARDEN)
        & (target_defs >= 0)
        & (jnp.asarray(cards.TYPE)[safe_target_defs] == CardType.ENTITY)
        & jnp.asarray(cards.HAS_IKZ_COST)[safe_target_defs]
        & (jnp.asarray(cards.IKZ_COST)[safe_target_defs].astype(jnp.int32) <= 2)
    )
  enter_confirm = place & hand_available & any_bounce_target

  is_garden = pz == jnp.int8(Zone.GARDEN)
  stepped = stepped._replace(
      entities_played_garden_turn=stepped.entities_played_garden_turn.at[
          acting
      ].add((place & is_garden).astype(jnp.uint8)),
      entities_played_alley_turn=stepped.entities_played_alley_turn.at[
          acting
      ].add((place & ~is_garden).astype(jnp.uint8)),
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          place.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(place, 0, stepped.next_play_cost_reduction[acting])
      ),
      ab_source=jnp.where(enter_confirm, safe.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(enter_confirm, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(enter_confirm, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(enter_confirm, True, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          enter_confirm, False, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          enter_confirm, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          enter_confirm, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          enter_confirm, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(enter_confirm, jnp.int8(1), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          enter_confirm, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(enter_confirm, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(enter_confirm, jnp.int8(1), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          enter_confirm,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_phase=jnp.where(
          enter_confirm, jnp.int8(AbilityPhase.CONFIRMATION), stepped.ab_phase
      ),
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      pz == jnp.int8(Zone.GARDEN),
      jnp.asarray(Act.PLAY_ENTITY_TO_GARDEN, jnp.int32),
      jnp.asarray(Act.PLAY_ENTITY_TO_ALLEY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_confirm_azk01_022_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-022 optional confirmation into hand discard selection."""
  from azuki_jax import cards

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.CONFIRMATION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-022"])
      & state.ab_is_optional
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 1)
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )
  do_confirm = (
      do_action
      & (action[0] == Act.CONFIRM_ABILITY)
      & source_ok
  )

  stepped = state._replace(
      ab_phase=jnp.where(
          do_confirm, jnp.int8(AbilityPhase.COST_SELECTION), state.ab_phase
      ),
      ab_costs_applied=jnp.where(do_confirm, False, state.ab_costs_applied),
      ab_cost_selected=jnp.where(do_confirm, jnp.int8(0), state.ab_cost_selected),
      ab_cost_max=jnp.where(do_confirm, jnp.int8(1), state.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_confirm, jnp.int8(0), state.ab_eff_selected),
      ab_eff_min=jnp.where(do_confirm, jnp.int8(1), state.ab_eff_min),
      ab_eff_max=jnp.where(do_confirm, jnp.int8(1), state.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_eff_targets, -1),
          state.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_eff_target_players, -1),
          state.ab_eff_target_players,
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.CONFIRM_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_cost_azk01_022_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-022 cost: discard selected hand card, then select bounce."""
  from azuki_jax import cards
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import discard, hand_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  hand_idx = action[1].astype(jnp.int32)
  target = hand_instance(state, owner, hand_idx)
  safe_target = jnp.maximum(target, 0)
  source_def = state.def_id[owner, src]
  target_ok = (
      (target >= 0)
      & (target != src)
      & (state.zone[owner, safe_target] == Zone.HAND)
  )
  ok = (
      do_action
      & (action[0] == Act.SELECT_COST_TARGET)
      & (state.ab_phase == AbilityPhase.COST_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_def == cards.CODE_TO_ID["AZK01-022"])
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 1)
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
      & target_ok
  )

  stepped = state._replace(
      ab_cost_targets=state.ab_cost_targets.at[0].set(
          jnp.where(ok, safe_target.astype(jnp.int8), state.ab_cost_targets[0])
      ),
      ab_cost_target_players=state.ab_cost_target_players.at[0].set(
          jnp.where(ok, owner.astype(jnp.int8), state.ab_cost_target_players[0])
      ),
      ab_cost_selected=jnp.where(ok, jnp.int8(1), state.ab_cost_selected),
  )
  stepped = discard(stepped, owner, safe_target, do=ok)

  any_bounce_target = jnp.zeros((), jnp.bool_)
  for player in (0, 1):
    target_defs = stepped.def_id[player]
    safe_target_defs = jnp.maximum(target_defs, 0)
    any_bounce_target = any_bounce_target | jnp.any(
        (stepped.zone[player] == Zone.GARDEN)
        & (target_defs >= 0)
        & (jnp.asarray(cards.TYPE)[safe_target_defs] == CardType.ENTITY)
        & jnp.asarray(cards.HAS_IKZ_COST)[safe_target_defs]
        & (jnp.asarray(cards.IKZ_COST)[safe_target_defs].astype(jnp.int32) <= 2)
    )
  to_effect = ok & any_bounce_target
  exhausted = ok & ~any_bounce_target
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(ok, True, stepped.ab_costs_applied),
      ab_phase=jnp.where(
          to_effect, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
      ab_eff_min=jnp.where(to_effect, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(to_effect, jnp.int8(1), stepped.ab_eff_max),
  )
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(exhausted, a, b), cleared, stepped
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_COST_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_022_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-022 effect: return one low-cost any-garden entity."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import return_to_hand
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-022"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )

  target_index = action[1].astype(jnp.int32)
  target_player = jnp.where(target_index < GARDEN_SIZE, owner, opp)
  target_slot = jnp.where(
      target_index < GARDEN_SIZE, target_index, target_index - GARDEN_SIZE
  )
  target = card_at_slot(state, target_player, Zone.GARDEN, target_slot)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[target_player, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_ok = (
      (target_index >= 0)
      & (target_index < (2 * GARDEN_SIZE))
      & (target >= 0)
      & (state.zone[target_player, safe_target] == Zone.GARDEN)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_target_def]
      & (jnp.asarray(cards.IKZ_COST)[safe_target_def].astype(jnp.int32) <= 2)
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(
              do_select,
              target_player.astype(jnp.int8),
              state.ab_eff_target_players[0],
          )
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = return_to_hand(stepped, target_player, safe_target, do_select)
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(do_select, a, b), cleared, state)
  stepped = recompute_passives(stepped)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_cost_azk01_032_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-032 cost return into optional enemy return selection."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import return_to_hand
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.COST_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-032"])
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 1)
  )
  target = card_at_slot(state, owner, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_ok = (
      (target >= 0)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_target_def]
      & (jnp.asarray(cards.IKZ_COST)[safe_target_def].astype(jnp.int32) >= 2)
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_COST_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_cost_targets=state.ab_cost_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_cost_targets[0])
      ),
      ab_cost_target_players=state.ab_cost_target_players.at[0].set(
          jnp.where(do_select, owner.astype(jnp.int8), state.ab_cost_target_players[0])
      ),
      ab_cost_selected=jnp.where(
          do_select, jnp.int8(1), state.ab_cost_selected
      ),
  )
  stepped = return_to_hand(stepped, owner, safe_target, do_select)
  stepped = recompute_passives(stepped)

  opp_defs = stepped.def_id[opp]
  opp_safe_defs = jnp.maximum(opp_defs, 0)
  effect_targets = (
      (stepped.zone[opp] == Zone.GARDEN)
      & (opp_defs >= 0)
      & (jnp.asarray(cards.TYPE)[opp_safe_defs] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[opp_safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[opp_safe_defs].astype(jnp.int32) <= 4)
  )
  remaining = jnp.sum(effect_targets, dtype=jnp.int32)
  to_effect = do_select & (remaining > 0)
  exhausted = do_select & (remaining == 0)
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(do_select, True, stepped.ab_costs_applied),
      ab_phase=jnp.where(
          to_effect, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
      ab_eff_min=jnp.where(to_effect, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(
          to_effect, jnp.minimum(remaining, 1).astype(jnp.int8), stepped.ab_eff_max
      ),
  )
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(exhausted, a, b), cleared, stepped
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_COST_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_032_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-032 optional enemy return selection, then clear context."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import return_to_hand
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-032"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 1)
  )
  target = card_at_slot(state, opp, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_ok = (
      (target >= 0)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_target_def]
      & (jnp.asarray(cards.IKZ_COST)[safe_target_def].astype(jnp.int32) <= 4)
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  do_skip = do_action & (action_type == Act.NOOP) & source_ok

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = return_to_hand(stepped, opp, safe_target, do_select)
  stepped = recompute_passives(stepped)
  cleared = _clear_context(stepped)
  finish = do_select | do_skip
  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), cleared, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_040_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-040 effect selection, then resolve clean pending combat."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import leader_instance
  from azuki_jax.engine.phases import combat_resolve
  from azuki_jax.engine.triggers import TIMING_WHEN_TAKES_DAMAGE, has_queued, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.phase == Phase.COMBAT_RESOLVE)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-040"])
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 1)
  )
  do_skip = do_action & (action_type == Act.NOOP) & source_ok
  target_index = action[1].astype(jnp.int32)
  target_player = jnp.where(target_index == 0, owner, (owner + 1) % 2)
  target = leader_instance(state, target_player)
  safe_target = jnp.maximum(target, 0)
  target_ok = (target_index >= 0) & (target_index <= 1) & (target >= 0)
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  selected = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(
              do_select,
              target_player.astype(jnp.int8),
              state.ab_eff_target_players[0],
          )
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  selected = deal_effect_damage(selected, target_player, safe_target, 1, do_select)
  before_clear = jax.tree.map(
      lambda a, b: jnp.where(do_select, a, b), selected, state
  )
  needs_clear = do_skip | do_select
  cleared = _clear_context(before_clear)
  auto_combat = (cleared.phase == Phase.COMBAT_RESOLVE) & ~has_queued(cleared)
  cleared = jax.lax.cond(
      auto_combat,
      lambda st: combat_resolve(st, do=True),
      lambda st: st,
      cleared,
  )
  recomputed = recompute_passives(cleared)
  cleared = jax.tree.map(
      lambda a, b: jnp.where(auto_combat, a, b), recomputed, cleared
  )
  head_src = jnp.maximum(cleared.trig_source[0].astype(jnp.int32), 0)
  head_owner = jnp.maximum(cleared.trig_owner[0].astype(jnp.int32), 0)
  azk01_062_combat_fizzle = (
      auto_combat
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (cleared.def_id[head_owner, head_src] == cards.CODE_TO_ID["AZK01-062"])
      & ~cleared.last_dmg_from_effect[head_owner, head_src]
  )
  popped, _, _, _ = pop_effect(cleared)
  cleared = jax.tree.map(
      lambda a, b: jnp.where(azk01_062_combat_fizzle, a, b),
      popped,
      cleared,
  )
  stepped = jax.tree.map(lambda a, b: jnp.where(needs_clear, a, b), cleared, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_gate_portal_simple_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Narrow fast path for gate portal into garden, including full-slot replacement."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.cards_batch4 import _azk01_120_ocp
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import (
      card_at_slot,
      gate_instance,
  )
  from azuki_jax.engine.triggers import queue_enter_garden

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  state = recompute_passives(state)
  alley_index = action[1]
  garden_index = action[2]
  inst = card_at_slot(state, acting, Zone.ALLEY, alley_index)
  safe = jnp.maximum(inst, 0)
  gate = gate_instance(state, acting)
  safe_gate = jnp.maximum(gate, 0)
  place = do_action & (inst >= 0)

  stepped = _enter_board_slot(state, acting, safe, Zone.GARDEN, garden_index, place)
  stepped = stepped._replace(
      tapped=stepped.tapped.at[acting, safe_gate].set(
          jnp.where(place & (gate >= 0), True, stepped.tapped[acting, safe_gate])
      ),
  )
  stepped = queue_enter_garden(stepped, acting, safe, do=place)

  gate_def = stepped.def_id[acting, safe_gate]
  portaled_def = stepped.def_id[acting, safe]
  gate_power = jnp.where(
      portaled_def >= 0,
      jnp.asarray(cards.GATE_POINTS)[jnp.maximum(portaled_def, 0)],
      0,
  ).astype(jnp.int32)

  enter_stt02_002 = (
      place
      & (gate >= 0)
      & (gate_def == cards.CODE_TO_ID["STT02-002"])
      & (gate_power > 0)
  )
  ikz_tapped = (stepped.zone[acting] == Zone.IKZ_AREA) & stepped.tapped[acting]
  ikz_zpos = stepped.zpos[acting].astype(jnp.int32)
  ikz_rank = jnp.sum(
      (ikz_zpos[None, :] < ikz_zpos[:, None]) & ikz_tapped[None, :],
      axis=1,
  )
  untap_ikz = ikz_tapped & (ikz_rank < gate_power)
  stepped = stepped._replace(
      tapped=stepped.tapped.at[acting].set(
          jnp.where(enter_stt02_002 & untap_ikz, False, stepped.tapped[acting])
      )
  )

  row_defs = stepped.def_id[acting]
  safe_defs = jnp.maximum(row_defs, 0)
  rushfire_row = (
      (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[safe_defs].astype(jnp.int32) <= gate_power)
      & (gate_power > 0)
  )
  rushfire_count = jnp.sum(
      (stepped.zone[acting] == Zone.HAND) & rushfire_row,
      dtype=jnp.int32,
  )
  enter_azk01_122 = (
      place
      & (gate >= 0)
      & (gate_def == cards.CODE_TO_ID["AZK01-122"])
      & (rushfire_count > 0)
  )
  stepped = stepped._replace(
      ab_source=jnp.where(
          enter_azk01_122, safe_gate.astype(jnp.int8), stepped.ab_source
      ),
      ab_owner=jnp.where(
          enter_azk01_122, acting.astype(jnp.int8), stepped.ab_owner
      ),
      ab_slot=jnp.where(enter_azk01_122, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(enter_azk01_122, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          enter_azk01_122, True, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          enter_azk01_122, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          enter_azk01_122, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          enter_azk01_122, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(
          enter_azk01_122, jnp.int8(0), stepped.ab_cost_max
      ),
      ab_eff_selected=jnp.where(
          enter_azk01_122, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(enter_azk01_122, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(enter_azk01_122, jnp.int8(0), stepped.ab_eff_max),
      ab_scratch=stepped.ab_scratch.at[0]
      .set(jnp.where(enter_azk01_122, safe.astype(jnp.int16), stepped.ab_scratch[0]))
      .at[1]
      .set(
          jnp.where(
              enter_azk01_122,
              garden_index.astype(jnp.int16),
              stepped.ab_scratch[1],
          )
      )
      .at[2]
      .set(jnp.where(enter_azk01_122, jnp.int16(1), stepped.ab_scratch[2])),
  )
  stepped = sel_mod.move_matching_zone_to_selection(
      stepped, Zone.HAND, rushfire_row, 1, do=enter_azk01_122
  )

  echoed_row = (
      (jnp.asarray(cards.TYPE)[safe_defs] == CardType.SPELL)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[safe_defs].astype(jnp.int32) <= gate_power)
  )
  echoed_count = jnp.sum(
      (stepped.zone[acting] == Zone.DISCARD) & echoed_row,
      dtype=jnp.int32,
  )
  enter_azk01_126 = (
      place
      & (gate >= 0)
      & (gate_def == cards.CODE_TO_ID["AZK01-126"])
      & (gate_power > 0)
      & (echoed_count > 0)
  )
  stepped = stepped._replace(
      ab_source=jnp.where(
          enter_azk01_126, safe_gate.astype(jnp.int8), stepped.ab_source
      ),
      ab_owner=jnp.where(
          enter_azk01_126, acting.astype(jnp.int8), stepped.ab_owner
      ),
      ab_slot=jnp.where(enter_azk01_126, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(enter_azk01_126, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          enter_azk01_126, True, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          enter_azk01_126, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          enter_azk01_126, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          enter_azk01_126, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(
          enter_azk01_126, jnp.int8(0), stepped.ab_cost_max
      ),
      ab_eff_selected=jnp.where(
          enter_azk01_126, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(enter_azk01_126, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(enter_azk01_126, jnp.int8(0), stepped.ab_eff_max),
      ab_scratch=stepped.ab_scratch.at[0]
      .set(jnp.where(enter_azk01_126, safe.astype(jnp.int16), stepped.ab_scratch[0]))
      .at[1]
      .set(
          jnp.where(
              enter_azk01_126,
              garden_index.astype(jnp.int16),
              stepped.ab_scratch[1],
          )
      )
      .at[2]
      .set(jnp.where(enter_azk01_126, jnp.int16(1), stepped.ab_scratch[2])),
  )
  stepped = sel_mod.move_matching_zone_to_selection(
      stepped, Zone.DISCARD, echoed_row, 1, do=enter_azk01_126
  )

  attached_120 = (
      (stepped.zone[acting] == Zone.ATTACHED)
      & (stepped.attached_to[acting] >= 0)
  )
  host_120 = jnp.clip(
      stepped.attached_to[acting].astype(jnp.int32), 0, stepped.zone.shape[1] - 1
  )
  host_zone_120 = stepped.zone[acting][host_120]
  host_is_leader_120 = host_zone_120 == Zone.LEADER
  host_ok_120 = host_is_leader_120 | (host_zone_120 == Zone.GARDEN)
  garden_count_120 = jnp.sum(stepped.zone[acting] == Zone.GARDEN, dtype=jnp.int32)
  other_host_120 = jnp.where(host_is_leader_120, garden_count_120 >= 1, True)
  reequip_row = (
      attached_120
      & host_ok_120
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.WEAPON)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[safe_defs].astype(jnp.int32) <= gate_power)
      & (gate_power > 0)
      & other_host_120
  )
  enter_azk01_120 = (
      place
      & (gate >= 0)
      & (gate_def == cards.CODE_TO_ID["AZK01-120"])
      & jnp.any(reequip_row)
  )
  pre_120 = stepped._replace(
      ab_source=jnp.where(enter_azk01_120, safe_gate.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(enter_azk01_120, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(enter_azk01_120, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(enter_azk01_120, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          enter_azk01_120, True, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          enter_azk01_120, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          enter_azk01_120, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          enter_azk01_120, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(
          enter_azk01_120, jnp.int8(0), stepped.ab_cost_max
      ),
      ab_eff_selected=jnp.where(
          enter_azk01_120, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(enter_azk01_120, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(enter_azk01_120, jnp.int8(0), stepped.ab_eff_max),
      ab_scratch=stepped.ab_scratch.at[0]
      .set(jnp.where(enter_azk01_120, safe.astype(jnp.int16), stepped.ab_scratch[0]))
      .at[1]
      .set(
          jnp.where(
              enter_azk01_120,
              garden_index.astype(jnp.int16),
              stepped.ab_scratch[1],
          )
      )
      .at[2]
      .set(jnp.where(enter_azk01_120, jnp.int16(1), stepped.ab_scratch[2])),
  )
  stepped_120 = _azk01_120_ocp(pre_120)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(enter_azk01_120, a, b), stepped_120, stepped
  )

  is_stt03_002 = gate_def == cards.CODE_TO_ID["STT03-002"]
  safe_portaled_def = jnp.maximum(portaled_def, 0)
  stt03_target_row = (
      (stepped.zone[acting] == Zone.GARDEN)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_BASE_STATS)[safe_defs]
      & (jnp.asarray(cards.BASE_HP)[safe_defs].astype(jnp.int32) <= gate_power)
      & ~jnp.asarray(cards.INHERENT_DEFENDER)[safe_defs]
      & ~stepped.grant_defender[acting]
      & ~jnp.any(stepped.timed_tag[acting] != 0, axis=1)
  )
  stt03_target_ok = jnp.any(stt03_target_row)
  enter_stt03_002 = place & (gate >= 0) & is_stt03_002 & stt03_target_ok
  stepped = stepped._replace(
      ab_phase=jnp.where(
          enter_stt03_002,
          jnp.int8(AbilityPhase.EFFECT_SELECTION),
          stepped.ab_phase,
      ),
      ab_source=jnp.where(
          enter_stt03_002, safe_gate.astype(jnp.int8), stepped.ab_source
      ),
      ab_owner=jnp.where(
          enter_stt03_002, acting.astype(jnp.int8), stepped.ab_owner
      ),
      ab_slot=jnp.where(enter_stt03_002, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(enter_stt03_002, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          enter_stt03_002, False, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          enter_stt03_002, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          enter_stt03_002, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          enter_stt03_002, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(
          enter_stt03_002, jnp.int8(0), stepped.ab_cost_max
      ),
      ab_eff_selected=jnp.where(
          enter_stt03_002, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(enter_stt03_002, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(enter_stt03_002, jnp.int8(1), stepped.ab_eff_max),
      ab_cost_targets=jnp.where(
          enter_stt03_002,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          enter_stt03_002,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_targets=jnp.where(
          enter_stt03_002,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          enter_stt03_002,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_scratch=stepped.ab_scratch.at[0]
      .set(
          jnp.where(enter_stt03_002, safe.astype(jnp.int16), stepped.ab_scratch[0])
      )
      .at[1]
      .set(
          jnp.where(
              enter_stt03_002,
              garden_index.astype(jnp.int16),
              stepped.ab_scratch[1],
          )
      )
      .at[2]
      .set(jnp.where(enter_stt03_002, jnp.int16(1), stepped.ab_scratch[2])),
  )

  stt04_target_row = (
      (stepped.zone[acting] == Zone.GARDEN)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
      & stepped.took_damage_turn[acting]
  )
  stt04_target_ok = jnp.any(stt04_target_row)
  enter_stt04_002 = (
      place
      & (gate >= 0)
      & (gate_def == cards.CODE_TO_ID["STT04-002"])
      & stt04_target_ok
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          enter_stt04_002,
          jnp.int8(AbilityPhase.EFFECT_SELECTION),
          stepped.ab_phase,
      ),
      ab_source=jnp.where(
          enter_stt04_002, safe_gate.astype(jnp.int8), stepped.ab_source
      ),
      ab_owner=jnp.where(
          enter_stt04_002, acting.astype(jnp.int8), stepped.ab_owner
      ),
      ab_slot=jnp.where(enter_stt04_002, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(enter_stt04_002, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          enter_stt04_002, False, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          enter_stt04_002, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          enter_stt04_002, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          enter_stt04_002, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(
          enter_stt04_002, jnp.int8(0), stepped.ab_cost_max
      ),
      ab_eff_selected=jnp.where(
          enter_stt04_002, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(enter_stt04_002, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(enter_stt04_002, jnp.int8(1), stepped.ab_eff_max),
      ab_cost_targets=jnp.where(
          enter_stt04_002,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          enter_stt04_002,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_targets=jnp.where(
          enter_stt04_002,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          enter_stt04_002,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_scratch=stepped.ab_scratch.at[0]
      .set(
          jnp.where(enter_stt04_002, safe.astype(jnp.int16), stepped.ab_scratch[0])
      )
      .at[1]
      .set(
          jnp.where(
              enter_stt04_002,
              garden_index.astype(jnp.int16),
              stepped.ab_scratch[1],
          )
      )
      .at[2]
      .set(jnp.where(enter_stt04_002, jnp.int16(1), stepped.ab_scratch[2])),
  )


  opp = (acting + 1) % 2
  azk01_124_cost_row = (
      (stepped.zone[acting] == Zone.GARDEN)
      & (jnp.arange(stepped.zone.shape[1]) != safe)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
      & ~stepped.tapped[acting]
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[safe_defs].astype(jnp.int32) <= gate_power)
      & (gate_power > 0)
  )
  azk01_124_cost_ok = jnp.any(azk01_124_cost_row)
  opp_defs = stepped.def_id[opp]
  opp_safe_defs = jnp.maximum(opp_defs, 0)
  azk01_124_effect_ok = jnp.any(
      (stepped.zone[opp] == Zone.GARDEN)
      & (opp_defs >= 0)
      & (jnp.asarray(cards.TYPE)[opp_safe_defs] == CardType.ENTITY)
  )
  enter_azk01_124 = (
      place
      & (gate >= 0)
      & (gate_def == cards.CODE_TO_ID["AZK01-124"])
      & azk01_124_cost_ok
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          enter_azk01_124,
          jnp.int8(AbilityPhase.CONFIRMATION),
          stepped.ab_phase,
      ),
      ab_source=jnp.where(
          enter_azk01_124, safe_gate.astype(jnp.int8), stepped.ab_source
      ),
      ab_owner=jnp.where(
          enter_azk01_124, acting.astype(jnp.int8), stepped.ab_owner
      ),
      ab_slot=jnp.where(enter_azk01_124, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(enter_azk01_124, True, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          enter_azk01_124, False, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          enter_azk01_124, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          enter_azk01_124, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          enter_azk01_124, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(enter_azk01_124, jnp.int8(1), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          enter_azk01_124,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          enter_azk01_124,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          enter_azk01_124, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(enter_azk01_124, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(
          enter_azk01_124,
          jnp.where(azk01_124_effect_ok, jnp.int8(1), jnp.int8(0)),
          stepped.ab_eff_max,
      ),
      ab_eff_targets=jnp.where(
          enter_azk01_124,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          enter_azk01_124,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_scratch=stepped.ab_scratch.at[0]
      .set(jnp.where(enter_azk01_124, safe.astype(jnp.int16), stepped.ab_scratch[0]))
      .at[1]
      .set(
          jnp.where(
              enter_azk01_124,
              garden_index.astype(jnp.int16),
              stepped.ab_scratch[1],
          )
      )
      .at[2]
      .set(jnp.where(enter_azk01_124, jnp.int16(1), stepped.ab_scratch[2])),
  )

  is_stt01_002 = gate_def == cards.CODE_TO_ID["STT01-002"]
  stt01_002_row = (
      (stepped.zone[acting] == Zone.DISCARD)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.WEAPON)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[safe_defs].astype(jnp.int32) <= gate_power)
      & (gate_power > 0)
  )
  enter_stt01_002 = (
      place
      & (gate >= 0)
      & is_stt01_002
      & jnp.any(stt01_002_row)
  )
  stt01_002_base = stepped._replace(
      ab_source=jnp.where(
          enter_stt01_002, safe_gate.astype(jnp.int8), stepped.ab_source
      ),
      ab_owner=jnp.where(
          enter_stt01_002, acting.astype(jnp.int8), stepped.ab_owner
      ),
      ab_slot=jnp.where(enter_stt01_002, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(enter_stt01_002, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          enter_stt01_002, True, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          enter_stt01_002, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          enter_stt01_002, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          enter_stt01_002, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(enter_stt01_002, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          enter_stt01_002,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          enter_stt01_002,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          enter_stt01_002, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(enter_stt01_002, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(enter_stt01_002, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          enter_stt01_002,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          enter_stt01_002,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )
  stt01_002 = sel_mod.move_matching_zone_to_selection(
      stt01_002_base, Zone.DISCARD, stt01_002_row, 1, do=enter_stt01_002
  )
  stt01_002_moved = enter_stt01_002 & (stt01_002.ab_sel_count > 0)
  stt01_002 = stt01_002._replace(
      ab_scratch=stt01_002.ab_scratch.at[0]
      .set(
          jnp.where(
              stt01_002_moved,
              gate_power.astype(jnp.int16),
              stt01_002.ab_scratch[0],
          )
      )
      .at[2]
      .set(jnp.where(stt01_002_moved, jnp.int16(2), stt01_002.ab_scratch[2])),
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(enter_stt01_002, a, b), stt01_002, stepped
  )
  stepped = recompute_passives(stepped)
  stepped = _begin_queued_stt03_013_enter_trigger(stepped, place)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.GATE_PORTAL, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_confirm_azk01_124_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-124 gate-portal confirmation into cost selection."""
  from azuki_jax import cards

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.CONFIRMATION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-124"])
      & state.ab_is_optional
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 1)
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_scratch[2] == 1)
  )
  do_confirm = do_action & (action[0] == Act.CONFIRM_ABILITY) & source_ok

  stepped = state._replace(
      ab_phase=jnp.where(
          do_confirm, jnp.int8(AbilityPhase.COST_SELECTION), state.ab_phase
      ),
      ab_costs_applied=jnp.where(do_confirm, False, state.ab_costs_applied),
      ab_cost_selected=jnp.where(do_confirm, jnp.int8(0), state.ab_cost_selected),
      ab_cost_max=jnp.where(do_confirm, jnp.int8(1), state.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.CONFIRM_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_cost_azk01_124_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-124 cost sacrifice after gate-portal confirmation."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import sacrifice_card
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  portaled = jnp.maximum(state.ab_scratch[0].astype(jnp.int32), 0)
  portaled_def = state.def_id[owner, portaled]
  gate_power = jnp.where(
      portaled_def >= 0,
      jnp.asarray(cards.GATE_POINTS)[jnp.maximum(portaled_def, 0)],
      0,
  ).astype(jnp.int32)
  target = card_at_slot(state, owner, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.COST_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-124"])
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 1)
      & (state.ab_scratch[2] == 1)
  )
  target_ok = (
      (target >= 0)
      & (target != portaled)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & ~state.tapped[owner, safe_target]
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_target_def]
      & (jnp.asarray(cards.IKZ_COST)[safe_target_def].astype(jnp.int32) <= gate_power)
      & (gate_power > 0)
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_COST_TARGET)
      & source_ok
      & target_ok
  )

  damage = jnp.maximum(state.cur_hp[owner, safe_target].astype(jnp.int16), 0)
  stepped = sacrifice_card(state, owner, safe_target, do_select)
  stepped = recompute_passives(stepped)
  to_effect = do_select & (state.ab_eff_max > 0)
  to_clear = do_select & ~to_effect
  stepped = stepped._replace(
      ab_cost_targets=stepped.ab_cost_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), stepped.ab_cost_targets[0])
      ),
      ab_cost_target_players=stepped.ab_cost_target_players.at[0].set(
          jnp.where(do_select, owner.astype(jnp.int8),
                    stepped.ab_cost_target_players[0])
      ),
      ab_cost_selected=jnp.where(do_select, jnp.int8(1), stepped.ab_cost_selected),
      ab_costs_applied=jnp.where(do_select, True, stepped.ab_costs_applied),
      ab_phase=jnp.where(
          to_effect, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
      ab_scratch=stepped.ab_scratch.at[0]
      .set(jnp.where(do_select, damage, stepped.ab_scratch[0]).astype(jnp.int16))
      .at[2]
      .set(jnp.where(do_select, jnp.int16(3), stepped.ab_scratch[2])),
  )
  cleared = _clear_context(stepped)
  cleared = _begin_queued_stt03_013_enter_trigger(cleared, to_clear)
  stepped = jax.tree.map(lambda a, b: jnp.where(to_clear, a, b), cleared, stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_COST_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_124_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-124 optional enemy-garden damage or skip."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot
  from azuki_jax.engine.triggers import TIMING_WHEN_DESTROYED, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-124"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 1)
      & (state.ab_scratch[2] == 3)
  )
  target = card_at_slot(state, opp, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_ok = (
      (target >= 0)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  do_skip = do_action & (action_type == Act.NOOP) & source_ok
  damage = state.ab_scratch[0].astype(jnp.int16)
  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = deal_effect_damage(stepped, opp, safe_target, damage, do_select)
  stepped = recompute_passives(stepped)
  finish = do_select | do_skip
  cleared = _clear_context(stepped)
  popped, trig_src, trig_owner, trig_timing = pop_effect(cleared)
  trig_src_i = jnp.maximum(trig_src.astype(jnp.int32), 0)
  trig_owner_i = jnp.maximum(trig_owner.astype(jnp.int32), 0)
  begin_stt03_006 = (
      finish
      & (cleared.trig_count > 0)
      & (trig_timing == jnp.int8(TIMING_WHEN_DESTROYED))
      & (cleared.def_id[trig_owner_i, trig_src_i] == cards.CODE_TO_ID["STT03-006"])
  )
  begun = resolve_triggered_effect(popped, trig_src, trig_owner, trig_timing)
  cleared = jax.tree.map(lambda a, b: jnp.where(begin_stt03_006, a, b), begun, cleared)
  cleared = _begin_queued_stt03_013_enter_trigger(cleared, finish)
  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), cleared, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_select_azk01_122_place_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    placement_zone: int,
    episode_cap: int = 0,
):
  """Fast AZK01-122 selection-to-garden/alley placement."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.abilities.cards_impl import apply_charge_grant
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import (
      GRANT_PHASE_NONE,
      _detach_from_location,
      attr_force_tapped,
      discard,
      passive_zone_event,
  )
  from azuki_jax.engine.triggers import queue_enter_garden, queue_on_play

  placement_zone = int(placement_zone)
  action_id = (
      Act.SELECT_TO_GARDEN
      if placement_zone == int(Zone.GARDEN)
      else Act.SELECT_TO_ALLEY
  )

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  sel_idx = action[1].astype(jnp.int32)
  slot = action[2].astype(jnp.int32)
  idx = jnp.clip(sel_idx, 0, state.ab_sel_cards.shape[0] - 1)
  inst = state.ab_sel_cards[idx]
  target = jnp.maximum(inst.astype(jnp.int32), 0)
  source_def = state.def_id[owner, src]
  target_def = state.def_id[owner, target]
  portaled = jnp.clip(
      state.ab_scratch[0].astype(jnp.int32), 0, state.def_id.shape[1] - 1
  )
  portaled_def = state.def_id[owner, portaled]
  gate_power = jnp.where(
      (state.ab_scratch[2] == 1) & (portaled_def >= 0),
      jnp.asarray(cards.GATE_POINTS)[jnp.maximum(portaled_def, 0)],
      0,
  ).astype(jnp.int32)
  target_ok = (
      (target_def >= 0)
      & (jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[jnp.maximum(target_def, 0)]
      & (jnp.asarray(cards.IKZ_COST)[jnp.maximum(target_def, 0)].astype(jnp.int32) <= gate_power)
      & (gate_power > 0)
  )
  source_ok = (
      (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_def == cards.CODE_TO_ID["AZK01-122"])
      & state.ab_costs_applied
      & (state.ab_sel_pick_max == 1)
      & (state.ab_sel_picked_count == 0)
  )
  pick_ok = (
      (action[0] == jnp.asarray(action_id, jnp.int32))
      & (sel_idx >= 0)
      & (sel_idx < state.ab_sel_count.astype(jnp.int32))
      & (inst >= 0)
      & ((state.zone[owner, target] == Zone.SELECTION) | (state.zone[owner, target] == Zone.HAND))
      & target_ok
  )
  do_action = do_action & source_ok & pick_ok
  target_is_stt04_005 = do_action & (
      target_def == cards.CODE_TO_ID["STT04-005"]
  )
  target_is_stt04_004 = do_action & (
      target_def == cards.CODE_TO_ID["STT04-004"]
  )
  target_is_azk01_056 = do_action & (
      target_def == cards.CODE_TO_ID["AZK01-056"]
  )

  if placement_zone == int(Zone.GARDEN):
    stepped = _enter_board_slot(
        state, owner, target, Zone.GARDEN, slot, do_action
    )
    stepped = stepped._replace(
        entities_played_garden_turn=stepped.entities_played_garden_turn.at[
            owner
        ].add(do_action.astype(jnp.uint8)),
        cards_played_turn=stepped.cards_played_turn.at[owner].add(
            do_action.astype(jnp.uint8)
        ),
        next_play_cost_reduction=stepped.next_play_cost_reduction.at[
            owner
        ].set(
            jnp.where(do_action, 0, stepped.next_play_cost_reduction[owner])
        ),
    )
    stepped = queue_enter_garden(stepped, owner, target, do=do_action)
    stepped = queue_on_play(
        stepped,
        owner,
        target,
        do=do_action
        & ~(target_is_stt04_005 | target_is_stt04_004 | target_is_azk01_056),
    )
  else:
    slot_match = (state.zone[owner] == Zone.ALLEY) & (
        state.zpos[owner] == slot.astype(jnp.int8)
    )
    occupied = jnp.any(slot_match)
    occupied_inst = jnp.where(occupied, jnp.argmax(slot_match), -1)
    full = jnp.sum(state.zone[owner] == Zone.ALLEY, dtype=jnp.int32) >= GARDEN_SIZE
    stepped = discard(
        state,
        owner,
        jnp.maximum(occupied_inst, 0),
        reason_replacement=True,
        ignore_godmode=True,
        do=do_action & occupied & full,
    )
    stepped = _detach_from_location(stepped, owner, target, do_action)
    stepped = stepped._replace(
        zone=stepped.zone.at[owner, target].set(
            jnp.where(do_action, jnp.int8(Zone.ALLEY), stepped.zone[owner, target])
        ),
        zpos=stepped.zpos.at[owner, target].set(
            jnp.where(do_action, slot.astype(jnp.int8), stepped.zpos[owner, target])
        ),
        board_seq=stepped.board_seq.at[owner, target].set(
            jnp.where(do_action, stepped.seq_counter, stepped.board_seq[owner, target])
        ),
        seq_counter=(stepped.seq_counter + jnp.where(do_action, 1, 0)).astype(
            jnp.int16
        ),
        tapped=stepped.tapped.at[owner, target].set(
            jnp.where(do_action, False, stepped.tapped[owner, target])
        ),
        cooldown=stepped.cooldown.at[owner, target].set(
            jnp.where(do_action, 0, stepped.cooldown[owner, target])
        ),
        entities_played_alley_turn=stepped.entities_played_alley_turn.at[
            owner
        ].add(do_action.astype(jnp.uint8)),
        cards_played_turn=stepped.cards_played_turn.at[owner].add(
            do_action.astype(jnp.uint8)
        ),
        next_play_cost_reduction=stepped.next_play_cost_reduction.at[
            owner
        ].set(
            jnp.where(do_action, 0, stepped.next_play_cost_reduction[owner])
        ),
    )
    stepped = passive_zone_event(
        stepped, owner, Zone.ALLEY, target, True, do=do_action
    )
    stepped = queue_on_play(
        stepped,
        owner,
        target,
        do=do_action
        & ~(target_is_stt04_005 | target_is_stt04_004 | target_is_azk01_056),
    )

  stepped = stepped._replace(
      ab_sel_picked=stepped.ab_sel_picked.at[0].set(
          jnp.where(do_action, inst, stepped.ab_sel_picked[0])
      ),
      ab_sel_picked_count=jnp.where(
          do_action, jnp.int8(1), stepped.ab_sel_picked_count
      ),
      ab_sel_cards=stepped.ab_sel_cards.at[idx].set(
          jnp.where(do_action, jnp.int8(-1), stepped.ab_sel_cards[idx])
      ),
  )
  stepped = apply_charge_grant(
      stepped, owner, target, GRANT_PHASE_NONE, -1, do_action
  )
  force = attr_force_tapped(stepped, owner, target)
  stepped = stepped._replace(
      tapped=stepped.tapped.at[owner, target].set(
          jnp.where(do_action, force | stepped.tapped[owner, target],
                    stepped.tapped[owner, target])
      ),
      cooldown=stepped.cooldown.at[owner, target].set(
          jnp.where(do_action, 0, stepped.cooldown[owner, target]).astype(jnp.uint8)
      ),
  )
  stepped = sel_mod.return_remaining_to_hand(stepped, do_action)
  cleared = runtime._clear_context(stepped)
  stt04_004 = cleared._replace(
      ab_source=jnp.where(
          target_is_stt04_004, target.astype(jnp.int8), cleared.ab_source
      ),
      ab_owner=jnp.where(
          target_is_stt04_004, owner.astype(jnp.int8), cleared.ab_owner
      ),
      ab_slot=jnp.where(target_is_stt04_004, jnp.int8(0), cleared.ab_slot),
      ab_is_optional=jnp.where(
          target_is_stt04_004, True, cleared.ab_is_optional
      ),
      ab_costs_applied=jnp.where(
          target_is_stt04_004, False, cleared.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          target_is_stt04_004, jnp.int8(-1), cleared.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          target_is_stt04_004, False, cleared.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          target_is_stt04_004, jnp.int8(0), cleared.ab_cost_selected
      ),
      ab_cost_max=jnp.where(
          target_is_stt04_004, jnp.int8(0), cleared.ab_cost_max
      ),
      ab_eff_selected=jnp.where(
          target_is_stt04_004, jnp.int8(0), cleared.ab_eff_selected
      ),
      ab_eff_min=jnp.where(
          target_is_stt04_004, jnp.int8(1), cleared.ab_eff_min
      ),
      ab_eff_max=jnp.where(
          target_is_stt04_004, jnp.int8(1), cleared.ab_eff_max
      ),
      ab_phase=jnp.where(
          target_is_stt04_004,
          jnp.int8(AbilityPhase.CONFIRMATION),
          cleared.ab_phase,
      ),
  )
  cleared = jax.tree.map(
      lambda a, b: jnp.where(target_is_stt04_004, a, b), stt04_004, cleared
  )
  azk01_056 = cleared._replace(
      ab_source=jnp.where(
          target_is_azk01_056, target.astype(jnp.int8), cleared.ab_source
      ),
      ab_owner=jnp.where(
          target_is_azk01_056, owner.astype(jnp.int8), cleared.ab_owner
      ),
      ab_slot=jnp.where(target_is_azk01_056, jnp.int8(0), cleared.ab_slot),
      ab_is_optional=jnp.where(
          target_is_azk01_056, False, cleared.ab_is_optional
      ),
      ab_costs_applied=jnp.where(
          target_is_azk01_056, True, cleared.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          target_is_azk01_056, jnp.int8(-1), cleared.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          target_is_azk01_056, False, cleared.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          target_is_azk01_056, jnp.int8(0), cleared.ab_cost_selected
      ),
      ab_cost_max=jnp.where(
          target_is_azk01_056, jnp.int8(0), cleared.ab_cost_max
      ),
      ab_eff_selected=jnp.where(
          target_is_azk01_056, jnp.int8(0), cleared.ab_eff_selected
      ),
      ab_eff_min=jnp.where(
          target_is_azk01_056, jnp.int8(0), cleared.ab_eff_min
      ),
      ab_eff_max=jnp.where(
          target_is_azk01_056, jnp.int8(0), cleared.ab_eff_max
      ),
  )
  azk01_056 = sel_mod.reveal_top_into_selection(
      azk01_056, 5, 1, target_is_azk01_056
  )
  revealed = azk01_056.ab_sel_count > 0
  scorchweaver = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Scorchweaver")], jnp.bool_
  )
  sel_cards = azk01_056.ab_sel_cards
  sel_safe = jnp.maximum(sel_cards.astype(jnp.int32), 0)
  sel_defs = azk01_056.def_id[owner, sel_safe]
  sel_valid = sel_cards >= 0
  matching = jnp.any(sel_valid & scorchweaver[jnp.maximum(sel_defs, 0)])
  next_phase = jnp.where(
      matching,
      jnp.int8(AbilityPhase.SELECTION_PICK),
      jnp.int8(AbilityPhase.BOTTOM_DECK),
  )
  azk01_056 = azk01_056._replace(
      ab_phase=jnp.where(
          target_is_azk01_056 & revealed,
          next_phase,
          azk01_056.ab_phase,
      )
  )
  cleared = jax.tree.map(
      lambda a, b: jnp.where(target_is_azk01_056, a, b),
      azk01_056,
      cleared,
  )

  stt04 = cleared._replace(
      ab_source=jnp.where(
          target_is_stt04_005, target.astype(jnp.int8), cleared.ab_source
      ),
      ab_owner=jnp.where(
          target_is_stt04_005, owner.astype(jnp.int8), cleared.ab_owner
      ),
      ab_slot=jnp.where(target_is_stt04_005, jnp.int8(0), cleared.ab_slot),
      ab_is_optional=jnp.where(
          target_is_stt04_005, False, cleared.ab_is_optional
      ),
      ab_costs_applied=jnp.where(
          target_is_stt04_005, True, cleared.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          target_is_stt04_005, jnp.int8(-1), cleared.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          target_is_stt04_005, False, cleared.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          target_is_stt04_005, jnp.int8(0), cleared.ab_cost_selected
      ),
      ab_cost_max=jnp.where(
          target_is_stt04_005, jnp.int8(0), cleared.ab_cost_max
      ),
      ab_eff_selected=jnp.where(
          target_is_stt04_005, jnp.int8(0), cleared.ab_eff_selected
      ),
      ab_eff_min=jnp.where(
          target_is_stt04_005, jnp.int8(0), cleared.ab_eff_min
      ),
      ab_eff_max=jnp.where(
          target_is_stt04_005, jnp.int8(0), cleared.ab_eff_max
      ),
  )
  stt04 = sel_mod.reveal_top_into_selection(stt04, 5, 1, target_is_stt04_005)
  revealed = stt04.ab_sel_count > 0
  pyreskin = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Pyreskin")], jnp.bool_
  )
  sel_cards = stt04.ab_sel_cards
  sel_safe = jnp.maximum(sel_cards.astype(jnp.int32), 0)
  sel_defs = stt04.def_id[owner, sel_safe]
  sel_valid = sel_cards >= 0
  matching = jnp.any(sel_valid & pyreskin[jnp.maximum(sel_defs, 0)])
  next_phase = jnp.where(
      matching,
      jnp.int8(AbilityPhase.SELECTION_PICK),
      jnp.int8(AbilityPhase.BOTTOM_DECK),
  )
  stt04 = stt04._replace(
      ab_phase=jnp.where(
          target_is_stt04_005 & revealed,
          next_phase,
          stt04.ab_phase,
      )
  )
  stt04_done = runtime._clear_context(stt04)
  stt04 = jax.tree.map(
      lambda a, b: jnp.where(target_is_stt04_005 & ~revealed, a, b),
      stt04_done,
      stt04,
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(target_is_stt04_005, a, b),
      stt04,
      cleared,
  )
  stepped = jax.tree.map(lambda a, b: jnp.where(do_action, a, b), stepped, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(action_id, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_confirm_stt03_011_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT03-011 optional confirmation into effect selection."""
  from azuki_jax import cards

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.CONFIRMATION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT03-011"])
      & state.ab_is_optional
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 0)
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 1)
  )
  do_confirm = do_action & (action[0] == Act.CONFIRM_ABILITY) & source_ok

  stepped = state._replace(
      ab_is_optional=jnp.where(do_confirm, False, state.ab_is_optional),
      ab_eff_min=jnp.where(do_confirm, jnp.int8(1), state.ab_eff_min),
      ab_phase=jnp.where(
          do_confirm, jnp.int8(AbilityPhase.EFFECT_SELECTION), state.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.CONFIRM_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_confirm_clear_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast confirmation resolution when confirm/decline only clears context."""
  from azuki_jax import cards
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.triggers import (
      TIMING_WHEN_DESTROYED,
      TIMING_WHEN_RETURNED_TO_HAND,
      pop_effect,
  )

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)

  prev = state
  stepped = _clear_context(state)
  popped, src, owner, timing = pop_effect(stepped)
  src_i = jnp.maximum(src.astype(jnp.int32), 0)
  owner_i = jnp.maximum(owner.astype(jnp.int32), 0)
  begin_stt02_010 = (
      (stepped.trig_count > 0)
      & (timing == jnp.int8(TIMING_WHEN_RETURNED_TO_HAND))
      & (stepped.def_id[owner_i, src_i] == cards.CODE_TO_ID["STT02-010"])
  )
  begin_stt03_006 = (
      (stepped.trig_count > 0)
      & (timing == jnp.int8(TIMING_WHEN_DESTROYED))
      & (stepped.def_id[owner_i, src_i] == cards.CODE_TO_ID["STT03-006"])
  )
  begun = resolve_triggered_effect(popped, src, owner, timing)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(begin_stt02_010 | begin_stt03_006, a, b), begun, stepped
  )
  stepped = _begin_queued_stt03_013_enter_trigger(stepped, jnp.bool_(True))
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, noop_had_alternatives
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_confirm_stt02_010_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-010 optional trigger confirmation: tap, draw, then drain queue."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.triggers import (
      TIMING_WHEN_DESTROYED,
      TIMING_WHEN_RETURNED_TO_HAND,
      pop_effect,
  )

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.CONFIRMATION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT02-010"])
      & state.ab_is_optional
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 0)
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 0)
  )
  do_confirm = (
      ~(did_reset | zero_legal)
      & (action_type == Act.CONFIRM_ABILITY)
      & source_ok
  )

  stepped = runtime.process_confirm(state, do_confirm)
  stepped = recompute_passives(stepped)

  popped, trig_src, trig_owner, trig_timing = pop_effect(stepped)
  trig_src_i = jnp.maximum(trig_src.astype(jnp.int32), 0)
  trig_owner_i = jnp.maximum(trig_owner.astype(jnp.int32), 0)
  can_begin_next = (
      do_confirm
      & (stepped.winner == -1)
      & (stepped.ab_phase == AbilityPhase.NONE)
      & (stepped.trig_count > 0)
  )
  begin_stt02_010 = (
      can_begin_next
      & (trig_timing == jnp.int8(TIMING_WHEN_RETURNED_TO_HAND))
      & (stepped.def_id[trig_owner_i, trig_src_i] == cards.CODE_TO_ID["STT02-010"])
  )
  begin_stt03_006 = (
      can_begin_next
      & (trig_timing == jnp.int8(TIMING_WHEN_DESTROYED))
      & (stepped.def_id[trig_owner_i, trig_src_i] == cards.CODE_TO_ID["STT03-006"])
  )
  begun = resolve_triggered_effect(popped, trig_src, trig_owner, trig_timing)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(begin_stt02_010 | begin_stt03_006, a, b),
      begun,
      stepped,
  )
  stepped = _begin_queued_stt03_013_enter_trigger(stepped, do_confirm)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )
  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations


def step_confirm_azk01_006_when_attacked_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-006 optional when-attacked confirmation/decline."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import return_to_hand
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.phases import combat_resolve

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.phase == Phase.COMBAT_RESOLVE)
      & (state.ab_phase == AbilityPhase.CONFIRMATION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-006"])
      & state.ab_is_optional
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 0)
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 0)
  )
  do_confirm = do_action & (action_type == Act.CONFIRM_ABILITY) & source_ok
  do_decline = do_action & (action_type == Act.NOOP) & source_ok
  finish = do_confirm | do_decline

  stepped = return_to_hand(state, owner, src, do_confirm)
  stepped = _clear_context(stepped)
  stepped = recompute_passives(stepped)
  stepped = combat_resolve(stepped, do=finish)
  stepped = recompute_passives(stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), stepped, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type.astype(jnp.int32), noop_had_alternatives
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_confirm_stt03_013_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT03-013 optional enters-garden confirmation."""
  from azuki_jax import cards
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.abilities.passives import recompute_passives

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.CONFIRMATION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT03-013"])
      & state.ab_is_optional
  )
  do_confirm = do_action & (action_type == Act.CONFIRM_ABILITY) & source_ok
  do_decline = do_action & (action_type == Act.NOOP) & source_ok
  confirmed = state._replace(
      tapped=state.tapped.at[owner, src].set(
          jnp.where(do_confirm, True, state.tapped[owner, src])
      )
  )
  confirmed = _clear_context(confirmed)
  declined = _clear_context(state)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_decline, a, b), declined, state
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_confirm, a, b), confirmed, stepped
  )
  stepped = recompute_passives(stepped)
  acted = do_confirm | do_decline
  stepped = jax.tree.map(lambda a, b: jnp.where(acted, a, b), stepped, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type.astype(jnp.int32), noop_had_alternatives
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_confirm_stt01_007_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-007 confirmation into friendly-hand discard selection."""
  from azuki_jax import cards
  from azuki_jax.abilities.runtime import _clear_context

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.CONFIRMATION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT01-007"])
      & state.ab_is_optional
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 1)
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 0)
  )
  do_confirm = (
      do_action
      & (action[0] == Act.CONFIRM_ABILITY)
      & source_ok
  )
  has_cost_target = jnp.any(
      (state.zone[owner] == Zone.HAND)
      & (jnp.arange(state.zone.shape[1]) != src)
  )
  proceed = do_confirm & has_cost_target
  clear = do_confirm & ~has_cost_target

  stepped = state._replace(
      ab_phase=jnp.where(
          proceed, jnp.int8(AbilityPhase.COST_SELECTION), state.ab_phase
      ),
      ab_cost_max=jnp.where(proceed, jnp.int8(1), state.ab_cost_max),
  )
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(clear, a, b), cleared, stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(do_confirm, a, b), stepped, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.CONFIRM_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_confirm_stt01_002_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-002 confirmation into discard weapon selection."""
  from azuki_jax import cards
  from azuki_jax.abilities import selection as sel_mod

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  portaled = jnp.clip(
      state.ab_scratch[0].astype(jnp.int32), 0, state.def_id.shape[1] - 1
  )
  portaled_def = state.def_id[owner, portaled]
  source_def = state.def_id[owner, src]
  source_stt01_002 = (
      (source_def == cards.CODE_TO_ID["STT01-002"])
      & (state.ab_scratch[2] == 1)
  )
  source_azk01_098 = source_def == cards.CODE_TO_ID["AZK01-098"]
  gate_power = jnp.where(
      (state.ab_scratch[2] == 1) & (portaled_def >= 0),
      jnp.asarray(cards.GATE_POINTS)[jnp.maximum(portaled_def, 0)],
      0,
  ).astype(jnp.int32)
  source_ok = (
      (state.ab_phase == AbilityPhase.CONFIRMATION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_stt01_002 | source_azk01_098)
      & state.ab_is_optional
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 0)
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 0)
  )
  max_cost = jnp.where(source_azk01_098, jnp.int32(3), gate_power)
  src_zone = jnp.where(
      source_azk01_098, jnp.int8(Zone.HAND), jnp.int8(Zone.DISCARD)
  )

  def_id_row = state.def_id[owner]
  safe_defs = jnp.maximum(def_id_row, 0)
  eligible = (
      (state.zone[owner] == src_zone)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.WEAPON)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[safe_defs].astype(jnp.int32) <= max_cost)
      & (max_cost > 0)
  )
  has_eligible = jnp.any(eligible)
  do_confirm = (
      do_action
      & (action[0] == Act.CONFIRM_ABILITY)
      & source_ok
      & has_eligible
  )

  stepped = state._replace(
      ab_costs_applied=jnp.where(do_confirm, True, state.ab_costs_applied),
      tapped=state.tapped.at[owner, src].set(
          jnp.where(do_confirm & source_azk01_098, True, state.tapped[owner, src])
      ),
  )
  stepped = sel_mod.move_matching_zone_to_selection(
      stepped, src_zone, eligible, 1, do=do_confirm
  )
  moved = do_confirm & (stepped.ab_sel_count > 0)
  stepped = stepped._replace(
      ab_scratch=stepped.ab_scratch.at[0]
      .set(
          jnp.where(
              moved & source_stt01_002,
              gate_power.astype(jnp.int16),
              stepped.ab_scratch[0],
          )
      )
      .at[2]
      .set(
          jnp.where(
              moved & source_stt01_002, jnp.int16(2), stepped.ab_scratch[2]
          )
      ),
  )
  stepped = jax.tree.map(lambda a, b: jnp.where(do_confirm, a, b), stepped, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.CONFIRM_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_stt01_002_equip_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-002 selection pick into weapon equip."""
  from azuki_jax import cards
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.helpers import leader_instance
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.engine.triggers import TIMING_ON_PLAY, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  sel_idx = jnp.clip(action[1].astype(jnp.int32), 0, state.ab_sel_cards.shape[0] - 1)
  selected = state.ab_sel_cards[sel_idx]
  safe_selected = jnp.maximum(selected.astype(jnp.int32), 0)
  source_def = state.def_id[owner, src]
  source_stt01_002 = (
      (source_def == cards.CODE_TO_ID["STT01-002"])
      & state.ab_costs_applied
      & (state.ab_scratch[2] == 2)
  )
  source_azk01_120 = (
      (source_def == cards.CODE_TO_ID["AZK01-120"])
      & state.ab_costs_applied
      & (state.ab_scratch[2] == 1)
  )
  source_azk01_098 = (
      (source_def == cards.CODE_TO_ID["AZK01-098"])
      & state.ab_costs_applied
  )
  source_ok = (
      (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_stt01_002 | source_azk01_120 | source_azk01_098)
      & (state.ab_sel_pick_max == 1)
      & (state.ab_sel_picked_count == 0)
  )
  do_equip = (
      do_action
      & (action[0] == Act.SELECT_TO_EQUIP)
      & source_ok
  )

  stepped = sel_mod.process_selection_to_equip(
      state, action[1].astype(jnp.int32), action[2].astype(jnp.int32), do_equip
  )
  stepped = recompute_passives(stepped)
  leader = leader_instance(stepped, owner)
  safe_leader = jnp.maximum(leader, 0)
  host = stepped.attached_to[owner, safe_selected].astype(jnp.int32)
  begin_stt01_013 = (
      do_equip
      & (selected >= 0)
      & (stepped.def_id[owner, safe_selected] == cards.CODE_TO_ID["STT01-013"])
      & (stepped.trig_count > 0)
      & (stepped.trig_source[0] == safe_selected.astype(jnp.int8))
      & (stepped.trig_owner[0] == owner.astype(jnp.int8))
      & (leader >= 0)
      & (host >= 0)
      & (stepped.cur_hp[owner, safe_leader] >= 1)
  )
  popped, trig_src, trig_owner, trig_timing = pop_effect(stepped)
  begin_stt01_013 = (
      begin_stt01_013
      & (trig_timing == jnp.int8(TIMING_ON_PLAY))
      & (trig_src == safe_selected.astype(jnp.int8))
      & (trig_owner == owner.astype(jnp.int8))
  )
  begun = popped._replace(
      ab_phase=jnp.where(
          begin_stt01_013,
          jnp.int8(AbilityPhase.CONFIRMATION),
          popped.ab_phase,
      ),
      ab_source=jnp.where(
          begin_stt01_013, safe_selected.astype(jnp.int8), popped.ab_source
      ),
      ab_owner=jnp.where(
          begin_stt01_013, owner.astype(jnp.int8), popped.ab_owner
      ),
      ab_slot=jnp.where(begin_stt01_013, jnp.int8(0), popped.ab_slot),
      ab_is_optional=jnp.where(begin_stt01_013, True, popped.ab_is_optional),
      ab_costs_applied=jnp.where(
          begin_stt01_013, False, popped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          begin_stt01_013, jnp.int8(-1), popped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          begin_stt01_013, False, popped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          begin_stt01_013, jnp.int8(0), popped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(
          begin_stt01_013, jnp.int8(0), popped.ab_cost_max
      ),
      ab_cost_targets=jnp.where(
          begin_stt01_013,
          jnp.full_like(popped.ab_cost_targets, -1),
          popped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          begin_stt01_013,
          jnp.full_like(popped.ab_cost_target_players, -1),
          popped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          begin_stt01_013, jnp.int8(0), popped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(begin_stt01_013, jnp.int8(0), popped.ab_eff_min),
      ab_eff_max=jnp.where(begin_stt01_013, jnp.int8(0), popped.ab_eff_max),
      ab_eff_targets=jnp.where(
          begin_stt01_013,
          jnp.full_like(popped.ab_eff_targets, -1),
          popped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          begin_stt01_013,
          jnp.full_like(popped.ab_eff_target_players, -1),
          popped.ab_eff_target_players,
      ),
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(begin_stt01_013, a, b), begun, stepped
  )
  begin_stt01_014 = (
      do_equip
      & (selected >= 0)
      & (stepped.def_id[owner, safe_selected] == cards.CODE_TO_ID["STT01-014"])
      & (stepped.trig_count > 0)
      & (trig_timing == jnp.int8(TIMING_ON_PLAY))
      & (trig_src == safe_selected.astype(jnp.int8))
      & (trig_owner == owner.astype(jnp.int8))
  )
  begun_stt01_014 = resolve_triggered_effect(
      popped, trig_src, trig_owner, trig_timing
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(begin_stt01_014, a, b),
      begun_stt01_014,
      stepped,
  )
  stepped = jax.tree.map(lambda a, b: jnp.where(do_equip, a, b), stepped, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_TO_EQUIP, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_select_azk01_086_pick_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-086 selection pick via the shared selection runtime."""
  from azuki_jax import cards
  from azuki_jax.abilities import selection as sel_mod

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_def = state.def_id[owner, src]
  source_ok = (
      (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (
          (source_def == cards.CODE_TO_ID["AZK01-084"])
          | (source_def == cards.CODE_TO_ID["AZK01-086"])
      )
      & state.ab_costs_applied
      & (state.ab_sel_pick_max > 0)
      & (state.ab_sel_picked_count < state.ab_sel_pick_max)
  )
  do_pick = (
      do_action
      & (action[0] == Act.SELECT_FROM_SELECTION)
      & source_ok
  )

  stepped = sel_mod.process_selection_pick(
      state, action[1].astype(jnp.int32), do_pick
  )
  stepped = jax.tree.map(lambda a, b: jnp.where(do_pick, a, b), stepped, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_FROM_SELECTION, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_select_azk01_126_pick_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-126 selection pick: return picked discard spell to hand."""
  from azuki_jax import cards
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.abilities.passives import recompute_passives

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-126"])
      & state.ab_costs_applied
      & (state.ab_scratch[2] == 1)
      & (state.ab_sel_pick_max == 1)
      & (state.ab_sel_picked_count == 0)
  )
  do_pick = (
      do_action
      & (action[0] == Act.SELECT_FROM_SELECTION)
      & source_ok
  )

  stepped = sel_mod.process_selection_pick(
      state, action[1].astype(jnp.int32), do_pick
  )
  stepped = jax.tree.map(lambda a, b: jnp.where(do_pick, a, b), stepped, state)
  stepped = recompute_passives(stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_FROM_SELECTION, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_cost_stt01_007_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-007 cost: discard selected hand card, then draw 1."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import draw_with_deckout
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import discard, hand_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  hand_idx = action[1].astype(jnp.int32)
  target = hand_instance(state, owner, hand_idx)
  safe_target = jnp.maximum(target, 0)
  source_def = state.def_id[owner, src]
  target_ok = (
      (target >= 0)
      & (target != src)
      & (state.zone[owner, safe_target] == Zone.HAND)
  )
  ok = (
      do_action
      & (action[0] == Act.SELECT_COST_TARGET)
      & (state.ab_phase == AbilityPhase.COST_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_def == cards.CODE_TO_ID["STT01-007"])
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 1)
      & target_ok
  )

  stepped = state._replace(
      ab_cost_targets=state.ab_cost_targets.at[0].set(
          jnp.where(ok, safe_target.astype(jnp.int8), state.ab_cost_targets[0])
      ),
      ab_cost_target_players=state.ab_cost_target_players.at[0].set(
          jnp.where(ok, owner.astype(jnp.int8), state.ab_cost_target_players[0])
      ),
      ab_cost_selected=jnp.where(ok, jnp.int8(1), state.ab_cost_selected),
  )
  stepped = discard(stepped, owner, safe_target, do=ok)
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(ok, True, stepped.ab_costs_applied)
  )
  stepped = draw_with_deckout(stepped, owner, 1, ok)
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(ok, a, b), cleared, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_COST_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_confirm_stt01_013_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-013 optional confirmation."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  leader = leader_instance(state, owner)
  safe_leader = jnp.maximum(leader, 0)
  host = state.attached_to[owner, src].astype(jnp.int32)
  safe_host = jnp.maximum(host, 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.CONFIRMATION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT01-013"])
      & state.ab_is_optional
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 0)
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 0)
      & (leader >= 0)
      & (host >= 0)
  )
  do_confirm = (
      do_action
      & (action[0] == Act.CONFIRM_ABILITY)
      & source_ok
  )

  stepped = deal_effect_damage(state, owner, safe_leader, 1, do=do_confirm)
  weapon_atk = (stepped.cur_atk[owner, src].astype(jnp.int16) + 1).astype(
      jnp.int8
  )
  host_atk = (stepped.cur_atk[owner, safe_host].astype(jnp.int16) + 1).astype(
      jnp.int8
  )
  stepped = stepped._replace(
      cur_atk=stepped.cur_atk.at[owner, src].set(
          jnp.where(do_confirm, weapon_atk, stepped.cur_atk[owner, src])
      )
  )
  stepped = stepped._replace(
      cur_atk=stepped.cur_atk.at[owner, safe_host].set(
          jnp.where(do_confirm, host_atk, stepped.cur_atk[owner, safe_host])
      )
  )
  cleared = _clear_context(stepped)
  cleared = recompute_passives(cleared)
  stepped = jax.tree.map(lambda a, b: jnp.where(do_confirm, a, b), cleared, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.CONFIRM_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_confirm_stt01_004_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-004 confirmation into weapon cost selection."""
  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  stepped = state._replace(
      ab_phase=jnp.where(
          do_action, jnp.int8(AbilityPhase.COST_SELECTION), state.ab_phase
      ),
      ab_costs_applied=jnp.where(do_action, False, state.ab_costs_applied),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), state.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(1), state.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), state.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(0), state.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(0), state.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_targets, -1),
          state.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_target_players, -1),
          state.ab_eff_target_players,
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.CONFIRM_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_confirm_stt04_004_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT04-004 confirmation into effect selection."""
  from azuki_jax.abilities.cards_impl import sacrifice_card
  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  do_confirm = do_action & (action[0] == Act.CONFIRM_ABILITY)
  stepped = state._replace(
      ab_phase=jnp.where(
          do_confirm, jnp.int8(AbilityPhase.EFFECT_SELECTION), state.ab_phase
      ),
      ab_costs_applied=jnp.where(do_confirm, True, state.ab_costs_applied),
      ab_cost_selected=jnp.where(do_confirm, jnp.int8(0), state.ab_cost_selected),
      ab_cost_max=jnp.where(do_confirm, jnp.int8(0), state.ab_cost_max),
      ab_eff_selected=jnp.where(do_confirm, jnp.int8(0), state.ab_eff_selected),
      ab_eff_min=jnp.where(do_confirm, jnp.int8(1), state.ab_eff_min),
      ab_eff_max=jnp.where(do_confirm, jnp.int8(1), state.ab_eff_max),
      ab_cost_targets=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
      ab_eff_targets=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_eff_targets, -1),
          state.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_confirm,
          jnp.full_like(state.ab_eff_target_players, -1),
          state.ab_eff_target_players,
      ),
  )
  stepped = sacrifice_card(stepped, owner, src, do=do_confirm)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.CONFIRM_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_cost_stt01_004_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-004 weapon cost discard into reveal selection."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.abilities.cards_impl import sacrifice_card
  from azuki_jax.engine.helpers import hand_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  hand_idx = action[1].astype(jnp.int32)
  target = hand_instance(state, owner, hand_idx)
  safe_target = jnp.maximum(target, 0)
  source_def = state.def_id[owner, src]
  target_def = state.def_id[owner, safe_target]
  target_ok = (
      (target >= 0)
      & (state.zone[owner, safe_target] == Zone.HAND)
      & (jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)] == CardType.WEAPON)
  )
  ok = (
      do_action
      & (action[0] == Act.SELECT_COST_TARGET)
      & (state.ab_phase == AbilityPhase.COST_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_def == cards.CODE_TO_ID["STT01-004"])
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 1)
      & target_ok
  )

  stepped = state._replace(
      ab_cost_targets=state.ab_cost_targets.at[0].set(
          jnp.where(ok, safe_target.astype(jnp.int8), state.ab_cost_targets[0])
      ),
      ab_cost_target_players=state.ab_cost_target_players.at[0].set(
          jnp.where(ok, owner.astype(jnp.int8), state.ab_cost_target_players[0])
      ),
      ab_cost_selected=jnp.where(ok, jnp.int8(1), state.ab_cost_selected),
  )
  stepped = sacrifice_card(stepped, owner, safe_target, do=ok)
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(ok, True, stepped.ab_costs_applied)
  )
  stepped = sel_mod.reveal_top_into_selection(stepped, 5, 1, ok)
  revealed = stepped.ab_sel_count > 0
  sel_cards = stepped.ab_sel_cards
  sel_safe = jnp.maximum(sel_cards.astype(jnp.int32), 0)
  sel_defs = stepped.def_id[owner, sel_safe]
  sel_valid = sel_cards >= 0
  matching = jnp.any(
      sel_valid
      & (jnp.asarray(cards.TYPE)[jnp.maximum(sel_defs, 0)] == CardType.WEAPON)
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          ok & revealed,
          jnp.where(
              matching,
              jnp.int8(AbilityPhase.SELECTION_PICK),
              jnp.int8(AbilityPhase.BOTTOM_DECK),
          ),
          stepped.ab_phase,
      )
  )
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(ok & ~revealed, a, b),
      cleared,
      stepped,
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_COST_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_confirm_azk01_058_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast confirm for AZK01-058 after-attacking into effect selection."""
  from azuki_jax.abilities.cards_impl import sacrifice_card
  from azuki_jax.abilities.passives import recompute_passives

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  stepped = sacrifice_card(state, owner, src, do=do_action)
  stepped = recompute_passives(stepped)
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(1), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.CONFIRM_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_058_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast single target effect for AZK01-058: +2 attack until EOT."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_attack_modifier
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot, leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  target_index = jnp.clip(action[1].astype(jnp.int32), 0, 11)

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  friendly_garden = target_index < GARDEN_SIZE
  enemy_garden = (target_index >= GARDEN_SIZE) & (target_index < 2 * GARDEN_SIZE)
  own_leader = target_index == 2 * GARDEN_SIZE
  enemy_leader = target_index == (2 * GARDEN_SIZE + 1)

  garden_player = jnp.where(friendly_garden, owner, opp)
  garden_slot = jnp.where(
      friendly_garden,
      target_index,
      target_index - GARDEN_SIZE,
  )
  garden_inst = card_at_slot(state, garden_player, Zone.GARDEN, garden_slot)
  target_player = jnp.where(
      own_leader | friendly_garden,
      owner,
      opp,
  ).astype(jnp.int32)
  target_inst = jnp.where(
      friendly_garden | enemy_garden,
      garden_inst,
      jnp.where(own_leader, leader_instance(state, owner), leader_instance(state, opp)),
  ).astype(jnp.int32)
  safe_target = jnp.maximum(target_inst, 0)
  safe_player = jnp.clip(target_player, 0, 1)
  target_def = state.def_id[safe_player, safe_target]
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)],
      jnp.int8(-1),
  )
  valid_target = (
      ((own_leader | enemy_leader) & (target_inst >= 0))
      | (
          friendly_garden
          & (target_inst >= 0)
          & (target_type == CardType.ENTITY)
      )
  )
  do_action = ~(did_reset | zero_legal) & valid_target

  stepped = apply_attack_modifier(
      state, safe_player, safe_target, 2, expires_eot=True, do=do_action
  )
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_007_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-007 effect: friendly garden entity gets +1 ATK EOT."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_attack_modifier
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-007"]
  target = card_at_slot(state, owner, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)],
      jnp.int8(-1),
  )
  valid_target = (target >= 0) & (target_type == CardType.ENTITY)
  do_action = (
      ~(did_reset | zero_legal)
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & source_ok
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
      & valid_target
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_action, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_action, owner.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = apply_attack_modifier(
      stepped, owner, safe_target, 1, expires_eot=True, do=do_action
  )
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action, a, b), cleared, state
  )
  stepped = recompute_passives(stepped)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_stt03_011_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT03-011 optional effect: destroy one clean low-base-HP enemy garden entity."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import destroy_card
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT03-011"])
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )

  target_index = action[1].astype(jnp.int32)
  target = card_at_slot(state, opp, Zone.GARDEN, target_index)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_ok = (
      (target_index >= 0)
      & (target_index < GARDEN_SIZE)
      & (target >= 0)
      & (state.zone[opp, safe_target] == Zone.GARDEN)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_BASE_STATS)[safe_target_def]
      & (jnp.asarray(cards.BASE_HP)[safe_target_def].astype(jnp.int32) <= 2)
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  do_skip = jnp.bool_(False)

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = destroy_card(stepped, opp, safe_target, do=do_select)
  cleared = _clear_context(stepped)
  finish = do_select | do_skip
  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), cleared, state)
  stepped = recompute_passives(stepped)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_070_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-070 effect: enemy garden entity gets -1 ATK EOT."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_attack_modifier, deal_effect_damage
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot, tap

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-070"]
  target = card_at_slot(state, opp, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)],
      jnp.int8(-1),
  )
  valid_target = (
      (target >= 0)
      & (target_type == CardType.ENTITY)
      & (state.cur_hp[opp, safe_target] > 0)
  )
  do_action = (
      ~(did_reset | zero_legal)
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & source_ok
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
      & valid_target
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_action, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_action, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = tap(stepped, owner, src, do=do_action)
  stepped = deal_effect_damage(
      stepped, owner, src, 1, do=do_action, allow_redirect=False
  )
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied)
  )
  stepped = apply_attack_modifier(
      stepped, opp, safe_target, -1, expires_eot=True, do=do_action
  )
  cleared = recompute_passives(_clear_context(stepped))
  cleared = _close_response_combat_if_idle(cleared)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_059_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast triggered effect for AZK01-059: another garden entity gets +1 ATK."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_attack_modifier
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot
  from azuki_jax.engine.triggers import TIMING_WHEN_TAKES_DAMAGE, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-059"]
  target = card_at_slot(state, owner, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)],
      jnp.int8(-1),
  )
  valid_target = (
      (target >= 0)
      & (safe_target != src)
      & (target_type == CardType.ENTITY)
  )
  do_action = (
      ~(did_reset | zero_legal)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & source_ok
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
      & valid_target
  )

  stepped = apply_attack_modifier(
      state, owner, safe_target, 1, expires_eot=True, do=do_action
  )
  once_used = stepped.once_per_turn_used[owner, src]
  stepped = stepped._replace(
      once_per_turn_used=stepped.once_per_turn_used.at[owner, src].set(
          jnp.where(do_action, once_used | jnp.uint8(1), once_used)
      )
  )
  cleared = _clear_context(stepped)
  trig_src = jnp.maximum(cleared.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(cleared.trig_owner[0].astype(jnp.int32), 0)
  queued_azk01_059 = (
      do_action
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (cleared.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-059"])
      & ((cleared.once_per_turn_used[trig_owner, trig_src] & jnp.uint8(1)) == 0)
  )
  popped, next_src, next_owner, next_timing = pop_effect(cleared)
  begun = resolve_triggered_effect(popped, next_src, next_owner, next_timing)
  cleared = jax.tree.map(
      lambda a, b: jnp.where(queued_azk01_059, a, b), begun, cleared
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_stt04_001_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT04-001 effect: damage a friendly board entity, then +1 ATK EOT."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import (
      apply_attack_modifier,
      deal_effect_damage,
  )
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.triggers import TIMING_WHEN_TAKES_DAMAGE, pop_effect
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = state.def_id[owner, src] == cards.CODE_TO_ID["STT04-001"]
  target_index = action[1].astype(jnp.int32)
  garden_target = card_at_slot(state, owner, Zone.GARDEN, target_index)
  alley_target = card_at_slot(state, owner, Zone.ALLEY, target_index - GARDEN_SIZE)
  target = jnp.where(target_index < GARDEN_SIZE, garden_target, alley_target)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  target_zone = state.zone[owner, safe_target]
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)],
      jnp.int8(-1),
  )
  valid_target = (
      (target >= 0)
      & ((target_zone == Zone.GARDEN) | (target_zone == Zone.ALLEY))
      & (target_type == CardType.ENTITY)
      & (state.cur_hp[owner, safe_target] > 0)
  )
  do_action = (
      ~(did_reset | zero_legal)
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & source_ok
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
      & valid_target
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_action, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_action, owner.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = deal_effect_damage(stepped, owner, safe_target, 1, do=do_action)
  still_friendly = (
      (stepped.zone[owner, safe_target] == Zone.GARDEN)
      | (stepped.zone[owner, safe_target] == Zone.ALLEY)
  ) & (
      jnp.asarray(cards.TYPE)[jnp.maximum(stepped.def_id[owner, safe_target], 0)]
      == CardType.ENTITY
  )
  stepped = apply_attack_modifier(
      stepped,
      owner,
      safe_target,
      1,
      expires_eot=True,
      do=do_action & still_friendly,
  )
  cleared = recompute_passives(_clear_context(stepped))
  trig_src = jnp.maximum(cleared.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(cleared.trig_owner[0].astype(jnp.int32), 0)
  begin_azk01_062 = (
      do_action
      & (cleared.redirect_count > 0)
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (cleared.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-062"])
  )
  popped, src2, owner2, timing2 = pop_effect(cleared)
  begun = resolve_triggered_effect(popped, src2, owner2, timing2)
  cleared = jax.tree.map(
      lambda a, b: jnp.where(begin_azk01_062, a, b), begun, cleared
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_stt02_011_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-011 effect: grant EffectImmune 2 to friendly garden entity."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_effect_immune
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = state.def_id[owner, src] == cards.CODE_TO_ID["STT02-011"]
  target = card_at_slot(state, owner, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)],
      jnp.int8(-1),
  )
  valid_target = (
      (target >= 0)
      & (safe_target != src)
      & (target_type == CardType.ENTITY)
      & (state.cur_hp[owner, safe_target] > 0)
  )
  do_action = (
      ~(did_reset | zero_legal)
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & source_ok
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
      & valid_target
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_action, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_action, owner.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = apply_effect_immune(stepped, owner, safe_target, 2, do=do_action)
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_effect_azk01_123_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-123 effect: friendly garden entity gets +1 HP EOT."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_health_modifier
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-123"]
  target = card_at_slot(state, owner, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)],
      jnp.int8(-1),
  )
  valid_target = (
      (target >= 0)
      & (target_type == CardType.ENTITY)
      & (state.cur_hp[owner, safe_target] > 0)
  )
  do_action = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.MAIN)
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & source_ok
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
      & valid_target
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_action, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_action, owner.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = apply_health_modifier(
      stepped, owner, safe_target, 1, expires_eot=True, do=do_action
  )
  once_used = stepped.once_per_turn_used[owner, src]
  stepped = stepped._replace(
      once_per_turn_used=stepped.once_per_turn_used.at[owner, src].set(
          jnp.where(do_action, once_used | jnp.uint8(1), once_used)
      )
  )
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_activate_azk01_103_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-103 activation: enter Earth sacrifice cost selection."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  source = card_at_slot(state, acting, Zone.GARDEN, action[1])
  safe_source = jnp.maximum(source, 0)
  def_row = state.def_id[acting]
  safe_defs = jnp.maximum(def_row, 0)
  earth_garden = (
      (state.zone[acting] == Zone.GARDEN)
      & (def_row >= 0)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
      & (jnp.asarray(cards.ELEMENT)[safe_defs] == 3)
      & ~state.tapped[acting]
  )
  cost_available = jnp.any(
      earth_garden & (jnp.arange(state.zone.shape[1]) != safe_source)
  )
  do_action = (
      do_action
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (action[0] == Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
      & (action[2] == 0)
      & (source >= 0)
      & (state.def_id[acting, safe_source] == cards.CODE_TO_ID["AZK01-103"])
      & (state.frozen_dur[acting, safe_source] == 0)
      & ~state.tapped[acting, safe_source]
      & cost_available
  )

  stepped = state._replace(
      ab_source=jnp.where(do_action, safe_source.astype(jnp.int8), state.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), state.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), state.ab_slot),
      ab_is_optional=jnp.where(do_action, False, state.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, False, state.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), state.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, state.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), state.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(1), state.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), state.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(1), state.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(1), state.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_targets, -1),
          state.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_target_players, -1),
          state.ab_eff_target_players,
      ),
      ab_phase=jnp.where(
          do_action, jnp.int8(AbilityPhase.COST_SELECTION), state.ab_phase
      ),
  )
  stepped = recompute_passives(stepped)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_cost_azk01_103_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-103 cost: tap source, sacrifice Earth garden target."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import sacrifice_card
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.COST_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-103"])
      & ~state.ab_costs_applied
      & (state.ab_cost_selected == 0)
      & (state.ab_cost_max == 1)
  )
  target = card_at_slot(state, owner, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_ok = (
      (target >= 0)
      & (target != src)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & (jnp.asarray(cards.ELEMENT)[safe_target_def] == 3)
      & ~state.tapped[owner, safe_target]
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_COST_TARGET)
      & source_ok
      & target_ok
  )
  hp = state.cur_hp[owner, safe_target].astype(jnp.int16)
  damage = jnp.clip(hp, 0, 5)
  draw_after = hp >= 3

  stepped = state._replace(
      tapped=state.tapped.at[owner, src].set(
          jnp.where(do_select, True, state.tapped[owner, src])
      ),
      ab_cost_targets=state.ab_cost_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_cost_targets[0])
      ),
      ab_cost_target_players=state.ab_cost_target_players.at[0].set(
          jnp.where(do_select, owner.astype(jnp.int8), state.ab_cost_target_players[0])
      ),
      ab_cost_selected=jnp.where(
          do_select, jnp.int8(1), state.ab_cost_selected
      ),
      ab_scratch=state.ab_scratch.at[0]
      .set(jnp.where(do_select, damage, state.ab_scratch[0]))
      .at[1]
      .set(jnp.where(do_select & draw_after, jnp.int16(1), jnp.int16(0))),
  )
  stepped = sacrifice_card(stepped, owner, safe_target, do_select)
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(do_select, True, stepped.ab_costs_applied),
      ab_phase=jnp.where(
          do_select, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_COST_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_103_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-103 effect: damage either leader, then optional draw."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage, draw_with_deckout
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-103"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )
  target_index = action[1].astype(jnp.int32)
  target_player = jnp.where(target_index == 0, owner, opp)
  target = leader_instance(state, target_player)
  safe_target = jnp.maximum(target, 0)
  target_ok = (
      (target_index >= 0)
      & (target_index <= 1)
      & (target >= 0)
      & (state.zone[target_player, safe_target] == Zone.LEADER)
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, target_player.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  damage = stepped.ab_scratch[0].astype(jnp.int16)
  stepped = deal_effect_damage(
      stepped, target_player, safe_target, damage, do_select & (damage > 0)
  )
  stepped = draw_with_deckout(
      stepped, owner, 1, do_select & (stepped.ab_scratch[1] != 0)
  )
  cleared = _clear_context(stepped)
  cleared = recompute_passives(cleared)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_select, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_activate_azk01_105_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-105 main activation: sacrifice, then damage target prompt."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import sacrifice_card
  from azuki_jax.engine.helpers import card_at_slot, leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  source = card_at_slot(state, acting, Zone.GARDEN, action[1])
  safe_source = jnp.maximum(source, 0)
  opp = (acting + 1) % 2
  leader = leader_instance(state, opp)
  enemy_garden_entities = (
      (state.zone[opp] == Zone.GARDEN)
      & (state.def_id[opp] >= 0)
      & (
          jnp.asarray(cards.TYPE)[jnp.maximum(state.def_id[opp], 0)]
          == CardType.ENTITY
      )
  )
  target_available = (leader >= 0) | jnp.any(enemy_garden_entities)
  damage = state.cur_hp[acting, safe_source].astype(jnp.int16)
  do_action = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (action[0] == Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
      & (action[2] == 0)
      & (source >= 0)
      & (state.def_id[acting, safe_source] == cards.CODE_TO_ID["AZK01-105"])
      & (state.frozen_dur[acting, safe_source] == 0)
      & (state.cur_hp[acting, safe_source] > 0)
      & target_available
  )

  stepped = state._replace(
      ab_source=jnp.where(do_action, safe_source.astype(jnp.int8), state.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), state.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), state.ab_slot),
      ab_is_optional=jnp.where(do_action, False, state.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, False, state.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), state.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, state.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), state.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(0), state.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), state.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(1), state.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(1), state.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_targets, -1),
          state.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_target_players, -1),
          state.ab_eff_target_players,
      ),
      ab_scratch=state.ab_scratch.at[0].set(
          jnp.where(do_action, damage, state.ab_scratch[0])
      ),
  )
  stepped = sacrifice_card(stepped, acting, safe_source, do=do_action)
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied),
      ab_phase=jnp.where(
          do_action, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_105_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-105 effect: stored HP damage to enemy leader/garden."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot, leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-105"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )

  target_index = action[1].astype(jnp.int32)
  target_is_leader = target_index == GARDEN_SIZE
  garden_slot = jnp.clip(target_index, 0, GARDEN_SIZE - 1)
  garden_target = card_at_slot(state, opp, Zone.GARDEN, garden_slot)
  leader = leader_instance(state, opp)
  target = jnp.where(target_is_leader, leader, garden_target)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[safe_target_def],
      jnp.int8(-1),
  )
  target_ok = (
      (target_index >= 0)
      & (target_index <= GARDEN_SIZE)
      & (target >= 0)
      & (
          (
              target_is_leader
              & (state.zone[opp, safe_target] == Zone.LEADER)
          )
          | (
              ~target_is_leader
              & (state.zone[opp, safe_target] == Zone.GARDEN)
              & (target_type == CardType.ENTITY)
          )
      )
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  damage = stepped.ab_scratch[0].astype(jnp.int16)
  stepped = deal_effect_damage(
      stepped, opp, safe_target, damage, do_select & (damage > 0)
  )
  cleared = recompute_passives(_clear_context(stepped))
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_select, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_stt04_004_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT04-004 effect selection after sacrificed-source cost."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.engine.triggers import TIMING_WHEN_TAKES_DAMAGE, pop_effect
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  target_index = action[1].astype(jnp.int32)
  target_player = jnp.where(target_index < GARDEN_SIZE, owner, opp)
  target_slot = jnp.where(
      target_index < GARDEN_SIZE, target_index, target_index - GARDEN_SIZE
  )
  target = card_at_slot(state, target_player, Zone.GARDEN, target_slot)
  safe_target = jnp.maximum(target, 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT04-004"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )
  target_def = state.def_id[target_player, safe_target]
  target_ok = (
      (target >= 0)
      & (target_index >= 0)
      & (target_index < 2 * GARDEN_SIZE)
      & (target_def >= 0)
      & (
          jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)]
          == CardType.ENTITY
      )
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(
              do_select,
              target_player.astype(jnp.int8),
              state.ab_eff_target_players[0],
          )
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = deal_effect_damage(stepped, target_player, safe_target, 1, do_select)
  cleared = recompute_passives(_clear_context(stepped))
  trig_src = jnp.maximum(cleared.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(cleared.trig_owner[0].astype(jnp.int32), 0)
  begin_azk01_062 = (
      do_select
      & (cleared.redirect_count > 0)
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (cleared.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-062"])
  )
  popped, src2, owner2, timing2 = pop_effect(cleared)
  begun = resolve_triggered_effect(popped, src2, owner2, timing2)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(begin_azk01_062, a, b), begun, cleared
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_stt01_006_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-006 effect selection, then open response/combat gate."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot, leader_instance
  from azuki_jax.engine.phases import combat_resolve, phase_gate
  from azuki_jax.engine.triggers import TIMING_WHEN_ATTACKING, has_queued, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  target_index = action[1].astype(jnp.int32)
  target_is_leader = target_index == GARDEN_SIZE
  garden_target = card_at_slot(state, opp, Zone.GARDEN, target_index)
  leader_target = leader_instance(state, opp)
  target = jnp.where(target_is_leader, leader_target, garden_target)
  safe_target = jnp.maximum(target, 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT01-006"])
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )
  target_ok = (
      (target >= 0)
      & (target_index >= 0)
      & (target_index <= GARDEN_SIZE)
      & (
          target_is_leader
          | (
              jnp.asarray(cards.TYPE)[
                  jnp.maximum(state.def_id[opp, safe_target], 0)
              ]
              == CardType.ENTITY
          )
      )
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = deal_effect_damage(stepped, opp, safe_target, 1, do_select)
  stepped = _clear_context(stepped)
  trig_src = jnp.maximum(stepped.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(stepped.trig_owner[0].astype(jnp.int32), 0)
  stt01_012_trigger = (
      do_select
      & (stepped.trig_count > 0)
      & (stepped.trig_timing[0] == TIMING_WHEN_ATTACKING)
      & (stepped.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["STT01-012"])
  )
  popped, src2, owner2, timing2 = pop_effect(stepped)
  resolved = resolve_triggered_effect(popped, src2, owner2, timing2)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(stt01_012_trigger, a, b), resolved, stepped
  )
  run_gate = do_select & (stepped.winner == -1)
  gated = phase_gate(stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(run_gate, a, b), gated, stepped)
  auto_combat = run_gate & (stepped.phase == Phase.COMBAT_RESOLVE) & ~has_queued(stepped)
  stepped = jax.lax.cond(
      auto_combat,
      lambda st: combat_resolve(st, do=True),
      lambda st: st,
      stepped,
  )
  stepped = recompute_passives(stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_stt03_002_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT03-002 effect selection: timed Defender grant."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_timed_tag_grant
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import (
      GRANT_PHASE_START,
      TAG_DEFENDER,
      card_at_slot,
  )

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  target_index = action[1].astype(jnp.int32)
  target = card_at_slot(state, owner, Zone.GARDEN, target_index)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  portaled = jnp.clip(
      state.ab_scratch[0].astype(jnp.int32), 0, state.def_id.shape[1] - 1
  )
  portaled_def = state.def_id[owner, portaled]
  safe_portaled_def = jnp.maximum(portaled_def, 0)
  gate_power = jnp.where(
      state.ab_scratch[2] == 1,
      jnp.asarray(cards.GATE_POINTS)[safe_portaled_def],
      0,
  ).astype(jnp.int32)
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT03-002"])
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 1)
      & (state.ab_scratch[2] == 1)
  )
  target_ok = (
      (target >= 0)
      & (target_index >= 0)
      & (target_index < GARDEN_SIZE)
      & (target_def >= 0)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_BASE_STATS)[safe_target_def]
      & (jnp.asarray(cards.BASE_HP)[safe_target_def].astype(jnp.int32) <= gate_power)
      & ~jnp.asarray(cards.INHERENT_DEFENDER)[safe_target_def]
      & ~state.grant_defender[owner, safe_target]
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  do_skip = do_action & (action_type == Act.NOOP) & source_ok

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, owner.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped, _ = apply_timed_tag_grant(
      stepped,
      owner,
      safe_target,
      TAG_DEFENDER,
      GRANT_PHASE_START,
      2,
      do_select,
  )
  cleared = recompute_passives(_clear_context(stepped))
  finish = do_select | do_skip
  cleared = _begin_queued_stt03_013_enter_trigger(cleared, finish)
  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), cleared, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_effect_stt04_002_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT04-002 gate effect selection: optional Ragefire buff."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_attack_modifier
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  target_index = action[1].astype(jnp.int32)
  target = card_at_slot(state, owner, Zone.GARDEN, target_index)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  portaled = jnp.clip(
      state.ab_scratch[0].astype(jnp.int32), 0, state.def_id.shape[1] - 1
  )
  portaled_def = state.def_id[owner, portaled]
  safe_portaled_def = jnp.maximum(portaled_def, 0)
  gate_power = jnp.where(
      state.ab_scratch[2] == 1,
      jnp.asarray(cards.GATE_POINTS)[safe_portaled_def],
      0,
  ).astype(jnp.int32)
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT04-002"])
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 1)
      & (state.ab_scratch[2] == 1)
  )
  target_ok = (
      (target >= 0)
      & (target_index >= 0)
      & (target_index < GARDEN_SIZE)
      & (target_def >= 0)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & state.took_damage_turn[owner, safe_target]
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  do_skip = do_action & (action_type == Act.NOOP) & source_ok

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, owner.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = apply_attack_modifier(
      stepped, owner, safe_target, gate_power, expires_eot=True,
      do=do_select & (gate_power > 0),
  )
  cleared = recompute_passives(_clear_context(stepped))
  cleared = _begin_queued_stt03_013_enter_trigger(cleared, do_select | do_skip)
  acted = do_select | do_skip
  stepped = jax.tree.map(lambda a, b: jnp.where(acted, a, b), cleared, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type,
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_stt03_006_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT03-006 effect selection: discard one friendly hand card."""
  from azuki_jax import cards
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot, discard
  from azuki_jax.engine.triggers import TIMING_WHEN_DESTROYED, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  target_index = action[1].astype(jnp.int32)
  target = card_at_slot(state, owner, Zone.HAND, target_index)
  safe_target = jnp.maximum(target, 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT03-006"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )
  target_ok = (
      (target >= 0)
      & (target_index >= 0)
      & (state.zone[owner, safe_target] == Zone.HAND)
  )
  do_select = (
      do_action
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, owner.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = discard(stepped, owner, safe_target, do=do_select)
  stepped = _clear_context(stepped)
  stepped = recompute_passives(stepped)
  trig_src = jnp.maximum(stepped.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(stepped.trig_owner[0].astype(jnp.int32), 0)
  begin_stt03_006 = (
      do_select
      & (stepped.trig_count > 0)
      & (stepped.trig_timing[0] == TIMING_WHEN_DESTROYED)
      & (stepped.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["STT03-006"])
  )
  popped, src2, owner2, timing2 = pop_effect(stepped)
  resolved = resolve_triggered_effect(popped, src2, owner2, timing2)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(begin_stt03_006, a, b), resolved, stepped
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_azk01_003_pick_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-003 reveal pick-to-hand action."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  sel_idx = action[1].astype(jnp.int32)
  idx = jnp.clip(sel_idx, 0, state.ab_sel_cards.shape[0] - 1)
  inst = state.ab_sel_cards[idx]
  target = jnp.maximum(inst.astype(jnp.int32), 0)

  black_jade = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("BlackJade")], jnp.bool_
  )
  source_def = state.def_id[owner, src]
  target_def = state.def_id[owner, target]
  ok = (
      (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_def == cards.CODE_TO_ID["AZK01-003"])
      & (sel_idx >= 0)
      & (sel_idx < state.ab_sel_count.astype(jnp.int32))
      & (inst >= 0)
      & black_jade[jnp.maximum(target_def, 0)]
      & (target_def != cards.CODE_TO_ID["AZK01-003"])
  )
  do_action = do_action & ok

  stepped = state._replace(
      ab_sel_picked=state.ab_sel_picked.at[0].set(
          jnp.where(do_action, inst, state.ab_sel_picked[0])
      ),
      ab_sel_picked_count=jnp.where(
          do_action, jnp.int8(1), state.ab_sel_picked_count
      ),
      ab_sel_cards=state.ab_sel_cards.at[idx].set(
          jnp.where(do_action, jnp.int8(-1), state.ab_sel_cards[idx])
      ),
  )
  stepped = sel_mod.move_picked_to_hand(stepped, do=do_action)
  remaining = sel_mod.remaining_count(stepped)
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action & (remaining > 0),
          jnp.int8(AbilityPhase.BOTTOM_DECK),
          stepped.ab_phase,
      )
  )
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action & (remaining == 0), a, b),
      cleared,
      stepped,
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_FROM_SELECTION, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_select_azk01_031_pick_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-021/AZK01-031 reveal pick of one matching card to hand."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod
  driftward = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Driftward")], jnp.bool_
  )

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  sel_idx = action[1].astype(jnp.int32)
  idx = jnp.clip(sel_idx, 0, state.ab_sel_cards.shape[0] - 1)
  inst = state.ab_sel_cards[idx]
  target = jnp.maximum(inst.astype(jnp.int32), 0)

  source_def = state.def_id[owner, src]
  target_def = state.def_id[owner, target]
  source_azk01_021 = source_def == cards.CODE_TO_ID["AZK01-021"]
  source_azk01_031 = source_def == cards.CODE_TO_ID["AZK01-031"]
  ok = (
      (action[0] == Act.SELECT_FROM_SELECTION)
      & (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_azk01_021 | source_azk01_031)
      & (sel_idx >= 0)
      & (sel_idx < state.ab_sel_count.astype(jnp.int32))
      & (inst >= 0)
      & (
          (
              source_azk01_031
              & (jnp.asarray(cards.ELEMENT)[jnp.maximum(target_def, 0)] == 2)
          )
          | (source_azk01_021 & driftward[jnp.maximum(target_def, 0)])
      )
  )
  do_action = do_action & ok

  stepped = state._replace(
      ab_sel_picked=state.ab_sel_picked.at[0].set(
          jnp.where(do_action, inst, state.ab_sel_picked[0])
      ),
      ab_sel_picked_count=jnp.where(
          do_action, jnp.int8(1), state.ab_sel_picked_count
      ),
      ab_sel_cards=state.ab_sel_cards.at[idx].set(
          jnp.where(do_action, jnp.int8(-1), state.ab_sel_cards[idx])
      ),
  )
  stepped = sel_mod.move_picked_to_hand(stepped, do=do_action)
  remaining = sel_mod.remaining_count(stepped)
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action & (remaining > 0),
          jnp.int8(AbilityPhase.BOTTOM_DECK),
          stepped.ab_phase,
      )
  )
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action & (remaining == 0), a, b),
      cleared,
      stepped,
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_FROM_SELECTION, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations
def step_select_stt02_003_pick_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-003 selection pick to hand."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  sel_idx = action[1].astype(jnp.int32)
  idx = jnp.clip(sel_idx, 0, state.ab_sel_cards.shape[0] - 1)
  inst = state.ab_sel_cards[idx]
  target = jnp.maximum(inst.astype(jnp.int32), 0)

  watercrafting = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Watercrafting")], jnp.bool_
  )
  source_def = state.def_id[owner, src]
  target_def = state.def_id[owner, target]
  ok = (
      (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_def == cards.CODE_TO_ID["STT02-003"])
      & (sel_idx >= 0)
      & (sel_idx < state.ab_sel_count.astype(jnp.int32))
      & (inst >= 0)
      & watercrafting[jnp.maximum(target_def, 0)]
  )
  do_action = do_action & ok

  stepped = state._replace(
      ab_sel_picked=state.ab_sel_picked.at[0].set(
          jnp.where(do_action, inst, state.ab_sel_picked[0])
      ),
      ab_sel_picked_count=jnp.where(
          do_action, jnp.int8(1), state.ab_sel_picked_count
      ),
      ab_sel_cards=state.ab_sel_cards.at[idx].set(
          jnp.where(do_action, jnp.int8(-1), state.ab_sel_cards[idx])
      ),
  )
  stepped = sel_mod.move_picked_to_hand(stepped, do=do_action)
  remaining = sel_mod.remaining_count(stepped)
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action & (remaining > 0),
          jnp.int8(AbilityPhase.BOTTOM_DECK),
          stepped.ab_phase,
      )
  )
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action & (remaining == 0), a, b),
      cleared,
      stepped,
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_FROM_SELECTION, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_stt02_013_pick_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-013/AZK01-092 reveal pick to hand or board."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.apply import _enter_board_slot
  from azuki_jax.engine.helpers import (
      _detach_from_location,
      discard,
      passive_zone_event,
  )
  from azuki_jax.engine.triggers import queue_enter_garden, queue_on_play

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_step = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  sel_idx = action[1].astype(jnp.int32)
  slot = action[2].astype(jnp.int32)
  idx = jnp.clip(sel_idx, 0, state.ab_sel_cards.shape[0] - 1)
  inst = state.ab_sel_cards[idx]
  target = jnp.maximum(inst.astype(jnp.int32), 0)

  source_def = state.def_id[owner, src]
  source_is_stt02_013 = source_def == cards.CODE_TO_ID["STT02-013"]
  source_is_azk01_092 = source_def == cards.CODE_TO_ID["AZK01-092"]
  target_def = state.def_id[owner, target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_is_stt02_003 = target_def == cards.CODE_TO_ID["STT02-003"]
  target_type = jnp.asarray(cards.TYPE)[safe_target_def]
  target_is_entity = target_type == CardType.ENTITY
  water_le2 = (
      jnp.asarray(cards.HAS_IKZ_COST)[safe_target_def]
      & (jnp.asarray(cards.IKZ_COST)[safe_target_def] <= 2)
      & (jnp.asarray(cards.ELEMENT)[safe_target_def] == 2)
  )
  source_ok = (
      (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_is_stt02_013 | source_is_azk01_092)
      & (state.ab_sel_pick_max == 1)
      & (state.ab_sel_picked_count == 0)
      & (sel_idx >= 0)
      & (sel_idx < state.ab_sel_count.astype(jnp.int32))
      & (inst >= 0)
      & (state.zone[owner, target] == Zone.SELECTION)
      & water_le2
  )
  to_hand = action_type == Act.SELECT_FROM_SELECTION
  to_garden = action_type == Act.SELECT_TO_GARDEN
  to_alley = action_type == Act.SELECT_TO_ALLEY

  alley_match = (state.zone[owner] == Zone.ALLEY) & (
      state.zpos[owner] == slot.astype(jnp.int8)
  )
  alley_occupied = jnp.any(alley_match)
  alley_occupied_inst = jnp.where(alley_occupied, jnp.argmax(alley_match), -1)
  alley_full = jnp.sum(state.zone[owner] == Zone.ALLEY, dtype=jnp.int32) >= GARDEN_SIZE
  alley_slot_ok = (slot >= 0) & (slot < GARDEN_SIZE) & (
      ~alley_occupied | alley_full
  )

  garden_match = (state.zone[owner] == Zone.GARDEN) & (
      state.zpos[owner] == slot.astype(jnp.int8)
  )
  garden_occupied = jnp.any(garden_match)
  garden_full = jnp.sum(state.zone[owner] == Zone.GARDEN, dtype=jnp.int32) >= GARDEN_SIZE
  garden_slot_ok = (slot >= 0) & (slot < GARDEN_SIZE) & (
      ~garden_occupied | garden_full
  )

  do_pick = do_step & source_ok & to_hand
  do_garden = (
      do_step
      & source_ok
      & source_is_azk01_092
      & to_garden
      & target_is_entity
      & garden_slot_ok
  )
  do_place = (
      do_step
      & source_ok
      & to_alley
      & target_is_entity
      & alley_slot_ok
  )
  finish = do_pick | do_garden | do_place

  gardened = _enter_board_slot(
      state, owner, target, Zone.GARDEN, slot, do_garden
  )
  gardened = gardened._replace(
      entities_played_garden_turn=gardened.entities_played_garden_turn.at[
          owner
      ].add(do_garden.astype(jnp.uint8)),
      cards_played_turn=gardened.cards_played_turn.at[owner].add(
          do_garden.astype(jnp.uint8)
      ),
      next_play_cost_reduction=gardened.next_play_cost_reduction.at[owner].set(
          jnp.where(do_garden, 0, gardened.next_play_cost_reduction[owner])
      ),
  )
  gardened = queue_enter_garden(gardened, owner, target, do=do_garden)
  gardened = queue_on_play(gardened, owner, target, do=do_garden)
  gardened = _apply_simple_implemented_play_trigger(
      gardened, owner, target, do_garden
  )
  gardened = _clear_simple_unimplemented_play_triggers(
      gardened, owner, target, Zone.GARDEN, do_garden
  )
  gardened = recompute_passives(gardened)

  placed = discard(
      state,
      owner,
      jnp.maximum(alley_occupied_inst, 0),
      reason_replacement=True,
      ignore_godmode=True,
      do=do_place & alley_occupied & alley_full,
  )
  placed = _detach_from_location(placed, owner, target, do_place)
  placed = placed._replace(
      zone=placed.zone.at[owner, target].set(
          jnp.where(do_place, jnp.int8(Zone.ALLEY), placed.zone[owner, target])
      ),
      zpos=placed.zpos.at[owner, target].set(
          jnp.where(do_place, slot.astype(jnp.int8), placed.zpos[owner, target])
      ),
      board_seq=placed.board_seq.at[owner, target].set(
          jnp.where(do_place, placed.seq_counter, placed.board_seq[owner, target])
      ),
      seq_counter=(placed.seq_counter + jnp.where(do_place, 1, 0)).astype(
          jnp.int16
      ),
      tapped=placed.tapped.at[owner, target].set(
          jnp.where(do_place, False, placed.tapped[owner, target])
      ),
      cooldown=placed.cooldown.at[owner, target].set(
          jnp.where(do_place, 0, placed.cooldown[owner, target])
      ),
      entities_played_alley_turn=placed.entities_played_alley_turn.at[
          owner
      ].add(do_place.astype(jnp.uint8)),
      cards_played_turn=placed.cards_played_turn.at[owner].add(
          do_place.astype(jnp.uint8)
      ),
      next_play_cost_reduction=placed.next_play_cost_reduction.at[owner].set(
          jnp.where(do_place, 0, placed.next_play_cost_reduction[owner])
      ),
  )
  placed = passive_zone_event(placed, owner, Zone.ALLEY, target, True, do=do_place)
  placed = queue_on_play(placed, owner, target, do=do_place & target_is_stt02_003)
  placed = _apply_simple_implemented_play_trigger(
      placed, owner, target, do_place & ~target_is_stt02_003
  )
  placed = _clear_simple_unimplemented_play_triggers(
      placed, owner, target, Zone.ALLEY, do_place & ~target_is_stt02_003
  )
  placed = recompute_passives(placed)

  stepped = jax.tree.map(lambda a, b: jnp.where(do_garden, a, b), gardened, state)
  stepped = jax.tree.map(lambda a, b: jnp.where(do_place, a, b), placed, stepped)

  stepped = stepped._replace(
      ab_sel_picked=stepped.ab_sel_picked.at[0].set(
          jnp.where(finish, inst, stepped.ab_sel_picked[0])
      ),
      ab_sel_picked_count=jnp.where(
          finish, jnp.int8(1), stepped.ab_sel_picked_count
      ),
      ab_sel_cards=stepped.ab_sel_cards.at[idx].set(
          jnp.where(finish, jnp.int8(-1), stepped.ab_sel_cards[idx])
      ),
  )
  stepped = sel_mod.move_picked_to_hand(stepped, do=do_pick)
  remaining = sel_mod.remaining_count(stepped)
  stepped = stepped._replace(
      ab_phase=jnp.where(
          finish & (remaining > 0),
          jnp.int8(AbilityPhase.BOTTOM_DECK),
          stepped.ab_phase,
      )
  )
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(finish & (remaining == 0), a, b),
      cleared,
      stepped,
  )
  # STT02-013/AZK01-092 use the C helper variant that moves a picked card to
  # hand if it is still parented to the selection zone during completion. C
  # applies TO_GARDEN/TO_ALLEY reparent writes in the readonly ability stage, so
  # the completion hook still observes the old selection parent and the later
  # hand reparent wins. Keep the placement side effects above, then bounce the
  # card and refresh passives from the final zone layout.
  bounce_to_hand = do_garden | do_place
  bounced = sel_mod._bounce_pick_to_hand(stepped, owner, target, bounce_to_hand)
  bounced = recompute_passives(bounced)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(bounce_to_hand, a, b),
      bounced,
      stepped,
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type.astype(jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_azk01_033_pick_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-033/AZK01-069 selection pick to hand."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  sel_idx = action[1].astype(jnp.int32)
  idx = jnp.clip(sel_idx, 0, state.ab_sel_cards.shape[0] - 1)
  inst = state.ab_sel_cards[idx]
  target = jnp.maximum(inst.astype(jnp.int32), 0)

  steelborn = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Steelborn")], jnp.bool_
  )
  beanz = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Beanz")], jnp.bool_
  )
  source_def = state.def_id[owner, src]
  target_def = state.def_id[owner, target]
  source_azk01_033 = source_def == cards.CODE_TO_ID["AZK01-033"]
  source_azk01_069 = source_def == cards.CODE_TO_ID["AZK01-069"]
  target_safe_def = jnp.maximum(target_def, 0)
  subtype_ok = (
      (source_azk01_033 & steelborn[target_safe_def])
      | (source_azk01_069 & beanz[target_safe_def])
  )
  ok = (
      (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_azk01_033 | source_azk01_069)
      & (sel_idx >= 0)
      & (sel_idx < state.ab_sel_count.astype(jnp.int32))
      & (inst >= 0)
      & subtype_ok
  )
  do_action = do_action & ok

  stepped = state._replace(
      ab_sel_picked=state.ab_sel_picked.at[0].set(
          jnp.where(do_action, inst, state.ab_sel_picked[0])
      ),
      ab_sel_picked_count=jnp.where(
          do_action, jnp.int8(1), state.ab_sel_picked_count
      ),
      ab_sel_cards=state.ab_sel_cards.at[idx].set(
          jnp.where(do_action, jnp.int8(-1), state.ab_sel_cards[idx])
      ),
  )
  stepped = sel_mod.move_picked_to_hand(stepped, do=do_action)
  remaining = sel_mod.remaining_count(stepped)
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action & (remaining > 0),
          jnp.int8(AbilityPhase.BOTTOM_DECK),
          stepped.ab_phase,
      )
  )
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action & (remaining == 0), a, b),
      cleared,
      stepped,
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_FROM_SELECTION, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_azk01_045_pick_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-045 selection pick to hand."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  sel_idx = action[1].astype(jnp.int32)
  idx = jnp.clip(sel_idx, 0, state.ab_sel_cards.shape[0] - 1)
  inst = state.ab_sel_cards[idx]
  target = jnp.maximum(inst.astype(jnp.int32), 0)

  obsidian = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Obsidian")], jnp.bool_
  )
  source_def = state.def_id[owner, src]
  target_def = state.def_id[owner, target]
  ok = (
      (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_def == cards.CODE_TO_ID["AZK01-045"])
      & (sel_idx >= 0)
      & (sel_idx < state.ab_sel_count.astype(jnp.int32))
      & (inst >= 0)
      & obsidian[jnp.maximum(target_def, 0)]
  )
  do_action = do_action & ok

  stepped = state._replace(
      ab_sel_picked=state.ab_sel_picked.at[0].set(
          jnp.where(do_action, inst, state.ab_sel_picked[0])
      ),
      ab_sel_picked_count=jnp.where(
          do_action, jnp.int8(1), state.ab_sel_picked_count
      ),
      ab_sel_cards=state.ab_sel_cards.at[idx].set(
          jnp.where(do_action, jnp.int8(-1), state.ab_sel_cards[idx])
      ),
  )
  stepped = sel_mod.move_picked_to_hand(stepped, do=do_action)
  remaining = sel_mod.remaining_count(stepped)
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action & (remaining > 0),
          jnp.int8(AbilityPhase.BOTTOM_DECK),
          stepped.ab_phase,
      )
  )
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action & (remaining == 0), a, b),
      cleared,
      stepped,
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_FROM_SELECTION, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_azk01_056_pick_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-056 selection pick to hand."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  sel_idx = action[1].astype(jnp.int32)
  idx = jnp.clip(sel_idx, 0, state.ab_sel_cards.shape[0] - 1)
  inst = state.ab_sel_cards[idx]
  target = jnp.maximum(inst.astype(jnp.int32), 0)

  scorchweaver = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Scorchweaver")], jnp.bool_
  )
  source_def = state.def_id[owner, src]
  target_def = state.def_id[owner, target]
  ok = (
      (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_def == cards.CODE_TO_ID["AZK01-056"])
      & (sel_idx >= 0)
      & (sel_idx < state.ab_sel_count.astype(jnp.int32))
      & (inst >= 0)
      & scorchweaver[jnp.maximum(target_def, 0)]
  )
  do_action = do_action & ok

  stepped = state._replace(
      ab_sel_picked=state.ab_sel_picked.at[0].set(
          jnp.where(do_action, inst, state.ab_sel_picked[0])
      ),
      ab_sel_picked_count=jnp.where(
          do_action, jnp.int8(1), state.ab_sel_picked_count
      ),
      ab_sel_cards=state.ab_sel_cards.at[idx].set(
          jnp.where(do_action, jnp.int8(-1), state.ab_sel_cards[idx])
      ),
  )
  stepped = sel_mod.move_picked_to_hand(stepped, do=do_action)
  remaining = sel_mod.remaining_count(stepped)
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action & (remaining > 0),
          jnp.int8(AbilityPhase.BOTTOM_DECK),
          stepped.ab_phase,
      )
  )
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action & (remaining == 0), a, b),
      cleared,
      stepped,
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_FROM_SELECTION, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_stt04_005_pick_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT04-005 selection pick to hand."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  sel_idx = action[1].astype(jnp.int32)
  idx = jnp.clip(sel_idx, 0, state.ab_sel_cards.shape[0] - 1)
  inst = state.ab_sel_cards[idx]
  target = jnp.maximum(inst.astype(jnp.int32), 0)

  pyreskin = jnp.asarray(
      cards.SUBTYPE_MATRIX[:, cards.subtype_index("Pyreskin")], jnp.bool_
  )
  source_def = state.def_id[owner, src]
  target_def = state.def_id[owner, target]
  ok = (
      (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_def == cards.CODE_TO_ID["STT04-005"])
      & (sel_idx >= 0)
      & (sel_idx < state.ab_sel_count.astype(jnp.int32))
      & (inst >= 0)
      & pyreskin[jnp.maximum(target_def, 0)]
  )
  do_action = do_action & ok

  stepped = state._replace(
      ab_sel_picked=state.ab_sel_picked.at[0].set(
          jnp.where(do_action, inst, state.ab_sel_picked[0])
      ),
      ab_sel_picked_count=jnp.where(
          do_action, jnp.int8(1), state.ab_sel_picked_count
      ),
      ab_sel_cards=state.ab_sel_cards.at[idx].set(
          jnp.where(do_action, jnp.int8(-1), state.ab_sel_cards[idx])
      ),
  )
  stepped = sel_mod.move_picked_to_hand(stepped, do=do_action)
  remaining = sel_mod.remaining_count(stepped)
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action & (remaining > 0),
          jnp.int8(AbilityPhase.BOTTOM_DECK),
          stepped.ab_phase,
      )
  )
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action & (remaining == 0), a, b),
      cleared,
      stepped,
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_FROM_SELECTION, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_select_stt01_004_pick_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-004 revealed weapon pick to hand."""
  from azuki_jax import cards
  from azuki_jax.abilities import runtime
  from azuki_jax.abilities import selection as sel_mod

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  sel_idx = action[1].astype(jnp.int32)
  idx = jnp.clip(sel_idx, 0, state.ab_sel_cards.shape[0] - 1)
  inst = state.ab_sel_cards[idx]
  target = jnp.maximum(inst.astype(jnp.int32), 0)

  source_def = state.def_id[owner, src]
  target_def = state.def_id[owner, target]
  ok = (
      (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_def == cards.CODE_TO_ID["STT01-004"])
      & (sel_idx >= 0)
      & (sel_idx < state.ab_sel_count.astype(jnp.int32))
      & (inst >= 0)
      & (jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)] == CardType.WEAPON)
  )
  do_action = do_action & ok

  stepped = state._replace(
      ab_sel_picked=state.ab_sel_picked.at[0].set(
          jnp.where(do_action, inst, state.ab_sel_picked[0])
      ),
      ab_sel_picked_count=jnp.where(
          do_action, jnp.int8(1), state.ab_sel_picked_count
      ),
      ab_sel_cards=state.ab_sel_cards.at[idx].set(
          jnp.where(do_action, jnp.int8(-1), state.ab_sel_cards[idx])
      ),
  )
  stepped = sel_mod.move_picked_to_hand(stepped, do=do_action)
  remaining = sel_mod.remaining_count(stepped)
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action & (remaining > 0),
          jnp.int8(AbilityPhase.BOTTOM_DECK),
          stepped.ab_phase,
      )
  )
  cleared = runtime._clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action & (remaining == 0), a, b),
      cleared,
      stepped,
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_FROM_SELECTION, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_selection_pick_noop_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast optional SELECTION_PICK NOOP via the shared selection runtime."""
  from azuki_jax.abilities.selection import process_skip_selection

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)

  prev = state
  do_action = (
      ~(did_reset | zero_legal)
      & (action_type == Act.NOOP)
      & (state.ab_phase == AbilityPhase.SELECTION_PICK)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
  )
  stepped = process_skip_selection(state, do_action)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.NOOP, jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_top_deck_card_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast TOP_DECK_CARD selection action."""
  from azuki_jax.abilities.selection import process_top_deck

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  stepped = process_top_deck(state, action[1], ~(did_reset | zero_legal))
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.TOP_DECK_CARD, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations
def step_bottom_deck_card_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast BOTTOM_DECK_CARD selection action."""
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.selection import process_bottom_deck
  from azuki_jax.engine.triggers import pop_effect
  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  stepped = process_bottom_deck(
      state, action[1], ~(did_reset | zero_legal)
  )
  done_selection = (
      ~(did_reset | zero_legal)
      & (state.ab_phase == AbilityPhase.BOTTOM_DECK)
      & (stepped.ab_phase == AbilityPhase.NONE)
  )
  passived = recompute_passives(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(done_selection, a, b), passived, stepped
  )
  has_trigger = done_selection & (stepped.trig_count > 0)
  popped, trig_src, trig_owner, trig_timing = pop_effect(stepped)
  begun = resolve_triggered_effect(popped, trig_src, trig_owner, trig_timing)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(has_trigger, a, b), begun, stepped
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.BOTTOM_DECK_CARD, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_bottom_deck_all_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast BOTTOM_DECK_ALL selection action."""
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.selection import process_bottom_deck_all
  from azuki_jax.engine.triggers import pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  stepped = process_bottom_deck_all(state, ~(did_reset | zero_legal))
  done_selection = (
      ~(did_reset | zero_legal)
      & (state.ab_phase == AbilityPhase.BOTTOM_DECK)
      & (stepped.ab_phase == AbilityPhase.NONE)
  )
  passived = recompute_passives(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(done_selection, a, b), passived, stepped
  )
  has_trigger = done_selection & (stepped.trig_count > 0)
  popped, trig_src, trig_owner, trig_timing = pop_effect(stepped)
  begun = resolve_triggered_effect(popped, trig_src, trig_owner, trig_timing)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(has_trigger, a, b), begun, stepped
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.BOTTOM_DECK_ALL, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_065_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-065 effect: pay self-damage cost, then deal 5 damage."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot, leader_instance
  from azuki_jax.engine.triggers import TIMING_WHEN_TAKES_DAMAGE, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  target_index = jnp.clip(action[1].astype(jnp.int32), 0, 11)

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-065"]

  friendly_garden = target_index < GARDEN_SIZE
  enemy_garden = (target_index >= GARDEN_SIZE) & (target_index < 2 * GARDEN_SIZE)
  own_leader = target_index == 2 * GARDEN_SIZE
  enemy_leader = target_index == (2 * GARDEN_SIZE + 1)

  garden_player = jnp.where(friendly_garden, owner, opp)
  garden_slot = jnp.where(
      friendly_garden,
      target_index,
      target_index - GARDEN_SIZE,
  )
  garden_inst = card_at_slot(state, garden_player, Zone.GARDEN, garden_slot)
  target_player = jnp.where(
      own_leader | friendly_garden,
      owner,
      opp,
  ).astype(jnp.int32)
  target_inst = jnp.where(
      friendly_garden | enemy_garden,
      garden_inst,
      jnp.where(own_leader, leader_instance(state, owner), leader_instance(state, opp)),
  ).astype(jnp.int32)
  safe_target = jnp.maximum(target_inst, 0)
  safe_player = jnp.clip(target_player, 0, 1)
  target_def = state.def_id[safe_player, safe_target]
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)],
      jnp.int8(-1),
  )
  valid_target = (
      ((own_leader | enemy_leader) & (target_inst >= 0))
      | (
          (friendly_garden | enemy_garden)
          & (target_inst >= 0)
          & (target_type == CardType.ENTITY)
      )
  )
  own_leader_inst = leader_instance(state, owner)
  safe_own_leader = jnp.maximum(own_leader_inst, 0)
  do_action = (
      ~(did_reset | zero_legal)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & source_ok
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
      & ~state.ab_costs_applied
      & (own_leader_inst >= 0)
      & valid_target
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_action, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(
              do_action,
              safe_player.astype(jnp.int8),
              state.ab_eff_target_players[0],
          )
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = deal_effect_damage(
      stepped, owner, safe_own_leader, 3, do_action
  )
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied)
  )
  stepped = deal_effect_damage(
      stepped, safe_player, safe_target, 5, do_action
  )
  cleared = recompute_passives(_clear_context(stepped))
  trig_src062 = jnp.maximum(cleared.trig_source[0].astype(jnp.int32), 0)
  trig_owner062 = jnp.maximum(cleared.trig_owner[0].astype(jnp.int32), 0)
  begin_azk01_062 = (
      do_action
      & (cleared.redirect_count > 0)
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (cleared.def_id[trig_owner062, trig_src062] == cards.CODE_TO_ID["AZK01-062"])
  )
  popped062, src062, owner062, timing062 = pop_effect(cleared)
  begun062 = resolve_triggered_effect(popped062, src062, owner062, timing062)
  cleared = jax.tree.map(
      lambda a, b: jnp.where(begin_azk01_062, a, b), begun062, cleared
  )
  trig_src0 = jnp.maximum(cleared.trig_source[0].astype(jnp.int32), 0)
  trig_owner0 = jnp.maximum(cleared.trig_owner[0].astype(jnp.int32), 0)
  damage_trigger_in_play0 = (
      (cleared.zone[trig_owner0, trig_src0] == Zone.GARDEN)
      | (cleared.zone[trig_owner0, trig_src0] == Zone.ALLEY)
  )
  dead_damage_fizzle = (
      do_action
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (
          (cleared.def_id[trig_owner0, trig_src0] == cards.CODE_TO_ID["AZK01-059"])
          | (
              cleared.def_id[trig_owner0, trig_src0]
              == cards.CODE_TO_ID["STT04-009"]
          )
      )
      & ~damage_trigger_in_play0
  )
  popped_fizzle, _, _, _ = pop_effect(cleared)
  cleared = jax.tree.map(
      lambda a, b: jnp.where(dead_damage_fizzle, a, b),
      popped_fizzle,
      cleared,
  )
  trig_src = jnp.maximum(cleared.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(cleared.trig_owner[0].astype(jnp.int32), 0)
  azk01_059_in_play = (
      (cleared.zone[trig_owner, trig_src] == Zone.GARDEN)
      | (cleared.zone[trig_owner, trig_src] == Zone.ALLEY)
  )
  azk01_059_trigger = (
      do_action
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == TIMING_WHEN_TAKES_DAMAGE)
      & (cleared.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-059"])
      & ((cleared.once_per_turn_used[trig_owner, trig_src] & 1) == 0)
      & azk01_059_in_play
  )
  popped, src2, owner2, _ = pop_effect(cleared)
  owner2_i32 = owner2.astype(jnp.int32)
  needs_transfer = popped.active_player != owner2.astype(jnp.int8)
  begun = popped._replace(
      ab_source=src2.astype(jnp.int8),
      ab_owner=owner2.astype(jnp.int8),
      ab_is_optional=jnp.bool_(False),
      ab_costs_applied=jnp.bool_(False),
      ab_saved_active=jnp.where(needs_transfer, popped.active_player, jnp.int8(-1)),
      ab_restores_active=needs_transfer,
      ab_cost_selected=jnp.int8(0),
      ab_cost_max=jnp.int8(0),
      ab_eff_selected=jnp.int8(0),
      ab_eff_min=jnp.int8(1),
      ab_eff_max=jnp.int8(1),
      ab_phase=jnp.int8(AbilityPhase.EFFECT_SELECTION),
      active_player=jnp.where(
          needs_transfer, owner2_i32.astype(jnp.int8), popped.active_player
      ),
  )
  cleared = jax.tree.map(
      lambda a, b: jnp.where(azk01_059_trigger, a, b), begun, cleared
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_009_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-009 effect: grant Charge to a low-cost garden entity."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_charge_grant
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import GRANT_PHASE_END, card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  target_index = action[1].astype(jnp.int32)

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-009"]

  friendly_garden = target_index < GARDEN_SIZE
  enemy_garden = (target_index >= GARDEN_SIZE) & (target_index < 2 * GARDEN_SIZE)
  target_player = jnp.where(friendly_garden, owner, opp).astype(jnp.int32)
  target_slot = jnp.where(
      friendly_garden,
      target_index,
      target_index - GARDEN_SIZE,
  )
  target = card_at_slot(state, target_player, Zone.GARDEN, target_slot)
  safe_target = jnp.maximum(target, 0)
  safe_player = jnp.clip(target_player, 0, 1)
  target_def = state.def_id[safe_player, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_ok = (
      (friendly_garden | enemy_garden)
      & (target >= 0)
      & (target_def >= 0)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & (jnp.asarray(cards.IKZ_COST)[safe_target_def].astype(jnp.int32) <= 4)
  )
  do_action = (
      ~(did_reset | zero_legal)
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & source_ok
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_action, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_action, safe_player.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(1), state.ab_eff_selected),
      ab_costs_applied=jnp.where(do_action, True, state.ab_costs_applied),
  )
  stepped = apply_charge_grant(
      stepped, safe_player, safe_target, GRANT_PHASE_END, 1, do_action
  )
  stepped = _clear_context(stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_effect_azk01_117_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-117 effect: grant Charge to a cost-5 garden entity."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_charge_grant, deal_effect_damage
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import GRANT_PHASE_END, card_at_slot, leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  target_index = action[1].astype(jnp.int32)

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-117"]

  friendly_garden = target_index < GARDEN_SIZE
  enemy_garden = (target_index >= GARDEN_SIZE) & (target_index < 2 * GARDEN_SIZE)
  target_player = jnp.where(friendly_garden, owner, opp).astype(jnp.int32)
  target_slot = jnp.where(
      friendly_garden,
      target_index,
      target_index - GARDEN_SIZE,
  )
  target = card_at_slot(state, target_player, Zone.GARDEN, target_slot)
  safe_target = jnp.maximum(target, 0)
  safe_player = jnp.clip(target_player, 0, 1)
  target_def = state.def_id[safe_player, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_ok = (
      (friendly_garden | enemy_garden)
      & (target >= 0)
      & (target_def >= 0)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[safe_target_def]
      & (jnp.asarray(cards.IKZ_COST)[safe_target_def].astype(jnp.int32) <= 5)
  )
  leader = leader_instance(state, owner)
  safe_leader = jnp.maximum(leader, 0)
  do_action = (
      ~(did_reset | zero_legal)
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & source_ok
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
      & target_ok
      & (leader >= 0)
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_action, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_action, safe_player.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(1), state.ab_eff_selected),
  )
  cost_do = do_action & ~state.ab_costs_applied
  stepped = deal_effect_damage(
      stepped,
      owner,
      safe_leader,
      2,
      cost_do,
      src_player=owner,
      src_inst=src,
  )
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied)
  )
  stepped = apply_charge_grant(
      stepped, safe_player, safe_target, GRANT_PHASE_END, 1, do_action
  )
  stepped = _clear_context(stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_effect_azk01_042_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-042 effect selection; 3/2/1 damage applies after pick 3."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  selected = jnp.clip(
      state.ab_eff_selected.astype(jnp.int32), 0, MAX_ABILITY_SELECTION - 1
  )
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-042"])
      & ~state.ab_costs_applied
      & (state.ab_eff_selected < 3)
      & (state.ab_eff_min == 3)
      & (state.ab_eff_max == 3)
  )

  target_index = action[1].astype(jnp.int32)
  target = card_at_slot(state, opp, Zone.GARDEN, target_index)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[safe_target_def],
      jnp.int8(-1),
  )
  k = jnp.arange(MAX_ABILITY_SELECTION)
  already_selected = jnp.any(
      (k < state.ab_eff_selected.astype(jnp.int32))
      & (state.ab_eff_target_players == opp.astype(jnp.int8))
      & (state.ab_eff_targets == safe_target.astype(jnp.int8))
  )
  target_ok = (
      (target_index >= 0)
      & (target_index < GARDEN_SIZE)
      & (target >= 0)
      & (state.zone[opp, safe_target] == Zone.GARDEN)
      & (target_type == CardType.ENTITY)
      & ~already_selected
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[selected].set(
          jnp.where(
              do_select,
              safe_target.astype(jnp.int8),
              state.ab_eff_targets[selected],
          )
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[selected].set(
          jnp.where(
              do_select,
              opp.astype(jnp.int8),
              state.ab_eff_target_players[selected],
          )
      ),
      ab_eff_selected=jnp.where(
          do_select, state.ab_eff_selected + 1, state.ab_eff_selected
      ).astype(jnp.int8),
  )

  finish = do_select & (stepped.ab_eff_selected >= stepped.ab_eff_max)
  for target_slot, damage in enumerate((3, 2, 1)):
    tp = jnp.maximum(stepped.ab_eff_target_players[target_slot].astype(jnp.int32), 0)
    ti = jnp.maximum(stepped.ab_eff_targets[target_slot].astype(jnp.int32), 0)
    has = (
        (stepped.ab_eff_target_players[target_slot] >= 0)
        & (stepped.ab_eff_targets[target_slot] >= 0)
        & (
            jnp.asarray(target_slot, jnp.int32)
            < stepped.ab_eff_selected.astype(jnp.int32)
        )
    )
    stepped = deal_effect_damage(stepped, tp, ti, damage, finish & has)

  cleared = recompute_passives(_clear_context(stepped))
  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), cleared, stepped)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_effect_azk01_127_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-127 effect: deal 1 damage to an enemy garden entity."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import (
      apply_attack_modifier,
      deal_effect_damage,
  )
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.helpers import card_at_slot
  from azuki_jax.engine.phases import (
      combat_resolve,
      defender_can_respond,
      transition_to_combat_resolve,
  )
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.engine.triggers import TIMING_WHEN_TAKES_DAMAGE, has_queued, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  target_index = jnp.clip(action[1].astype(jnp.int32), 0, GARDEN_SIZE - 1)

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-127"]
  target = card_at_slot(state, opp, Zone.GARDEN, target_index)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)],
      jnp.int8(-1),
  )
  valid_target = (
      (action[1] >= 0)
      & (action[1] < GARDEN_SIZE)
      & (target >= 0)
      & (target_type == CardType.ENTITY)
  )
  do_action = (
      ~(did_reset | zero_legal)
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & source_ok
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
      & (state.combat_attacker >= 0)
      & valid_target
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_action, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_action, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(1), state.ab_eff_selected),
      ab_costs_applied=jnp.where(do_action, True, state.ab_costs_applied),
  )
  stepped = deal_effect_damage(stepped, opp, safe_target, 1, do_action)
  cleared = recompute_passives(_clear_context(stepped))
  trig_src0 = jnp.maximum(cleared.trig_source[0].astype(jnp.int32), 0)
  trig_owner0 = jnp.maximum(cleared.trig_owner[0].astype(jnp.int32), 0)
  begin_azk01_062 = (
      do_action
      & (cleared.redirect_count > 0)
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (cleared.def_id[trig_owner0, trig_src0] == cards.CODE_TO_ID["AZK01-062"])
  )
  begin_stt04_009 = (
      do_action
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (cleared.def_id[trig_owner0, trig_src0] == cards.CODE_TO_ID["STT04-009"])
      & cleared.last_dmg_from_effect[trig_owner0, trig_src0]
  )
  popped, src2, owner2, timing2 = pop_effect(cleared)
  begun = resolve_triggered_effect(popped, src2, owner2, timing2)
  cleared = jax.tree.map(
      lambda a, b: jnp.where(begin_azk01_062 | begin_stt04_009, a, b),
      begun,
      cleared,
  )
  trig_src_stt04_007 = jnp.maximum(cleared.trig_source[0].astype(jnp.int32), 0)
  trig_owner_stt04_007 = jnp.maximum(cleared.trig_owner[0].astype(jnp.int32), 0)
  stt04_007_trigger = (
      do_action
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (
          cleared.def_id[trig_owner_stt04_007, trig_src_stt04_007]
          == cards.CODE_TO_ID["STT04-007"]
      )
  )
  stt04_007_to_effect = (
      stt04_007_trigger
      & (
          (cleared.once_per_turn_used[trig_owner_stt04_007, trig_src_stt04_007]
           & jnp.uint8(1))
          == 0
      )
      & (
          (cleared.zone[trig_owner_stt04_007, trig_src_stt04_007] == Zone.GARDEN)
          | (cleared.zone[trig_owner_stt04_007, trig_src_stt04_007] == Zone.ALLEY)
      )
  )
  popped_stt04_007, _, _, _ = pop_effect(cleared)
  buffed_stt04_007 = apply_attack_modifier(
      popped_stt04_007,
      trig_owner_stt04_007,
      trig_src_stt04_007,
      1,
      expires_eot=True,
      do=stt04_007_to_effect,
  )
  stt04_007_once = buffed_stt04_007.once_per_turn_used[
      trig_owner_stt04_007, trig_src_stt04_007
  ]
  buffed_stt04_007 = buffed_stt04_007._replace(
      once_per_turn_used=buffed_stt04_007.once_per_turn_used.at[
          trig_owner_stt04_007, trig_src_stt04_007
      ].set(
          jnp.where(
              stt04_007_to_effect,
              stt04_007_once | jnp.uint8(1),
              stt04_007_once,
          )
      )
  )
  cleared = jax.tree.map(
      lambda a, b: jnp.where(stt04_007_trigger, a, b),
      buffed_stt04_007,
      cleared,
  )
  trig_src059 = jnp.maximum(cleared.trig_source[0].astype(jnp.int32), 0)
  trig_owner059 = jnp.maximum(cleared.trig_owner[0].astype(jnp.int32), 0)
  azk01_059_trigger = (
      do_action
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (cleared.def_id[trig_owner059, trig_src059] == cards.CODE_TO_ID["AZK01-059"])
      & ((cleared.once_per_turn_used[trig_owner059, trig_src059] & 1) == 0)
      & (
          (cleared.zone[trig_owner059, trig_src059] == Zone.GARDEN)
          | (cleared.zone[trig_owner059, trig_src059] == Zone.ALLEY)
      )
  )
  popped059, src059, owner059, _ = pop_effect(cleared)
  owner059_i32 = owner059.astype(jnp.int32)
  needs_transfer059 = popped059.active_player != owner059.astype(jnp.int8)
  begun059 = popped059._replace(
      ab_source=src059.astype(jnp.int8),
      ab_owner=owner059.astype(jnp.int8),
      ab_is_optional=jnp.bool_(False),
      ab_costs_applied=jnp.bool_(False),
      ab_saved_active=jnp.where(
          needs_transfer059, popped059.active_player, jnp.int8(-1)
      ),
      ab_restores_active=needs_transfer059,
      ab_cost_selected=jnp.int8(0),
      ab_cost_max=jnp.int8(0),
      ab_eff_selected=jnp.int8(0),
      ab_eff_min=jnp.int8(1),
      ab_eff_max=jnp.int8(1),
      ab_phase=jnp.int8(AbilityPhase.EFFECT_SELECTION),
      active_player=jnp.where(
          needs_transfer059, owner059_i32.astype(jnp.int8), popped059.active_player
      ),
  )
  cleared = jax.tree.map(
      lambda a, b: jnp.where(azk01_059_trigger, a, b), begun059, cleared
  )
  close_response = (
      (cleared.phase == Phase.RESPONSE_WINDOW)
      & (cleared.ab_phase == AbilityPhase.NONE)
      & ~has_queued(cleared)
      & ~defender_can_respond(cleared, cleared.active_player)
  )
  cleared = jax.lax.cond(
      close_response,
      lambda st: transition_to_combat_resolve(st, do=True),
      lambda st: st,
      cleared,
  )
  auto_combat = (cleared.phase == Phase.COMBAT_RESOLVE) & ~has_queued(cleared)
  cleared = jax.lax.cond(
      auto_combat,
      lambda st: combat_resolve(st, do=True),
      lambda st: st,
      cleared,
  )
  trig_src = jnp.maximum(cleared.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(cleared.trig_owner[0].astype(jnp.int32), 0)
  azk01_062_combat_fizzle = (
      auto_combat
      & (cleared.redirect_count == 0)
      & (cleared.trig_count > 0)
      & (cleared.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (cleared.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-062"])
  )
  popped_fizzle, _, _, _ = pop_effect(cleared)
  cleared = jax.tree.map(
      lambda a, b: jnp.where(azk01_062_combat_fizzle, a, b),
      popped_fizzle,
      cleared,
  )
  cleared = _begin_queued_stt03_006_destroy_trigger(
      cleared, auto_combat | azk01_062_combat_fizzle
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_128_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-128 effect: destroy the current low-HP attacking entity."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import destroy_card
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.helpers import card_at_slot
  from azuki_jax.engine.phases import (
      combat_resolve,
      defender_can_respond,
      transition_to_combat_resolve,
  )
  from azuki_jax.engine.triggers import TIMING_WHEN_DESTROYED, has_queued, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  target_index = jnp.clip(action[1].astype(jnp.int32), 0, GARDEN_SIZE - 1)

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-128"]
  target = card_at_slot(state, opp, Zone.GARDEN, target_index)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[jnp.maximum(target_def, 0)],
      jnp.int8(-1),
  )
  valid_target = (
      (action[1] >= 0)
      & (action[1] < GARDEN_SIZE)
      & (target >= 0)
      & (target == state.combat_attacker.astype(jnp.int32))
      & (target_type == CardType.ENTITY)
      & (state.cur_hp[opp, safe_target] <= 2)
  )
  safe_attacker = jnp.maximum(state.combat_attacker.astype(jnp.int32), 0)
  attacker_def = state.def_id[opp, safe_attacker]
  attacker_type = jnp.where(
      attacker_def >= 0,
      jnp.asarray(cards.TYPE)[jnp.maximum(attacker_def, 0)],
      jnp.int8(-1),
  )
  target_available = (
      (state.combat_attacker >= 0)
      & (state.zone[opp, safe_attacker] == Zone.GARDEN)
      & (attacker_type == CardType.ENTITY)
      & (state.cur_hp[opp, safe_attacker] <= 2)
  )
  can_resolve = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & source_ok
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
      & (state.combat_attacker >= 0)
  )
  do_select = can_resolve & (action[0] == Act.SELECT_EFFECT_TARGET) & valid_target
  do_skip = can_resolve & (action[0] == Act.NOOP) & ~target_available
  do_action = do_select | do_skip

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
      ab_costs_applied=jnp.where(do_action, True, state.ab_costs_applied),
  )
  stepped = destroy_card(stepped, opp, safe_target, do_select)
  cleared = recompute_passives(_clear_context(stepped))
  popped, trig_src, trig_owner, trig_timing = pop_effect(cleared)
  trig_src_i = jnp.maximum(trig_src.astype(jnp.int32), 0)
  trig_owner_i = jnp.maximum(trig_owner.astype(jnp.int32), 0)
  begin_stt03_006 = (
      do_action
      & (cleared.trig_count > 0)
      & (trig_timing == jnp.int8(TIMING_WHEN_DESTROYED))
      & (cleared.def_id[trig_owner_i, trig_src_i] == cards.CODE_TO_ID["STT03-006"])
  )
  begun = resolve_triggered_effect(popped, trig_src, trig_owner, trig_timing)
  cleared = jax.tree.map(lambda a, b: jnp.where(begin_stt03_006, a, b), begun, cleared)
  close_response = (
      (cleared.phase == Phase.RESPONSE_WINDOW)
      & (cleared.ab_phase == AbilityPhase.NONE)
      & ~has_queued(cleared)
      & ~defender_can_respond(cleared, cleared.active_player)
  )
  cleared = jax.lax.cond(
      close_response,
      lambda st: transition_to_combat_resolve(st, do=True),
      lambda st: st,
      cleared,
  )
  auto_combat = (cleared.phase == Phase.COMBAT_RESOLVE) & ~has_queued(cleared)
  cleared = jax.lax.cond(
      auto_combat,
      lambda st: combat_resolve(st, do=True),
      lambda st: st,
      cleared,
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations




def step_activate_stt01_001_fast(
    state: State,
    actions: jax.Array,
    prev_terminals: jax.Array,
    prev_truncations: jax.Array,
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-001 main activation into charge target selection."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  leader = leader_instance(state, acting)
  safe_leader = jnp.maximum(leader, 0)
  row_defs = state.def_id[acting]
  row_safe_defs = jnp.maximum(row_defs, 0)
  weapon_targets = (
      (state.zone[acting] == Zone.GARDEN)
      & (row_defs >= 0)
      & (jnp.asarray(cards.TYPE)[row_safe_defs] == CardType.ENTITY)
      & (state.cooldown[acting] != 0)
  )
  attached = state.zone[acting] == Zone.ATTACHED
  weapon_count = jnp.zeros_like(weapon_targets, dtype=jnp.int8)
  for i in range(state.zone.shape[1]):
    host = jnp.clip(
        state.attached_to[acting, i].astype(jnp.int32),
        0,
        state.zone.shape[1] - 1,
    )
    is_weapon = (
        attached[i]
        & (
            jnp.asarray(cards.TYPE)[jnp.maximum(state.def_id[acting, i], 0)]
            == CardType.WEAPON
        )
    )
    weapon_count = weapon_count.at[host].add(is_weapon.astype(jnp.int8))
  target_available = jnp.any(weapon_targets & (weapon_count > 0))
  use_token = action[3] != 0
  pay_ok = ikz.can_pay(state, acting, 1, use_token)
  do_action = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (action[0] == Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
      & (action[1] == GARDEN_SIZE)
      & (action[2] == 0)
      & (leader >= 0)
      & (state.def_id[acting, safe_leader] == cards.CODE_TO_ID["STT01-001"])
      & (state.frozen_dur[acting, safe_leader] == 0)
      & ((state.once_per_turn_used[acting, safe_leader] & 1) == 0)
      & pay_ok
      & target_available
  )

  stepped = ikz.pay(state, acting, 1, use_token, do=do_action)
  stepped = stepped._replace(
      ab_source=jnp.where(do_action, safe_leader.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(do_action, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(1), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_phase=jnp.where(
          do_action, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )
  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )
  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations


def step_effect_stt01_001_fast(
    state: State,
    actions: jax.Array,
    prev_terminals: jax.Array,
    prev_truncations: jax.Array,
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-001 effect: grant Charge to a weaponed cooldown entity."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_charge_grant
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import GRANT_PHASE_END, card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT01-001"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )
  target_index = action[1].astype(jnp.int32)
  target = card_at_slot(state, owner, Zone.GARDEN, target_index)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[owner, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  attached = state.zone[owner] == Zone.ATTACHED
  weapon_count = jnp.int8(0)
  for i in range(state.zone.shape[1]):
    is_weapon = (
        attached[i]
        & (state.attached_to[owner, i].astype(jnp.int32) == safe_target)
        & (
            jnp.asarray(cards.TYPE)[jnp.maximum(state.def_id[owner, i], 0)]
            == CardType.WEAPON
        )
    )
    weapon_count = weapon_count + is_weapon.astype(jnp.int8)
  target_ok = (
      (target_index >= 0)
      & (target_index < GARDEN_SIZE)
      & (target >= 0)
      & (target_def >= 0)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & (weapon_count > 0)
      & (state.cooldown[owner, safe_target] != 0)
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, owner.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = apply_charge_grant(
      stepped, owner, safe_target, GRANT_PHASE_END, 1, do_select
  )
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(do_select, a, b), cleared, state)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )
  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )
  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations

def step_activate_stt02_001_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-001 response activation into enemy target selection."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  leader = leader_instance(state, acting)
  safe_leader = jnp.maximum(leader, 0)
  opp = (acting + 1) % 2
  opp_leader = leader_instance(state, opp)
  opp_defs = state.def_id[opp]
  enemy_garden_entities = (
      (state.zone[opp] == Zone.GARDEN)
      & (opp_defs >= 0)
      & (jnp.asarray(cards.TYPE)[jnp.maximum(opp_defs, 0)] == CardType.ENTITY)
  )
  target_available = (opp_leader >= 0) | jnp.any(enemy_garden_entities)
  use_token = action[3] != 0
  pay_ok = ikz.can_pay(state, acting, 1, use_token)
  once_used = state.once_per_turn_used[acting, safe_leader]
  do_action = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == AbilityPhase.NONE)
      & (action[0] == Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
      & (action[1] == GARDEN_SIZE)
      & (action[2] == 0)
      & (leader >= 0)
      & (state.def_id[acting, safe_leader] == cards.CODE_TO_ID["STT02-001"])
      & (state.frozen_dur[acting, safe_leader] == 0)
      & ((once_used & 1) == 0)
      & pay_ok
      & target_available
  )

  stepped = ikz.pay(state, acting, 1, use_token, do=do_action)
  stepped = stepped._replace(
      ab_source=jnp.where(do_action, safe_leader.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(do_action, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      once_per_turn_used=stepped.once_per_turn_used.at[acting, safe_leader].set(
          jnp.where(do_action, once_used | jnp.uint8(1), once_used)
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(1), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_phase=jnp.where(
          do_action, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_stt02_001_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-001 effect: enemy leader/garden target gets -1 ATK EOT."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_attack_modifier
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot, leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  source_ok = (
      (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT02-001"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )

  target_index = action[1].astype(jnp.int32)
  target_is_leader = target_index == GARDEN_SIZE
  garden_slot = jnp.clip(target_index, 0, GARDEN_SIZE - 1)
  garden_target = card_at_slot(state, opp, Zone.GARDEN, garden_slot)
  leader = leader_instance(state, opp)
  target = jnp.where(target_is_leader, leader, garden_target)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_type = jnp.where(
      target_def >= 0,
      jnp.asarray(cards.TYPE)[safe_target_def],
      jnp.int8(-1),
  )
  target_ok = (
      (target_index >= 0)
      & (target_index <= GARDEN_SIZE)
      & (target >= 0)
      & (
          (
              target_is_leader
              & (state.zone[opp, safe_target] == Zone.LEADER)
          )
          | (
              ~target_is_leader
              & (state.zone[opp, safe_target] == Zone.GARDEN)
              & (target_type == CardType.ENTITY)
          )
      )
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = apply_attack_modifier(
      stepped, opp, safe_target, -1, expires_eot=True, do=do_select
  )
  cleared = _close_response_combat_if_idle(_clear_context(stepped))
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_select, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_activate_stt03_001_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT03-001 leader main ability: pay 1 IKZ, arm Bobu latch."""
  from azuki_jax import cards
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  leader = leader_instance(state, acting)
  safe_leader = jnp.maximum(leader, 0)
  use_token = action[3] != 0
  pay_ok = ikz.can_pay(state, acting, 1, use_token)
  do_action = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (action[0] == Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
      & (action[1] == GARDEN_SIZE)
      & (action[2] == 0)
      & (leader >= 0)
      & (state.def_id[acting, safe_leader] == cards.CODE_TO_ID["STT03-001"])
      & (state.frozen_dur[acting, safe_leader] == 0)
      & ((state.once_per_turn_used[acting, safe_leader] & 1) == 0)
      & pay_ok
  )

  stepped = ikz.pay(state, acting, 1, use_token, do=do_action)
  stepped = stepped._replace(
      bobu_expires_turn=stepped.bobu_expires_turn.at[acting].set(
          jnp.where(
              do_action,
              (stepped.turn_number + 2).astype(jnp.int16),
              stepped.bobu_expires_turn[acting],
          )
      ),
      ab_source=jnp.where(do_action, safe_leader.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), stepped.ab_owner),
      ab_phase=jnp.where(do_action, jnp.int8(AbilityPhase.NONE), stepped.ab_phase),
  )
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_activate_stt03_004_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT03-004 main ability: sacrifice self, heal leader 1."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import heal_leader, sacrifice_card
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  activate_garden = action[0] == Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY
  activate_alley = action[0] == Act.ACTIVATE_ALLEY_ABILITY
  source = jnp.where(
      activate_alley,
      card_at_slot(state, acting, Zone.ALLEY, action[2]),
      card_at_slot(state, acting, Zone.GARDEN, action[1]),
  )
  safe_source = jnp.maximum(source, 0)
  ability_index_ok = jnp.where(activate_alley, action[1] == 0, action[2] == 0)
  do_action = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (activate_garden | activate_alley)
      & ability_index_ok
      & (source >= 0)
      & (state.def_id[acting, safe_source] == cards.CODE_TO_ID["STT03-004"])
      & ~state.tapped[acting, safe_source]
  )

  stepped = sacrifice_card(state, acting, safe_source, do_action)
  stepped = heal_leader(stepped, acting, 1, do_action)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  action_type = jnp.where(
      activate_alley,
      jnp.asarray(Act.ACTIVATE_ALLEY_ABILITY, jnp.int32),
      jnp.asarray(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY, jnp.int32),
  )
  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type,
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_activate_azk01_119_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-119 leader main activation into equipped-entity selection."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  leader = leader_instance(state, acting)
  safe_leader = jnp.maximum(leader, 0)
  use_token = action[3] != 0
  pay_ok = ikz.can_pay(state, acting, 3, use_token)
  once_used = state.once_per_turn_used[acting, safe_leader]
  idx = jnp.arange(state.zone.shape[1])
  row_defs = state.def_id[acting]
  safe_defs = jnp.maximum(row_defs, 0)
  attached_weapon = (
      (state.zone[acting] == Zone.ATTACHED)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.WEAPON)
  )
  equipped_counts = jax.vmap(
      lambda i: jnp.sum(
          attached_weapon
          & (state.attached_to[acting].astype(jnp.int32) == i),
          dtype=jnp.int32,
      )
  )(idx)
  friendly_equipped_entity = (
      (state.zone[acting] == Zone.GARDEN)
      & (row_defs >= 0)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
      & (equipped_counts > 0)
  )
  target_available = jnp.any(friendly_equipped_entity)
  do_action = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (action[0] == Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
      & (action[1] == GARDEN_SIZE)
      & (action[2] == 0)
      & (leader >= 0)
      & (state.def_id[acting, safe_leader] == cards.CODE_TO_ID["AZK01-119"])
      & (state.frozen_dur[acting, safe_leader] == 0)
      & ((once_used & 1) == 0)
      & pay_ok
      & target_available
  )

  stepped = ikz.pay(state, acting, 3, use_token, do=do_action)
  stepped = stepped._replace(
      once_per_turn_used=stepped.once_per_turn_used.at[acting, safe_leader].set(
          jnp.where(do_action, once_used | jnp.uint8(1), once_used)
      ),
      ab_source=jnp.where(do_action, safe_leader.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(do_action, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(1), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_phase=jnp.where(
          do_action, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )
  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )
  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations


def step_effect_azk01_119_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-119 effect: equipped friendly garden entity gets +ATK EOT."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_attack_modifier
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  target_index = action[1].astype(jnp.int32)
  target = card_at_slot(state, owner, Zone.GARDEN, target_index)
  safe_target = jnp.maximum(target, 0)
  row_defs = state.def_id[owner]
  safe_defs = jnp.maximum(row_defs, 0)
  attached_weapon = (
      (state.zone[owner] == Zone.ATTACHED)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.WEAPON)
      & (state.attached_to[owner].astype(jnp.int32) == safe_target)
  )
  target_def = state.def_id[owner, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  target_ok = (
      (target_index >= 0)
      & (target_index < GARDEN_SIZE)
      & (target >= 0)
      & (target_def >= 0)
      & (state.zone[owner, safe_target] == Zone.GARDEN)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
      & jnp.any(attached_weapon)
  )
  source_ok = (
      (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-119"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  discard_weapons = (
      (state.zone[owner] == Zone.DISCARD)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.WEAPON)
  )
  buff = jnp.minimum(jnp.sum(discard_weapons, dtype=jnp.int32), 3).astype(jnp.int8)
  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, owner.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = apply_attack_modifier(
      stepped, owner, safe_target, buff, expires_eot=True, do=do_select & (buff > 0)
  )
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(do_select, a, b), cleared, state)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )
  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )
  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations


def step_activate_azk01_121_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-121 leader main ability: pay 1 IKZ, gain capped ATK."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_attack_modifier
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  leader = leader_instance(state, acting)
  safe_leader = jnp.maximum(leader, 0)
  total_played = (
      state.entities_played_garden_turn[acting].astype(jnp.int32)
      + state.entities_played_alley_turn[acting].astype(jnp.int32)
  )
  buff = jnp.minimum(total_played, 2)
  use_token = action[3] != 0
  pay_ok = ikz.can_pay(state, acting, 1, use_token)
  do_action = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (action[0] == Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
      & (action[1] == GARDEN_SIZE)
      & (action[2] == 0)
      & (leader >= 0)
      & (state.def_id[acting, safe_leader] == cards.CODE_TO_ID["AZK01-121"])
      & (state.frozen_dur[acting, safe_leader] == 0)
      & ((state.once_per_turn_used[acting, safe_leader] & 1) == 0)
      & pay_ok
      & (total_played > 0)
  )

  stepped = ikz.pay(state, acting, 1, use_token, do=do_action)
  stepped = apply_attack_modifier(
      stepped, acting, safe_leader, buff, expires_eot=True, do=do_action
  )
  stepped = stepped._replace(
      ab_source=jnp.where(do_action, safe_leader.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), stepped.ab_owner),
      ab_phase=jnp.where(do_action, jnp.int8(AbilityPhase.NONE), stepped.ab_phase),
  )
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_activate_azk01_125_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-125 leader main ability: pay 1 IKZ, reduce next play cost by 2."""
  from azuki_jax import cards
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  leader = leader_instance(state, acting)
  safe_leader = jnp.maximum(leader, 0)
  use_token = action[3] != 0
  pay_ok = ikz.can_pay(state, acting, 1, use_token)
  once_used = state.once_per_turn_used[acting, safe_leader]
  do_action = (
      ~(did_reset | zero_legal)
      & ((state.phase == Phase.MAIN) | (state.phase == Phase.RESPONSE_WINDOW))
      & (state.ab_phase == AbilityPhase.NONE)
      & (action[0] == Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
      & (action[1] == GARDEN_SIZE)
      & (action[2] == 0)
      & (leader >= 0)
      & (state.def_id[acting, safe_leader] == cards.CODE_TO_ID["AZK01-125"])
      & (state.frozen_dur[acting, safe_leader] == 0)
      & ((once_used & 1) == 0)
      & (state.discarded_cards_turn[acting] > 0)
      & pay_ok
  )

  stepped = ikz.pay(state, acting, 1, use_token, do=do_action)
  stepped = stepped._replace(
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(
              do_action,
              stepped.next_play_cost_reduction[acting] + jnp.uint8(2),
              stepped.next_play_cost_reduction[acting],
          )
      ),
      once_per_turn_used=stepped.once_per_turn_used.at[acting, safe_leader].set(
          jnp.where(do_action, once_used | jnp.uint8(1), once_used)
      ),
      ab_source=jnp.where(do_action, safe_leader.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), stepped.ab_owner),
      ab_phase=jnp.where(do_action, jnp.int8(AbilityPhase.NONE), stepped.ab_phase),
  )
  cleared = _close_response_combat_if_idle(_clear_context(stepped))
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action, a, b), cleared, state
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_activate_azk01_123_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-123 leader main ability: pay 1 IKZ, then target +1 HP EOT."""
  from azuki_jax import cards
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  leader = leader_instance(state, acting)
  safe_leader = jnp.maximum(leader, 0)
  row_def = state.def_id[acting]
  safe_defs = jnp.maximum(row_def, 0)
  friendly_garden_entities = (
      (state.zone[acting] == Zone.GARDEN)
      & (row_def >= 0)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
      & (state.cur_hp[acting] > 0)
  )
  use_token = action[3] != 0
  pay_ok = ikz.can_pay(state, acting, 1, use_token)
  do_action = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (action[0] == Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
      & (action[1] == GARDEN_SIZE)
      & (action[2] == 0)
      & (leader >= 0)
      & (state.def_id[acting, safe_leader] == cards.CODE_TO_ID["AZK01-123"])
      & (state.frozen_dur[acting, safe_leader] == 0)
      & ((state.once_per_turn_used[acting, safe_leader] & 1) == 0)
      & pay_ok
      & jnp.any(friendly_garden_entities)
  )

  stepped = ikz.pay(state, acting, 1, use_token, do=do_action)
  stepped = stepped._replace(
      ab_source=jnp.where(do_action, safe_leader.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(do_action, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(1), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(1), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_phase=jnp.where(
          do_action, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_activate_stt04_001_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT04-001 leader activation into friendly board effect selection."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.engine.helpers import leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  leader = leader_instance(state, acting)
  safe_leader = jnp.maximum(leader, 0)
  row_zone = state.zone[acting]
  row_def = state.def_id[acting]
  safe_defs = jnp.maximum(row_def, 0)
  friendly_targets = (
      ((row_zone == Zone.GARDEN) | (row_zone == Zone.ALLEY))
      & (row_def >= 0)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
  )
  do_action = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (action[0] == Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
      & (action[1] == GARDEN_SIZE)
      & (action[2] == 0)
      & (leader >= 0)
      & (state.def_id[acting, safe_leader] == cards.CODE_TO_ID["STT04-001"])
      & (state.frozen_dur[acting, safe_leader] == 0)
      & ((state.once_per_turn_used[acting, safe_leader] & 1) == 0)
      & (state.cur_hp[acting, safe_leader] > 0)
      & jnp.any(friendly_targets)
  )

  stepped = state._replace(
      ab_source=jnp.where(do_action, safe_leader.astype(jnp.int8), state.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), state.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), state.ab_slot),
      ab_is_optional=jnp.where(do_action, False, state.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, False, state.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), state.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, state.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), state.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(0), state.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), state.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(1), state.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(1), state.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_targets, -1),
          state.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_target_players, -1),
          state.ab_eff_target_players,
      ),
  )
  stepped = deal_effect_damage(
      stepped, acting, safe_leader, 1, do=do_action, allow_redirect=False
  )
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied),
      ab_phase=jnp.where(
          do_action, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_activate_azk01_070_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-070 response activation into enemy-garden effect selection."""
  from azuki_jax import cards
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  source = card_at_slot(state, acting, Zone.GARDEN, action[1])
  safe_source = jnp.maximum(source, 0)
  opp = (acting + 1) % 2
  row_def = state.def_id[opp]
  safe_defs = jnp.maximum(row_def, 0)
  enemy_targets = (
      (state.zone[opp] == Zone.GARDEN)
      & (row_def >= 0)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
  )
  do_action = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == AbilityPhase.NONE)
      & (action[0] == Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
      & (action[2] == 0)
      & (source >= 0)
      & (state.def_id[acting, safe_source] == cards.CODE_TO_ID["AZK01-070"])
      & ~state.tapped[acting, safe_source]
      & (state.cooldown[acting, safe_source] == 0)
      & (state.cur_hp[acting, safe_source] > 1)
      & jnp.any(enemy_targets)
  )

  stepped = state._replace(
      ab_source=jnp.where(do_action, safe_source.astype(jnp.int8), state.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), state.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), state.ab_slot),
      ab_is_optional=jnp.where(do_action, False, state.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, False, state.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), state.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, state.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), state.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(0), state.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), state.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(1), state.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(1), state.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_targets, -1),
          state.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_target_players, -1),
          state.ab_eff_target_players,
      ),
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_activate_stt02_011_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT02-011 main activation: sacrifice, then choose immune target."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import sacrifice_card
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  source = card_at_slot(state, acting, Zone.GARDEN, action[1])
  safe_source = jnp.maximum(source, 0)
  row_zone = state.zone[acting]
  row_def = state.def_id[acting]
  safe_defs = jnp.maximum(row_def, 0)
  inst = jnp.arange(row_zone.shape[0], dtype=jnp.int32)
  friendly_targets = (
      (row_zone == Zone.GARDEN)
      & (row_def >= 0)
      & (jnp.asarray(cards.TYPE)[safe_defs] == CardType.ENTITY)
      & (inst != safe_source)
      & (state.cur_hp[acting] > 0)
  )
  do_action = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (action[0] == Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
      & (action[2] == 0)
      & (source >= 0)
      & (state.def_id[acting, safe_source] == cards.CODE_TO_ID["STT02-011"])
      & (state.frozen_dur[acting, safe_source] == 0)
      & jnp.any(friendly_targets)
  )

  stepped = state._replace(
      ab_source=jnp.where(do_action, safe_source.astype(jnp.int8), state.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), state.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), state.ab_slot),
      ab_is_optional=jnp.where(do_action, False, state.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, False, state.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), state.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, state.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), state.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(0), state.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), state.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(1), state.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(1), state.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_targets, -1),
          state.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_target_players, -1),
          state.ab_eff_target_players,
      ),
  )
  stepped = sacrifice_card(stepped, acting, safe_source, do=do_action)
  stepped = stepped._replace(
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied),
      ab_phase=jnp.where(
          do_action, jnp.int8(AbilityPhase.EFFECT_SELECTION), stepped.ab_phase
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_activate_azk01_111_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-111 alley activation: sacrifice, then damage prompt."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import sacrifice_card
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_base = ~(did_reset | zero_legal)

  prev = state
  source = card_at_slot(state, acting, Zone.ALLEY, action[2])
  safe_source = jnp.maximum(source, 0)
  opp = (acting + 1) % 2
  opp_defs = state.def_id[opp]
  enemy_garden_entity = jnp.any(
      (state.zone[opp] == Zone.GARDEN)
      & (jnp.asarray(cards.TYPE)[jnp.maximum(opp_defs, 0)] == CardType.ENTITY)
  )
  own_defs = state.def_id[acting]
  eligible_hand_entity = jnp.any(
      (state.zone[acting] == Zone.HAND)
      & (jnp.asarray(cards.TYPE)[jnp.maximum(own_defs, 0)] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[jnp.maximum(own_defs, 0)]
      & (jnp.asarray(cards.IKZ_COST)[jnp.maximum(own_defs, 0)].astype(jnp.int32) <= 2)
  )
  do_action = (
      do_base
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (action[0] == Act.ACTIVATE_ALLEY_ABILITY)
      & (action[1] == 0)
      & (source >= 0)
      & (state.def_id[acting, safe_source] == cards.CODE_TO_ID["AZK01-111"])
      & enemy_garden_entity
  )

  stepped = state._replace(
      ab_phase=jnp.where(
          do_action, jnp.int8(AbilityPhase.EFFECT_SELECTION), state.ab_phase
      ),
      ab_source=jnp.where(do_action, safe_source.astype(jnp.int8), state.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), state.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), state.ab_slot),
      ab_is_optional=jnp.where(do_action, False, state.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, True, state.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), state.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, state.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), state.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(0), state.ab_cost_max),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), state.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(0), state.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(1), state.ab_eff_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_targets, -1),
          state.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_target_players, -1),
          state.ab_eff_target_players,
      ),
  )
  stepped = sacrifice_card(stepped, acting, safe_source, do=do_action)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ACTIVATE_ALLEY_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_azk01_111_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-111 effect: optional damage, then small hand-entity selection."""
  from azuki_jax import cards
  from azuki_jax.abilities import selection as sel_mod
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_base = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  opp = (owner + 1) % 2
  target_index = action[1].astype(jnp.int32)
  target = card_at_slot(state, opp, Zone.GARDEN, target_index)
  safe_target = jnp.maximum(target, 0)
  target_def = state.def_id[opp, safe_target]
  safe_target_def = jnp.maximum(target_def, 0)
  own_defs = state.def_id[owner]
  eligible_hand_entity = jnp.any(
      (state.zone[owner] == Zone.HAND)
      & (jnp.asarray(cards.TYPE)[jnp.maximum(own_defs, 0)] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[jnp.maximum(own_defs, 0)]
      & (jnp.asarray(cards.IKZ_COST)[jnp.maximum(own_defs, 0)].astype(jnp.int32) <= 2)
  )
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-111"])
      & state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 1)
  )
  target_ok = (
      (target >= 0)
      & (target_index >= 0)
      & (target_index < GARDEN_SIZE)
      & (target_def >= 0)
      & (jnp.asarray(cards.TYPE)[safe_target_def] == CardType.ENTITY)
  )
  do_select = (
      do_base
      & (action_type == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  do_skip = do_base & (action_type == Act.NOOP) & source_ok

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, opp.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = deal_effect_damage(stepped, opp, safe_target, 2, do_select)
  finished = recompute_passives(stepped)
  hand_defs = finished.def_id[owner]
  hand_safe_defs = jnp.maximum(hand_defs, 0)
  small_hand = (
      (finished.zone[owner] == Zone.HAND)
      & (jnp.asarray(cards.TYPE)[hand_safe_defs] == CardType.ENTITY)
      & jnp.asarray(cards.HAS_IKZ_COST)[hand_safe_defs]
      & (jnp.asarray(cards.IKZ_COST)[hand_safe_defs].astype(jnp.int32) <= 2)
  )
  finished = sel_mod.move_matching_zone_to_selection(
      finished, Zone.HAND, small_hand, 1, do_select | do_skip
  )
  has_selection = finished.ab_sel_count > 0
  cleared = _clear_context(finished)
  finished = jax.tree.map(
      lambda a, b: jnp.where((do_select | do_skip) & ~has_selection, a, b),
      cleared,
      finished,
  )
  acted = do_select | do_skip
  stepped = jax.tree.map(lambda a, b: jnp.where(acted, a, b), finished, state)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      action_type,
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_activate_stt01_005_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-005 alley ability: sacrifice, draw 3, discard 2 prompt."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import draw_with_deckout, sacrifice_card
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot
  from azuki_jax.zones import zone_count

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  source = card_at_slot(state, acting, Zone.ALLEY, action[2])
  safe_source = jnp.maximum(source, 0)
  deck_before = zone_count(state.zone[acting], Zone.DECK)
  hand_before = zone_count(state.zone[acting], Zone.HAND)
  discard_count = jnp.minimum(hand_before, 2).astype(jnp.int8)
  do_action = (
      ~(did_reset | zero_legal)
      & (state.phase == Phase.MAIN)
      & (state.ab_phase == AbilityPhase.NONE)
      & (action[0] == Act.ACTIVATE_ALLEY_ABILITY)
      & (action[1] == 0)
      & (source >= 0)
      & (state.def_id[acting, safe_source] == cards.CODE_TO_ID["STT01-005"])
      & (deck_before > 3)
  )

  stepped = state._replace(
      ab_source=jnp.where(do_action, safe_source.astype(jnp.int8), state.ab_source),
      ab_owner=jnp.where(do_action, acting.astype(jnp.int8), state.ab_owner),
      ab_slot=jnp.where(do_action, jnp.int8(0), state.ab_slot),
      ab_is_optional=jnp.where(do_action, False, state.ab_is_optional),
      ab_costs_applied=jnp.where(do_action, False, state.ab_costs_applied),
      ab_saved_active=jnp.where(do_action, jnp.int8(-1), state.ab_saved_active),
      ab_restores_active=jnp.where(do_action, False, state.ab_restores_active),
      ab_cost_selected=jnp.where(do_action, jnp.int8(0), state.ab_cost_selected),
      ab_cost_max=jnp.where(do_action, jnp.int8(0), state.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_action, jnp.int8(0), state.ab_eff_selected),
      ab_eff_min=jnp.where(do_action, jnp.int8(2), state.ab_eff_min),
      ab_eff_max=jnp.where(do_action, jnp.int8(2), state.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_targets, -1),
          state.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_action,
          jnp.full_like(state.ab_eff_target_players, -1),
          state.ab_eff_target_players,
      ),
  )
  stepped = sacrifice_card(stepped, acting, safe_source, do=do_action)
  stepped = draw_with_deckout(stepped, acting, 3, do_action)
  has_discard_effect = discard_count > 0
  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_action & has_discard_effect,
          jnp.int8(AbilityPhase.EFFECT_SELECTION),
          stepped.ab_phase,
      ),
      ab_costs_applied=jnp.where(do_action, True, stepped.ab_costs_applied),
      ab_eff_min=jnp.where(do_action, discard_count, stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_action, discard_count, stepped.ab_eff_max),
  )
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(do_action & ~has_discard_effect, a, b),
      cleared,
      stepped,
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ACTIVATE_ALLEY_ABILITY, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_stt01_005_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-005 discard-target selection."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import sacrifice_card
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import hand_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  selected = jnp.clip(
      state.ab_eff_selected.astype(jnp.int32), 0, MAX_ABILITY_SELECTION - 1
  )
  target = hand_instance(state, owner, action[1])
  safe_target = jnp.maximum(target, 0)
  eff_max = state.ab_eff_max.astype(jnp.int8)
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT01-005"])
      & state.ab_costs_applied
      & (state.ab_eff_min == eff_max)
      & (eff_max >= 1)
      & (eff_max <= 2)
      & (state.ab_eff_selected < eff_max)
  )
  do_select = (
      ~(did_reset | zero_legal)
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & (target >= 0)
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[selected].set(
          jnp.where(
              do_select,
              safe_target.astype(jnp.int8),
              state.ab_eff_targets[selected],
          )
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[selected].set(
          jnp.where(
              do_select,
              owner.astype(jnp.int8),
              state.ab_eff_target_players[selected],
          )
      ),
      ab_eff_selected=jnp.where(
          do_select, state.ab_eff_selected + 1, state.ab_eff_selected
      ).astype(jnp.int8),
  )

  finish = do_select & (stepped.ab_eff_selected >= stepped.ab_eff_max)
  first = jnp.maximum(stepped.ab_eff_targets[0].astype(jnp.int32), 0)
  second = jnp.maximum(stepped.ab_eff_targets[1].astype(jnp.int32), 0)
  first_has = finish & (stepped.ab_eff_targets[0] >= 0)
  second_has = finish & (stepped.ab_eff_targets[1] >= 0)
  stepped = sacrifice_card(stepped, owner, first, do=first_has)
  second_not_discarded = stepped.zone[owner, second] != Zone.DISCARD
  stepped = sacrifice_card(
      stepped, owner, second, do=second_has & second_not_discarded
  )
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(finish, a, b), cleared, stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_effect_stt01_014_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-014 leader damage target selection."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import deal_effect_damage
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  target_index = action[1].astype(jnp.int32)
  target_player = jnp.where(target_index == 0, owner, (owner + 1) % 2)
  target = leader_instance(state, target_player)
  safe_target = jnp.maximum(target, 0)

  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (state.def_id[owner, src] == cards.CODE_TO_ID["STT01-014"])
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 0)
      & (state.ab_eff_max == 1)
  )
  target_ok = (
      (target_index >= 0)
      & (target_index <= 1)
      & (target >= 0)
      & (state.zone[target_player, safe_target] == Zone.LEADER)
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )

  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(
              do_select,
              target_player.astype(jnp.int8),
              state.ab_eff_target_players[0],
          )
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  stepped = deal_effect_damage(
      stepped, target_player, safe_target, 1, do=do_select
  )
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(lambda a, b: jnp.where(do_select, a, b), cleared, stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_attach_weapon_simple_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Narrow fast path for simple weapon attach and STT01-014 on-play."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.cards_impl import apply_charge_grant
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import (
      _detach_from_location,
      GRANT_PHASE_NONE,
      card_at_slot,
      hand_instance,
      leader_instance,
      weapons_of,
  )
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  entity_index = action[2]
  use_token = action[3] != 0
  weapon = hand_instance(state, acting, hand_index)
  safe_weapon = jnp.maximum(weapon, 0)
  target_is_leader = entity_index == GARDEN_SIZE
  leader = leader_instance(state, acting)
  garden_target = card_at_slot(state, acting, Zone.GARDEN, entity_index)
  target = jnp.where(target_is_leader, leader, garden_target)
  safe_target = jnp.maximum(target, 0)
  do_attach = do_action & (weapon >= 0) & (target >= 0)

  weapon_count = jnp.sum(weapons_of(state, acting, safe_target), dtype=jnp.int32)
  stepped = _detach_from_location(state, acting, safe_weapon, do_attach)
  stepped = stepped._replace(
      zone=stepped.zone.at[acting, safe_weapon].set(
          jnp.where(do_attach, jnp.int8(Zone.ATTACHED), stepped.zone[acting, safe_weapon])
      ),
      zpos=stepped.zpos.at[acting, safe_weapon].set(
          jnp.where(
              do_attach, weapon_count.astype(jnp.int8), stepped.zpos[acting, safe_weapon]
          )
      ),
      attached_to=stepped.attached_to.at[acting, safe_weapon].set(
          jnp.where(
              do_attach,
              safe_target.astype(jnp.int8),
              stepped.attached_to[acting, safe_weapon],
          )
      ),
  )

  weapon_atk = stepped.cur_atk[acting, safe_weapon].astype(jnp.int16)
  host_atk = stepped.cur_atk[acting, safe_target].astype(jnp.int16)
  new_atk = jnp.maximum(host_atk + weapon_atk, 0).astype(jnp.int8)
  stepped = stepped._replace(
      cur_atk=stepped.cur_atk.at[acting, safe_target].set(
          jnp.where(do_attach, new_atk, stepped.cur_atk[acting, safe_target])
      )
  )

  is_018 = stepped.def_id[acting, safe_weapon] == cards.CODE_TO_ID["AZK01-018"]
  apply_mod = do_attach & is_018 & target_is_leader
  stepped = stepped._replace(
      cmb_in_perm=stepped.cmb_in_perm.at[acting, safe_target].add(
          jnp.where(apply_mod, -1, 0).astype(jnp.int8)
      )
  )
  is_stt01_015 = stepped.def_id[acting, safe_weapon] == cards.CODE_TO_ID["STT01-015"]
  discard_count = jnp.sum(stepped.zone[acting] == Zone.DISCARD, dtype=jnp.int32)
  apply_stt01_015 = do_attach & is_stt01_015 & (discard_count >= 15)
  stt01_015_delta = jnp.where(apply_stt01_015, 1, 0).astype(jnp.int8)
  stepped = stepped._replace(
      cur_atk=stepped.cur_atk.at[acting, safe_weapon]
      .add(stt01_015_delta)
      .at[acting, safe_target]
      .add(stt01_015_delta)
  )
  target_def = stepped.def_id[acting, safe_target]
  apply_azk01_039 = do_attach & (target_def == cards.CODE_TO_ID["AZK01-039"])
  stepped = apply_charge_grant(
      stepped, acting, safe_target, GRANT_PHASE_NONE, -1, apply_azk01_039
  )



  cost = effective_play_cost(stepped, acting, safe_weapon)
  stepped = ikz.pay(stepped, acting, cost, use_token, do=do_attach)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_attach.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_attach, 0, stepped.next_play_cost_reduction[acting])
      ),
  )
  stepped = recompute_passives(stepped)
  begin_stt01_014 = do_attach & (
      stepped.def_id[acting, safe_weapon] == cards.CODE_TO_ID["STT01-014"]
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          begin_stt01_014,
          jnp.int8(AbilityPhase.EFFECT_SELECTION),
          stepped.ab_phase,
      ),
      ab_source=jnp.where(
          begin_stt01_014, safe_weapon.astype(jnp.int8), stepped.ab_source
      ),
      ab_owner=jnp.where(
          begin_stt01_014, acting.astype(jnp.int8), stepped.ab_owner
      ),
      ab_slot=jnp.where(begin_stt01_014, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(begin_stt01_014, False, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          begin_stt01_014, False, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          begin_stt01_014, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          begin_stt01_014, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          begin_stt01_014, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(begin_stt01_014, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          begin_stt01_014,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          begin_stt01_014,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          begin_stt01_014, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(begin_stt01_014, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(begin_stt01_014, jnp.int8(1), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          begin_stt01_014,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          begin_stt01_014,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )

  # HandleResponseAction head parity: after a response-window attach, C's
  # next tick auto-transitions to combat resolve (and resolves clean combat)
  # when no queued effects remain and the defender has no further response
  # options. The host mask only admits response rows whose combat is clean
  # (no when-attacked/after-attacking/damage/destroy timings), so
  # transition_to_combat_resolve queues nothing and combat_resolve leaves no
  # queued work.
  from azuki_jax.engine.phases import (
      combat_resolve,
      defender_can_respond,
      transition_to_combat_resolve,
  )
  from azuki_jax.engine.triggers import has_queued

  response_attach = (
      do_attach
      & (stepped.phase == Phase.RESPONSE_WINDOW)
      & (stepped.ab_phase == 0)
      & (stepped.combat_attacker >= 0)
      & (stepped.combat_defender_player == acting.astype(jnp.int8))
  )
  auto_close = (
      response_attach
      & ~has_queued(stepped)
      & ~defender_can_respond(stepped, acting)
  )
  resolved = transition_to_combat_resolve(stepped, do=auto_close)
  resolved = combat_resolve(resolved, do=auto_close & ~has_queued(resolved))
  resolved = recompute_passives(resolved)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(auto_close, a, b), resolved, stepped
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ATTACH_WEAPON_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_attach_stt01_013_confirm_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-013 attach that opens optional on-play confirmation."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine import ikz
  from azuki_jax.engine.helpers import (
      _detach_from_location,
      card_at_slot,
      hand_instance,
      leader_instance,
      weapons_of,
  )
  from azuki_jax.engine.validate import effective_play_cost

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  hand_index = action[1]
  entity_index = action[2]
  use_token = action[3] != 0
  weapon = hand_instance(state, acting, hand_index)
  safe_weapon = jnp.maximum(weapon, 0)
  target_is_leader = entity_index == GARDEN_SIZE
  leader = leader_instance(state, acting)
  safe_leader = jnp.maximum(leader, 0)
  garden_target = card_at_slot(state, acting, Zone.GARDEN, entity_index)
  target = jnp.where(target_is_leader, leader, garden_target)
  safe_target = jnp.maximum(target, 0)
  is_dagger = state.def_id[acting, safe_weapon] == cards.CODE_TO_ID["STT01-013"]
  leader_hp_ok = (leader >= 0) & (state.cur_hp[acting, safe_leader] >= 1)
  do_attach = do_action & (weapon >= 0) & (target >= 0) & is_dagger & leader_hp_ok

  weapon_count = jnp.sum(weapons_of(state, acting, safe_target), dtype=jnp.int32)
  stepped = _detach_from_location(state, acting, safe_weapon, do_attach)
  stepped = stepped._replace(
      zone=stepped.zone.at[acting, safe_weapon].set(
          jnp.where(do_attach, jnp.int8(Zone.ATTACHED), stepped.zone[acting, safe_weapon])
      ),
      zpos=stepped.zpos.at[acting, safe_weapon].set(
          jnp.where(
              do_attach, weapon_count.astype(jnp.int8), stepped.zpos[acting, safe_weapon]
          )
      ),
      attached_to=stepped.attached_to.at[acting, safe_weapon].set(
          jnp.where(
              do_attach,
              safe_target.astype(jnp.int8),
              stepped.attached_to[acting, safe_weapon],
          )
      ),
  )

  weapon_atk = stepped.cur_atk[acting, safe_weapon].astype(jnp.int16)
  host_atk = stepped.cur_atk[acting, safe_target].astype(jnp.int16)
  new_atk = jnp.maximum(host_atk + weapon_atk, 0).astype(jnp.int8)
  stepped = stepped._replace(
      cur_atk=stepped.cur_atk.at[acting, safe_target].set(
          jnp.where(do_attach, new_atk, stepped.cur_atk[acting, safe_target])
      )
  )

  cost = effective_play_cost(stepped, acting, safe_weapon)
  stepped = ikz.pay(stepped, acting, cost, use_token, do=do_attach)
  stepped = stepped._replace(
      cards_played_turn=stepped.cards_played_turn.at[acting].add(
          do_attach.astype(jnp.uint8)
      ),
      next_play_cost_reduction=stepped.next_play_cost_reduction.at[acting].set(
          jnp.where(do_attach, 0, stepped.next_play_cost_reduction[acting])
      ),
  )
  stepped = recompute_passives(stepped)

  stepped = stepped._replace(
      ab_phase=jnp.where(
          do_attach, jnp.int8(AbilityPhase.CONFIRMATION), stepped.ab_phase
      ),
      ab_source=jnp.where(do_attach, safe_weapon.astype(jnp.int8), stepped.ab_source),
      ab_owner=jnp.where(do_attach, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(do_attach, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(do_attach, True, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(do_attach, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(do_attach, jnp.int8(-1), stepped.ab_saved_active),
      ab_restores_active=jnp.where(do_attach, False, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(do_attach, jnp.int8(0), stepped.ab_cost_selected),
      ab_cost_max=jnp.where(do_attach, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          do_attach,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          do_attach,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(do_attach, jnp.int8(0), stepped.ab_eff_selected),
      ab_eff_min=jnp.where(do_attach, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(do_attach, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          do_attach,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          do_attach,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ATTACH_WEAPON_FROM_HAND, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_attack_entity_mutual_destroy_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Narrow fast path for simple garden entity combat."""
  from azuki_jax import cards
  from azuki_jax.abilities import tables as ab_tables
  from azuki_jax.abilities.cards_impl import (
      apply_attack_modifier,
      apply_shocked,
      draw_with_deckout,
      heal_leader,
  )
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.helpers import card_at_slot, discard, total_carapace
  from azuki_jax.engine.triggers import (
      TIMING_WHEN_TAKES_DAMAGE,
      pop_effect,
      record_damage_event,
  )
  from azuki_jax.zones import zone_count

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  opp = (acting + 1) % 2
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  attacker = card_at_slot(state, acting, Zone.GARDEN, action[1])
  defender = card_at_slot(state, opp, Zone.GARDEN, action[2])
  safe_attacker = jnp.maximum(attacker, 0)
  safe_defender = jnp.maximum(defender, 0)
  attacker_def = state.def_id[acting, safe_attacker]
  defender_def = state.def_id[opp, safe_defender]
  attacker_godmode = (
      jnp.asarray(cards.INHERENT_GODMODE)[jnp.maximum(attacker_def, 0)]
      | state.grant_godmode[acting, safe_attacker]
  )
  defender_godmode = (
      jnp.asarray(cards.INHERENT_GODMODE)[jnp.maximum(defender_def, 0)]
      | state.grant_godmode[opp, safe_defender]
  )
  defender_frozen = state.frozen_dur[opp, safe_defender] > 0
  do_combat = do_action & (attacker >= 0) & (defender >= 0)

  raw_damage_to_defender = jnp.maximum(
      state.cur_atk[acting, safe_attacker].astype(jnp.int16)
      - total_carapace(state, opp, safe_defender),
      0,
  )
  raw_damage_to_attacker = jnp.maximum(
      state.cur_atk[opp, safe_defender].astype(jnp.int16)
      - total_carapace(state, acting, safe_attacker),
      0,
  )
  damage_to_defender = jnp.where(
      defender_frozen, 0, raw_damage_to_defender
  )
  damage_to_attacker = jnp.where(
      defender_frozen, 0, raw_damage_to_attacker
  )
  attacker_hp = state.cur_hp[acting, safe_attacker].astype(jnp.int16)
  defender_hp = state.cur_hp[opp, safe_defender].astype(jnp.int16)
  new_attacker_hp = attacker_hp - damage_to_attacker
  new_defender_hp = defender_hp - damage_to_defender
  # C damage_util: godmode does NOT prevent damage — it clamps negative HP
  # at 0 and prevents death. A godmode entity can sit in play at 0 HP.
  new_attacker_hp = jnp.where(
      attacker_godmode & (new_attacker_hp < 0), 0, new_attacker_hp
  )
  new_defender_hp = jnp.where(
      defender_godmode & (new_defender_hp < 0), 0, new_defender_hp
  )
  attacker_dead = do_combat & (new_attacker_hp <= 0) & ~attacker_godmode
  defender_dead = do_combat & (new_defender_hp <= 0) & ~defender_godmode

  stepped = state._replace(
      tapped=state.tapped.at[acting, safe_attacker].set(
          jnp.where(
              do_combat, True, state.tapped[acting, safe_attacker]
          )
      ),
      cur_hp=state.cur_hp.at[acting, safe_attacker]
      .set(
          jnp.where(
              do_combat,
              new_attacker_hp.astype(jnp.int8),
              state.cur_hp[acting, safe_attacker],
          )
      )
      .at[opp, safe_defender]
      .set(
          jnp.where(
              do_combat,
              new_defender_hp.astype(jnp.int8),
              state.cur_hp[opp, safe_defender],
          )
      ),
  )
  attacker_stt03_006_destroyed = (
      attacker_dead & (attacker_def == cards.CODE_TO_ID["STT03-006"])
  )
  defender_stt03_006_destroyed = (
      defender_dead & (defender_def == cards.CODE_TO_ID["STT03-006"])
  )
  stt03_006_destroyed = (
      attacker_stt03_006_destroyed | defender_stt03_006_destroyed
  )
  stt03_006_owner = jnp.where(
      attacker_stt03_006_destroyed, acting, opp
  ).astype(jnp.int32)
  stt03_006_source = jnp.where(
      attacker_stt03_006_destroyed, safe_attacker, safe_defender
  ).astype(jnp.int32)
  azk01_047_heal = (
      do_combat
      & (attacker_def == cards.CODE_TO_ID["AZK01-047"])
      & ((state.once_per_turn_used[acting, safe_attacker] & jnp.uint8(1)) == 0)
  )
  stepped = heal_leader(stepped, acting, 1, azk01_047_heal)
  once_used = stepped.once_per_turn_used[acting, safe_attacker]
  stepped = stepped._replace(
      once_per_turn_used=stepped.once_per_turn_used.at[
          acting, safe_attacker
      ].set(jnp.where(azk01_047_heal, once_used | jnp.uint8(1), once_used))
  )
  stepped = record_damage_event(
      stepped,
      opp,
      safe_defender,
      acting,
      safe_attacker,
      attacker_hp - new_attacker_hp,
      do_combat,
      from_effect=False,
  )
  stepped = record_damage_event(
      stepped,
      acting,
      safe_attacker,
      opp,
      safe_defender,
      defender_hp - new_defender_hp,
      do_combat,
      from_effect=False,
  )
  attacker_azk01_062_fizzle = (
      do_combat
      & (state.redirect_count == 0)
      & (state.def_id[acting, safe_attacker] == cards.CODE_TO_ID["AZK01-062"])
      & (stepped.trig_count > 0)
      & (stepped.trig_owner[0] == acting.astype(jnp.int8))
      & (stepped.trig_source[0] == safe_attacker.astype(jnp.int8))
      & (stepped.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
  )
  defender_azk01_062_fizzle = (
      do_combat
      & (state.redirect_count == 0)
      & (state.def_id[opp, safe_defender] == cards.CODE_TO_ID["AZK01-062"])
      & (stepped.trig_count > 0)
      & (stepped.trig_owner[0] == opp.astype(jnp.int8))
      & (stepped.trig_source[0] == safe_defender.astype(jnp.int8))
      & (stepped.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
  )
  azk01_062_fizzle = attacker_azk01_062_fizzle | defender_azk01_062_fizzle
  popped_sources = jnp.roll(stepped.trig_source, -1).at[-1].set(-1)
  popped_owners = jnp.roll(stepped.trig_owner, -1).at[-1].set(-1)
  popped_timings = jnp.roll(stepped.trig_timing, -1).at[-1].set(-1)
  stepped = stepped._replace(
      trig_source=jnp.where(azk01_062_fizzle, popped_sources, stepped.trig_source),
      trig_owner=jnp.where(azk01_062_fizzle, popped_owners, stepped.trig_owner),
      trig_timing=jnp.where(azk01_062_fizzle, popped_timings, stepped.trig_timing),
      trig_count=jnp.where(
          azk01_062_fizzle,
          jnp.maximum(stepped.trig_count - 1, 0),
          stepped.trig_count,
      ).astype(jnp.int8),
  )
  safe_defender_def = jnp.maximum(defender_def, 0)
  azk01_036_when_attacked = (
      do_combat & (defender_def == cards.CODE_TO_ID["AZK01-036"])
  )
  stepped = apply_shocked(
      stepped, acting, safe_attacker, 1, do=azk01_036_when_attacked
  )
  unimplemented_when_attacked = (
      do_combat
      & (defender_def >= 0)
      & jnp.asarray(ab_tables.TIMING_WHEN_ATTACKED)[safe_defender_def]
      & jnp.asarray(ab_tables.HAS_ABILITY)[safe_defender_def]
      & ~jnp.asarray(ab_tables.IMPLEMENTED)[safe_defender_def]
  )
  stepped = stepped._replace(
      ab_scratch=stepped.ab_scratch.at[3].add(
          unimplemented_when_attacked.astype(jnp.int16)
      )
  )
  stepped = discard(stepped, acting, safe_attacker, do=attacker_dead)
  stepped = discard(stepped, opp, safe_defender, do=defender_dead)
  stepped = stepped._replace(
      combat_attacker=jnp.int8(-1),
      combat_defender=jnp.int8(-1),
      combat_defender_player=jnp.int8(-1),
      combat_intercepted=jnp.bool_(False),
      phase=jnp.int8(Phase.MAIN),
  )
  stepped = recompute_passives(stepped)
  head_src = jnp.maximum(stepped.trig_source[0].astype(jnp.int32), 0)
  head_owner = jnp.maximum(stepped.trig_owner[0].astype(jnp.int32), 0)
  stt04_007_trigger = (
      do_combat
      & (stepped.trig_count > 0)
      & (stepped.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (stepped.def_id[head_owner, head_src] == cards.CODE_TO_ID["STT04-007"])
  )
  stt04_007_to_effect = (
      stt04_007_trigger
      & ((stepped.once_per_turn_used[head_owner, head_src] & jnp.uint8(1)) == 0)
      & (
          (stepped.zone[head_owner, head_src] == Zone.GARDEN)
          | (stepped.zone[head_owner, head_src] == Zone.ALLEY)
      )
  )
  popped_stt04_007, _, _, _ = pop_effect(stepped)
  buffed_stt04_007 = apply_attack_modifier(
      popped_stt04_007, head_owner, head_src, 1, True, stt04_007_to_effect
  )
  stt04_007_once = buffed_stt04_007.once_per_turn_used[head_owner, head_src]
  buffed_stt04_007 = buffed_stt04_007._replace(
      once_per_turn_used=buffed_stt04_007.once_per_turn_used.at[
          head_owner, head_src
      ].set(
          jnp.where(
              stt04_007_to_effect,
              stt04_007_once | jnp.uint8(1),
              stt04_007_once,
          )
      )
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(stt04_007_trigger, a, b),
      buffed_stt04_007,
      stepped,
  )
  head_src = jnp.maximum(stepped.trig_source[0].astype(jnp.int32), 0)
  head_owner = jnp.maximum(stepped.trig_owner[0].astype(jnp.int32), 0)
  stt04_009_combat_fizzle = (
      do_combat
      & (stepped.trig_count > 0)
      & (stepped.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (stepped.def_id[head_owner, head_src] == cards.CODE_TO_ID["STT04-009"])
      & ~stepped.last_dmg_from_effect[head_owner, head_src]
  )
  popped_fizzle, _, _, _ = pop_effect(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(stt04_009_combat_fizzle, a, b),
      popped_fizzle,
      stepped,
  )
  trig_src0 = jnp.maximum(stepped.trig_source[0].astype(jnp.int32), 0)
  trig_owner0 = jnp.maximum(stepped.trig_owner[0].astype(jnp.int32), 0)
  azk01_059_in_play = (
      (stepped.zone[trig_owner0, trig_src0] == Zone.GARDEN)
      | (stepped.zone[trig_owner0, trig_src0] == Zone.ALLEY)
  )
  azk01_059_dead_fizzle = (
      do_combat
      & (stepped.trig_count > 0)
      & (stepped.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (stepped.def_id[trig_owner0, trig_src0] == cards.CODE_TO_ID["AZK01-059"])
      & ~azk01_059_in_play
  )
  popped_azk01_059_fizzle, _, _, _ = pop_effect(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(azk01_059_dead_fizzle, a, b),
      popped_azk01_059_fizzle,
      stepped,
  )
  trig_src = jnp.maximum(stepped.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(stepped.trig_owner[0].astype(jnp.int32), 0)
  azk01_059_to_effect = (
      do_combat
      & (stepped.trig_count > 0)
      & (stepped.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (stepped.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-059"])
      & ((stepped.once_per_turn_used[trig_owner, trig_src] & jnp.uint8(1)) == 0)
      & (
          (stepped.zone[trig_owner, trig_src] == Zone.GARDEN)
          | (stepped.zone[trig_owner, trig_src] == Zone.ALLEY)
      )
  )
  popped, src2, owner2, _ = pop_effect(stepped)
  owner2_i32 = owner2.astype(jnp.int32)
  needs_transfer = popped.active_player != owner2.astype(jnp.int8)
  begun = popped._replace(
      ab_source=src2.astype(jnp.int8),
      ab_owner=owner2.astype(jnp.int8),
      ab_is_optional=jnp.bool_(False),
      ab_costs_applied=jnp.bool_(False),
      ab_saved_active=jnp.where(needs_transfer, popped.active_player, jnp.int8(-1)),
      ab_restores_active=needs_transfer,
      ab_cost_selected=jnp.int8(0),
      ab_cost_max=jnp.int8(0),
      ab_eff_selected=jnp.int8(0),
      ab_eff_min=jnp.int8(1),
      ab_eff_max=jnp.int8(1),
      ab_phase=jnp.int8(AbilityPhase.EFFECT_SELECTION),
      active_player=jnp.where(
          needs_transfer, owner2_i32.astype(jnp.int8), popped.active_player
      ),
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(azk01_059_to_effect, a, b), begun, stepped
  )
  stepped = draw_with_deckout(
      stepped, stt03_006_owner, 1, stt03_006_destroyed
  )
  stt03_006_to_effect = (
      stt03_006_destroyed
      & (stepped.winner == -1)
      & (zone_count(stepped.zone[stt03_006_owner], Zone.HAND) > 0)
  )
  stt03_006_transfer = stt03_006_to_effect & (
      stepped.active_player != stt03_006_owner.astype(jnp.int8)
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          stt03_006_to_effect,
          jnp.int8(AbilityPhase.EFFECT_SELECTION),
          stepped.ab_phase,
      ),
      ab_source=jnp.where(
          stt03_006_to_effect, stt03_006_source.astype(jnp.int8), stepped.ab_source
      ),
      ab_owner=jnp.where(
          stt03_006_to_effect, stt03_006_owner.astype(jnp.int8), stepped.ab_owner
      ),
      active_player=jnp.where(
          stt03_006_to_effect,
          stt03_006_owner.astype(jnp.int8),
          stepped.active_player,
      ),
      ab_slot=jnp.where(stt03_006_to_effect, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(
          stt03_006_to_effect, jnp.bool_(False), stepped.ab_is_optional
      ),
      ab_costs_applied=jnp.where(
          stt03_006_to_effect, jnp.bool_(True), stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          stt03_006_to_effect,
          jnp.where(stt03_006_transfer, stepped.active_player, jnp.int8(-1)),
          stepped.ab_saved_active,
      ),
      ab_restores_active=jnp.where(
          stt03_006_to_effect, stt03_006_transfer, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          stt03_006_to_effect, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(
          stt03_006_to_effect, jnp.int8(0), stepped.ab_cost_max
      ),
      ab_eff_selected=jnp.where(
          stt03_006_to_effect, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(
          stt03_006_to_effect, jnp.int8(1), stepped.ab_eff_min
      ),
      ab_eff_max=jnp.where(
          stt03_006_to_effect, jnp.int8(1), stepped.ab_eff_max
      ),
      ab_cost_targets=jnp.where(
          stt03_006_to_effect,
          jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          stt03_006_to_effect,
          jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
          stepped.ab_cost_target_players,
      ),
      ab_eff_targets=jnp.where(
          stt03_006_to_effect,
          jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          stt03_006_to_effect,
          jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
          stepped.ab_eff_target_players,
      ),
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ATTACK, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_response_noop_combat_fizzle_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast response pass when pending combat will fizzle."""
  from azuki_jax.engine.phases import combat_resolve, transition_to_combat_resolve
  from azuki_jax.abilities.passives import recompute_passives

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  can_pass = (
      ~(did_reset | zero_legal)
      & (action_type == Act.NOOP)
      & (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == AbilityPhase.NONE)
      & (state.trig_count == 0)
      & (state.redirect_count == 0)
      & (state.combat_attacker >= 0)
      & (state.combat_defender >= 0)
      & (state.combat_defender_player == acting.astype(jnp.int8))
  )

  prev = state
  stepped = transition_to_combat_resolve(state, do=can_pass)
  stepped = combat_resolve(stepped, do=can_pass)
  stepped = recompute_passives(stepped)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.NOOP, jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_response_noop_entity_combat_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast response pass for clean entity-vs-entity combat."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.cards_impl import apply_attack_modifier, draw_with_deckout
  from azuki_jax.engine.helpers import discard, total_carapace
  from azuki_jax.engine.phases import _lightning_kanabo
  from azuki_jax.engine.triggers import (
      TIMING_WHEN_TAKES_DAMAGE,
      pop_effect,
      record_damage_event,
  )
  from azuki_jax.zones import zone_count

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)

  prev = state
  defender_p = acting
  attacker_p = (defender_p + 1) % 2
  attacker = state.combat_attacker.astype(jnp.int32)
  defender = state.combat_defender.astype(jnp.int32)
  safe_attacker = jnp.maximum(attacker, 0)
  safe_defender = jnp.maximum(defender, 0)
  do_combat = (
      ~(did_reset | zero_legal)
      & (action_type == Act.NOOP)
      & (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == 0)
      & (state.combat_attacker >= 0)
      & (state.combat_defender >= 0)
      & (state.combat_defender_player == defender_p.astype(jnp.int8))
      & (state.zone[attacker_p, safe_attacker] == Zone.GARDEN)
      & (
          (state.zone[defender_p, safe_defender] == Zone.GARDEN)
          | (state.zone[defender_p, safe_defender] == Zone.ALLEY)
      )
  )

  damage_to_attacker = jnp.maximum(
      state.cur_atk[defender_p, safe_defender].astype(jnp.int16)
      - total_carapace(state, attacker_p, safe_attacker),
      0,
  )
  damage_to_defender = jnp.maximum(
      state.cur_atk[attacker_p, safe_attacker].astype(jnp.int16)
      - total_carapace(state, defender_p, safe_defender),
      0,
  )
  attacker_hp = state.cur_hp[attacker_p, safe_attacker].astype(jnp.int16)
  defender_hp = state.cur_hp[defender_p, safe_defender].astype(jnp.int16)
  new_attacker_hp = attacker_hp - damage_to_attacker
  new_defender_hp = defender_hp - damage_to_defender
  both_started_alive = (attacker_hp > 0) & (defender_hp > 0)
  resolve = do_combat & both_started_alive
  attacker_dead = resolve & (new_attacker_hp <= 0)
  defender_dead = resolve & (new_defender_hp <= 0)
  attacker_def = state.def_id[attacker_p, safe_attacker]
  defender_def = state.def_id[defender_p, safe_defender]
  attacker_stt03_006_destroyed = (
      attacker_dead & (attacker_def == cards.CODE_TO_ID["STT03-006"])
  )
  defender_stt03_006_destroyed = (
      defender_dead & (defender_def == cards.CODE_TO_ID["STT03-006"])
  )
  stt03_006_destroyed = (
      attacker_stt03_006_destroyed | defender_stt03_006_destroyed
  )
  stt03_006_owner = jnp.where(
      attacker_stt03_006_destroyed, attacker_p, defender_p
  ).astype(jnp.int32)
  stt03_006_source = jnp.where(
      attacker_stt03_006_destroyed, safe_attacker, safe_defender
  ).astype(jnp.int32)

  stepped = state._replace(
      cur_hp=state.cur_hp.at[attacker_p, safe_attacker]
      .set(
          jnp.where(
              resolve,
              new_attacker_hp.astype(jnp.int8),
              state.cur_hp[attacker_p, safe_attacker],
          )
      )
      .at[defender_p, safe_defender]
      .set(
          jnp.where(
              resolve,
              new_defender_hp.astype(jnp.int8),
              state.cur_hp[defender_p, safe_defender],
          )
      ),
  )
  stepped = _lightning_kanabo(
      stepped,
      attacker_p,
      safe_attacker,
      defender_p,
      safe_defender,
      jnp.where(resolve, damage_to_defender, 0),
  )
  stepped = _lightning_kanabo(
      stepped,
      defender_p,
      safe_defender,
      attacker_p,
      safe_attacker,
      jnp.where(resolve, damage_to_attacker, 0),
  )
  stepped = record_damage_event(
      stepped,
      defender_p,
      safe_defender,
      attacker_p,
      safe_attacker,
      damage_to_attacker,
      resolve,
      from_effect=False,
  )
  stepped = record_damage_event(
      stepped,
      attacker_p,
      safe_attacker,
      defender_p,
      safe_defender,
      damage_to_defender,
      resolve,
      from_effect=False,
  )
  azk01_062_attacker_fizzle = (
      resolve
      & (state.redirect_count == 0)
      & (attacker_def == cards.CODE_TO_ID["AZK01-062"])
      & (stepped.trig_count > 0)
      & (stepped.trig_owner[0] == attacker_p.astype(jnp.int8))
      & (stepped.trig_source[0] == safe_attacker.astype(jnp.int8))
      & (stepped.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
  )
  azk01_062_defender_fizzle = (
      resolve
      & (state.redirect_count == 0)
      & (defender_def == cards.CODE_TO_ID["AZK01-062"])
      & (stepped.trig_count > 0)
      & (stepped.trig_owner[0] == defender_p.astype(jnp.int8))
      & (stepped.trig_source[0] == safe_defender.astype(jnp.int8))
      & (stepped.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
  )
  azk01_062_fizzle = azk01_062_attacker_fizzle | azk01_062_defender_fizzle
  popped_sources = jnp.roll(stepped.trig_source, -1).at[-1].set(-1)
  popped_owners = jnp.roll(stepped.trig_owner, -1).at[-1].set(-1)
  popped_timings = jnp.roll(stepped.trig_timing, -1).at[-1].set(-1)
  stepped = stepped._replace(
      trig_source=jnp.where(azk01_062_fizzle, popped_sources, stepped.trig_source),
      trig_owner=jnp.where(azk01_062_fizzle, popped_owners, stepped.trig_owner),
      trig_timing=jnp.where(azk01_062_fizzle, popped_timings, stepped.trig_timing),
      trig_count=jnp.where(
          azk01_062_fizzle,
          jnp.maximum(stepped.trig_count - 1, 0),
          stepped.trig_count,
      ).astype(jnp.int8),
  )
  head_src_stt04_009 = jnp.maximum(stepped.trig_source[0].astype(jnp.int32), 0)
  head_owner_stt04_009 = jnp.maximum(stepped.trig_owner[0].astype(jnp.int32), 0)
  stt04_009_combat_fizzle = (
      resolve
      & (stepped.trig_count > 0)
      & (stepped.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (
          stepped.def_id[head_owner_stt04_009, head_src_stt04_009]
          == cards.CODE_TO_ID["STT04-009"]
      )
      & ~stepped.last_dmg_from_effect[head_owner_stt04_009, head_src_stt04_009]
  )
  popped_stt04_009, _, _, _ = pop_effect(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(stt04_009_combat_fizzle, a, b),
      popped_stt04_009,
      stepped,
  )
  stepped = discard(stepped, attacker_p, safe_attacker, do=attacker_dead)
  stepped = discard(stepped, defender_p, safe_defender, do=defender_dead)
  stepped = recompute_passives(stepped)
  stepped = stepped._replace(
      combat_attacker=jnp.where(resolve, jnp.int8(-1), stepped.combat_attacker),
      combat_defender=jnp.where(resolve, jnp.int8(-1), stepped.combat_defender),
      combat_defender_player=jnp.where(
          resolve, jnp.int8(-1), stepped.combat_defender_player
      ),
      combat_intercepted=jnp.where(resolve, False, stepped.combat_intercepted),
      phase=jnp.where(resolve, jnp.int8(Phase.MAIN), stepped.phase),
      active_player=jnp.where(
          resolve, attacker_p.astype(jnp.int8), stepped.active_player
      ),
  )
  trig_src_stt04_007 = jnp.maximum(stepped.trig_source[0].astype(jnp.int32), 0)
  trig_owner_stt04_007 = jnp.maximum(stepped.trig_owner[0].astype(jnp.int32), 0)
  stt04_007_trigger = (
      resolve
      & (stepped.trig_count > 0)
      & (stepped.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (
          stepped.def_id[trig_owner_stt04_007, trig_src_stt04_007]
          == cards.CODE_TO_ID["STT04-007"]
      )
  )
  stt04_007_to_effect = (
      stt04_007_trigger
      & (
          (stepped.once_per_turn_used[trig_owner_stt04_007, trig_src_stt04_007]
           & jnp.uint8(1))
          == 0
      )
      & (
          (stepped.zone[trig_owner_stt04_007, trig_src_stt04_007] == Zone.GARDEN)
          | (stepped.zone[trig_owner_stt04_007, trig_src_stt04_007] == Zone.ALLEY)
      )
  )
  popped_stt04_007, _, _, _ = pop_effect(stepped)
  buffed_stt04_007 = apply_attack_modifier(
      popped_stt04_007,
      trig_owner_stt04_007,
      trig_src_stt04_007,
      1,
      expires_eot=True,
      do=stt04_007_to_effect,
  )
  stt04_007_once = buffed_stt04_007.once_per_turn_used[
      trig_owner_stt04_007, trig_src_stt04_007
  ]
  buffed_stt04_007 = buffed_stt04_007._replace(
      once_per_turn_used=buffed_stt04_007.once_per_turn_used.at[
          trig_owner_stt04_007, trig_src_stt04_007
      ].set(
          jnp.where(
              stt04_007_to_effect,
              stt04_007_once | jnp.uint8(1),
              stt04_007_once,
          )
      )
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(stt04_007_trigger, a, b),
      buffed_stt04_007,
      stepped,
  )
  trig_src_059 = jnp.maximum(stepped.trig_source[0].astype(jnp.int32), 0)
  trig_owner_059 = jnp.maximum(stepped.trig_owner[0].astype(jnp.int32), 0)
  owner_defs_059 = stepped.def_id[trig_owner_059]
  owner_safe_defs_059 = jnp.maximum(owner_defs_059, 0)
  inst_axis_059 = jnp.arange(stepped.zone.shape[1], dtype=jnp.int32)
  azk01_059_other_garden_entity = jnp.any(
      (stepped.zone[trig_owner_059] == Zone.GARDEN)
      & (inst_axis_059 != trig_src_059)
      & (owner_defs_059 >= 0)
      & (jnp.asarray(cards.TYPE)[owner_safe_defs_059] == CardType.ENTITY)
  )
  azk01_059_to_effect = (
      resolve
      & (stepped.trig_count > 0)
      & (stepped.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (stepped.def_id[trig_owner_059, trig_src_059] == cards.CODE_TO_ID["AZK01-059"])
      & ((stepped.once_per_turn_used[trig_owner_059, trig_src_059] & jnp.uint8(1)) == 0)
      & azk01_059_other_garden_entity
  )
  popped_059, src_059, owner_059, _ = pop_effect(stepped)
  owner_059_i32 = owner_059.astype(jnp.int32)
  needs_transfer_059 = popped_059.active_player != owner_059.astype(jnp.int8)
  begun_059 = popped_059._replace(
      ab_phase=jnp.int8(AbilityPhase.EFFECT_SELECTION),
      ab_source=src_059.astype(jnp.int8),
      ab_owner=owner_059.astype(jnp.int8),
      ab_slot=jnp.int8(0),
      ab_is_optional=jnp.bool_(False),
      ab_costs_applied=jnp.bool_(False),
      ab_saved_active=jnp.where(
          needs_transfer_059, popped_059.active_player, jnp.int8(-1)
      ),
      ab_restores_active=needs_transfer_059,
      ab_cost_selected=jnp.int8(0),
      ab_cost_max=jnp.int8(0),
      ab_cost_targets=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_cost_target_players=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_eff_selected=jnp.int8(0),
      ab_eff_min=jnp.int8(1),
      ab_eff_max=jnp.int8(1),
      ab_eff_targets=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_eff_target_players=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      active_player=jnp.where(
          needs_transfer_059, owner_059_i32.astype(jnp.int8), popped_059.active_player
      ),
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(azk01_059_to_effect, a, b), begun_059, stepped
  )
  stepped = draw_with_deckout(
      stepped, stt03_006_owner, 1, stt03_006_destroyed
  )
  stt03_006_to_effect = (
      stt03_006_destroyed
      & (stepped.winner == -1)
      & (zone_count(stepped.zone[stt03_006_owner], Zone.HAND) > 0)
  )
  stt03_006_transfer = stt03_006_to_effect & (
      stepped.active_player != stt03_006_owner.astype(jnp.int8)
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          stt03_006_to_effect,
          jnp.int8(AbilityPhase.EFFECT_SELECTION),
          stepped.ab_phase,
      ),
      ab_source=jnp.where(
          stt03_006_to_effect, stt03_006_source.astype(jnp.int8), stepped.ab_source
      ),
      ab_owner=jnp.where(
          stt03_006_to_effect, stt03_006_owner.astype(jnp.int8), stepped.ab_owner
      ),
      active_player=jnp.where(
          stt03_006_to_effect,
          stt03_006_owner.astype(jnp.int8),
          stepped.active_player,
      ),
      ab_slot=jnp.where(stt03_006_to_effect, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(
          stt03_006_to_effect, jnp.bool_(False), stepped.ab_is_optional
      ),
      ab_costs_applied=jnp.where(
          stt03_006_to_effect, jnp.bool_(True), stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          stt03_006_to_effect,
          jnp.where(stt03_006_transfer, stepped.active_player, jnp.int8(-1)),
          stepped.ab_saved_active,
      ),
      ab_restores_active=jnp.where(
          stt03_006_to_effect, stt03_006_transfer, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          stt03_006_to_effect, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(
          stt03_006_to_effect, jnp.int8(0), stepped.ab_cost_max
      ),
      ab_eff_selected=jnp.where(
          stt03_006_to_effect, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(
          stt03_006_to_effect, jnp.int8(1), stepped.ab_eff_min
      ),
      ab_eff_max=jnp.where(
          stt03_006_to_effect, jnp.int8(1), stepped.ab_eff_max
      ),
      ab_cost_targets=jnp.where(
          stt03_006_to_effect,
          jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          stt03_006_to_effect,
          jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
          stepped.ab_cost_target_players,
      ),
      ab_eff_targets=jnp.where(
          stt03_006_to_effect,
          jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          stt03_006_to_effect,
          jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
          stepped.ab_eff_target_players,
      ),
  )

  azk01_058_confirm = (
      resolve
      & ~attacker_dead
      & ~stt03_006_to_effect
      & (attacker_def == cards.CODE_TO_ID["AZK01-058"])
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          azk01_058_confirm,
          jnp.int8(AbilityPhase.CONFIRMATION),
          stepped.ab_phase,
      ),
      ab_source=jnp.where(
          azk01_058_confirm, safe_attacker.astype(jnp.int8), stepped.ab_source
      ),
      ab_owner=jnp.where(
          azk01_058_confirm, attacker_p.astype(jnp.int8), stepped.ab_owner
      ),
      ab_slot=jnp.where(azk01_058_confirm, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(azk01_058_confirm, True, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          azk01_058_confirm, False, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          azk01_058_confirm, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          azk01_058_confirm, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          azk01_058_confirm, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(
          azk01_058_confirm, jnp.int8(0), stepped.ab_cost_max
      ),
      ab_cost_targets=jnp.where(
          azk01_058_confirm,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          azk01_058_confirm,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          azk01_058_confirm, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(
          azk01_058_confirm, jnp.int8(1), stepped.ab_eff_min
      ),
      ab_eff_max=jnp.where(
          azk01_058_confirm, jnp.int8(1), stepped.ab_eff_max
      ),
      ab_eff_targets=jnp.where(
          azk01_058_confirm,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          azk01_058_confirm,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.NOOP, jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_response_noop_azk01_040_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast response pass that queues AZK01-040's when-attacked effect."""
  from azuki_jax import cards
  from azuki_jax.engine.phases import transition_to_combat_resolve
  from azuki_jax.engine.triggers import TIMING_WHEN_ATTACKED, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)

  prev = state
  defender = jnp.maximum(state.combat_defender.astype(jnp.int32), 0)
  can_pass = (
      ~(did_reset | zero_legal)
      & (action_type == Act.NOOP)
      & (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == AbilityPhase.NONE)
      & (state.trig_count == 0)
      & (state.redirect_count == 0)
      & (state.combat_attacker >= 0)
      & (state.combat_defender >= 0)
      & (state.combat_defender_player == acting.astype(jnp.int8))
      & (state.def_id[acting, defender] == cards.CODE_TO_ID["AZK01-040"])
  )

  stepped = transition_to_combat_resolve(state, do=can_pass)
  popped, src, owner, timing = pop_effect(stepped)
  is_azk01_040 = (
      can_pass
      & (timing == jnp.int8(TIMING_WHEN_ATTACKED))
      & (owner == acting.astype(jnp.int8))
      & (src == defender.astype(jnp.int8))
  )
  entered = popped._replace(
      active_player=owner.astype(jnp.int8),
      ab_phase=jnp.int8(AbilityPhase.EFFECT_SELECTION),
      ab_source=src.astype(jnp.int8),
      ab_owner=owner.astype(jnp.int8),
      ab_slot=jnp.int8(0),
      ab_is_optional=jnp.bool_(False),
      ab_costs_applied=jnp.bool_(False),
      ab_saved_active=popped.active_player.astype(jnp.int8),
      ab_restores_active=jnp.bool_(True),
      ab_cost_selected=jnp.int8(0),
      ab_cost_max=jnp.int8(0),
      ab_cost_targets=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_cost_target_players=jnp.full(
          (MAX_ABILITY_SELECTION,), -1, jnp.int8
      ),
      ab_eff_selected=jnp.int8(0),
      ab_eff_min=jnp.int8(0),
      ab_eff_max=jnp.int8(1),
      ab_eff_targets=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_eff_target_players=jnp.full(
          (MAX_ABILITY_SELECTION,), -1, jnp.int8
      ),
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(is_azk01_040, a, b), entered, popped
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )
  state = jax.tree.map(lambda a, b: jnp.where(is_azk01_040, a, b), state, prev)

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.NOOP, jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_attack_azk01_060_confirm_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast declaration for AZK01-060 when-attacking confirmation."""
  from azuki_jax import cards
  from azuki_jax.engine.helpers import card_at_slot, leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  opp = (acting + 1) % 2
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  attacker = card_at_slot(state, acting, Zone.GARDEN, action[1])
  target_is_leader = action[2] == GARDEN_SIZE
  garden_defender = card_at_slot(state, opp, Zone.GARDEN, action[2])
  leader_defender = leader_instance(state, opp)
  defender = jnp.where(target_is_leader, leader_defender, garden_defender)
  safe_attacker = jnp.maximum(attacker, 0)
  safe_defender = jnp.maximum(defender, 0)
  can_declare = (
      do_action
      & (action[2] >= 0)
      & (action[2] <= GARDEN_SIZE)
      & (attacker >= 0)
      & (defender >= 0)
      & (state.def_id[acting, safe_attacker] == cards.CODE_TO_ID["AZK01-060"])
  )

  stepped = state._replace(
      tapped=state.tapped.at[acting, safe_attacker].set(
          jnp.where(can_declare, True, state.tapped[acting, safe_attacker])
      ),
      combat_attacker=jnp.where(
          can_declare, safe_attacker.astype(jnp.int8), state.combat_attacker
      ),
      combat_defender=jnp.where(
          can_declare, safe_defender.astype(jnp.int8), state.combat_defender
      ),
      combat_defender_player=jnp.where(
          can_declare, opp.astype(jnp.int8), state.combat_defender_player
      ),
      combat_intercepted=jnp.where(
          can_declare, False, state.combat_intercepted
      ),
      combat_attacker_is_leader=jnp.where(
          can_declare, False, state.combat_attacker_is_leader
      ),
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          can_declare,
          jnp.int8(AbilityPhase.CONFIRMATION),
          stepped.ab_phase,
      ),
      ab_source=jnp.where(
          can_declare, safe_attacker.astype(jnp.int8), stepped.ab_source
      ),
      ab_owner=jnp.where(can_declare, acting.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(can_declare, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(can_declare, True, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(can_declare, False, stepped.ab_costs_applied),
      ab_saved_active=jnp.where(
          can_declare, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          can_declare, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          can_declare, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(can_declare, jnp.int8(0), stepped.ab_cost_max),
      ab_cost_targets=jnp.where(
          can_declare,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          can_declare,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          can_declare, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(can_declare, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(can_declare, jnp.int8(0), stepped.ab_eff_max),
      ab_eff_targets=jnp.where(
          can_declare,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          can_declare,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
      ab_sel_cards=jnp.where(
          can_declare,
          jnp.full_like(stepped.ab_sel_cards, -1),
          stepped.ab_sel_cards,
      ),
      ab_sel_count=jnp.where(can_declare, jnp.int8(0), stepped.ab_sel_count),
      ab_sel_picked=jnp.where(
          can_declare,
          jnp.full_like(stepped.ab_sel_picked, -1),
          stepped.ab_sel_picked,
      ),
      ab_sel_picked_count=jnp.where(
          can_declare, jnp.int8(0), stepped.ab_sel_picked_count
      ),
      ab_sel_pick_max=jnp.where(
          can_declare, jnp.int8(0), stepped.ab_sel_pick_max
      ),
      ab_scratch=jnp.where(
          can_declare, jnp.zeros_like(stepped.ab_scratch), stepped.ab_scratch
      ),
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ATTACK, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_declare_defender_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast response defender declaration when another response action remains."""
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.apply import apply_declare_defender
  from azuki_jax.engine.helpers import card_at_slot, has_defender_kw, has_infiltrate

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  attacker_p = (acting + 1) % 2
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  defender = card_at_slot(state, acting, Zone.GARDEN, action[1])
  safe_defender = jnp.maximum(defender, 0)
  attacker = jnp.maximum(state.combat_attacker.astype(jnp.int32), 0)
  can_declare = (
      do_action
      & (action[0] == Act.DECLARE_DEFENDER)
      & (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == AbilityPhase.NONE)
      & ~state.combat_intercepted
      & (state.combat_attacker >= 0)
      & (defender >= 0)
      & has_defender_kw(state, acting, safe_defender)
      & ~state.tapped[acting, safe_defender]
      & ~has_infiltrate(state, attacker_p, attacker)
  )

  stepped = apply_declare_defender(state, action[1], do=can_declare)
  stepped = _close_response_combat_if_idle(stepped)
  stepped = recompute_passives(stepped)

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.DECLARE_DEFENDER, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_attack_leader_response_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast declaration for a clean leader attack that opens response window."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import (
      deal_effect_damage,
      garden_seq_order,
      heal_leader,
      mill_with_deckout,
  )
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.helpers import (
      attr_attack_alley,
      attr_leaders_only,
      card_at_slot,
      leader_instance,
  )

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  opp = (acting + 1) % 2
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  state = recompute_passives(state)
  attacker_is_leader = action[1] == GARDEN_SIZE
  garden_attacker = card_at_slot(state, acting, Zone.GARDEN, action[1])
  leader_attacker = leader_instance(state, acting)
  attacker = jnp.where(attacker_is_leader, leader_attacker, garden_attacker)
  target_index = action[2].astype(jnp.int32)
  defender_is_leader = target_index == GARDEN_SIZE
  defender_is_alley = target_index > GARDEN_SIZE
  garden_defender = card_at_slot(state, opp, Zone.GARDEN, target_index)
  alley_defender = card_at_slot(
      state, opp, Zone.ALLEY, target_index - (GARDEN_SIZE + 1)
  )
  leader_defender = leader_instance(state, opp)
  defender = jnp.where(
      defender_is_leader,
      leader_defender,
      jnp.where(defender_is_alley, alley_defender, garden_defender),
  )
  safe_attacker = jnp.maximum(attacker, 0)
  safe_defender = jnp.maximum(defender, 0)
  stt01_012_attached = (
      (state.zone[acting] == Zone.ATTACHED)
      & (state.attached_to[acting] == safe_attacker.astype(jnp.int8))
      & (state.def_id[acting] == cards.CODE_TO_ID["STT01-012"])
  )
  deck_count = jnp.sum(state.zone[acting] == Zone.DECK, dtype=jnp.int32)
  has_stt01_012 = jnp.any(stt01_012_attached) & (deck_count > 0)
  target_index_ok = (target_index >= 0) & (target_index <= GARDEN_SIZE + ALLEY_SIZE)
  can_declare = (
      do_action
      & target_index_ok
      & (attacker >= 0)
      & (defender >= 0)
      & (~defender_is_alley | attr_attack_alley(state, acting, safe_attacker))
      & (~attr_leaders_only(state, acting, safe_attacker) | defender_is_leader)
  )

  stepped = state._replace(
      tapped=state.tapped.at[acting, safe_attacker].set(
          jnp.where(can_declare, True, state.tapped[acting, safe_attacker])
      ),
      combat_attacker=jnp.where(
          can_declare, safe_attacker.astype(jnp.int8), state.combat_attacker
      ),
      combat_defender=jnp.where(
          can_declare, safe_defender.astype(jnp.int8), state.combat_defender
      ),
      combat_defender_player=jnp.where(
          can_declare, opp.astype(jnp.int8), state.combat_defender_player
      ),
      combat_intercepted=jnp.where(
          can_declare, False, state.combat_intercepted
      ),
      combat_attacker_is_leader=jnp.where(
          can_declare, attacker_is_leader, state.combat_attacker_is_leader
      ),
      phase=jnp.where(
          can_declare, jnp.int8(Phase.RESPONSE_WINDOW), state.phase
      ),
      active_player=jnp.where(
          can_declare, opp.astype(jnp.int8), state.active_player
      ),
  )
  stepped = mill_with_deckout(
      stepped,
      acting,
      1,
      can_declare & has_stt01_012,
      max_n=1,
  )
  attacker_def_for_triggers = stepped.def_id[acting, safe_attacker]
  stt01_016_attached = (
      (stepped.zone[acting] == Zone.ATTACHED)
      & (stepped.attached_to[acting] == safe_attacker.astype(jnp.int8))
      & (stepped.def_id[acting] == cards.CODE_TO_ID["STT01-016"])
  )
  stt01_016_weapon = jnp.argmax(stt01_016_attached)
  raizan = jnp.where(
      attacker_def_for_triggers >= 0,
      jnp.asarray(cards.SUBTYPE_MATRIX)[
          jnp.maximum(attacker_def_for_triggers, 0),
          cards.subtype_index("Raizan"),
      ],
      False,
  )
  stt01_016_fire = (
      can_declare
      & jnp.any(stt01_016_attached)
      & raizan
      & (stepped.winner == -1)
  )
  garden_order, in_opp_garden = garden_seq_order(stepped, opp)
  for k in range(GARDEN_SIZE):
    target = garden_order[k]
    stepped = deal_effect_damage(
        stepped,
        opp,
        target,
        1,
        do=stt01_016_fire & in_opp_garden[target],
        src_player=acting,
        src_inst=stt01_016_weapon,
    )
  attacker_def = state.def_id[acting, safe_attacker]
  azk01_047_heal = (
      can_declare
      & (attacker_def == cards.CODE_TO_ID["AZK01-047"])
      & ((state.once_per_turn_used[acting, safe_attacker] & jnp.uint8(1)) == 0)
  )
  stepped = heal_leader(stepped, acting, 1, azk01_047_heal)
  once_used = stepped.once_per_turn_used[acting, safe_attacker]
  stepped = stepped._replace(
      once_per_turn_used=stepped.once_per_turn_used.at[
          acting, safe_attacker
      ].set(jnp.where(azk01_047_heal, once_used | jnp.uint8(1), once_used))
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ATTACK, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_attack_leader_garden_simple_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast clean attack declaration/resolution with no response window."""
  from azuki_jax import cards
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.helpers import (
      attr_attack_alley,
      attr_leaders_only,
      card_at_slot,
      leader_instance,
  )
  from azuki_jax.engine.phases import combat_resolve, phase_gate
  from azuki_jax.engine.triggers import (
      TIMING_WHEN_TAKES_DAMAGE,
      has_queued,
      pop_effect,
  )

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  opp = (acting + 1) % 2
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  attacker_is_leader = action[1] == GARDEN_SIZE
  attacker = jnp.where(
      attacker_is_leader,
      leader_instance(state, acting),
      card_at_slot(state, acting, Zone.GARDEN, action[1]),
  )
  target_index = action[2].astype(jnp.int32)
  defender_is_leader = target_index == GARDEN_SIZE
  defender_is_garden = target_index < GARDEN_SIZE
  defender_is_alley = target_index > GARDEN_SIZE
  garden_defender = card_at_slot(state, opp, Zone.GARDEN, target_index)
  alley_defender = card_at_slot(
      state, opp, Zone.ALLEY, target_index - (GARDEN_SIZE + 1)
  )
  defender = jnp.where(
      defender_is_leader,
      leader_instance(state, opp),
      jnp.where(defender_is_alley, alley_defender, garden_defender),
  )
  safe_attacker = jnp.maximum(attacker, 0)
  safe_defender = jnp.maximum(defender, 0)
  target_index_ok = (target_index >= 0) & (target_index <= GARDEN_SIZE + ALLEY_SIZE)
  can_declare = (
      do_action
      & (action[0] == Act.ATTACK)
      & (action[1] >= 0)
      & (action[1] <= GARDEN_SIZE)
      & target_index_ok
      & (attacker >= 0)
      & (defender >= 0)
      & (~defender_is_alley | attr_attack_alley(state, acting, safe_attacker))
      & (~attr_leaders_only(state, acting, safe_attacker) | defender_is_leader)
  )

  stepped = state._replace(
      tapped=state.tapped.at[acting, safe_attacker].set(
          jnp.where(can_declare, True, state.tapped[acting, safe_attacker])
      ),
      combat_attacker=jnp.where(
          can_declare, safe_attacker.astype(jnp.int8), state.combat_attacker
      ),
      combat_defender=jnp.where(
          can_declare, safe_defender.astype(jnp.int8), state.combat_defender
      ),
      combat_defender_player=jnp.where(
          can_declare, opp.astype(jnp.int8), state.combat_defender_player
      ),
      combat_intercepted=jnp.where(
          can_declare, False, state.combat_intercepted
      ),
      combat_attacker_is_leader=jnp.where(
          can_declare, attacker_is_leader, state.combat_attacker_is_leader
      ),
  )
  run_gate = can_declare & (stepped.winner == -1)
  gated = phase_gate(stepped)
  auto_combat = run_gate & (gated.phase == Phase.COMBAT_RESOLVE) & ~has_queued(gated)
  gated = jax.lax.cond(
      auto_combat,
      lambda st: combat_resolve(st, do=True),
      lambda st: st,
      gated,
  )
  trig_src = jnp.maximum(gated.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(gated.trig_owner[0].astype(jnp.int32), 0)
  azk01_062_combat_fizzle = (
      auto_combat
      & (gated.redirect_count == 0)
      & (gated.trig_count > 0)
      & (gated.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (gated.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-062"])
  )
  popped_fizzle, _, _, _ = pop_effect(gated)
  gated = jax.tree.map(
      lambda a, b: jnp.where(azk01_062_combat_fizzle, a, b),
      popped_fizzle,
      gated,
  )
  recomputed = recompute_passives(gated)
  gated = jax.tree.map(
      lambda a, b: jnp.where(auto_combat, a, b),
      recomputed,
      gated,
  )
  trig_src = jnp.maximum(gated.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(gated.trig_owner[0].astype(jnp.int32), 0)
  azk01_061_combat_fizzle = (
      auto_combat
      & (gated.trig_count > 0)
      & (gated.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (gated.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-061"])
      & (gated.dmg_src_count[trig_owner, trig_src].astype(jnp.int32) < 3)
  )
  popped_fizzle, _, _, _ = pop_effect(gated)
  gated = jax.tree.map(
      lambda a, b: jnp.where(azk01_061_combat_fizzle, a, b),
      popped_fizzle,
      gated,
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(run_gate, a, b), gated, stepped
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ATTACK, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_attack_stt01_012_response_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast attack with attached STT01-012 mill trigger, then phase gate."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import mill_with_deckout
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot, leader_instance
  from azuki_jax.engine.phases import combat_resolve, phase_gate
  from azuki_jax.engine.triggers import has_queued

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  opp = (acting + 1) % 2
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  attacker_is_leader = action[1] == GARDEN_SIZE
  garden_attacker = card_at_slot(state, acting, Zone.GARDEN, action[1])
  leader_attacker = leader_instance(state, acting)
  attacker = jnp.where(attacker_is_leader, leader_attacker, garden_attacker)
  defender_is_leader = action[2] == GARDEN_SIZE
  garden_defender = card_at_slot(state, opp, Zone.GARDEN, action[2])
  leader_defender = leader_instance(state, opp)
  defender = jnp.where(defender_is_leader, leader_defender, garden_defender)
  safe_attacker = jnp.maximum(attacker, 0)
  safe_defender = jnp.maximum(defender, 0)
  stt01_012_attached = (
      (state.zone[acting] == Zone.ATTACHED)
      & (state.attached_to[acting] == safe_attacker.astype(jnp.int8))
      & (state.def_id[acting] == cards.CODE_TO_ID["STT01-012"])
  )
  weapon = jnp.where(jnp.any(stt01_012_attached), jnp.argmax(stt01_012_attached), -1)
  can_declare = (
      do_action
      & (action[0] == Act.ATTACK)
      & (action[1] >= 0)
      & (action[1] <= GARDEN_SIZE)
      & (action[2] >= 0)
      & (action[2] <= GARDEN_SIZE)
      & (attacker >= 0)
      & (defender >= 0)
      & (weapon >= 0)
  )

  stepped = state._replace(
      tapped=state.tapped.at[acting, safe_attacker].set(
          jnp.where(can_declare, True, state.tapped[acting, safe_attacker])
      ),
      combat_attacker=jnp.where(
          can_declare, safe_attacker.astype(jnp.int8), state.combat_attacker
      ),
      combat_defender=jnp.where(
          can_declare, safe_defender.astype(jnp.int8), state.combat_defender
      ),
      combat_defender_player=jnp.where(
          can_declare, opp.astype(jnp.int8), state.combat_defender_player
      ),
      combat_intercepted=jnp.where(
          can_declare, False, state.combat_intercepted
      ),
      combat_attacker_is_leader=jnp.where(
          can_declare, attacker_is_leader, state.combat_attacker_is_leader
      ),
  )
  stepped = mill_with_deckout(stepped, acting, 1, can_declare, max_n=1)
  cleared = _clear_context(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(can_declare, a, b), cleared, stepped
  )
  run_gate = can_declare & (stepped.winner == -1)
  gated = phase_gate(stepped)
  auto_combat = run_gate & (gated.phase == Phase.COMBAT_RESOLVE) & ~has_queued(gated)
  gated = jax.lax.cond(
      auto_combat,
      lambda st: combat_resolve(st, do=True),
      lambda st: st,
      gated,
  )
  stepped = jax.tree.map(
      lambda a, b: jnp.where(run_gate, a, b), gated, stepped
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ATTACK, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_response_noop_leader_combat_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Narrow response-pass fast path for clean garden attacker into leader."""
  from azuki_jax import cards
  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)
  do_action = ~(did_reset | zero_legal) & (action_type == Act.NOOP)

  prev = state
  defender_p = acting
  attacker_p = (defender_p + 1) % 2
  attacker = state.combat_attacker.astype(jnp.int32)
  defender = state.combat_defender.astype(jnp.int32)
  safe_attacker = jnp.maximum(attacker, 0)
  safe_defender = jnp.maximum(defender, 0)
  can_resolve = (
      do_action
      & (state.phase == Phase.RESPONSE_WINDOW)
      & (state.ab_phase == 0)
      & (state.combat_attacker >= 0)
      & (state.combat_defender >= 0)
      & (state.combat_defender_player == defender_p.astype(jnp.int8))
  )

  damage = jnp.maximum(
      state.cur_atk[attacker_p, safe_attacker].astype(jnp.int16), 0
  )
  defender_hp = state.cur_hp[defender_p, safe_defender].astype(jnp.int16)
  new_hp = (defender_hp - damage).astype(jnp.int8)
  deal = can_resolve & (damage > 0)

  stepped = state._replace(
      cur_hp=state.cur_hp.at[defender_p, safe_defender].set(
          jnp.where(deal, new_hp, state.cur_hp[defender_p, safe_defender])
      ),
      took_damage_turn=state.took_damage_turn.at[defender_p, safe_defender].set(
          jnp.where(
              deal,
              True,
              state.took_damage_turn[defender_p, safe_defender],
          )
      ),
      last_dmg_taken=state.last_dmg_taken.at[defender_p, safe_defender].set(
          jnp.where(
              deal,
              damage.astype(jnp.int8),
              state.last_dmg_taken[defender_p, safe_defender],
          )
      ),
      last_dmg_src_player=state.last_dmg_src_player.at[
          defender_p, safe_defender
      ].set(
          jnp.where(
              deal,
              attacker_p.astype(jnp.int8),
              state.last_dmg_src_player[defender_p, safe_defender],
          )
      ),
      last_dmg_src_inst=state.last_dmg_src_inst.at[
          defender_p, safe_defender
      ].set(
          jnp.where(
              deal,
              safe_attacker.astype(jnp.int8),
              state.last_dmg_src_inst[defender_p, safe_defender],
          )
      ),
      last_dmg_from_effect=state.last_dmg_from_effect.at[
          defender_p, safe_defender
      ].set(
          jnp.where(
              deal,
              False,
              state.last_dmg_from_effect[defender_p, safe_defender],
          )
      ),
      dealt_damage_turn=state.dealt_damage_turn.at[attacker_p, safe_attacker].set(
          jnp.where(
              deal,
              True,
              state.dealt_damage_turn[attacker_p, safe_attacker],
          )
      ),
      combat_attacker=jnp.where(
          can_resolve, jnp.int8(-1), state.combat_attacker
      ),
      combat_defender=jnp.where(
          can_resolve, jnp.int8(-1), state.combat_defender
      ),
      combat_defender_player=jnp.where(
          can_resolve, jnp.int8(-1), state.combat_defender_player
      ),
      combat_intercepted=jnp.where(
          can_resolve, False, state.combat_intercepted
      ),
      phase=jnp.where(can_resolve, jnp.int8(Phase.MAIN), state.phase),
      active_player=jnp.where(
          can_resolve, attacker_p.astype(jnp.int8), state.active_player
      ),
      winner=jnp.where(
          deal & (new_hp <= 0),
          attacker_p.astype(jnp.int8),
          state.winner,
      ),
  )

  key = (
      attacker_p.astype(jnp.int16) * jnp.int16(256)
      + safe_attacker.astype(jnp.int16)
  )
  src_count = state.dmg_src_count[defender_p, safe_defender].astype(jnp.int32)
  row_keys = state.dmg_src_keys[defender_p, safe_defender]
  seen = jnp.any((jnp.arange(8) < src_count) & (row_keys == key))
  add_source = deal & ~seen & (src_count < 8)
  slot = jnp.clip(src_count, 0, 7)
  stepped = stepped._replace(
      dmg_src_keys=stepped.dmg_src_keys.at[defender_p, safe_defender, slot].set(
          jnp.where(
              add_source,
              key,
              stepped.dmg_src_keys[defender_p, safe_defender, slot],
          )
      ),
      dmg_src_count=stepped.dmg_src_count.at[defender_p, safe_defender].set(
          jnp.where(add_source, src_count + 1, src_count).astype(jnp.int8)
      ),
  )

  attacker_def = state.def_id[attacker_p, safe_attacker]
  azk01_058_confirm = (
      deal
      & (attacker_def == cards.CODE_TO_ID["AZK01-058"])
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          azk01_058_confirm,
          jnp.int8(AbilityPhase.CONFIRMATION),
          stepped.ab_phase,
      ),
      ab_source=jnp.where(
          azk01_058_confirm, safe_attacker.astype(jnp.int8), stepped.ab_source
      ),
      ab_owner=jnp.where(
          azk01_058_confirm, attacker_p.astype(jnp.int8), stepped.ab_owner
      ),
      ab_slot=jnp.where(azk01_058_confirm, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(azk01_058_confirm, True, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          azk01_058_confirm, False, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          azk01_058_confirm, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          azk01_058_confirm, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          azk01_058_confirm, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(
          azk01_058_confirm, jnp.int8(0), stepped.ab_cost_max
      ),
      ab_cost_targets=jnp.where(
          azk01_058_confirm,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          azk01_058_confirm,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          azk01_058_confirm, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(
          azk01_058_confirm, jnp.int8(1), stepped.ab_eff_min
      ),
      ab_eff_max=jnp.where(
          azk01_058_confirm, jnp.int8(1), stepped.ab_eff_max
      ),
      ab_eff_targets=jnp.where(
          azk01_058_confirm,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          azk01_058_confirm,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.NOOP, jnp.int32),
      noop_had_alternatives,
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations



def step_attack_azk01_014_effect_fast(
    state: State,
    actions: jax.Array,
    prev_terminals: jax.Array,
    prev_truncations: jax.Array,
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-014 attack declaration into friendly buff selection."""
  from azuki_jax import cards
  from azuki_jax.engine.helpers import card_at_slot, leader_instance

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  opp = (acting + 1) % 2
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  attacker = card_at_slot(state, acting, Zone.GARDEN, action[1])
  target_is_leader = action[2] == GARDEN_SIZE
  target_is_garden = (action[2] >= 0) & (action[2] < GARDEN_SIZE)
  garden_defender = card_at_slot(state, opp, Zone.GARDEN, action[2])
  leader_defender = leader_instance(state, opp)
  defender = jnp.where(target_is_leader, leader_defender, garden_defender)
  safe_attacker = jnp.maximum(attacker, 0)
  attacker_def = state.def_id[acting, safe_attacker]
  safe_defender = jnp.maximum(defender, 0)
  can_declare = (
      do_action
      & (action[0] == Act.ATTACK)
      & (target_is_leader | target_is_garden)
      & (attacker >= 0)
      & (defender >= 0)
      & (
          (attacker_def == cards.CODE_TO_ID["AZK01-014"])
          | (attacker_def == cards.CODE_TO_ID["AZK01-072"])
      )
  )

  stepped = state._replace(
      tapped=state.tapped.at[acting, safe_attacker].set(
          jnp.where(can_declare, True, state.tapped[acting, safe_attacker])
      ),
      combat_attacker=jnp.where(
          can_declare, safe_attacker.astype(jnp.int8), state.combat_attacker
      ),
      combat_defender=jnp.where(
          can_declare, safe_defender.astype(jnp.int8), state.combat_defender
      ),
      combat_defender_player=jnp.where(
          can_declare, opp.astype(jnp.int8), state.combat_defender_player
      ),
      combat_intercepted=jnp.where(can_declare, False, state.combat_intercepted),
      combat_attacker_is_leader=jnp.where(
          can_declare, False, state.combat_attacker_is_leader
      ),
      ab_phase=jnp.where(
          can_declare, jnp.int8(AbilityPhase.EFFECT_SELECTION), state.ab_phase
      ),
      ab_source=jnp.where(
          can_declare, safe_attacker.astype(jnp.int8), state.ab_source
      ),
      ab_owner=jnp.where(can_declare, acting.astype(jnp.int8), state.ab_owner),
      ab_slot=jnp.where(can_declare, jnp.int8(0), state.ab_slot),
      ab_is_optional=jnp.where(can_declare, False, state.ab_is_optional),
      ab_costs_applied=jnp.where(can_declare, False, state.ab_costs_applied),
      ab_saved_active=jnp.where(
          can_declare, jnp.int8(-1), state.ab_saved_active
      ),
      ab_restores_active=jnp.where(can_declare, False, state.ab_restores_active),
      ab_cost_selected=jnp.where(
          can_declare, jnp.int8(0), state.ab_cost_selected
      ),
      ab_cost_max=jnp.where(can_declare, jnp.int8(0), state.ab_cost_max),
      ab_cost_targets=jnp.where(
          can_declare, jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          can_declare, jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          can_declare, jnp.int8(0), state.ab_eff_selected
      ),
      ab_eff_min=jnp.where(can_declare, jnp.int8(1), state.ab_eff_min),
      ab_eff_max=jnp.where(can_declare, jnp.int8(1), state.ab_eff_max),
      ab_eff_targets=jnp.where(
          can_declare, jnp.full_like(state.ab_eff_targets, -1),
          state.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          can_declare, jnp.full_like(state.ab_eff_target_players, -1),
          state.ab_eff_target_players,
      ),
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )
  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, jnp.asarray(Act.ATTACK, jnp.int32), jnp.bool_(False)
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )
  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations


def step_effect_azk01_014_fast(
    state: State,
    actions: jax.Array,
    prev_terminals: jax.Array,
    prev_truncations: jax.Array,
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-014 target selection, then response/combat gate."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_attack_modifier
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import card_at_slot
  from azuki_jax.engine.phases import combat_resolve, phase_gate
  from azuki_jax.engine.triggers import has_queued

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  target = card_at_slot(state, owner, Zone.GARDEN, action[1])
  safe_target = jnp.maximum(target, 0)
  source_def = state.def_id[owner, src]
  source_azk01_014 = source_def == cards.CODE_TO_ID["AZK01-014"]
  source_azk01_072 = source_def == cards.CODE_TO_ID["AZK01-072"]
  source_ok = (
      (state.ab_phase == AbilityPhase.EFFECT_SELECTION)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.ab_source >= 0)
      & (source_azk01_014 | source_azk01_072)
      & ~state.ab_costs_applied
      & (state.ab_eff_selected == 0)
      & (state.ab_eff_min == 1)
      & (state.ab_eff_max == 1)
  )
  beanz_idx = cards.subtype_index("Beanz")
  target_beanz = jnp.asarray(cards.SUBTYPE_MATRIX[:, beanz_idx])[jnp.maximum(state.def_id[owner, safe_target], 0)].astype(jnp.bool_)
  target_ok = (
      (action[1] >= 0)
      & (action[1] < GARDEN_SIZE)
      & (target >= 0)
      & (target != src)
      & (
          source_azk01_014
          | (source_azk01_072 & target_beanz)
      )
  )
  do_select = (
      do_action
      & (action[0] == Act.SELECT_EFFECT_TARGET)
      & source_ok
      & target_ok
  )
  stepped = state._replace(
      ab_eff_targets=state.ab_eff_targets.at[0].set(
          jnp.where(do_select, safe_target.astype(jnp.int8), state.ab_eff_targets[0])
      ),
      ab_eff_target_players=state.ab_eff_target_players.at[0].set(
          jnp.where(do_select, owner.astype(jnp.int8), state.ab_eff_target_players[0])
      ),
      ab_eff_selected=jnp.where(do_select, jnp.int8(1), state.ab_eff_selected),
  )
  buff_amount = jnp.where(source_azk01_072, 1, 2)
  stepped = apply_attack_modifier(
      stepped, owner, safe_target, buff_amount, True, do_select
  )
  stepped = _clear_context(stepped)
  stepped = phase_gate(stepped)
  auto_combat = (stepped.phase == Phase.COMBAT_RESOLVE) & ~has_queued(stepped)
  resolved = jax.lax.cond(
      auto_combat,
      lambda st: combat_resolve(st, do=True),
      lambda st: st,
      stepped,
  )
  recomputed = recompute_passives(resolved)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(auto_combat, a, b), recomputed, resolved
  )
  stepped = jax.tree.map(lambda a, b: jnp.where(do_select, a, b), stepped, state)
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )
  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.SELECT_EFFECT_TARGET, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )
  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)
  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))
  state = state._replace(episode_returns=state.episode_returns + rewards)
  return state, rewards, terminals, truncations

def step_attack_stt01_006_effect_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast STT01-006 attack declaration into when-attacking effect selection."""
  from azuki_jax import cards
  from azuki_jax.engine.helpers import card_at_slot, leader_instance
  from azuki_jax.engine.triggers import TIMING_WHEN_ATTACKING, queue_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  opp = (acting + 1) % 2
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  attacker = card_at_slot(state, acting, Zone.GARDEN, action[1])
  target_is_leader = action[2] == GARDEN_SIZE
  garden_defender = card_at_slot(state, opp, Zone.GARDEN, action[2])
  leader_defender = leader_instance(state, opp)
  defender = jnp.where(target_is_leader, leader_defender, garden_defender)
  safe_attacker = jnp.maximum(attacker, 0)
  safe_defender = jnp.maximum(defender, 0)
  stt01_012_attached = (
      (state.zone[acting] == Zone.ATTACHED)
      & (state.attached_to[acting] == safe_attacker.astype(jnp.int8))
      & (state.def_id[acting] == cards.CODE_TO_ID["STT01-012"])
  )
  stt01_012_weapon = jnp.where(
      jnp.any(stt01_012_attached), jnp.argmax(stt01_012_attached), -1
  )
  can_declare = (
      do_action
      & (action[0] == Act.ATTACK)
      & (action[2] >= 0)
      & (action[2] <= GARDEN_SIZE)
      & (attacker >= 0)
      & (defender >= 0)
      & (state.def_id[acting, safe_attacker] == cards.CODE_TO_ID["STT01-006"])
  )

  stepped = state._replace(
      tapped=state.tapped.at[acting, safe_attacker].set(
          jnp.where(can_declare, True, state.tapped[acting, safe_attacker])
      ),
      combat_attacker=jnp.where(
          can_declare, safe_attacker.astype(jnp.int8), state.combat_attacker
      ),
      combat_defender=jnp.where(
          can_declare, safe_defender.astype(jnp.int8), state.combat_defender
      ),
      combat_defender_player=jnp.where(
          can_declare, opp.astype(jnp.int8), state.combat_defender_player
      ),
      combat_intercepted=jnp.where(
          can_declare, False, state.combat_intercepted
      ),
      combat_attacker_is_leader=jnp.where(
          can_declare, False, state.combat_attacker_is_leader
      ),
      ab_phase=jnp.where(
          can_declare,
          jnp.int8(AbilityPhase.EFFECT_SELECTION),
          state.ab_phase,
      ),
      ab_source=jnp.where(
          can_declare, safe_attacker.astype(jnp.int8), state.ab_source
      ),
      ab_owner=jnp.where(can_declare, acting.astype(jnp.int8), state.ab_owner),
      ab_slot=jnp.where(can_declare, jnp.int8(0), state.ab_slot),
      ab_is_optional=jnp.where(can_declare, False, state.ab_is_optional),
      ab_costs_applied=jnp.where(
          can_declare, False, state.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          can_declare, jnp.int8(-1), state.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          can_declare, False, state.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          can_declare, jnp.int8(0), state.ab_cost_selected
      ),
      ab_cost_max=jnp.where(can_declare, jnp.int8(0), state.ab_cost_max),
      ab_cost_targets=jnp.where(
          can_declare,
          jnp.full_like(state.ab_cost_targets, -1),
          state.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          can_declare,
          jnp.full_like(state.ab_cost_target_players, -1),
          state.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          can_declare, jnp.int8(0), state.ab_eff_selected
      ),
      ab_eff_min=jnp.where(can_declare, jnp.int8(1), state.ab_eff_min),
      ab_eff_max=jnp.where(can_declare, jnp.int8(1), state.ab_eff_max),
      ab_eff_targets=jnp.where(
          can_declare,
          jnp.full_like(state.ab_eff_targets, -1),
          state.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          can_declare,
          jnp.full_like(state.ab_eff_target_players, -1),
          state.ab_eff_target_players,
      ),
      ab_sel_cards=jnp.where(
          can_declare,
          jnp.full_like(state.ab_sel_cards, -1),
          state.ab_sel_cards,
      ),
      ab_sel_count=jnp.where(can_declare, jnp.int8(0), state.ab_sel_count),
      ab_sel_picked=jnp.where(
          can_declare,
          jnp.full_like(state.ab_sel_picked, -1),
          state.ab_sel_picked,
      ),
      ab_sel_picked_count=jnp.where(
          can_declare, jnp.int8(0), state.ab_sel_picked_count
      ),
      ab_sel_pick_max=jnp.where(
          can_declare, jnp.int8(0), state.ab_sel_pick_max
      ),
      ab_scratch=jnp.where(
          can_declare, jnp.zeros_like(state.ab_scratch), state.ab_scratch
      ),
  )
  stepped = queue_effect(
      stepped,
      acting,
      jnp.maximum(stt01_012_weapon, 0),
      TIMING_WHEN_ATTACKING,
      can_declare & (stt01_012_weapon >= 0),
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ATTACK, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations

def step_attack_azk01_006_when_attacked_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast clean attack into AZK01-006's optional when-attacked trigger."""
  from azuki_jax import cards
  from azuki_jax.engine.helpers import card_at_slot

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  opp = (acting + 1) % 2
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  attacker = card_at_slot(state, acting, Zone.GARDEN, action[1])
  defender = card_at_slot(state, opp, Zone.GARDEN, action[2])
  safe_attacker = jnp.maximum(attacker, 0)
  safe_defender = jnp.maximum(defender, 0)
  can_declare = (
      do_action
      & (action[0] == Act.ATTACK)
      & (action[1] >= 0)
      & (action[1] < GARDEN_SIZE)
      & (action[2] >= 0)
      & (action[2] < GARDEN_SIZE)
      & (attacker >= 0)
      & (defender >= 0)
      & (state.def_id[opp, safe_defender] == cards.CODE_TO_ID["AZK01-006"])
  )

  stepped = state._replace(
      tapped=state.tapped.at[acting, safe_attacker].set(
          jnp.where(can_declare, True, state.tapped[acting, safe_attacker])
      ),
      combat_attacker=jnp.where(
          can_declare, safe_attacker.astype(jnp.int8), state.combat_attacker
      ),
      combat_defender=jnp.where(
          can_declare, safe_defender.astype(jnp.int8), state.combat_defender
      ),
      combat_defender_player=jnp.where(
          can_declare, opp.astype(jnp.int8), state.combat_defender_player
      ),
      combat_intercepted=jnp.where(
          can_declare, False, state.combat_intercepted
      ),
      combat_attacker_is_leader=jnp.where(
          can_declare, False, state.combat_attacker_is_leader
      ),
  )
  stepped = stepped._replace(
      phase=jnp.where(
          can_declare, jnp.int8(Phase.COMBAT_RESOLVE), stepped.phase
      ),
      active_player=jnp.where(
          can_declare, opp.astype(jnp.int8), stepped.active_player
      ),
      ab_phase=jnp.where(
          can_declare, jnp.int8(AbilityPhase.CONFIRMATION), stepped.ab_phase
      ),
      ab_source=jnp.where(
          can_declare, safe_defender.astype(jnp.int8), stepped.ab_source
      ),
      ab_owner=jnp.where(can_declare, opp.astype(jnp.int8), stepped.ab_owner),
      ab_slot=jnp.where(can_declare, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(can_declare, True, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          can_declare, False, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          can_declare, acting.astype(jnp.int8), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(can_declare, True, stepped.ab_restores_active),
      ab_cost_selected=jnp.where(
          can_declare, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(can_declare, jnp.int8(0), stepped.ab_cost_max),
      ab_eff_selected=jnp.where(
          can_declare, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(can_declare, jnp.int8(0), stepped.ab_eff_min),
      ab_eff_max=jnp.where(can_declare, jnp.int8(0), stepped.ab_eff_max),
      ab_cost_targets=jnp.where(
          can_declare,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          can_declare,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_targets=jnp.where(
          can_declare,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          can_declare,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ATTACK, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_confirm_azk01_060_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast confirm/decline for AZK01-060, then resolve clean entity combat."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import (
      apply_attack_modifier,
      apply_timed_tag_grant,
  )
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import (
      GRANT_PHASE_END,
      TAG_INFILTRATE,
      TAG_SACRIFICE_EOT,
      discard,
      total_carapace,
  )
  from azuki_jax.engine.triggers import (
      TIMING_WHEN_TAKES_DAMAGE,
      pop_effect,
      record_damage_event,
  )

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.CONFIRMATION)
      & (state.ab_source >= 0)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-060"])
      & (state.combat_attacker == src.astype(jnp.int8))
  )
  confirm = action_type == Act.CONFIRM_ABILITY
  decline = action_type == Act.NOOP
  do_action = ~(did_reset | zero_legal) & source_ok & (confirm | decline)
  do_confirm = do_action & confirm

  stepped = state
  stepped, _ = apply_timed_tag_grant(
      stepped, owner, src, TAG_INFILTRATE, GRANT_PHASE_END, 1, do_confirm
  )
  stepped, _ = apply_timed_tag_grant(
      stepped, owner, src, TAG_SACRIFICE_EOT, GRANT_PHASE_END, 1, do_confirm
  )
  stepped = apply_attack_modifier(
      stepped, owner, src, 1, expires_eot=True, do=do_confirm
  )
  stepped = _clear_context(stepped)

  defender_player = jnp.maximum(
      stepped.combat_defender_player.astype(jnp.int32), 0
  )
  defender = jnp.maximum(stepped.combat_defender.astype(jnp.int32), 0)
  do_combat = (
      do_action
      & (stepped.combat_attacker >= 0)
      & (stepped.combat_defender >= 0)
      & (stepped.combat_defender_player >= 0)
  )
  damage_to_defender = jnp.maximum(
      stepped.cur_atk[owner, src].astype(jnp.int16)
      - total_carapace(stepped, defender_player, defender),
      0,
  )
  damage_to_attacker = jnp.maximum(
      stepped.cur_atk[defender_player, defender].astype(jnp.int16)
      - total_carapace(stepped, owner, src),
      0,
  )
  attacker_hp = stepped.cur_hp[owner, src].astype(jnp.int16)
  defender_hp = stepped.cur_hp[defender_player, defender].astype(jnp.int16)
  new_attacker_hp = attacker_hp - damage_to_attacker
  new_defender_hp = defender_hp - damage_to_defender
  attacker_dead = do_combat & (new_attacker_hp <= 0)
  defender_dead = do_combat & (new_defender_hp <= 0)
  defender_is_leader = stepped.zone[defender_player, defender] == Zone.LEADER

  stepped = stepped._replace(
      cur_hp=stepped.cur_hp.at[owner, src]
      .set(
          jnp.where(
              do_combat,
              new_attacker_hp.astype(jnp.int8),
              stepped.cur_hp[owner, src],
          )
      )
      .at[defender_player, defender]
      .set(
          jnp.where(
              do_combat,
              new_defender_hp.astype(jnp.int8),
              stepped.cur_hp[defender_player, defender],
          )
      ),
  )
  stepped = record_damage_event(
      stepped,
      defender_player,
      defender,
      owner,
      src,
      damage_to_attacker,
      do_combat,
      from_effect=False,
  )
  stepped = record_damage_event(
      stepped,
      owner,
      src,
      defender_player,
      defender,
      damage_to_defender,
      do_combat,
      from_effect=False,
  )
  trig_src = jnp.maximum(stepped.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(stepped.trig_owner[0].astype(jnp.int32), 0)
  azk01_062_combat_fizzle = (
      do_combat
      & (state.redirect_count == 0)
      & (stepped.trig_count > 0)
      & (stepped.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (stepped.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-062"])
  )
  popped_fizzle, _, _, _ = pop_effect(stepped)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(azk01_062_combat_fizzle, a, b),
      popped_fizzle,
      stepped,
  )
  stepped = discard(stepped, owner, src, do=attacker_dead)
  stepped = discard(
      stepped, defender_player, defender, do=defender_dead & ~defender_is_leader
  )
  stepped = recompute_passives(stepped)
  stepped = stepped._replace(
      winner=jnp.where(
          defender_dead & defender_is_leader,
          owner.astype(jnp.int8),
          stepped.winner,
      )
  )
  stepped = stepped._replace(
      combat_attacker=jnp.int8(-1),
      combat_defender=jnp.int8(-1),
      combat_defender_player=jnp.int8(-1),
      combat_intercepted=jnp.bool_(False),
      phase=jnp.int8(Phase.MAIN),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, noop_had_alternatives
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_confirm_azk01_060_response_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Fast AZK01-060 confirmation when combat must enter response window."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import (
      apply_attack_modifier,
      apply_timed_tag_grant,
  )
  from azuki_jax.abilities.runtime import _clear_context
  from azuki_jax.engine.helpers import (
      GRANT_PHASE_END,
      TAG_INFILTRATE,
      TAG_SACRIFICE_EOT,
  )
  from azuki_jax.engine.phases import phase_gate

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  action = actions[acting]
  action_type = action[0]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  noop_had_alternatives = (action_type == Act.NOOP) & (legal_count > 1)

  prev = state
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  source_ok = (
      (state.ab_phase == AbilityPhase.CONFIRMATION)
      & (state.ab_source >= 0)
      & (state.ab_owner == acting.astype(jnp.int8))
      & (state.def_id[owner, src] == cards.CODE_TO_ID["AZK01-060"])
      & (state.combat_attacker == src.astype(jnp.int8))
  )
  confirm = action_type == Act.CONFIRM_ABILITY
  decline = action_type == Act.NOOP
  do_action = (
      ~(did_reset | zero_legal)
      & source_ok
      & (confirm | decline)
      & (state.combat_defender >= 0)
      & (state.combat_defender_player >= 0)
  )
  do_confirm = do_action & confirm

  stepped = state
  stepped, _ = apply_timed_tag_grant(
      stepped, owner, src, TAG_INFILTRATE, GRANT_PHASE_END, 1, do_confirm
  )
  stepped, _ = apply_timed_tag_grant(
      stepped, owner, src, TAG_SACRIFICE_EOT, GRANT_PHASE_END, 1, do_confirm
  )
  stepped = apply_attack_modifier(
      stepped, owner, src, 1, expires_eot=True, do=do_confirm
  )
  stepped = phase_gate(_clear_context(stepped))
  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )
  state = jax.tree.map(lambda a, b: jnp.where(do_action, a, b), state, prev)

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state, prev, acting, action_type, noop_had_alternatives
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_attack_azk01_004_leader_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Narrow fast path for AZK01-004 attack buff before resolve/response."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import apply_attack_modifier
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.helpers import card_at_slot, leader_instance
  from azuki_jax.engine.phases import (
      combat_resolve,
      defender_can_respond,
      transition_to_combat_resolve,
  )

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  opp = (acting + 1) % 2
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  state = recompute_passives(state)
  target_index = action[2].astype(jnp.int32)
  defender_is_leader = target_index == GARDEN_SIZE
  defender_is_alley = target_index > GARDEN_SIZE
  attacker = card_at_slot(state, acting, Zone.GARDEN, action[1])
  safe_attacker = jnp.maximum(attacker, 0)
  garden_defender = card_at_slot(state, opp, Zone.GARDEN, target_index)
  alley_defender = card_at_slot(
      state, opp, Zone.ALLEY, target_index - (GARDEN_SIZE + 1)
  )
  leader_defender = leader_instance(state, opp)
  defender = jnp.where(
      defender_is_leader,
      leader_defender,
      jnp.where(defender_is_alley, alley_defender, garden_defender),
  )
  safe_defender = jnp.maximum(defender, 0)
  target_index_ok = (target_index >= 0) & (target_index <= GARDEN_SIZE + ALLEY_SIZE)
  source_ok = state.def_id[acting, safe_attacker] == cards.CODE_TO_ID["AZK01-004"]
  can_attack = (
      do_action & target_index_ok & (attacker >= 0) & (defender >= 0) & source_ok
  )

  buffed = apply_attack_modifier(
      state, acting, safe_attacker, 1, expires_eot=True, do=can_attack
  )
  declared = buffed._replace(
      tapped=buffed.tapped.at[acting, safe_attacker].set(
          jnp.where(can_attack, True, buffed.tapped[acting, safe_attacker])
      ),
      combat_attacker=jnp.where(
          can_attack, safe_attacker.astype(jnp.int8), buffed.combat_attacker
      ),
      combat_defender=jnp.where(
          can_attack, safe_defender.astype(jnp.int8), buffed.combat_defender
      ),
      combat_defender_player=jnp.where(
          can_attack, opp.astype(jnp.int8), buffed.combat_defender_player
      ),
      combat_intercepted=jnp.where(
          can_attack, False, buffed.combat_intercepted
      ),
      combat_attacker_is_leader=jnp.where(
          can_attack, False, buffed.combat_attacker_is_leader
      ),
      phase=jnp.where(
          can_attack, jnp.int8(Phase.RESPONSE_WINDOW), buffed.phase
      ),
      active_player=jnp.where(
          can_attack, opp.astype(jnp.int8), buffed.active_player
      ),
  )
  response_available = defender_can_respond(declared, opp)
  open_response = can_attack & response_available
  resolve_now = can_attack & ~response_available
  resolved = transition_to_combat_resolve(declared, do=resolve_now)
  resolved = combat_resolve(resolved, do=resolve_now)
  stepped = jax.tree.map(
      lambda a, b: jnp.where(resolve_now, a, b), resolved, declared
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ATTACK, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations


def step_attack_leader_simple_fast(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    legal_count: jax.Array,
    episode_cap: int = 0,
):
  """Narrow fast path for clean garden/leader attacker into leader."""
  from azuki_jax import cards
  from azuki_jax.abilities.cards_impl import (
      deal_effect_damage,
      garden_seq_order,
      heal_leader,
      mill_with_deckout,
  )
  from azuki_jax.abilities.effects import resolve_triggered_effect
  from azuki_jax.abilities.passives import recompute_passives
  from azuki_jax.engine.helpers import card_at_slot, leader_instance
  from azuki_jax.engine.phases import _lightning_kanabo
  from azuki_jax.engine.triggers import TIMING_WHEN_TAKES_DAMAGE, pop_effect

  was_done = (prev_terminals[0] & prev_terminals[1]) | (
      prev_truncations[0] & prev_truncations[1]
  )
  fresh = reset_state(state, pool)
  fresh = fresh._replace(
      completed_episodes=state.completed_episodes + 1, tick=jnp.int32(0)
  )
  state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
  did_reset = was_done

  state = state._replace(tick=state.tick + 1)
  acting = state.active_player.astype(jnp.int32)
  opp = (acting + 1) % 2
  action = actions[acting]

  legal_count = legal_count.astype(jnp.int32)
  zero_legal = (legal_count == 0) & ~did_reset
  do_action = ~(did_reset | zero_legal)

  prev = state
  state = recompute_passives(state)
  garden_attacker = card_at_slot(state, acting, Zone.GARDEN, action[1])
  leader_attacker = leader_instance(state, acting)
  attacker_is_leader = action[1] == GARDEN_SIZE
  attacker = jnp.where(attacker_is_leader, leader_attacker, garden_attacker)
  safe_attacker = jnp.maximum(attacker, 0)
  defender = leader_instance(state, opp)
  safe_defender = jnp.maximum(defender, 0)
  stt01_012_attached = (
      (state.zone[acting] == Zone.ATTACHED)
      & (state.attached_to[acting] == safe_attacker.astype(jnp.int8))
      & (state.def_id[acting] == cards.CODE_TO_ID["STT01-012"])
  )
  deck_count = jnp.sum(state.zone[acting] == Zone.DECK, dtype=jnp.int32)
  has_stt01_012 = jnp.any(stt01_012_attached) & (deck_count > 0)
  state = mill_with_deckout(
      state, acting, 1, do_action & (attacker >= 0) & has_stt01_012, max_n=1
  )
  attacker_def_for_triggers = state.def_id[acting, safe_attacker]
  stt01_016_attached = (
      (state.zone[acting] == Zone.ATTACHED)
      & (state.attached_to[acting] == safe_attacker.astype(jnp.int8))
      & (state.def_id[acting] == cards.CODE_TO_ID["STT01-016"])
  )
  stt01_016_weapon = jnp.argmax(stt01_016_attached)
  raizan = jnp.where(
      attacker_def_for_triggers >= 0,
      jnp.asarray(cards.SUBTYPE_MATRIX)[
          jnp.maximum(attacker_def_for_triggers, 0),
          cards.subtype_index("Raizan"),
      ],
      False,
  )
  stt01_016_fire = (
      do_action
      & (attacker >= 0)
      & jnp.any(stt01_016_attached)
      & raizan
      & (state.winner == -1)
  )
  declared_for_trigger = state._replace(
      tapped=state.tapped.at[acting, safe_attacker].set(
          jnp.where(stt01_016_fire, True, state.tapped[acting, safe_attacker])
      ),
      combat_attacker=jnp.where(
          stt01_016_fire, safe_attacker.astype(jnp.int8), state.combat_attacker
      ),
      combat_defender=jnp.where(
          stt01_016_fire, safe_defender.astype(jnp.int8), state.combat_defender
      ),
      combat_defender_player=jnp.where(
          stt01_016_fire, opp.astype(jnp.int8), state.combat_defender_player
      ),
      combat_intercepted=jnp.where(
          stt01_016_fire, False, state.combat_intercepted
      ),
      combat_attacker_is_leader=jnp.where(
          stt01_016_fire, attacker_is_leader, state.combat_attacker_is_leader
      ),
  )
  state = jax.tree.map(
      lambda a, b: jnp.where(stt01_016_fire, a, b), declared_for_trigger, state
  )
  garden_order, in_opp_garden = garden_seq_order(state, opp)
  for k in range(GARDEN_SIZE):
    target = garden_order[k]
    state = deal_effect_damage(
        state,
        opp,
        target,
        1,
        do=stt01_016_fire & in_opp_garden[target],
        src_player=acting,
        src_inst=stt01_016_weapon,
    )
  trig_src = jnp.maximum(state.trig_source[0].astype(jnp.int32), 0)
  trig_owner = jnp.maximum(state.trig_owner[0].astype(jnp.int32), 0)
  begin_azk01_062 = (
      stt01_016_fire
      & (state.redirect_count > 0)
      & (state.trig_count > 0)
      & (state.trig_timing[0] == jnp.int8(TIMING_WHEN_TAKES_DAMAGE))
      & (state.def_id[trig_owner, trig_src] == cards.CODE_TO_ID["AZK01-062"])
  )
  popped, src2, owner2, timing2 = pop_effect(state)
  begun = resolve_triggered_effect(popped, src2, owner2, timing2)
  state = jax.tree.map(
      lambda a, b: jnp.where(begin_azk01_062, a, b), begun, state
  )
  damage = jnp.maximum(state.cur_atk[acting, safe_attacker].astype(jnp.int16), 0)
  new_hp = (state.cur_hp[opp, safe_defender].astype(jnp.int16) - damage).astype(
      jnp.int8
  )
  deal = (
      do_action
      & (attacker >= 0)
      & (defender >= 0)
      & (damage > 0)
      & (state.winner == -1)
      & ~begin_azk01_062
  )

  stepped = state._replace(
      tapped=state.tapped.at[acting, safe_attacker].set(
          jnp.where(do_action & (attacker >= 0), True, state.tapped[acting, safe_attacker])
      ),
      cur_hp=state.cur_hp.at[opp, safe_defender].set(
          jnp.where(deal, new_hp, state.cur_hp[opp, safe_defender])
      ),
      took_damage_turn=state.took_damage_turn.at[opp, safe_defender].set(
          jnp.where(deal, True, state.took_damage_turn[opp, safe_defender])
      ),
      last_dmg_taken=state.last_dmg_taken.at[opp, safe_defender].set(
          jnp.where(deal, damage.astype(jnp.int8), state.last_dmg_taken[opp, safe_defender])
      ),
      last_dmg_src_player=state.last_dmg_src_player.at[opp, safe_defender].set(
          jnp.where(deal, acting.astype(jnp.int8), state.last_dmg_src_player[opp, safe_defender])
      ),
      last_dmg_src_inst=state.last_dmg_src_inst.at[opp, safe_defender].set(
          jnp.where(deal, safe_attacker.astype(jnp.int8), state.last_dmg_src_inst[opp, safe_defender])
      ),
      last_dmg_from_effect=state.last_dmg_from_effect.at[opp, safe_defender].set(
          jnp.where(deal, False, state.last_dmg_from_effect[opp, safe_defender])
      ),
      dealt_damage_turn=state.dealt_damage_turn.at[acting, safe_attacker].set(
          jnp.where(deal, True, state.dealt_damage_turn[acting, safe_attacker])
      ),
      combat_attacker=jnp.int8(-1),
      combat_defender=jnp.int8(-1),
      combat_defender_player=jnp.int8(-1),
      combat_intercepted=jnp.bool_(False),
      phase=jnp.int8(Phase.MAIN),
      winner=jnp.where(
          deal & (new_hp <= 0),
          acting.astype(jnp.int8),
          state.winner,
      ),
  )
  stepped = _lightning_kanabo(
      stepped,
      acting,
      safe_attacker,
      opp,
      safe_defender,
      jnp.where(deal, damage, 0),
  )
  key = (acting.astype(jnp.int16) * jnp.int16(256) + safe_attacker.astype(jnp.int16))
  src_count = state.dmg_src_count[opp, safe_defender].astype(jnp.int32)
  row_keys = state.dmg_src_keys[opp, safe_defender]
  seen = jnp.any((jnp.arange(8) < src_count) & (row_keys == key))
  add_source = deal & ~seen & (src_count < 8)
  slot = jnp.clip(src_count, 0, 7)
  stepped = stepped._replace(
      dmg_src_keys=stepped.dmg_src_keys.at[opp, safe_defender, slot].set(
          jnp.where(add_source, key, stepped.dmg_src_keys[opp, safe_defender, slot])
      ),
      dmg_src_count=stepped.dmg_src_count.at[opp, safe_defender].set(
          jnp.where(add_source, src_count + 1, src_count).astype(jnp.int8)
      ),
  )
  attacker_def = state.def_id[acting, safe_attacker]
  azk01_047_heal = (
      do_action
      & (attacker >= 0)
      & (attacker_def == cards.CODE_TO_ID["AZK01-047"])
      & ((state.once_per_turn_used[acting, safe_attacker] & jnp.uint8(1)) == 0)
  )
  stepped = heal_leader(stepped, acting, 1, azk01_047_heal)
  once_used = stepped.once_per_turn_used[acting, safe_attacker]
  stepped = stepped._replace(
      once_per_turn_used=stepped.once_per_turn_used.at[
          acting, safe_attacker
      ].set(jnp.where(azk01_047_heal, once_used | jnp.uint8(1), once_used))
  )
  azk01_058_confirm = (
      deal
      & (state.def_id[acting, safe_attacker] == cards.CODE_TO_ID["AZK01-058"])
  )
  stepped = stepped._replace(
      ab_phase=jnp.where(
          azk01_058_confirm,
          jnp.int8(AbilityPhase.CONFIRMATION),
          stepped.ab_phase,
      ),
      ab_source=jnp.where(
          azk01_058_confirm, safe_attacker.astype(jnp.int8), stepped.ab_source
      ),
      ab_owner=jnp.where(
          azk01_058_confirm, acting.astype(jnp.int8), stepped.ab_owner
      ),
      ab_slot=jnp.where(azk01_058_confirm, jnp.int8(0), stepped.ab_slot),
      ab_is_optional=jnp.where(azk01_058_confirm, True, stepped.ab_is_optional),
      ab_costs_applied=jnp.where(
          azk01_058_confirm, False, stepped.ab_costs_applied
      ),
      ab_saved_active=jnp.where(
          azk01_058_confirm, jnp.int8(-1), stepped.ab_saved_active
      ),
      ab_restores_active=jnp.where(
          azk01_058_confirm, False, stepped.ab_restores_active
      ),
      ab_cost_selected=jnp.where(
          azk01_058_confirm, jnp.int8(0), stepped.ab_cost_selected
      ),
      ab_cost_max=jnp.where(
          azk01_058_confirm, jnp.int8(0), stepped.ab_cost_max
      ),
      ab_cost_targets=jnp.where(
          azk01_058_confirm,
          jnp.full_like(stepped.ab_cost_targets, -1),
          stepped.ab_cost_targets,
      ),
      ab_cost_target_players=jnp.where(
          azk01_058_confirm,
          jnp.full_like(stepped.ab_cost_target_players, -1),
          stepped.ab_cost_target_players,
      ),
      ab_eff_selected=jnp.where(
          azk01_058_confirm, jnp.int8(0), stepped.ab_eff_selected
      ),
      ab_eff_min=jnp.where(
          azk01_058_confirm, jnp.int8(1), stepped.ab_eff_min
      ),
      ab_eff_max=jnp.where(
          azk01_058_confirm, jnp.int8(1), stepped.ab_eff_max
      ),
      ab_eff_targets=jnp.where(
          azk01_058_confirm,
          jnp.full_like(stepped.ab_eff_targets, -1),
          stepped.ab_eff_targets,
      ),
      ab_eff_target_players=jnp.where(
          azk01_058_confirm,
          jnp.full_like(stepped.ab_eff_target_players, -1),
          stepped.ab_eff_target_players,
      ),
  )

  stepped = jax.tree.map(
      lambda a, b: jnp.where(begin_azk01_062, a, b), state, stepped
  )

  state = jax.tree.map(
      lambda a, b: jnp.where(did_reset | zero_legal, b, a), stepped, state
  )

  game_over = (state.winner != -1) & ~did_reset & ~zero_legal
  timeout = (
      (episode_cap > 0) & (state.tick >= episode_cap)
      & ~game_over & ~did_reset & ~zero_legal
  )

  shaped_state, shaped = _shaped_rewards(
      state,
      prev,
      acting,
      jnp.asarray(Act.ATTACK, jnp.int32),
      jnp.bool_(False),
  )
  normal = ~did_reset & ~zero_legal & ~game_over & ~timeout
  state = jax.tree.map(
      lambda a, b: jnp.where(normal, a, b), shaped_state, state
  )

  rewards = jnp.where(
      game_over,
      _terminal_rewards(state),
      jnp.where(
          zero_legal | timeout,
          _truncation_rewards(state),
          jnp.where(normal, shaped, jnp.zeros(2, jnp.float32)),
      ),
  )
  rewards = jnp.where(did_reset, jnp.zeros(2, jnp.float32), rewards)

  terminals = jnp.broadcast_to(game_over, (2,))
  truncations = jnp.broadcast_to(zero_legal | timeout, (2,))

  state = state._replace(
      episode_returns=state.episode_returns + rewards,
  )
  return state, rewards, terminals, truncations
