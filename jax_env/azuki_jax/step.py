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
    PBRS_GARDEN_ATTACK_CAP,
    SHAPED_BOARD_DELTA_WEIGHT,
    SHAPED_LEADER_DELTA_WEIGHT,
    SHAPED_NOOP_PENALTY,
    TERMINAL_REWARD,
    TRUNCATION_BOARD_EDGE_WEIGHT,
    TRUNCATION_LEADER_EDGE_WEIGHT,
    TRUNCATION_TIMEOUT_PENALTY,
    PBRS_TIME_DECAY,
)
from azuki_jax.engine.step import engine_step
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


def step(
    state: State,
    actions: jax.Array,  # (2, 4) int32
    prev_terminals: jax.Array,  # (2,) bool
    prev_truncations: jax.Array,  # (2,) bool
    pool: DeckPoolTables,
    episode_cap: int = 0,
):
  """One vec_step-style transition. Returns (state, rewards, terms, truncs)."""
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

  legal, count, _ = build_mask(state)
  zero_legal = (count == 0) & ~did_reset
  noop_had_alternatives = (action[0] == Act.NOOP) & (count > 1)

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
