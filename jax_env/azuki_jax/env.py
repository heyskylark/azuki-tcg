"""Environment lifecycle: init/reset (step lives in engine/)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from azuki_jax.rng import (
    deck_seed_from_env_seed,
    starter_seed_from_env_seed,
    xorshift32,
)
from azuki_jax.setup import DeckPoolTables, new_game
from azuki_jax.state import State


def init_state(env_seed, pool: DeckPoolTables) -> State:
  """tcg.h init(): first episode of a fresh env instance."""
  env_seed = jnp.asarray(env_seed, jnp.uint32)
  starter_rng = xorshift32(starter_seed_from_env_seed(env_seed))
  starting_player = (starter_rng % 2).astype(jnp.int8)

  deck_rng = deck_seed_from_env_seed(env_seed)
  deck_rng = xorshift32(deck_rng)
  d0 = (deck_rng % jnp.uint32(pool.num_decks)).astype(jnp.int32)
  deck_rng = xorshift32(deck_rng)
  d1 = (deck_rng % jnp.uint32(pool.num_decks)).astype(jnp.int32)

  state = new_game(
      env_seed,
      starting_player,
      jnp.stack([d0, d1]),
      jnp.asarray(pool.def_ids),
      jnp.asarray(pool.init_zone),
      jnp.asarray(pool.init_zpos),
  )
  state = state._replace(
      env_seed=env_seed,
      starter_rng_state=starter_rng,
      deck_rng_state=deck_rng,
  )
  return reset_reward_tracking(state)


def init_state_with_decks(env_seed, deck_tables: DeckPoolTables) -> State:
  """binding env_reset_with_decks: explicit decks (rows 0/1), starter RNG
  re-derived from the seed and advanced once; deck RNG untouched."""
  env_seed = jnp.asarray(env_seed, jnp.uint32)
  starter_rng = xorshift32(starter_seed_from_env_seed(env_seed))
  starting_player = (starter_rng % 2).astype(jnp.int8)
  state = new_game(
      env_seed,
      starting_player,
      jnp.asarray([0, 1], jnp.int32),
      jnp.asarray(deck_tables.def_ids),
      jnp.asarray(deck_tables.init_zone),
      jnp.asarray(deck_tables.init_zpos),
  )
  state = state._replace(
      env_seed=env_seed,
      starter_rng_state=starter_rng,
      deck_rng_state=deck_seed_from_env_seed(env_seed),
      current_deck_indices=jnp.asarray([-1, -1], jnp.int16),
  )
  return reset_reward_tracking(state)


def reset_state(state: State, pool: DeckPoolTables) -> State:
  """tcg.h c_reset(): next episode; episode RNG streams advance."""
  starter_rng = xorshift32(state.starter_rng_state)
  starting_player = (starter_rng % 2).astype(jnp.int8)

  deck_rng = xorshift32(state.deck_rng_state)
  d0 = (deck_rng % jnp.uint32(pool.num_decks)).astype(jnp.int32)
  deck_rng = xorshift32(deck_rng)
  d1 = (deck_rng % jnp.uint32(pool.num_decks)).astype(jnp.int32)

  fresh = new_game(
      state.env_seed,
      starting_player,
      jnp.stack([d0, d1]),
      jnp.asarray(pool.def_ids),
      jnp.asarray(pool.init_zone),
      jnp.asarray(pool.init_zpos),
  )
  fresh = fresh._replace(
      env_seed=state.env_seed,
      starter_rng_state=starter_rng,
      deck_rng_state=deck_rng,
      completed_episodes=state.completed_episodes,
  )
  return reset_reward_tracking(fresh)


def reset_reward_tracking(state: State) -> State:
  """tcg.h reset_reward_tracking(): snapshot + phi at episode start."""
  from azuki_jax.rewards import compute_phi_pair, reward_snapshot

  leader_ratio, garden_attack, _, _ = reward_snapshot(state)
  phi = compute_phi_pair(state)
  return state._replace(
      time_weight=jnp.float32(1.0),
      episode_returns=jnp.zeros(2, jnp.float32),
      last_phi=phi,
      last_leader_ratio=leader_ratio,
      last_garden_attack=garden_attack,
      has_last_snapshot=jnp.bool_(True),
  )
