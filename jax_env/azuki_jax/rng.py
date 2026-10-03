"""Bit-exact JAX ports of the C engine's RNG primitives.

C sources: src/utils/deck_utils.c (deck_next_rand, shuffle_deck),
python/src/tcg.h (advance_episode_seed, seed mixers). All operate on uint32
xorshift32 state; zero state is replaced by 0x9E3779B9 before stepping.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from azuki_jax.constants import DECK_SEED_XOR, STARTER_SEED_XOR

GOLDEN = jnp.uint32(0x9E3779B9)


def xorshift32(state: jax.Array) -> jax.Array:
  """One xorshift32 step (identical to deck_next_rand / advance_episode_seed)."""
  x = jnp.where(state == 0, GOLDEN, state).astype(jnp.uint32)
  x = x ^ (x << jnp.uint32(13))
  x = x ^ (x >> jnp.uint32(17))
  x = x ^ (x << jnp.uint32(5))
  return x


def starter_seed_from_env_seed(seed: jax.Array) -> jax.Array:
  return seed.astype(jnp.uint32) ^ jnp.uint32(STARTER_SEED_XOR)


def deck_seed_from_env_seed(seed: jax.Array) -> jax.Array:
  return seed.astype(jnp.uint32) ^ jnp.uint32(DECK_SEED_XOR)


def fisher_yates(order: jax.Array, count: jax.Array, rng_state: jax.Array):
  """Shuffle `order[:count]` exactly like shuffle_deck().

  `order` is a fixed-size int32 vector; entries at index >= count are
  untouched. The C loop runs i from count-1 down to 1, drawing one rand per
  step and swapping order[i] <-> order[roll % (i+1)]. Returns (order, state).
  count <= 1 performs no draws (C early-out).
  """
  size = order.shape[0]

  def body(k, carry):
    arr, state = carry
    i = count - 1 - k  # i in [count-1 .. 1]
    active = i >= 1

    new_state = xorshift32(state)
    roll = new_state
    j = (roll % jnp.where(active, i + 1, 1).astype(jnp.uint32)).astype(jnp.int32)

    vi = arr[i]
    vj = arr[j]
    arr = arr.at[i].set(jnp.where(active, vj, vi))
    arr = arr.at[j].set(jnp.where(active, vi, vj))
    state = jnp.where(active, new_state, state)
    return arr, state

  return jax.lax.fori_loop(0, size - 1, body, (order, rng_state.astype(jnp.uint32)))
