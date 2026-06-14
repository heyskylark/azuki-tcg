"""IKZ payment semantics (zone_util.c get_tappable_ikz_cards et al.).

Payment order: ready IKZ token first (when use_ikz_token), then untapped
ikz_area cards in zone order, then untapped garden ikz-source cards in slot
order. Validation requires exactly `cost` sources; payment taps the first
`cost` of them.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from azuki_jax.constants import TOKEN_INSTANCE, Zone
from azuki_jax.engine.helpers import attr_counts_as_ikz
from azuki_jax.state import State

AZK_MAX_IKZ_PAYMENT = 11  # IKZ_AREA_SIZE + 1


def token_ready(state: State, p) -> jax.Array:
  return (state.zone[p, TOKEN_INSTANCE] == Zone.TOKEN) & (
      ~state.tapped[p, TOKEN_INSTANCE]
  )


def _payment_rank(state: State, p, use_token) -> jax.Array:
  """Rank (payment order) per instance; large = not a source. Token gets rank
  -1 when used so it is always first."""
  n = state.zone.shape[1]
  big = jnp.int32(1 << 20)
  ranks = jnp.full((n,), big, jnp.int32)

  in_area = (state.zone[p] == Zone.IKZ_AREA) & (~state.tapped[p])
  ranks = jnp.where(in_area, state.zpos[p].astype(jnp.int32), ranks)

  in_garden_src = (
      (state.zone[p] == Zone.GARDEN)
      & (~state.tapped[p])
      & attr_counts_as_ikz(state, p, jnp.arange(n))
  )
  ranks = jnp.where(in_garden_src, 1000 + state.zpos[p].astype(jnp.int32), ranks)

  token_ok = token_ready(state, p) & use_token
  ranks = ranks.at[TOKEN_INSTANCE].set(
      jnp.where(token_ok, -1, big)
  )
  return ranks


def count_tappable(state: State, p, include_token) -> jax.Array:
  """azk_count_tappable_ikz_sources (capped at AZK_MAX_IKZ_PAYMENT)."""
  ranks = _payment_rank(state, p, include_token)
  count = jnp.sum(ranks < (1 << 20), dtype=jnp.int32)
  return jnp.minimum(count, AZK_MAX_IKZ_PAYMENT)


def can_pay(state: State, p, cost, use_token) -> jax.Array:
  """fetch_ikz_payment validity: cost==0 ok; if use_token the token must be
  ready; total sources >= cost."""
  cost = jnp.asarray(cost, jnp.int32)
  token_needed_ok = ~use_token | token_ready(state, p)
  enough = count_tappable(state, p, use_token) >= cost
  return (cost == 0) | (token_needed_ok & enough)


def pay(state: State, p, cost, use_token, do=True) -> State:
  """Tap the first `cost` payment sources (in payment order)."""
  cost = jnp.asarray(cost, jnp.int32)
  ranks = _payment_rank(state, p, use_token)
  order = jnp.argsort(ranks)  # stable: payment order
  k = jnp.arange(ranks.shape[0])
  selected_mask = jnp.zeros_like(ranks, dtype=jnp.bool_)
  selected_mask = selected_mask.at[order].set(
      (k < cost) & (jnp.sort(ranks) < (1 << 20))
  )
  do_pay = jnp.asarray(do) & (cost > 0)
  tapped_row = jnp.where(do_pay & selected_mask, True, state.tapped[p])
  return state._replace(tapped=state.tapped.at[p].set(tapped_row))
