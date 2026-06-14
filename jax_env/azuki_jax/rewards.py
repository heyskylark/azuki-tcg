"""Reward computation mirroring tcg.h (PBRS phi, shaped/terminal/truncation)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from azuki_jax import cards
from azuki_jax.constants import (
    PBRS_GARDEN_ATTACK_CAP,
    PBRS_GARDEN_ATTACK_WEIGHT,
    PBRS_LEADER_WEIGHT,
    PBRS_UNTAPPED_GARDEN_CAP,
    PBRS_UNTAPPED_GARDEN_WEIGHT,
    PBRS_UNTAPPED_IKZ_CAP,
    PBRS_UNTAPPED_IKZ_WEIGHT,
    Zone,
)
from azuki_jax.state import State


def reward_snapshot(state: State):
  """AzkRewardSnapshot: per-player leader hp ratio, garden attack sum,
  untapped garden count, untapped ikz count. Each shape (2,) float32."""
  is_leader = state.zone == Zone.LEADER
  base_hp = jnp.where(
      state.def_id >= 0, jnp.asarray(cards.BASE_HP)[state.def_id], 0
  ).astype(jnp.float32)
  cur_hp = state.cur_hp.astype(jnp.float32)
  ratio = jnp.where(base_hp > 0, cur_hp / jnp.maximum(base_hp, 1.0), 0.0)
  ratio = jnp.clip(ratio, 0.0, 1.0)
  leader_ratio = jnp.sum(jnp.where(is_leader, ratio, 0.0), axis=1)

  in_garden = state.zone == Zone.GARDEN
  garden_attack = jnp.sum(
      jnp.where(in_garden, state.cur_atk.astype(jnp.float32), 0.0), axis=1
  )
  untapped = (~state.tapped) & (state.cooldown == 0)
  untapped_garden = jnp.sum(
      jnp.where(in_garden & untapped, 1.0, 0.0), axis=1
  )
  in_ikz = state.zone == Zone.IKZ_AREA
  untapped_ikz = jnp.sum(jnp.where(in_ikz & untapped, 1.0, 0.0), axis=1)
  return leader_ratio, garden_attack, untapped_garden, untapped_ikz


def leader_health_transform(x: jax.Array) -> jax.Array:
  x = jnp.clip(x, 0.0, 1.0)
  one_minus = 1.0 - x
  pow4 = (one_minus * one_minus) ** 2
  return 0.5 * (x + 1.0 - pow4)


def _safe_delta(numerator, denominator):
  return jnp.where(jnp.abs(denominator) <= 1e-6, 0.0, numerator / denominator)


def compute_phi_pair(state: State) -> jax.Array:
  """compute_phi_for_player for both players: shape (2,) float32."""
  leader_ratio, garden_attack, untapped_garden, untapped_ikz = reward_snapshot(
      state
  )
  me = jnp.arange(2)
  opp = 1 - me
  leader_term = PBRS_LEADER_WEIGHT * (
      leader_health_transform(leader_ratio[me])
      - leader_health_transform(leader_ratio[opp])
  )
  attack_term = PBRS_GARDEN_ATTACK_WEIGHT * _safe_delta(
      garden_attack[me] - garden_attack[opp], PBRS_GARDEN_ATTACK_CAP
  )
  garden_term = PBRS_UNTAPPED_GARDEN_WEIGHT * _safe_delta(
      untapped_garden[me] - untapped_garden[opp], PBRS_UNTAPPED_GARDEN_CAP
  )
  ikz_term = PBRS_UNTAPPED_IKZ_WEIGHT * _safe_delta(
      untapped_ikz[me] - untapped_ikz[opp], PBRS_UNTAPPED_IKZ_CAP
  )
  return jnp.tanh(leader_term + attack_term + garden_term + ikz_term).astype(
      jnp.float32
  )
