"""Game setup: deck pool tables and new-game initialization.

Mirrors world.c azk_world_init_with_decks_internal + tcg.h create_env_engine:
- instances created in CardInfo expansion order (leader -> leader zone,
  gate -> gate zone, IKZ -> ikz pile, entity/weapon/spell -> deck)
- second player gets the IKZ token
- for p in (0,1): shuffle deck (xorshift32 Fisher-Yates), draw 7 from the top
- phase = PREGAME_MULLIGAN, active = starting player
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from azuki_jax import cards
from azuki_jax.constants import (
    DECK_CARD_COUNT,
    INITIAL_DRAW_COUNT,
    MAX_DECK_SIZE,
    TOKEN_INSTANCE,
    CardType,
    Phase,
    Zone,
)
from azuki_jax.rng import fisher_yates
from azuki_jax.state import State, empty_state
from azuki_jax.zones import list_zone_order, move_top_n


class DeckPoolTables(NamedTuple):
  """Host-precomputed per-deck instance layout, shape (D, 62) unless noted."""

  def_ids: np.ndarray      # int16
  init_zone: np.ndarray    # int8
  init_zpos: np.ndarray    # int8
  num_decks: int


def build_deck_pool_tables(native_pool) -> DeckPoolTables:
  """Expand a NativeDeckPool (list of decks of (card_code, qty)) into tables."""
  all_ids, all_zone, all_zpos = [], [], []
  for deck in native_pool:
    ids, zones, zpos = [], [], []
    counts = {int(Zone.DECK): 0, int(Zone.IKZ_PILE): 0,
              int(Zone.LEADER): 0, int(Zone.GATE): 0}
    for code, qty in deck:
      def_id = cards.CODE_TO_ID[code]
      ctype = int(cards.TYPE[def_id])
      if ctype == CardType.LEADER:
        zone = int(Zone.LEADER)
      elif ctype == CardType.GATE:
        zone = int(Zone.GATE)
      elif ctype == CardType.IKZ:
        zone = int(Zone.IKZ_PILE)
      else:
        zone = int(Zone.DECK)
      for _ in range(int(qty)):
        ids.append(def_id)
        zones.append(zone)
        zpos.append(counts[zone])
        counts[zone] += 1
    if len(ids) != DECK_CARD_COUNT:
      raise ValueError(f"deck expands to {len(ids)} cards, expected {DECK_CARD_COUNT}")
    if counts[int(Zone.DECK)] != MAX_DECK_SIZE:
      raise ValueError(f"main deck has {counts[int(Zone.DECK)]} cards")
    all_ids.append(ids)
    all_zone.append(zones)
    all_zpos.append(zpos)
  return DeckPoolTables(
      def_ids=np.asarray(all_ids, np.int16),
      init_zone=np.asarray(all_zone, np.int8),
      init_zpos=np.asarray(all_zpos, np.int8),
      num_decks=len(all_ids),
  )


def deck_tables_from_card_lists(deck0, deck1) -> DeckPoolTables:
  """Tables for two explicit decks (list of (code, qty)); deck indices 0/1."""
  return build_deck_pool_tables([deck0, deck1])


def _shuffle_and_draw(state: State, p) -> State:
  """shuffle_deck + draw INITIAL_DRAW_COUNT for player p."""
  zone_row, zpos_row = state.zone[p], state.zpos[p]
  order = list_zone_order(zone_row, zpos_row, Zone.DECK, MAX_DECK_SIZE)
  count = jnp.sum(zone_row == Zone.DECK, dtype=jnp.int32)
  order, rng = fisher_yates(order, count, state.rng_state)
  # write back: instance order[k] gets zpos k
  positions = jnp.arange(MAX_DECK_SIZE, dtype=jnp.int32)
  zpos_row = zpos_row.at[order].set(
      positions.astype(zpos_row.dtype), mode="drop"
  )
  zone_row, zpos_row = move_top_n(
      zone_row, zpos_row, Zone.DECK, Zone.HAND,
      jnp.int32(INITIAL_DRAW_COUNT), INITIAL_DRAW_COUNT,
  )
  return state._replace(
      zone=state.zone.at[p].set(zone_row),
      zpos=state.zpos.at[p].set(zpos_row),
      rng_state=rng,
  )


def new_game(
    env_seed: jax.Array,
    starting_player: jax.Array,
    deck_indices: jax.Array,  # (2,) int32 rows into pool tables
    pool_def_ids: jax.Array,  # (D, 62) int16
    pool_init_zone: jax.Array,  # (D, 62) int8
    pool_init_zpos: jax.Array,  # (D, 62) int8
) -> State:
  """Fresh game state (pre-mulligan decision point)."""
  state = empty_state()

  def_id62 = pool_def_ids[deck_indices]          # (2, 62)
  zone62 = pool_init_zone[deck_indices]          # (2, 62)
  zpos62 = pool_init_zpos[deck_indices]          # (2, 62)

  def_id = state.def_id.at[:, :DECK_CARD_COUNT].set(def_id62)
  def_id = def_id.at[:, TOKEN_INSTANCE].set(jnp.int16(cards.IKZ_002_ID))
  zone = state.zone.at[:, :DECK_CARD_COUNT].set(zone62)
  zpos = state.zpos.at[:, :DECK_CARD_COUNT].set(zpos62)

  # IKZ token: exists only for the second player.
  second = (starting_player + 1) % 2
  token_zone = jnp.where(
      jnp.arange(2) == second, jnp.int8(Zone.TOKEN), jnp.int8(Zone.ABSENT)
  )
  zone = zone.at[:, TOKEN_INSTANCE].set(token_zone)

  valid = def_id >= 0
  base_atk = jnp.where(valid, jnp.asarray(cards.BASE_ATK)[def_id], 0)
  base_hp = jnp.where(valid, jnp.asarray(cards.BASE_HP)[def_id], 0)
  immune = jnp.where(valid, jnp.asarray(cards.COND_EFFECT_IMMUNE)[def_id], 0)

  state = state._replace(
      def_id=def_id,
      zone=zone,
      zpos=zpos,
      cur_atk=base_atk,
      cur_hp=base_hp,
      effect_immune_dur=immune,
      phase=jnp.int8(Phase.PREGAME_MULLIGAN),
      active_player=starting_player.astype(jnp.int8),
      starting_player=starting_player.astype(jnp.int8),
      winner=jnp.int8(-1),
      rng_state=env_seed.astype(jnp.uint32),
      current_deck_indices=deck_indices.astype(jnp.int16),
  )

  state = _shuffle_and_draw(state, 0)
  state = _shuffle_and_draw(state, 1)
  return state
