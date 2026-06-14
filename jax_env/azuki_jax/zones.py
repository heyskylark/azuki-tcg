"""List-zone primitives over (zone, zpos) per-player instance arrays.

Replicates flecs ordered-children semantics: appending puts a card at the end
(zpos = count); removing compacts positions above the removed slot. The deck
top is the highest zpos (C draws with from_index = count-1-i).

Functions operate on a single player's rows: zone_row/zpos_row have shape
(NUM_INSTANCES,). Garden/alley are slot zones — do not use these helpers for
them except zone_count.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def zone_count(zone_row: jax.Array, z) -> jax.Array:
  return jnp.sum(zone_row == z, dtype=jnp.int32)


def instance_at(zone_row: jax.Array, zpos_row: jax.Array, z, pos) -> jax.Array:
  """Instance index at zpos == pos in list zone z, or -1."""
  match = (zone_row == z) & (zpos_row == jnp.asarray(pos, zpos_row.dtype))
  return jnp.where(match.any(), jnp.argmax(match), -1).astype(jnp.int32)


def top_instance(zone_row: jax.Array, zpos_row: jax.Array, z) -> jax.Array:
  """Instance with the highest zpos in list zone z (the 'top'), or -1."""
  in_zone = zone_row == z
  key = jnp.where(in_zone, zpos_row.astype(jnp.int32), -1)
  return jnp.where(in_zone.any(), jnp.argmax(key), -1).astype(jnp.int32)


def remove_from_list_zone(zone_row: jax.Array, zpos_row: jax.Array, inst):
  """Compact out `inst` from its current list zone. Returns (zone, zpos) with
  the card left in place (caller sets its new zone/zpos)."""
  old_zone = zone_row[inst]
  old_pos = zpos_row[inst]
  above = (zone_row == old_zone) & (zpos_row > old_pos)
  zpos_row = jnp.where(above, zpos_row - 1, zpos_row)
  return zone_row, zpos_row


def append_to_list_zone(zone_row: jax.Array, zpos_row: jax.Array, inst, z):
  count = zone_count(zone_row, z)
  zone_row = zone_row.at[inst].set(jnp.asarray(z, zone_row.dtype))
  zpos_row = zpos_row.at[inst].set(count.astype(zpos_row.dtype))
  return zone_row, zpos_row


def move_to_list_zone(zone_row, zpos_row, inst, z):
  """Remove from current list zone and append to z. inst must be valid."""
  zone_row, zpos_row = remove_from_list_zone(zone_row, zpos_row, inst)
  return append_to_list_zone(zone_row, zpos_row, inst, z)


def move_top_n(zone_row, zpos_row, src_zone, dst_zone, n, max_n: int):
  """move_cards_to_zone(): move up to n cards from the top of src to dst,
  one at a time (each taken from the current top, appended to dst end)."""

  def body(i, carry):
    zr, pr = carry
    inst = top_instance(zr, pr, src_zone)
    do = (i < n) & (inst >= 0)
    safe_inst = jnp.maximum(inst, 0)
    nzr, npr = move_to_list_zone(zr, pr, safe_inst, dst_zone)
    return jnp.where(do, nzr, zr), jnp.where(do, npr, pr)

  return jax.lax.fori_loop(0, max_n, body, (zone_row, zpos_row))


def list_zone_order(zone_row, zpos_row, z, size: int) -> jax.Array:
  """Instance ids of cards in list zone z ordered by zpos, padded with -1.

  order[k] = instance with zpos == k (scatter; positions are unique per zone).
  """
  order = jnp.full((size,), -1, jnp.int32)
  in_zone = zone_row == z
  idx = jnp.where(in_zone, zpos_row.astype(jnp.int32), size)
  # scatter instance ids into their positions; drop out-of-zone via index=size
  src = jnp.arange(zone_row.shape[0], dtype=jnp.int32)
  return order.at[idx].set(jnp.where(in_zone, src, -1), mode="drop")
