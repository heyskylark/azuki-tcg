"""Packed TrainingObservationData emission (jit/vmap-safe).

`packed_observation_pair(state)` -> (2, ITEMSIZE) uint8: for both players,
EXACTLY the bytes the training wrapper stack produces per agent, i.e.
`observation_to_dict(c_struct)` emulated into the aligned numpy struct dtype
`dtype_from_space(build_observation_space())` (azk_puffer.emulation).

Host side (import): walk the struct dtype once; all offsets are static.
Device side: build every leaf from State and assemble the buffer with
fixed-offset writes; arrays-of-structs (gym Tuple fields) are packed as
vectorized (N, stride) byte blocks, so the op count stays small.

Parity notes (mirrors src/utils/training_observation_util.c +
python/src/observation.py):
- Only the ACTIVE player gets an action mask; the other player's block is
  all zeros with legal_action_count = 0. Game over zeroes both (build_mask
  already enumerates nothing once winner != -1).
- legal_* arrays are zero beyond legal_action_count (observation_to_dict
  zero-fill).
- Fields the vanilla engine does not track yet keep the C empty-state
  values: selection entries card_def_id=-1 & zone_index=i with
  selection_count=0, ability_context phase NONE / no source / zero counts
  (active_player_index is still emitted).
- recent actions are read from state.recent_actions, which the engine does
  NOT populate yet -- all zeros in training (tests may fill it on the host
  in the C clamped encoding [valid, primary, sub1..3 (0..49), was_noop]).

Requires python/src on sys.path (observation + azk_puffer).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from azk_puffer.emulation import dtype_from_space
from observation import (
    MAX_SELECTION_ZONE_SIZE as OBS_SELECTION_ZONE_SIZE,
    build_observation_space,
)

from azuki_jax import cards
from azuki_jax.constants import (
    ALLEY_SIZE,
    GARDEN_SIZE,
    IKZ_AREA_SIZE,
    MAX_ATTACHED_WEAPONS,
    MAX_DECK_SIZE,
    MAX_HAND_SIZE,
    NUM_INSTANCES,
    Phase,
    Zone,
)
from azuki_jax.engine.ikz import token_ready
from azuki_jax.masks import build_mask
from azuki_jax.state import State
from azuki_jax.zones import list_zone_order, zone_count

OBS_SPACE = build_observation_space()
STRUCT_DTYPE = dtype_from_space(OBS_SPACE)
ITEMSIZE = int(STRUCT_DTYPE.itemsize)


# ---------------------------------------------------------------------------
# Generic struct packer (host-recursive over the numpy dtype; static offsets)
# ---------------------------------------------------------------------------

def _leaf_bytes(value, leaf_dtype: np.dtype, batch_shape, logical_shape):
  expected = (*batch_shape, *logical_shape)
  value = jnp.asarray(value)
  if value.shape != expected:
    raise ValueError(
        f"leaf shape {value.shape} != expected {expected} (dtype {leaf_dtype})"
    )
  if leaf_dtype == np.bool_:
    out = value.astype(jnp.uint8)
  else:
    v = value.astype(leaf_dtype)
    if leaf_dtype.itemsize == 1:
      out = lax.bitcast_convert_type(v, jnp.uint8)
    else:
      out = lax.bitcast_convert_type(v, jnp.uint8)  # adds trailing itemsize axis
  n = int(np.prod(logical_shape, dtype=np.int64)) * leaf_dtype.itemsize
  return out.reshape(*batch_shape, n)


def _is_homogeneous_tuple(dtype: np.dtype) -> bool:
  names = list(dtype.fields)
  if not names or names[0] != "f0":
    return False
  if any(name != f"f{i}" for i, name in enumerate(names)):
    return False
  elem = dtype.fields["f0"][0]
  return all(dtype.fields[name][0] == elem for name in names)


def _pack(dtype: np.dtype, value, batch_shape) -> jax.Array:
  """uint8 bytes of shape (*batch_shape, dtype.itemsize); padding is zero."""
  if dtype.fields is None:
    if dtype.subdtype is not None:
      sub, shape = dtype.subdtype
      return _leaf_bytes(value, sub, batch_shape, shape)
    return _leaf_bytes(value, dtype, batch_shape, ())

  if _is_homogeneous_tuple(dtype):
    n = len(dtype.fields)
    elem = dtype.fields["f0"][0]
    offs = [dtype.fields[f"f{i}"][1] for i in range(n)]
    stride = (offs[1] - offs[0]) if n > 1 else elem.itemsize
    assert all(offs[i] == offs[0] + i * stride for i in range(n)), dtype
    elem_bytes = _pack(elem, value, (*batch_shape, n))
    if stride != elem.itemsize:
      padded = jnp.zeros((*batch_shape, n, stride), jnp.uint8)
      elem_bytes = padded.at[..., : elem.itemsize].set(elem_bytes)
    flat = elem_bytes.reshape(*batch_shape, n * stride)
    out = jnp.zeros((*batch_shape, dtype.itemsize), jnp.uint8)
    return out.at[..., offs[0] : offs[0] + n * stride].set(flat)

  out = jnp.zeros((*batch_shape, dtype.itemsize), jnp.uint8)
  for name, (fdtype, off) in dtype.fields.items():
    sub = _pack(fdtype, value[name], batch_shape)
    out = out.at[..., off : off + fdtype.itemsize].set(sub)
  return out


# ---------------------------------------------------------------------------
# Per-player value builders (q = player row, static python int)
# ---------------------------------------------------------------------------

def _gather(row, inst):
  """row[inst] with inst possibly -1 (caller masks the result)."""
  return row[jnp.maximum(inst, 0)]


def _list_block(state: State, q: int, zone: Zone, n: int):
  """(instances ordered by zpos padded -1, ids padded -1) for a list zone."""
  order = list_zone_order(state.zone[q], state.zpos[q], int(zone), n)
  ids = jnp.where(
      order >= 0, _gather(state.def_id[q], order), jnp.int16(-1)
  ).astype(jnp.int16)
  return order, ids


def _inherent(table, def_ids, present):
  ok = present & (def_ids >= 0)
  return jnp.where(ok, jnp.asarray(table)[jnp.maximum(def_ids, 0)], False)


def _weapons_for_host(state: State, q, host, present):
  """C set_attached_weapon_observations: weapons of `host` in attach order.
  `q` may be a static int or a traced row index."""
  is_w = (
      (state.zone[q] == Zone.ATTACHED)
      & (state.attached_to[q] == host.astype(jnp.int8))
      & (state.attached_to[q] >= 0)
      & present
  )
  idx = jnp.where(is_w, state.zpos[q].astype(jnp.int32), MAX_ATTACHED_WEAPONS)
  winst = jnp.full((MAX_ATTACHED_WEAPONS,), -1, jnp.int32).at[idx].set(
      jnp.arange(NUM_INSTANCES, dtype=jnp.int32), mode="drop"
  )
  ok = winst >= 0
  ids = jnp.where(ok, _gather(state.def_id[q], winst), jnp.int16(-1)).astype(jnp.int16)
  atk = jnp.where(ok, _gather(state.cur_atk[q], winst), 0).astype(jnp.int16)
  count = jnp.minimum(jnp.sum(is_w, dtype=jnp.int32), MAX_ATTACHED_WEAPONS)
  return ids, atk, count.astype(jnp.uint8)


def _leader_block(state: State, q: int):
  inst = jnp.argmax(state.zone[q] == Zone.LEADER).astype(jnp.int32)
  def_id = state.def_id[q][inst]
  wids, watk, wcnt = _weapons_for_host(state, q, inst, jnp.bool_(True))
  present = jnp.bool_(True)
  return {
      "card_def_id": def_id.astype(jnp.int16),
      "cooldown": state.cooldown[q][inst] != 0,
      "cur_atk": state.cur_atk[q][inst].astype(jnp.int16),
      "cur_hp": state.cur_hp[q][inst].astype(jnp.int16),
      "has_charge": _inherent(cards.INHERENT_CHARGE, def_id, present)
      | state.grant_charge[q][inst],
      "has_defender": _inherent(cards.INHERENT_DEFENDER, def_id, present)
      | state.grant_defender[q][inst],
      "has_infiltrate": _inherent(cards.INHERENT_INFILTRATE, def_id, present)
      | state.grant_infiltrate[q][inst],
      "tapped": state.tapped[q][inst],
      "weapon_count": wcnt,
      "weapons": {"card_def_id": wids, "cur_atk": watk},
  }


def _gate_block(state: State, q: int):
  inst = jnp.argmax(state.zone[q] == Zone.GATE).astype(jnp.int32)
  return {
      "card_def_id": state.def_id[q][inst].astype(jnp.int16),
      "cooldown": state.cooldown[q][inst] != 0,
      "tapped": state.tapped[q][inst],
  }


def _board_block(state: State, q: int, zone: Zone, size: int):
  """Garden/alley: card at each slot index (zpos == slot)."""
  in_z = state.zone[q] == zone
  idx = jnp.where(in_z, state.zpos[q].astype(jnp.int32), size)
  inst = jnp.full((size,), -1, jnp.int32).at[idx].set(
      jnp.arange(NUM_INSTANCES, dtype=jnp.int32), mode="drop"
  )
  occ = inst >= 0
  def_ids = jnp.where(occ, _gather(state.def_id[q], inst), jnp.int16(-1)).astype(jnp.int16)

  wids, watk, wcnt = jax.vmap(
      lambda h, o: _weapons_for_host(state, q, h, o)
  )(inst, occ)

  def stat(row, fill=0):
    return jnp.where(occ, _gather(row, inst), fill)

  return {
      "card_def_id": def_ids,
      "cooldown": stat(state.cooldown[q]) != 0,
      "cur_atk": stat(state.cur_atk[q]).astype(jnp.int16),
      "cur_hp": stat(state.cur_hp[q]).astype(jnp.int16),
      "has_charge": _inherent(cards.INHERENT_CHARGE, def_ids, occ)
      | (occ & stat(state.grant_charge[q], False)),
      "has_cur_stats": occ,  # entities on board always carry CurStats
      "has_defender": _inherent(cards.INHERENT_DEFENDER, def_ids, occ)
      | (occ & stat(state.grant_defender[q], False)),
      "has_infiltrate": _inherent(cards.INHERENT_INFILTRATE, def_ids, occ)
      | (occ & stat(state.grant_infiltrate[q], False)),
      "is_effect_immune": _inherent(cards.INHERENT_EFFECT_IMMUNE, def_ids, occ)
      | (occ & (stat(state.effect_immune_dur[q]) != 0)),
      "is_frozen": occ & (stat(state.frozen_dur[q]) != 0),
      "is_shocked": occ & (stat(state.shocked_dur[q]) != 0),
      "tapped": occ & stat(state.tapped[q], False),
      "weapon_count": wcnt,
      "weapons": {"card_def_id": wids, "cur_atk": watk},
      "zone_index": jnp.arange(size, dtype=jnp.uint8),
  }


def _selection_ctx_mode(state: State):
  """C training_observation_util.c:1060-1075: while the ability ctx holds a
  live selection (phase SELECTION_PICK/BOTTOM_DECK and ctx count > 0) the
  selection block of BOTH players' views lists ctx->selection.cards in ctx
  index order (holes for picked/bottomed entries, count = initial count);
  otherwise it lists the player's selection zone."""
  use_ctx = (
      (state.ab_phase == 4) | (state.ab_phase == 5)
  ) & (state.ab_sel_count > 0)
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  return use_ctx, owner


def _selection_block(state: State, q: int, size: int):
  """Selection contents as board-card entries: the ability-ctx view when a
  selection flow is live, else the zone in list (zpos) order
  (C: get_selection_from_ability_context /
  get_board_observation_array_for_zone(selection, use_zone_index=false))."""
  use_ctx, owner = _selection_ctx_mode(state)

  zone_inst, _ = _list_block(state, q, Zone.SELECTION, size)
  k = jnp.arange(size)
  ctx_raw = jnp.full((size,), -1, jnp.int32)
  ctx_limit = min(size, state.ab_sel_cards.shape[0])
  ctx_raw = ctx_raw.at[:ctx_limit].set(
      state.ab_sel_cards[:ctx_limit].astype(jnp.int32)
  )
  ctx_inst = jnp.where(
      (k < state.ab_sel_count) & (ctx_raw >= 0), ctx_raw, -1
  )

  row = jnp.where(use_ctx, owner, q)  # ctx cards belong to the ability owner
  inst = jnp.where(use_ctx, ctx_inst, zone_inst)
  occ = inst >= 0
  safe = jnp.maximum(inst, 0)
  def_ids = jnp.where(occ, state.def_id[row][safe], jnp.int16(-1)).astype(jnp.int16)

  wids, watk, wcnt = jax.vmap(
      lambda h, o: _weapons_for_host(state, row, h, o)
  )(safe, occ)

  def stat(arr, fill=0):
    return jnp.where(occ, arr[row][safe], fill)

  has_stats = _inherent(cards.HAS_BASE_STATS, def_ids, occ)
  return {
      "card_def_id": def_ids,
      "cooldown": occ & (stat(state.cooldown) != 0),
      "cur_atk": jnp.where(has_stats, stat(state.cur_atk), 0).astype(jnp.int16),
      "cur_hp": jnp.where(has_stats, stat(state.cur_hp), 0).astype(jnp.int16),
      "has_charge": _inherent(cards.INHERENT_CHARGE, def_ids, occ)
      | (occ & stat(state.grant_charge, False)),
      "has_cur_stats": has_stats,
      "has_defender": _inherent(cards.INHERENT_DEFENDER, def_ids, occ)
      | (occ & stat(state.grant_defender, False)),
      "has_infiltrate": _inherent(cards.INHERENT_INFILTRATE, def_ids, occ)
      | (occ & stat(state.grant_infiltrate, False)),
      "is_effect_immune": _inherent(cards.INHERENT_EFFECT_IMMUNE, def_ids, occ)
      | (occ & (stat(state.effect_immune_dur) != 0)),
      "is_frozen": occ & (stat(state.frozen_dur) != 0),
      "is_shocked": occ & (stat(state.shocked_dur) != 0),
      "tapped": occ & stat(state.tapped, False),
      "weapon_count": wcnt,
      "weapons": {"card_def_id": wids, "cur_atk": watk},
      "zone_index": jnp.arange(size, dtype=jnp.uint8),
  }


def _selection_count(state: State, q: int):
  use_ctx, _ = _selection_ctx_mode(state)
  return jnp.where(
      use_ctx,
      state.ab_sel_count.astype(jnp.int32),
      zone_count(state.zone[q], Zone.SELECTION),
  ).astype(jnp.uint8)


def _ikz_area_block(state: State, q: int):
  order, ids = _list_block(state, q, Zone.IKZ_AREA, IKZ_AREA_SIZE)
  present = order >= 0
  return {
      "card_def_id": ids,
      "cooldown": present & (_gather(state.cooldown[q], order) != 0),
      "tapped": present & _gather(state.tapped[q], order),
      "zone_index": jnp.arange(IKZ_AREA_SIZE, dtype=jnp.uint8),
  }


def _hand_block(state: State, q: int):
  _, ids = _list_block(state, q, Zone.HAND, MAX_HAND_SIZE)
  return {
      "card_def_id": ids,
      "zone_index": jnp.arange(MAX_HAND_SIZE, dtype=jnp.uint8),
  }


def _discard_block(state: State, q: int):
  _, ids = _list_block(state, q, Zone.DISCARD, MAX_DECK_SIZE)
  return {
      "card_def_id": ids,
      "zone_index": jnp.arange(MAX_DECK_SIZE, dtype=jnp.uint8),
  }


def _deck_ids_top_first(state: State, q: int):
  """critic_privileged decks: top of deck first (C reverses ordered children)."""
  order = list_zone_order(state.zone[q], state.zpos[q], int(Zone.DECK), MAX_DECK_SIZE)
  count = zone_count(state.zone[q], Zone.DECK)
  src = count - 1 - jnp.arange(MAX_DECK_SIZE, dtype=jnp.int32)
  inst = jnp.where(src >= 0, order[jnp.clip(src, 0, MAX_DECK_SIZE - 1)], -1)
  return jnp.where(
      inst >= 0, _gather(state.def_id[q], inst), jnp.int16(-1)
  ).astype(jnp.int16)


def _player_view(state: State, q: int):
  return {
      "alley": _board_block(state, q, Zone.ALLEY, ALLEY_SIZE),
      "deck_count": zone_count(state.zone[q], Zone.DECK).astype(jnp.uint8),
      "discard": _discard_block(state, q),
      "garden": _board_block(state, q, Zone.GARDEN, GARDEN_SIZE),
      "gate": _gate_block(state, q),
      "hand": _hand_block(state, q),
      "hand_count": zone_count(state.zone[q], Zone.HAND).astype(jnp.uint8),
      "has_ikz_token": token_ready(state, q),
      "ikz_area": _ikz_area_block(state, q),
      "ikz_pile_count": zone_count(state.zone[q], Zone.IKZ_PILE).astype(jnp.uint8),
      "leader": _leader_block(state, q),
      "selection": _selection_block(state, q, OBS_SELECTION_ZONE_SIZE),
      "selection_count": _selection_count(state, q),
  }


def _opponent_view(view):
  return {
      key: view[key]
      for key in (
          "alley", "deck_count", "discard", "garden", "gate", "hand_count",
          "has_ikz_token", "ikz_area", "ikz_pile_count", "leader",
      )
  }


def _recent_actions_block(state: State, row: int):
  ra = state.recent_actions[row].astype(jnp.int32)  # (4, 6)
  valid = ra[:, 0] != 0
  return {
      "primary": jnp.where(valid, ra[:, 1], 0),
      "sub1": jnp.where(valid, ra[:, 2], 0).astype(jnp.uint16),
      "sub2": jnp.where(valid, ra[:, 3], 0).astype(jnp.uint16),
      "sub3": jnp.where(valid, ra[:, 4], 0).astype(jnp.uint16),
      "valid": valid,
      "was_noop": ra[:, 5] != 0,
  }


def _ability_context_block(state: State):
  """build_ability_context_observation: live FSM fields + pending optional
  confirmations (current CONFIRMATION + queued optional effects of the
  active player)."""
  from azuki_jax.abilities import tables as ab_tables

  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  active = state.ab_phase != 0
  def_id = jnp.where(active, state.def_id[owner, src], jnp.int16(-1))
  has_src = active & (def_id >= 0)
  safe_def = jnp.maximum(def_id, 0)

  in_confirm = (
      (state.ab_phase == 1)
      & state.ab_is_optional
      & (state.ab_owner == state.active_player)
  )
  trig_def = jnp.where(
      state.trig_source >= 0,
      state.def_id[
          jnp.maximum(state.trig_owner.astype(jnp.int32), 0),
          jnp.maximum(state.trig_source.astype(jnp.int32), 0),
      ],
      -1,
  )
  k = jnp.arange(state.trig_source.shape[0])
  queued_optional = (
      (k < state.trig_count)
      & (state.trig_owner == state.active_player)
      & (trig_def >= 0)
      & jnp.asarray(ab_tables.HAS_ABILITY)[jnp.maximum(trig_def, 0)]
      & jnp.asarray(ab_tables.IS_OPTIONAL)[jnp.maximum(trig_def, 0)]
  )
  pending = in_confirm.astype(jnp.uint16) + jnp.sum(
      queued_optional, dtype=jnp.uint16
  )

  sel_count = jnp.clip(state.ab_sel_count.astype(jnp.int32), 0, 50)
  sel_picked = jnp.clip(state.ab_sel_picked_count.astype(jnp.int32), 0, 50)
  sel_max = jnp.clip(state.ab_sel_pick_max.astype(jnp.int32), 0, 50)

  return {
      "active_player_index": state.active_player.astype(jnp.int8),
      "cost_target_type": jnp.where(
          has_src, jnp.asarray(ab_tables.COST_TARGET_TYPE_C)[safe_def], 0
      ).astype(jnp.uint8),
      "effect_target_type": jnp.where(
          has_src, jnp.asarray(ab_tables.EFFECT_TARGET_TYPE_C)[safe_def], 0
      ).astype(jnp.uint8),
      "has_source_card_def_id": has_src,
      "pending_confirmation_count": pending,
      "phase": state.ab_phase.astype(jnp.int32),
      "selection_count": jnp.where(active, sel_count, 0).astype(jnp.uint8),
      "selection_pick_max": jnp.where(active, sel_max, 0).astype(jnp.uint8),
      "selection_picked": jnp.where(active, sel_picked, 0).astype(jnp.uint8),
      "source_card_def_id": jnp.where(has_src, def_id, -1).astype(jnp.int16),
  }


def _combat_entity_ctx(state: State, p: int, owner, inst_i8, have):
  have = have & (inst_i8 >= 0)
  owner = jnp.maximum(owner.astype(jnp.int32), 0)
  inst = jnp.maximum(inst_i8.astype(jnp.int32), 0)
  z = state.zone[owner, inst]
  is_leader = have & (z == Zone.LEADER)
  is_garden = have & (z == Zone.GARDEN)
  is_alley = have & (z == Zone.ALLEY)
  slot = jnp.where(
      is_garden | is_alley, state.zpos[owner, inst], 0
  ).astype(jnp.uint8)
  return {
      "card_def_id": jnp.where(
          have, state.def_id[owner, inst], jnp.int16(-1)
      ).astype(jnp.int16),
      "is_self": have & (owner == p),
      "is_leader": is_leader,
      "is_garden": is_garden,
      "is_alley": is_alley,
      "slot_index": slot,
  }


def _combat_context_block(state: State, p: int):
  active = state.combat_attacker >= 0
  dp = state.combat_defender_player.astype(jnp.int32)
  ap = jnp.where(dp >= 0, (dp + 1) % 2, state.active_player.astype(jnp.int32))
  atk = _combat_entity_ctx(state, p, ap, state.combat_attacker, active)
  tgt = _combat_entity_ctx(state, p, dp, state.combat_defender, active)
  return {
      "attacker_card_def_id": atk["card_def_id"],
      "attacker_is_alley": atk["is_alley"],
      "attacker_is_garden": atk["is_garden"],
      "attacker_is_leader": atk["is_leader"],
      "attacker_is_self": atk["is_self"],
      "attacker_slot_index": atk["slot_index"],
      "combat_active": active,
      "defender_intercepted": state.combat_intercepted,
      "response_window_active": state.phase == Phase.RESPONSE_WINDOW,
      "target_card_def_id": tgt["card_def_id"],
      "target_is_alley": tgt["is_alley"],
      "target_is_garden": tgt["is_garden"],
      "target_is_leader": tgt["is_leader"],
      "target_is_self": tgt["is_self"],
      "target_slot_index": tgt["slot_index"],
  }


def _action_mask_block(state: State, p: int, legal32, count, head0):
  is_active = state.active_player == p
  zero_col = jnp.zeros((legal32.shape[0],), jnp.int32)

  def col(k):
    return jnp.where(is_active, legal32[:, k], zero_col)

  return {
      "legal_action_count": jnp.where(is_active, count.astype(jnp.int32), 0),
      "legal_actions": {
          "legal_primary": col(0),
          "legal_sub1": col(1),
          "legal_sub2": col(2),
          "legal_sub3": col(3),
      },
      "primary_action_mask": head0 & is_active,
  }


def _obs_values(state: State, p: int, own_view, opp_view, legal32, count, head0):
  opp = 1 - p
  return {
      "ability_context": _ability_context_block(state),
      "action_mask": _action_mask_block(state, p, legal32, count, head0),
      "combat_context": _combat_context_block(state, p),
      "critic_privileged": {
          "opponent_hand": {
              "card_def_id": _list_block(state, opp, Zone.HAND, MAX_HAND_SIZE)[1],
              "zone_index": jnp.arange(MAX_HAND_SIZE, dtype=jnp.uint8),
          },
          "self_deck": {
              "card_def_id": _deck_ids_top_first(state, p),
              "zone_index": jnp.arange(MAX_DECK_SIZE, dtype=jnp.uint8),
          },
          "opponent_deck": {
              "card_def_id": _deck_ids_top_first(state, opp),
              "zone_index": jnp.arange(MAX_DECK_SIZE, dtype=jnp.uint8),
          },
      },
      "opp_recent_actions": _recent_actions_block(state, opp),
      "opponent": _opponent_view(opp_view),
      "phase": state.phase.astype(jnp.int32),
      "player": own_view,
      "self_recent_actions": _recent_actions_block(state, p),
  }


def packed_observation_with_mask(state: State):
  """((2, ITEMSIZE) uint8 obs, legal (1024, 4) uint8, count uint16).

  legal/count are the ACTIVE player's mask (build_mask), exposed so callers
  (e.g. JaxVecEnv legality checking) don't recompute it.
  """
  legal, count, head0 = build_mask(state)
  legal32 = legal.astype(jnp.int32)
  views = [_player_view(state, q) for q in (0, 1)]
  rows = [
      _pack(
          STRUCT_DTYPE,
          _obs_values(state, p, views[p], views[1 - p], legal32, count, head0),
          (),
      )
      for p in (0, 1)
  ]
  return jnp.stack(rows), legal, count


def packed_observation_pair(state: State) -> jax.Array:
  """(2, ITEMSIZE) uint8: packed TrainingObservationData for players 0 and 1."""
  return packed_observation_with_mask(state)[0]
