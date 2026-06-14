"""Card-level primitives mirroring src/utils/{card,zone,entity,status}_util.c.

All functions are single-env, jit/vmap-safe. Player and instance arguments may
be traced scalars. Conventions:
- `p` = player index (0/1), `inst` = instance index into (2, NUM_INSTANCES).
- Helpers that may be conditionally skipped take a `do` predicate and become
  no-ops when it is False (branchless conditioning).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from azuki_jax import cards
from azuki_jax.constants import (
    GARDEN_SIZE,
    MAX_ATTACHED_WEAPONS,
    Zone,
)
from azuki_jax.state import State
from azuki_jax.zones import (
    remove_from_list_zone,
    zone_count,
)

LIST_ZONES = (
    int(Zone.DECK), int(Zone.HAND), int(Zone.IKZ_PILE), int(Zone.IKZ_AREA),
    int(Zone.DISCARD), int(Zone.SELECTION),
)

# Timed tag grants (status_util.c TimedTagGrant): tag ids + tick phases.
TAG_CHARGE = 1
TAG_INFILTRATE = 2
TAG_SACRIFICE_EOT = 3
TAG_DEFENDER = 4
GRANT_FLAG_FIELDS = {
    TAG_CHARGE: "grant_charge",
    TAG_INFILTRATE: "grant_infiltrate",
    TAG_SACRIFICE_EOT: "sacrifice_eot",
    TAG_DEFENDER: "grant_defender",
}
GRANT_PHASE_NONE = 0
GRANT_PHASE_START = 1  # TAG_GRANT_TICK_START_OF_TURN
GRANT_PHASE_END = 2    # TAG_GRANT_TICK_END_OF_TURN


def _np(table) -> jnp.ndarray:
  return jnp.asarray(table)


def is_list_zone(z) -> jax.Array:
  out = jnp.zeros_like(z, dtype=jnp.bool_)
  for zone in LIST_ZONES:
    out = out | (z == zone)
  return out


# ---------------------------------------------------------------------------
# Keyword / attribute checks (inherent table flag OR granted state)
# ---------------------------------------------------------------------------

def has_charge(state: State, p, inst) -> jax.Array:
  return _inherent(cards.INHERENT_CHARGE, state, p, inst) | state.grant_charge[p, inst]


def has_defender_kw(state: State, p, inst) -> jax.Array:
  return _inherent(cards.INHERENT_DEFENDER, state, p, inst) | state.grant_defender[p, inst]


def has_infiltrate(state: State, p, inst) -> jax.Array:
  return (
      _inherent(cards.INHERENT_INFILTRATE, state, p, inst)
      | state.grant_infiltrate[p, inst]
  )


def has_taunt(state: State, p, inst) -> jax.Array:
  return _inherent(cards.INHERENT_TAUNT, state, p, inst) | state.grant_taunt[p, inst]


def has_rooted(state: State, p, inst) -> jax.Array:
  return _inherent(cards.INHERENT_ROOTED, state, p, inst) | state.grant_rooted[p, inst]


def has_godmode(state: State, p, inst) -> jax.Array:
  return _inherent(cards.INHERENT_GODMODE, state, p, inst) | state.grant_godmode[p, inst]


def is_frozen(state: State, p, inst) -> jax.Array:
  return state.frozen_dur[p, inst] != 0


def is_shocked(state: State, p, inst) -> jax.Array:
  return state.shocked_dur[p, inst] != 0


def is_effect_immune(state: State, p, inst) -> jax.Array:
  return (
      _inherent(cards.INHERENT_EFFECT_IMMUNE, state, p, inst)
      | (state.effect_immune_dur[p, inst] != 0)
  )


def _inherent(table: np.ndarray, state: State, p, inst) -> jax.Array:
  def_id = state.def_id[p, inst]
  return jnp.where(def_id >= 0, _np(table)[def_id], False)


def attr_force_tapped(state: State, p, inst) -> jax.Array:
  return _inherent(cards.ATTR_GARDEN_FORCE_TAPPED, state, p, inst)


def attr_counts_as_ikz(state: State, p, inst) -> jax.Array:
  return _inherent(cards.ATTR_COUNTS_AS_IKZ_SOURCE, state, p, inst)


def attr_attack_alley(state: State, p, inst) -> jax.Array:
  """Inherent attr OR the AZK01-043/095 passive: a leader equipped with an
  alley-targeting weapon gains AttrCanTargetTappedAndUntappedAlley
  (azk01_043.c sync_leader_alley_targeting keeps the tag == attached
  043/095 count > 0, host TLeader only; sole consumer is attack validation,
  action_validation.c:369)."""
  inherent = _inherent(cards.ATTR_TARGET_TAPPED_UNTAPPED_ALLEY, state, p, inst)
  is_leader = state.zone[p, inst] == Zone.LEADER
  alley_weapon = jnp.zeros_like(state.def_id[p], dtype=jnp.bool_)
  for code in ("AZK01-043", "AZK01-095"):
    alley_weapon = alley_weapon | (state.def_id[p] == cards.CODE_TO_ID[code])
  has_alley_weapon = jnp.any(
      alley_weapon
      & (state.zone[p] == Zone.ATTACHED)
      & (state.attached_to[p] == jnp.asarray(inst, jnp.int8))
  )
  return inherent | (is_leader & has_alley_weapon)


def attr_leaders_only(state: State, p, inst) -> jax.Array:
  return _inherent(cards.ATTR_TARGET_LEADER_ONLY, state, p, inst)


def in_play_zone(state: State, p, inst) -> jax.Array:
  z = state.zone[p, inst]
  return (
      (z == Zone.GARDEN) | (z == Zone.ALLEY) | (z == Zone.LEADER)
      | (z == Zone.ATTACHED)
  )


def godmode_in_play(state: State, p, inst) -> jax.Array:
  return has_godmode(state, p, inst) & in_play_zone(state, p, inst)


# ---------------------------------------------------------------------------
# Slot-zone lookups (garden / alley keep slot index in zpos)
# ---------------------------------------------------------------------------

def card_at_slot(state: State, p, zone, slot) -> jax.Array:
  """Instance at garden/alley slot, or -1 (find_card_in_zone_index)."""
  match = (state.zone[p] == zone) & (state.zpos[p] == jnp.asarray(slot, jnp.int8))
  return jnp.where(match.any(), jnp.argmax(match), -1).astype(jnp.int32)


def leader_instance(state: State, p) -> jax.Array:
  match = state.zone[p] == Zone.LEADER
  return jnp.where(match.any(), jnp.argmax(match), -1).astype(jnp.int32)


def gate_instance(state: State, p) -> jax.Array:
  match = state.zone[p] == Zone.GATE
  return jnp.where(match.any(), jnp.argmax(match), -1).astype(jnp.int32)


def hand_instance(state: State, p, hand_index) -> jax.Array:
  match = (state.zone[p] == Zone.HAND) & (
      state.zpos[p] == jnp.asarray(hand_index, jnp.int8)
  )
  return jnp.where(match.any(), jnp.argmax(match), -1).astype(jnp.int32)


def weapons_of(state: State, p, host) -> jax.Array:
  """Bool mask over instances: weapons attached to host (own player only)."""
  return (state.zone[p] == Zone.ATTACHED) & (state.attached_to[p] == jnp.asarray(host, jnp.int8))


# ---------------------------------------------------------------------------
# Effective combat numbers (buff aggregates; passives added separately later)
# ---------------------------------------------------------------------------

# Innate Carapace set by passive-init hooks at card creation (azk01_048.c
# sets CarapaceValue{1} via azk_sync_card_abilities when the instance is
# registered; never cleared in-episode — clear_card_temporary_state only
# drops CarapaceBuff pairs). status_util.c get_total_carapace_value =
# CarapaceValue + pairs, floored at 0.
_INNATE_CARAPACE = np.zeros(cards.CARD_DEF_COUNT, np.int16)
for _code in ("AZK01-048", "AZK01-109"):
  _INNATE_CARAPACE[cards.CODE_TO_ID[_code]] = 1


def total_carapace(state: State, p, inst) -> jax.Array:
  def_id = state.def_id[p, inst]
  innate = jnp.where(def_id >= 0, _np(_INNATE_CARAPACE)[jnp.maximum(def_id, 0)], 0)
  return jnp.maximum(
      innate.astype(jnp.int16)
      + state.carapace_perm[p, inst].astype(jnp.int16)
      + state.carapace_eot[p, inst].astype(jnp.int16),
      0,
  )


def total_cmb_in(state: State, p, inst) -> jax.Array:
  return (
      state.cmb_in_perm[p, inst].astype(jnp.int16)
      + state.cmb_in_eot[p, inst].astype(jnp.int16)
  )


def total_cmb_out(state: State, p, inst) -> jax.Array:
  return (
      state.cmb_out_perm[p, inst].astype(jnp.int16)
      + state.cmb_out_eot[p, inst].astype(jnp.int16)
  )


# ---------------------------------------------------------------------------
# Tap helpers
# ---------------------------------------------------------------------------

def can_tap(state: State, p, inst, ignore_cooldown=False) -> jax.Array:
  tapped = state.tapped[p, inst]
  cooldown = state.cooldown[p, inst] != 0
  ok = ~tapped
  if not ignore_cooldown:
    ok = ok & ~cooldown
  return ok


def tap(state: State, p, inst, do=True) -> State:
  tapped = state.tapped.at[p, inst].set(
      jnp.where(do, True, state.tapped[p, inst])
  )
  return state._replace(tapped=tapped)


# ---------------------------------------------------------------------------
# Discard / sacrifice / replacement (card_utils.c discard_card_internal)
# ---------------------------------------------------------------------------

def clear_temporary_state(state: State, p, inst, do=True) -> State:
  """clear_card_temporary_state: statuses, modifiers, timed grants.

  Inherent (prefab) EffectImmune keeps duration -1; granted keyword flags
  drop. Attack/health buff aggregates are NOT cleared here (C keeps pairs but
  the card leaves play; stats reset separately on discard)."""
  def_id = state.def_id[p, inst]
  prefab_immune = jnp.where(
      def_id >= 0, _np(cards.INHERENT_EFFECT_IMMUNE)[def_id], False
  )

  def set2(arr, value):
    return arr.at[p, inst].set(jnp.where(do, value, arr[p, inst]))

  state = state._replace(
      frozen_dur=set2(state.frozen_dur, 0),
      shocked_dur=set2(state.shocked_dur, 0),
      effect_immune_dur=set2(
          state.effect_immune_dur, jnp.where(prefab_immune, -1, 0).astype(jnp.int8)
      ),
      cmb_in_perm=set2(state.cmb_in_perm, 0),
      cmb_in_eot=set2(state.cmb_in_eot, 0),
      cmb_out_perm=set2(state.cmb_out_perm, 0),
      cmb_out_eot=set2(state.cmb_out_eot, 0),
      carapace_perm=set2(state.carapace_perm, 0),
      carapace_eot=set2(state.carapace_eot, 0),
      grant_charge=set2(state.grant_charge, False),
      grant_defender=set2(state.grant_defender, False),
      grant_infiltrate=set2(state.grant_infiltrate, False),
      grant_taunt=set2(state.grant_taunt, False),
      grant_rooted=set2(state.grant_rooted, False),
      grant_godmode=set2(state.grant_godmode, False),
      timed_tag=state.timed_tag.at[p, inst].set(
          jnp.where(do, 0, state.timed_tag[p, inst])
      ),
      timed_ticks=state.timed_ticks.at[p, inst].set(
          jnp.where(do, 0, state.timed_ticks[p, inst])
      ),
      timed_phase=state.timed_phase.at[p, inst].set(
          jnp.where(do, 0, state.timed_phase[p, inst])
      ),
      sacrifice_eot=set2(state.sacrifice_eot, False),
  )
  return state


def stt02_012_garden_event(state: State, event_player, is_removal, do) -> State:
  """Re-latch every STT02-012 on a garden add/remove event (C observers).

  Counts are taken POST-event; removal events subtract one more from the
  event-side garden (the C intends to exclude a still-counted mover, but
  flecs lists are already post-removal — replicating the off-by-one
  exactly). 012s not in their owner's garden latch off."""
  do = jnp.asarray(do)
  is_012 = state.def_id == cards.CODE_TO_ID["STT02-012"]
  any_012 = jnp.any(is_012)

  def count_garden(q):
    z = state.zone[q]
    def_ids = state.def_id[q]
    is_entity = jnp.where(
        def_ids >= 0, _np(cards.TYPE)[jnp.maximum(def_ids, 0)] == 2, False
    )
    return jnp.sum((z == Zone.GARDEN) & is_entity, dtype=jnp.int32)

  counts = jnp.stack([count_garden(0), count_garden(1)])  # (2,)
  adj = jnp.where(
      jnp.arange(2) == jnp.asarray(event_player),
      jnp.where(jnp.asarray(is_removal), 1, 0),
      0,
  )
  counts = counts - adj

  in_own_garden = (state.zone == Zone.GARDEN) & is_012
  q = jnp.arange(2)[:, None]
  diff = counts[q] - counts[1 - q]  # (2, 1) per owner perspective
  new_latch = in_own_garden & (diff >= 2)
  latch = jnp.where(do & any_012 & is_012, new_latch, state.stt02_012_latch)
  return state._replace(stt02_012_latch=latch)


def bobu_destroy_heal(state: State, p, eligible, do) -> State:
  """card_utils.c maybe_trigger_bobu_state (hardcoded destroy observer): when
  an EARTH ENTITY of player p is destroyed/sacrificed (reason != REPLACEMENT)
  from p's garden or alley while p's leader is STT03-001 with an active
  latch (expires_turn != 0 and turn < expires_turn), heal the leader 1 and
  clear the latch. `eligible` = the destroyed card matched the zone/type/
  element filter (computed by the caller PRE-move).

  Miharu (STT03-012) / Kurai (STT04-013) observers are NOT modeled — those
  cards are outside the training pool."""
  do = jnp.asarray(do) & jnp.asarray(eligible)
  leader = leader_instance(state, p)
  li = jnp.maximum(leader, 0)
  leader_def = state.def_id[p, li]
  is_bobu = (leader >= 0) & (leader_def == cards.CODE_TO_ID["STT03-001"])
  latch = state.bobu_expires_turn[p]
  active = (latch != 0) & (state.turn_number < latch)
  fire = do & is_bobu & active

  base = jnp.where(leader_def >= 0, _np(cards.BASE_HP)[jnp.maximum(leader_def, 0)], 0)
  cur = state.cur_hp[p, li].astype(jnp.int16)
  heal = jnp.clip(base.astype(jnp.int16) - cur, 0, 1)
  return state._replace(
      cur_hp=state.cur_hp.at[p, li].set(
          jnp.where(fire, (cur + heal).astype(jnp.int8), state.cur_hp[p, li])
      ),
      bobu_expires_turn=state.bobu_expires_turn.at[p].set(
          jnp.where(fire, jnp.int16(0), latch)
      ),
  )


def _is_earth_entity(state: State, p):
  """(N,) bool row: Type ENTITY and Element EARTH (card_utils bobu filter)."""
  def_id = state.def_id[p]
  valid = def_id >= 0
  sd = jnp.maximum(def_id, 0)
  return valid & (_np(cards.TYPE)[sd] == 2) & (_np(cards.ELEMENT)[sd] == 3)


def _detach_from_location(state: State, p, inst, do) -> State:
  """Remove inst from wherever it is (list zone compaction / slot clear)."""
  z = state.zone[p, inst]
  listy = is_list_zone(z)
  zone_row, zpos_row = state.zone[p], state.zpos[p]
  nzr, npr = remove_from_list_zone(zone_row, zpos_row, inst)
  use = do & listy
  zone = state.zone.at[p].set(jnp.where(use, nzr, zone_row))
  zpos = state.zpos.at[p].set(jnp.where(use, npr, zpos_row))
  return state._replace(zone=zone, zpos=zpos)


def discard(state: State, p, inst, *, reason_replacement=False,
            ignore_godmode=False, do=True) -> State:
  """discard_card / sacrifice_card / discard_card_for_replacement core.

  Triggers (when-destroyed etc.) and the hardcoded destroy observers are NOT
  handled here — callers queue them (ability layer). Weapon detach stat fixes
  are also the caller's job (discard_equipped_weapon_cards)."""
  if not ignore_godmode:
    do = do & ~godmode_in_play(state, p, inst)

  from_hand = state.zone[p, inst] == Zone.HAND
  from_garden = state.zone[p, inst] == Zone.GARDEN
  from_alley = state.zone[p, inst] == Zone.ALLEY
  bobu_eligible = (
      (from_garden | from_alley)
      & ~jnp.asarray(reason_replacement)
      & _is_earth_entity(state, p)[inst]
  )

  state = _detach_from_location(state, p, inst, do)

  def_id = state.def_id[p, inst]
  base_atk = jnp.where(def_id >= 0, _np(cards.BASE_ATK)[def_id], 0)
  base_hp = jnp.where(def_id >= 0, _np(cards.BASE_HP)[def_id], 0)

  def set2(arr, value):
    return arr.at[p, inst].set(jnp.where(do, value, arr[p, inst]))

  count = zone_count(state.zone[p], Zone.DISCARD)
  state = state._replace(
      tapped=set2(state.tapped, False),
      cooldown=set2(state.cooldown, 0),
      cur_atk=set2(state.cur_atk, base_atk.astype(jnp.int8)),
      cur_hp=set2(state.cur_hp, base_hp.astype(jnp.int8)),
      atk_buff_perm=set2(state.atk_buff_perm, 0),
      atk_buff_eot=set2(state.atk_buff_eot, 0),
      hp_buff_perm=set2(state.hp_buff_perm, 0),
      hp_buff_eot=set2(state.hp_buff_eot, 0),
      attached_to=set2(state.attached_to, -1),
      zone=state.zone.at[p, inst].set(
          jnp.where(do, jnp.int8(Zone.DISCARD), state.zone[p, inst])
      ),
      zpos=state.zpos.at[p, inst].set(
          jnp.where(do, count.astype(jnp.int8), state.zpos[p, inst])
      ),
  )
  state = clear_temporary_state(state, p, inst, do=do)

  inc = (do & from_hand & ~jnp.asarray(reason_replacement)).astype(jnp.uint8)
  state = state._replace(
      discarded_cards_turn=state.discarded_cards_turn.at[p].add(inc)
  )
  state = bobu_destroy_heal(state, p, bobu_eligible, do)
  return stt02_012_garden_event(state, p, True, do & from_garden)


def detach_weapon_stat_fix(state: State, p, host, weapon, do=True) -> State:
  """discard_weapon_card stat math: host.cur_atk -= weapon.cur_atk (>= 0),
  plus removal of the weapon's equipped combat modifier (AZK01-018 only)."""
  weapon_atk = state.cur_atk[p, weapon].astype(jnp.int16)
  host_atk = state.cur_atk[p, host].astype(jnp.int16)
  new_atk = jnp.maximum(host_atk - weapon_atk, 0).astype(jnp.int8)
  cur_atk = state.cur_atk.at[p, host].set(
      jnp.where(do, new_atk, state.cur_atk[p, host])
  )
  state = state._replace(cur_atk=cur_atk)
  # AZK01-018 modifier removal: it applies incoming -1 to leaders only
  is_018 = state.def_id[p, weapon] == cards.CODE_TO_ID["AZK01-018"]
  host_is_leader = state.zone[p, host] == Zone.LEADER
  undo = do & is_018 & host_is_leader
  cmb = state.cmb_in_perm.at[p, host].add(jnp.where(undo, 1, 0).astype(jnp.int8))
  return state._replace(cmb_in_perm=cmb)


def batch_discard(state: State, p, mask, order_key, do=True) -> State:
  """Discard every instance in `mask` at once (no triggers).

  order_key (int32 per instance, smaller = earlier) fixes the discard-pile
  append order to match the C iteration order. Resets stats/taps/statuses
  like discard_card_internal; hand-count bookkeeping is NOT handled (callers
  that may discard from hand use the single-card `discard`)."""
  do = jnp.asarray(do)
  mask = mask & do
  any_from_garden = jnp.any(mask & (state.zone[p] == Zone.GARDEN))
  # bobu observer fires per discard_card_internal call in C; the latch clears
  # on the first match so one batched heal == the C sequence (all batch_discard
  # call sites use DESTROY/SACRIFICE reasons, never REPLACEMENT)
  bobu_eligible = jnp.any(
      mask
      & ((state.zone[p] == Zone.GARDEN) | (state.zone[p] == Zone.ALLEY))
      & _is_earth_entity(state, p)
  )

  # list-zone compaction: for each list zone, subtract how many discarded
  # cards sat below each survivor
  zone_row, zpos_row = state.zone[p], state.zpos[p]
  new_zpos = zpos_row
  for z in LIST_ZONES:
    in_zone = zone_row == z
    removed = in_zone & mask
    below = (
        removed[None, :]
        & (zpos_row[None, :] < zpos_row[:, None])
    ).sum(axis=1).astype(zpos_row.dtype)
    new_zpos = jnp.where(in_zone & ~mask, new_zpos - below, new_zpos)

  discard_count = jnp.sum((zone_row == Zone.DISCARD) & ~mask, dtype=jnp.int32)
  rank = _rank_by_key(mask, order_key)
  new_zpos = jnp.where(mask, (discard_count + rank).astype(zpos_row.dtype), new_zpos)
  new_zone = jnp.where(mask, jnp.int8(Zone.DISCARD), zone_row)

  def_id = state.def_id[p]
  valid = def_id >= 0
  base_atk = jnp.where(valid, _np(cards.BASE_ATK)[def_id], 0).astype(jnp.int8)
  base_hp = jnp.where(valid, _np(cards.BASE_HP)[def_id], 0).astype(jnp.int8)
  prefab_immune = jnp.where(valid, _np(cards.INHERENT_EFFECT_IMMUNE)[def_id], False)

  def reset_row(arr, value):
    return arr.at[p].set(jnp.where(mask, value, arr[p]))

  state = state._replace(
      zone=state.zone.at[p].set(new_zone),
      zpos=state.zpos.at[p].set(new_zpos),
      tapped=reset_row(state.tapped, False),
      cooldown=reset_row(state.cooldown, 0),
      cur_atk=reset_row(state.cur_atk, base_atk),
      cur_hp=reset_row(state.cur_hp, base_hp),
      atk_buff_perm=reset_row(state.atk_buff_perm, 0),
      atk_buff_eot=reset_row(state.atk_buff_eot, 0),
      hp_buff_perm=reset_row(state.hp_buff_perm, 0),
      hp_buff_eot=reset_row(state.hp_buff_eot, 0),
      carapace_perm=reset_row(state.carapace_perm, 0),
      carapace_eot=reset_row(state.carapace_eot, 0),
      cmb_in_perm=reset_row(state.cmb_in_perm, 0),
      cmb_in_eot=reset_row(state.cmb_in_eot, 0),
      cmb_out_perm=reset_row(state.cmb_out_perm, 0),
      cmb_out_eot=reset_row(state.cmb_out_eot, 0),
      frozen_dur=reset_row(state.frozen_dur, 0),
      shocked_dur=reset_row(state.shocked_dur, 0),
      effect_immune_dur=reset_row(
          state.effect_immune_dur, jnp.where(prefab_immune, -1, 0).astype(jnp.int8)
      ),
      grant_charge=reset_row(state.grant_charge, False),
      grant_defender=reset_row(state.grant_defender, False),
      grant_infiltrate=reset_row(state.grant_infiltrate, False),
      grant_taunt=reset_row(state.grant_taunt, False),
      grant_rooted=reset_row(state.grant_rooted, False),
      grant_godmode=reset_row(state.grant_godmode, False),
      timed_tag=state.timed_tag.at[p].set(
          jnp.where(mask[:, None], 0, state.timed_tag[p])
      ),
      timed_ticks=state.timed_ticks.at[p].set(
          jnp.where(mask[:, None], 0, state.timed_ticks[p])
      ),
      timed_phase=state.timed_phase.at[p].set(
          jnp.where(mask[:, None], 0, state.timed_phase[p])
      ),
      sacrifice_eot=reset_row(state.sacrifice_eot, False),
      attached_to=reset_row(state.attached_to, -1),
  )
  state = bobu_destroy_heal(state, p, bobu_eligible, do)
  # one latch event after the batch: equals C's per-card event sequence
  # because only the LAST evaluation persists (same side, post-all counts)
  return stt02_012_garden_event(state, p, True, any_from_garden)


def _rank_by_key(mask, key) -> jax.Array:
  """Rank (0-based) of each masked element among masked elements by key asc."""
  big = jnp.int32(1 << 28)
  k = jnp.where(mask, key.astype(jnp.int32), big)
  return jnp.sum(
      (k[None, :] < k[:, None])
      | ((k[None, :] == k[:, None]) & (jnp.arange(k.shape[0])[None, :] < jnp.arange(k.shape[0])[:, None])),
      axis=1,
  ).astype(jnp.int32)


def batch_detach_weapons(state: State, p, host_mask, host_order, do=True) -> State:
  """Discard all weapons of all hosts in host_mask (entity_util semantics).

  host_order: int32 per instance — discard-order rank of each HOST (C iterates
  hosts in a zone order); weapons keep attach order within a host. Hosts lose
  the summed weapon attack (clamped at 0 per host) and AZK01-018 leader
  modifiers are reverted."""
  do = jnp.asarray(do)
  attached = state.zone[p] == Zone.ATTACHED
  host_idx = jnp.maximum(state.attached_to[p].astype(jnp.int32), 0)
  weapon_mask = attached & host_mask[host_idx] & (state.attached_to[p] >= 0) & do

  # host atk -= sum of its weapons' atk (clamp at 0, matching sequential C
  # subtraction of non-negative weapon attacks)
  contrib = jnp.where(weapon_mask, state.cur_atk[p].astype(jnp.int16), 0)
  sums = jnp.zeros((state.zone.shape[1],), jnp.int16).at[host_idx].add(contrib)
  new_atk = jnp.maximum(state.cur_atk[p].astype(jnp.int16) - sums, 0).astype(jnp.int8)
  state = state._replace(
      cur_atk=state.cur_atk.at[p].set(
          jnp.where(host_mask & do, new_atk, state.cur_atk[p])
      )
  )

  # AZK01-018 modifier revert on leader hosts
  is_018 = state.def_id[p] == cards.CODE_TO_ID["AZK01-018"]
  host_is_leader = state.zone[p][host_idx] == Zone.LEADER
  undo_mask = weapon_mask & is_018 & host_is_leader
  undo = jnp.zeros((state.zone.shape[1],), jnp.int8).at[host_idx].add(
      jnp.where(undo_mask, 1, 0).astype(jnp.int8)
  )
  state = state._replace(cmb_in_perm=state.cmb_in_perm.at[p].add(undo))

  order_key = host_order[host_idx] * 16 + state.zpos[p].astype(jnp.int32)
  return batch_discard(state, p, weapon_mask, order_key, do=do)


def discard_equipped_weapons(state: State, p, host, do=True) -> State:
  """Discard all weapons attached to one host (entity_util.c)."""
  n = state.zone.shape[1]
  host_mask = jnp.arange(n) == host
  return batch_detach_weapons(
      state, p, host_mask, jnp.zeros((n,), jnp.int32), do=do
  )


def tick_timed_grants(state: State, p, phase: int, do=True) -> State:
  """status_util tick_timed_tag_grants for one player's garden/alley/leader:
  decrement matching-phase grants; expired grants are removed and the keyword
  flag drops when no other grant record with that tag remains (inherent
  keywords stay via the has_* OR)."""
  do = jnp.asarray(do)
  z = state.zone[p]
  in_zones = ((z == Zone.GARDEN) | (z == Zone.ALLEY) | (z == Zone.LEADER)) & do

  tags = state.timed_tag[p]
  ticks = state.timed_ticks[p]
  phases = state.timed_phase[p]
  active = tags != 0
  match = active & (phases == phase) & (ticks > 0) & in_zones[:, None]
  new_ticks = jnp.where(match, ticks - 1, ticks)
  expired = match & (new_ticks <= 0)
  new_tags = jnp.where(expired, 0, tags).astype(tags.dtype)
  new_phases = jnp.where(expired, 0, phases).astype(phases.dtype)
  new_ticks = jnp.where(expired, 0, new_ticks).astype(ticks.dtype)

  state = state._replace(
      timed_tag=state.timed_tag.at[p].set(new_tags),
      timed_ticks=state.timed_ticks.at[p].set(new_ticks),
      timed_phase=state.timed_phase.at[p].set(new_phases),
  )
  for tag, field in GRANT_FLAG_FIELDS.items():
    had_expired = jnp.any(expired & (tags == tag), axis=1)
    still_granted = jnp.any(new_tags == tag, axis=1)
    arr = getattr(state, field)
    new_flags = jnp.where(had_expired & ~still_granted, False, arr[p])
    state = state._replace(**{field: arr.at[p].set(new_flags)})
  return state


def reset_entity_health(state: State, p, inst, do=True) -> State:
  """reset_entity_health via recalculate_health_from_buffs: hp = base + buffs."""
  def_id = state.def_id[p, inst]
  base_hp = jnp.where(def_id >= 0, _np(cards.BASE_HP)[def_id], 0).astype(jnp.int16)
  total = (
      base_hp
      + state.hp_buff_perm[p, inst].astype(jnp.int16)
      + state.hp_buff_eot[p, inst].astype(jnp.int16)
      + state.passive_hp[p, inst].astype(jnp.int16)  # passive HealthBuff pairs
  )
  total = jnp.clip(total, -128, 127).astype(jnp.int8)
  cur_hp = state.cur_hp.at[p, inst].set(jnp.where(do, total, state.cur_hp[p, inst]))
  return state._replace(cur_hp=cur_hp)
