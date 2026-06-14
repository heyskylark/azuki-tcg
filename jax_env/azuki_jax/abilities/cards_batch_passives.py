"""Passive-observer card ports (registered into cards_impl + passives on
import; see abilities/passives.py for the recompute framework and the
C-equivalence argument).

Each contribution mirrors the condition its C observers maintain:
  STT01-008  +1 atk to itself while >=1 weapon is attached (stt01_008.c).
  STT01-009  +2 atk to itself while in own garden and >=6 weapon cards are in
             its owner's discard pile (stt01_009.c).
  STT01-011  +1 atk to friendly STT01-016 weapons equipped to entities in the
             owner's garden while >=1 STT01-011 is in play (garden or alley);
             non-stacking across copies — the C buff source is the shared
             card PREFAB, so duplicate applies are dedup'ed (stt01_011.c).
             Weapons equipped to the LEADER are NOT buffed (the C update and
             attach observers only consider garden hosts).
  AZK01-010  +2 atk to itself while in play (garden/alley) and every card in
             the owner's garden is a Normal-element entity; an empty garden
             satisfies the condition vacuously (azk01_010.c).
  AZK01-019  +2 hp to itself, same condition as AZK01-010 (azk01_019.c).
  AZK01-073  +1/+1 to itself while in the owner's garden and every garden
             card is a Beanz entity (non-empty garden; the card itself is
             Beanz so its own presence qualifies) (azk01_073.c).

Implemented OUTSIDE the recompute (constant, zone-independent semantics):
  AZK01-048 / AZK01-109  Carapace 1: the C init hook sets CarapaceValue{1}
             when the card instance is created (azk01_048.c via
             azk_sync_card_abilities at deck registration) and it is never
             cleared in-episode (clear_card_temporary_state only removes
             CarapaceBuff pairs) — i.e. an innate keyword. Mirrored by the
             INNATE_CARAPACE table consumed by engine/helpers.total_carapace.
  AZK01-043 / AZK01-095  Leader gains AttrCanTargetTappedAndUntappedAlley
             while an alley-targeting weapon is attached (azk01_043.c,
             shared impl for both codes). The C observer syncs the tag
             synchronously from the attached-weapon count; mirrored as a
             derived check in engine/helpers.attr_attack_alley (its only
             consumer is attack validation, action_validation.c:369 /
             engine/validate.py).

NOT ported: STT02-012 — its C observers subtract the moving entity from
garden counts on EcsOnRemove ("the entity being removed is still counted"),
but flecs children lists are already post-removal when the callbacks run, so
the buff latch is off by one on every removal event (empirically: diff 3->2
via own-side death REMOVES the buff; diff 0->1 via enemy-side death GRANTS
it) and the latched state persists until the next garden event. That makes
the buff a function of the last event's kind, not of the current board —
unrepresentable in this stateless recompute. See the report for probe
transcripts.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from azuki_jax import cards
from azuki_jax.abilities.cards_impl import register
from azuki_jax.abilities.passives import register_passive
from azuki_jax.constants import CardType, Zone
from azuki_jax.state import State

ELEMENT_NORMAL = 0  # generated/card_defs.h CARD_ELEMENT_NORMAL


def _np(table):
  return jnp.asarray(table)


def _def_is(state: State, code: str) -> jax.Array:
  return state.def_id == cards.CODE_TO_ID[code]


def _def_table(state: State, table) -> jax.Array:
  """table[def_id] per instance ((2, N)), False/0 for empty slots."""
  return jnp.where(
      state.def_id >= 0, _np(table)[jnp.maximum(state.def_id, 0)], 0
  )


def _weapon_counts(state: State) -> jax.Array:
  """(2, N) number of weapons attached to each instance."""
  n = state.zone.shape[1]
  att = (state.zone == Zone.ATTACHED) & (state.attached_to >= 0)
  idx = jnp.clip(state.attached_to.astype(jnp.int32), 0, n - 1)
  return jax.vmap(
      lambda a, h: jnp.zeros((n,), jnp.int16).at[h].add(a.astype(jnp.int16))
  )(att, idx)


def _i16(mask, amount: int) -> jax.Array:
  return jnp.where(mask, jnp.int16(amount), jnp.int16(0))


# --- STT01-008: when equipped with a weapon card, +1 attack (self) ---------
def _stt01_008(state: State):
  has_weapon = _weapon_counts(state) >= 1  # implies the host is in play
  atk = _i16(_def_is(state, "STT01-008") & has_weapon, 1)
  return atk, jnp.zeros_like(atk)


# --- STT01-009: >=6 weapons in own discard -> +2 attack (self, garden) -----
def _stt01_009(state: State):
  is_weapon = _def_table(state, cards.TYPE == CardType.WEAPON)
  weapons_in_discard = jnp.sum(
      (state.zone == Zone.DISCARD) & is_weapon, axis=1, dtype=jnp.int16
  )  # (2,)
  cond = (
      _def_is(state, "STT01-009")
      & (state.zone == Zone.GARDEN)
      & (weapons_in_discard[:, None] >= 6)
  )
  atk = _i16(cond, 2)
  return atk, jnp.zeros_like(atk)


# --- STT01-011: +1 atk on friendly STT01-016 weapons (garden hosts) --------
def _stt01_011(state: State):
  n = state.zone.shape[1]
  in_play = (state.zone == Zone.GARDEN) | (state.zone == Zone.ALLEY)
  any_011 = jnp.any(_def_is(state, "STT01-011") & in_play, axis=1)  # (2,)
  host_idx = jnp.clip(state.attached_to.astype(jnp.int32), 0, n - 1)
  host_in_garden = jnp.take_along_axis(
      state.zone == Zone.GARDEN, host_idx, axis=1
  )
  target = (
      _def_is(state, "STT01-016")
      & (state.zone == Zone.ATTACHED)
      & (state.attached_to >= 0)
      & host_in_garden
  )
  atk = _i16(target & any_011[:, None], 1)
  return atk, jnp.zeros_like(atk)


# --- AZK01-010 / AZK01-019: own garden all Normal-element entities ---------
def _garden_all_normal(state: State) -> jax.Array:
  """(2,) every owner-garden card is a Normal-element entity (empty -> True);
  mirrors azk01_010.c owner_garden_has_only_normal_entities."""
  in_garden = state.zone == Zone.GARDEN
  ok = _def_table(
      state,
      (cards.TYPE == CardType.ENTITY) & (cards.ELEMENT == ELEMENT_NORMAL),
  ).astype(jnp.bool_)
  return jnp.all(~in_garden | ok, axis=1)


def _self_in_play(state: State, code: str) -> jax.Array:
  return _def_is(state, code) & (
      (state.zone == Zone.GARDEN) | (state.zone == Zone.ALLEY)
  )


def _azk01_010(state: State):
  cond = _self_in_play(state, "AZK01-010") & _garden_all_normal(state)[:, None]
  atk = _i16(cond, 2)
  return atk, jnp.zeros_like(atk)


def _azk01_019(state: State):
  cond = _self_in_play(state, "AZK01-019") & _garden_all_normal(state)[:, None]
  hp = _i16(cond, 2)
  return jnp.zeros_like(hp), hp


# --- AZK01-073: own garden all Beanz entities (non-empty) -> +1/+1 self ----
def _azk01_073(state: State):
  in_garden = state.zone == Zone.GARDEN
  beanz = _def_table(
      state,
      (cards.TYPE == CardType.ENTITY)
      & cards.SUBTYPE_MATRIX[:, cards.subtype_index("Beanz")].astype(bool),
  ).astype(jnp.bool_)
  all_beanz = jnp.all(~in_garden | beanz, axis=1) & jnp.any(in_garden, axis=1)
  cond = _def_is(state, "AZK01-073") & in_garden & all_beanz[:, None]
  atk = _i16(cond, 1)
  return atk, atk


# --- STT02-012: EVENT-LATCHED aura (+1/+1) — C observers evaluate per garden
# add/remove event with an off-by-one on removals; the engine updates
# state.stt02_012_latch at every garden-event site (helpers.discard /
# batch_discard, apply._enter_board_slot, cards_impl.return_to_hand). The
# contribution reads the latch (in-garden gated at event time; recompute
# zeroes out-of-play contributions).
def _stt02_012(state: State):
  cond = (
      _def_is(state, "STT02-012")
      & state.stt02_012_latch
      & (state.zone == Zone.GARDEN)
  )
  amt = _i16(cond, 1)
  return amt, amt


register_passive("STT01-008", _stt01_008)
register_passive("STT01-009", _stt01_009)
register_passive("STT01-011", _stt01_011)
register_passive("AZK01-010", _azk01_010)
register_passive("AZK01-019", _azk01_019)
register_passive("AZK01-073", _azk01_073)
register_passive("STT02-012", _stt02_012)

# cards_impl bookkeeping: pure passives have no invokable/triggered hooks
# (registry: no timing tags, NONE cost/effect reqs), so no-op registration
# just marks them IMPLEMENTED.
register("STT01-008")
register("STT01-009")
register("STT01-011")
register("AZK01-010")
register("AZK01-019")
register("AZK01-073")
register("STT02-012")
# table-driven passives (engine/helpers.py INNATE_CARAPACE / attr_attack_alley)
register("AZK01-048")
register("AZK01-109")
register("AZK01-043")
register("AZK01-095")
