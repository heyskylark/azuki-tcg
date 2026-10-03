"""Card ability ports, batch 2 (registered into cards_impl on import).

Add cards with:
    from azuki_jax.abilities.cards_impl import register
    register("CODE", effects=..., costs=..., validate=..., target_validator=...)
Hooks operate on State; ctx lives in the ab_* fields (see cards_impl._ctx).
Each card mirrors src/abilities/cards/<code>.c exactly (validate, cost/effect
target validators, apply_costs, apply_effects).

Batch 2 scope: when-equipped (AZK01-039, STT01-015), when-returned-to-hand
(STT02-010), start-of-each-turn (STT04-003), plus the AZK01-044 bookkeeping
registration (its behavior is hardcoded in engine/phases combat_resolve).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from azuki_jax.abilities.cards_impl import (
    _ctx,
    deal_effect_damage,
    draw_with_deckout,
    register,
)
from azuki_jax.constants import Zone
from azuki_jax.state import State


def _zone_of(state: State, p, inst):
  return state.zone[p, jnp.maximum(inst, 0)]


def _zone_count(state: State, p, zone):
  return jnp.sum(state.zone[p] == zone, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# When Equipped
# ---------------------------------------------------------------------------

# AZK01-039 (entity): When Equipped (a weapon is attached to this card); this
# card gains Charge permanently (apply_charge_grant TAG_GRANT_TICK_NONE, -1).
# C resolves weapon->host via ChildOf when the source is a weapon; AZK01-039
# is an entity, so the source card itself is always the grant target.
def _azk01_039_effects(state: State) -> State:
  from azuki_jax.abilities.cards_impl import apply_charge_grant
  from azuki_jax.engine.helpers import GRANT_PHASE_NONE

  owner, src = _ctx(state)
  return apply_charge_grant(state, owner, src, GRANT_PHASE_NONE, -1, True)


register("AZK01-039", effects=_azk01_039_effects)


# STT01-015 "Tenraku" (weapon): When Equipped; if you have 15+ cards in your
# discard pile, this weapon and its host each get +1 attack (direct CurStats
# bump, no buff record — mirrors the C hook exactly).
def _stt01_015_effects(state: State) -> State:
  owner, src = _ctx(state)
  host = state.attached_to[owner, src].astype(jnp.int32)
  safe_host = jnp.maximum(host, 0)
  # C returns early when the weapon has no host (ChildOf target is a zone)
  apply = (host >= 0) & (_zone_count(state, owner, Zone.DISCARD) >= 15)

  new_weapon_atk = (state.cur_atk[owner, src].astype(jnp.int16) + 1).astype(
      jnp.int8
  )
  state = state._replace(
      cur_atk=state.cur_atk.at[owner, src].set(
          jnp.where(apply, new_weapon_atk, state.cur_atk[owner, src])
      )
  )
  new_host_atk = (
      state.cur_atk[owner, safe_host].astype(jnp.int16) + 1
  ).astype(jnp.int8)
  return state._replace(
      cur_atk=state.cur_atk.at[owner, safe_host].set(
          jnp.where(apply, new_host_atk, state.cur_atk[owner, safe_host])
      )
  )


register("STT01-015", effects=_stt01_015_effects)


# ---------------------------------------------------------------------------
# When Returned To Hand
# ---------------------------------------------------------------------------

# STT02-010 (entity): Garden only; whenever an entity is returned to its
# owner's hand, you may tap this card (ignores cooldown): draw 1.
# Queue wiring lives in cards_impl.return_to_hand (self + garden observers).
def _stt02_010_validate(state: State, owner, src) -> jax.Array:
  from azuki_jax.engine.helpers import can_tap

  in_garden = _zone_of(state, owner, src) == Zone.GARDEN
  tappable = can_tap(state, owner, jnp.maximum(src, 0), ignore_cooldown=True)
  deck_ok = _zone_count(state, owner, Zone.DECK) > 0
  return in_garden & tappable & deck_ok


def _stt02_010_costs(state: State) -> State:
  from azuki_jax.engine.helpers import tap

  owner, src = _ctx(state)
  return tap(state, owner, src, do=True)


def _stt02_010_effects(state: State) -> State:
  owner, _ = _ctx(state)
  return draw_with_deckout(state, owner, 1, True)


register(
    "STT02-010",
    effects=_stt02_010_effects,
    costs=_stt02_010_costs,
    validate=_stt02_010_validate,
)


# ---------------------------------------------------------------------------
# Start Of Each Turn
# ---------------------------------------------------------------------------

# STT04-003 (entity): Start of each turn (garden or alley); deal 1 effect
# damage to this card.
def _stt04_003_validate(state: State, owner, src) -> jax.Array:
  z = _zone_of(state, owner, src)
  return (z == Zone.GARDEN) | (z == Zone.ALLEY)


def _stt04_003_effects(state: State) -> State:
  owner, src = _ctx(state)
  return deal_effect_damage(state, owner, src, 1, do=True)


register("STT04-003", effects=_stt04_003_effects, validate=_stt04_003_validate)


# ---------------------------------------------------------------------------
# Hardcoded-behavior registrations
# ---------------------------------------------------------------------------

# AZK01-044 "Lightning Kanabo" (weapon): registry timing is None — the C
# engine hardcodes shock-on-combat-damage (once per turn per weapon) in
# combat_util.c, mirrored by engine/phases._lightning_kanabo. Registered with
# no-op hooks so coverage bookkeeping counts it as implemented.
register("AZK01-044")
