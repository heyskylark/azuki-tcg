"""Card ability ports, batch 1 (registered into cards_impl on import).

Add cards with:
    from azuki_jax.abilities.cards_impl import register
    register("CODE", effects=..., costs=..., validate=..., target_validator=...)
Hooks operate on State; ctx lives in the ab_* fields (see cards_impl._ctx).
Each card mirrors src/abilities/cards/<code>.c exactly (validate, cost/effect
target validators, apply_costs, apply_effects).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from azuki_jax import cards
from azuki_jax.abilities.cards_impl import (
    _ctx,
    _cost_target,
    _eff_target,
    already_selected,
    apply_attack_modifier,
    apply_charge_grant,
    apply_effect_immune,
    apply_frozen,
    apply_health_modifier,
    apply_shocked,
    apply_timed_tag_grant,
    bottom_deck_from_play,
    deal_effect_damage,
    destroy_card,
    draw_with_deckout,
    garden_seq_order,
    heal_leader,
    ikz_grant_tapped,
    mill_with_deckout,
    register,
    return_to_hand,
    sacrifice_card,
)
from azuki_jax.constants import CardType, Zone
from azuki_jax.state import State


def _np(table):
  return jnp.asarray(table)


def _zone_of(state: State, p, inst):
  return state.zone[p, jnp.maximum(inst, 0)]


def _in_garden(state: State, p, inst):
  return _zone_of(state, p, inst) == Zone.GARDEN


def _leader_of(state: State, p):
  from azuki_jax.engine.helpers import leader_instance

  return leader_instance(state, p)


def _card_type(state: State, p, inst):
  def_id = state.def_id[p, jnp.maximum(inst, 0)]
  return jnp.where(def_id >= 0, _np(cards.TYPE)[jnp.maximum(def_id, 0)], -1)


def _ikz_cost(state: State, p, inst):
  def_id = state.def_id[p, jnp.maximum(inst, 0)]
  return jnp.where(def_id >= 0, _np(cards.IKZ_COST)[jnp.maximum(def_id, 0)], 0)


def _has_subtype(state: State, p, inst, name: str):
  def_id = state.def_id[p, jnp.maximum(inst, 0)]
  col = _np(cards.SUBTYPE_MATRIX[:, cards.subtype_index(name)])
  return jnp.where(def_id >= 0, col[jnp.maximum(def_id, 0)], False)


def _zone_count(state: State, p, zone):
  return jnp.sum(state.zone[p] == zone, dtype=jnp.int32)


def _attacker(state: State):
  """(attacker_player, attacker_inst, valid) from combat state."""
  ap = (jnp.maximum(state.combat_defender_player.astype(jnp.int32), 0) + 1) % 2
  inst = state.combat_attacker.astype(jnp.int32)
  return ap, jnp.maximum(inst, 0), inst >= 0


# ---------------------------------------------------------------------------
# Wave A: no-target on-play / main abilities
# ---------------------------------------------------------------------------

# STT02-005: On Play; if you played 2 other entities this turn, draw 1.
def _stt02_005_validate(state: State, owner, src) -> jax.Array:
  total = (
      state.entities_played_garden_turn[owner].astype(jnp.int32)
      + state.entities_played_alley_turn[owner].astype(jnp.int32)
  )
  return total >= 3


def _stt02_005_effects(state: State) -> State:
  owner, _ = _ctx(state)
  return draw_with_deckout(state, owner, 1, True)


register("STT02-005", effects=_stt02_005_effects, validate=_stt02_005_validate)


# AZK01-116: On Play; deal 3 effect damage to your own leader.
def _azk01_116_effects(state: State) -> State:
  owner, _ = _ctx(state)
  leader = _leader_of(state, owner)
  return deal_effect_damage(
      state, owner, jnp.maximum(leader, 0), 3, do=leader >= 0
  )


register("AZK01-116", effects=_azk01_116_effects)


# STT01-003: On Play; mill 3 (5 if no weapons in your discard pile).
def _stt01_003_effects(state: State) -> State:
  owner, _ = _ctx(state)
  def_ids = state.def_id[owner]
  is_weapon = jnp.where(
      def_ids >= 0, _np(cards.TYPE)[jnp.maximum(def_ids, 0)] == CardType.WEAPON,
      False,
  )
  weapons = jnp.sum((state.zone[owner] == Zone.DISCARD) & is_weapon)
  n = jnp.where(weapons == 0, 5, 3)
  return mill_with_deckout(state, owner, n, True, max_n=5)


register("STT01-003", effects=_stt01_003_effects)


# AZK01-113: On Play; if you played 3+ cards this turn, gain Charge until EOT.
def _azk01_113_validate(state: State, owner, src) -> jax.Array:
  return state.cards_played_turn[owner].astype(jnp.int32) >= 3


def _azk01_113_effects(state: State) -> State:
  from azuki_jax.engine.helpers import GRANT_PHASE_END

  owner, src = _ctx(state)
  return apply_charge_grant(state, owner, src, GRANT_PHASE_END, 1, True)


register("AZK01-113", effects=_azk01_113_effects, validate=_azk01_113_validate)


# STT03-009: On Play; put the top card of your IKZ pile into play tapped.
def _stt03_009_validate(state: State, owner, src) -> jax.Array:
  return _zone_count(state, owner, Zone.IKZ_PILE) > 0


def _stt03_009_effects(state: State) -> State:
  owner, _ = _ctx(state)
  return ikz_grant_tapped(state, owner, True)


register("STT03-009", effects=_stt03_009_effects, validate=_stt03_009_validate)


# ---------------------------------------------------------------------------
# Wave B: combat-timing abilities without target selection
# ---------------------------------------------------------------------------

# STT01-012 (weapon): When Attacking; mill 1.
def _stt01_012_validate(state: State, owner, src) -> jax.Array:
  return _zone_count(state, owner, Zone.DECK) > 0


def _stt01_012_effects(state: State) -> State:
  owner, _ = _ctx(state)
  return mill_with_deckout(state, owner, 1, True, max_n=1)


register("STT01-012", effects=_stt01_012_effects, validate=_stt01_012_validate)


# AZK01-036: When Attacked; shock the attacking card (duration 1).
def _azk01_036_validate(state: State, owner, src) -> jax.Array:
  z = _zone_of(state, owner, src)
  return (z == Zone.GARDEN) | (z == Zone.ALLEY)


def _azk01_036_effects(state: State) -> State:
  ap, inst, ok = _attacker(state)
  return apply_shocked(state, ap, inst, 1, do=ok)


register("AZK01-036", effects=_azk01_036_effects, validate=_azk01_036_validate)


# AZK01-047: [Once/Turn] When Attacking (garden); heal your leader 1.
def _azk01_047_validate(state: State, owner, src) -> jax.Array:
  return _in_garden(state, owner, src)


def _azk01_047_effects(state: State) -> State:
  owner, _ = _ctx(state)
  return heal_leader(state, owner, 1, True)


register("AZK01-047", effects=_azk01_047_effects, validate=_azk01_047_validate)


# AZK01-060: When Attacking (optional, garden); gains Infiltrate +
# SacrificeAtEndOfTurn until EOT and +1 attack until EOT.
def _azk01_060_validate(state: State, owner, src) -> jax.Array:
  return _in_garden(state, owner, src)


def _azk01_060_effects(state: State) -> State:
  from azuki_jax.engine.helpers import (
      GRANT_PHASE_END,
      TAG_INFILTRATE,
      TAG_SACRIFICE_EOT,
  )

  owner, src = _ctx(state)
  state, _ = apply_timed_tag_grant(
      state, owner, src, TAG_INFILTRATE, GRANT_PHASE_END, 1, True
  )
  state, _ = apply_timed_tag_grant(
      state, owner, src, TAG_SACRIFICE_EOT, GRANT_PHASE_END, 1, True
  )
  return apply_attack_modifier(state, owner, src, 1, expires_eot=True, do=True)


register("AZK01-060", effects=_azk01_060_effects, validate=_azk01_060_validate)


# STT03-013: When Enters Garden (optional); tap this card.
def _stt03_013_validate(state: State, owner, src) -> jax.Array:
  untapped = ~state.tapped[owner, jnp.maximum(src, 0)]
  return _in_garden(state, owner, src) & untapped


def _stt03_013_effects(state: State) -> State:
  from azuki_jax.engine.helpers import tap

  owner, src = _ctx(state)
  return tap(state, owner, src, do=True)


register("STT03-013", effects=_stt03_013_effects, validate=_stt03_013_validate)


# ---------------------------------------------------------------------------
# Wave C: single-effect-target abilities
# ---------------------------------------------------------------------------

# STT01-014: On Play; deal up to 1 damage to a leader.
def _stt01_014_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return deal_effect_damage(state, tp, ti, 1, do=has & picked)


register("STT01-014", effects=_stt01_014_effects)


# AZK01-007: On Play; a friendly garden entity gets +1 attack until EOT.
def _azk01_007_validate(state: State, owner, src) -> jax.Array:
  return _zone_count(state, owner, Zone.GARDEN) > 0


def _azk01_007_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  return apply_attack_modifier(state, tp, ti, 1, expires_eot=True, do=has)


register("AZK01-007", effects=_azk01_007_effects, validate=_azk01_007_validate)


# AZK01-014: When Attacking; ANOTHER friendly garden entity gets +2 atk EOT.
def _azk01_014_validate(state: State, owner, src) -> jax.Array:
  n = state.zone.shape[1]
  others = (state.zone[owner] == Zone.GARDEN) & (jnp.arange(n) != src)
  return jnp.any(others)


def _azk01_014_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  _, src = _ctx(state)
  return ti != src


def _azk01_014_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  return apply_attack_modifier(state, tp, ti, 2, expires_eot=True, do=has)


register(
    "AZK01-014",
    effects=_azk01_014_effects,
    validate=_azk01_014_validate,
    target_validator=_azk01_014_target,
)


# AZK01-072: When Attacking; another friendly garden Beanz gets +1 atk EOT.
def _azk01_072_validate(state: State, owner, src) -> jax.Array:
  n = state.zone.shape[1]
  idx = jnp.arange(n)
  beanz = jax.vmap(lambda i: _has_subtype(state, owner, i, "Beanz"))(idx)
  others = (state.zone[owner] == Zone.GARDEN) & (idx != src) & beanz
  return jnp.any(others)


def _azk01_072_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  _, src = _ctx(state)
  return (ti != src) & _has_subtype(state, tp, ti, "Beanz")


def _azk01_072_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  return apply_attack_modifier(state, tp, ti, 1, expires_eot=True, do=has)


register(
    "AZK01-072",
    effects=_azk01_072_effects,
    validate=_azk01_072_validate,
    target_validator=_azk01_072_target,
)


# STT01-006: [Once/Turn] When Attacking; deal 1 damage to a non-EffectImmune
# enemy leader or garden entity.
def _stt01_006_validate(state: State, owner, src) -> jax.Array:
  from azuki_jax.engine.helpers import is_effect_immune

  n = state.zone.shape[1]
  idx = jnp.arange(n)
  opp = (owner + 1) % 2
  non_immune = ~jax.vmap(lambda i: is_effect_immune(state, opp, i))(idx)
  garden_ok = jnp.any((state.zone[opp] == Zone.GARDEN) & non_immune)
  leader_ok = jnp.any((state.zone[opp] == Zone.LEADER) & non_immune)
  return garden_ok | leader_ok


def _stt01_006_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  from azuki_jax.engine.helpers import is_effect_immune

  return ~is_effect_immune(state, tp, ti)


def _stt01_006_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  return deal_effect_damage(state, tp, ti, 1, do=has)


register(
    "STT01-006",
    effects=_stt01_006_effects,
    validate=_stt01_006_validate,
    target_validator=_stt01_006_target,
)


# ---------------------------------------------------------------------------
# Wave D: first spells + when-attacked target ability
# ---------------------------------------------------------------------------

# AZK01-002 (spell, Main): heal 2 to your leader.
def _azk01_002_validate(state: State, owner, src) -> jax.Array:
  leader = _leader_of(state, owner)
  return state.cur_hp[owner, jnp.maximum(leader, 0)] > 0


def _azk01_002_effects(state: State) -> State:
  owner, _ = _ctx(state)
  return heal_leader(state, owner, 2, True)


register("AZK01-002", effects=_azk01_002_effects, validate=_azk01_002_validate)


# STT02-014 (spell, Main): freeze an enemy garden entity with cost <= 2 for 2.
def _stt02_014_valid_target(state: State, tp, ti) -> jax.Array:
  return (_card_type(state, tp, ti) == CardType.ENTITY) & (
      _ikz_cost(state, tp, ti) <= 2
  )


def _stt02_014_validate(state: State, owner, src) -> jax.Array:
  n = state.zone.shape[1]
  idx = jnp.arange(n)
  opp = (owner + 1) % 2
  ok = jax.vmap(lambda i: _stt02_014_valid_target(state, opp, i))(idx)
  return jnp.any((state.zone[opp] == Zone.GARDEN) & ok)


def _stt02_014_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return _stt02_014_valid_target(state, tp, ti)


def _stt02_014_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  return apply_frozen(state, tp, ti, 2, do=has)


register(
    "STT02-014",
    effects=_stt02_014_effects,
    validate=_stt02_014_validate,
    target_validator=_stt02_014_target,
)


# AZK01-127 (spell, Response): deal 1 damage to an enemy garden entity.
def _azk01_127_validate(state: State, owner, src) -> jax.Array:
  opp = (owner + 1) % 2
  return _zone_count(state, opp, Zone.GARDEN) > 0


def _azk01_127_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return _card_type(state, tp, ti) == CardType.ENTITY


def _azk01_127_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return deal_effect_damage(state, tp, ti, 1, do=has & picked)


register(
    "AZK01-127",
    effects=_azk01_127_effects,
    validate=_azk01_127_validate,
    target_validator=_azk01_127_target,
)


# AZK01-040: When Attacked; deal up to 1 damage to a leader.
def _azk01_040_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return deal_effect_damage(state, tp, ti, 1, do=has & picked)


register("AZK01-040", effects=_azk01_040_effects)


# AZK01-128 (spell, Response): destroy the attacking entity if its HP <= 2.
def _azk01_128_validate(state: State, owner, src) -> jax.Array:
  ap, inst, ok = _attacker(state)
  return ok & (state.cur_hp[ap, inst] <= 2)


def _azk01_128_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  ap, inst, ok = _attacker(state)
  return (
      ok
      & (tp == ap)
      & (ti == inst)
      & (_card_type(state, tp, ti) == CardType.ENTITY)
      & (state.cur_hp[tp, ti] <= 2)
  )


def _azk01_128_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return destroy_card(state, tp, ti, do=has & picked)


register(
    "AZK01-128",
    effects=_azk01_128_effects,
    validate=_azk01_128_validate,
    target_validator=_azk01_128_target,
)


# ---------------------------------------------------------------------------
# Wave E: multi-target / mass-effect spells
# ---------------------------------------------------------------------------

# STT01-017 (spell, Response): deal 1 damage to one or two DIFFERENT enemy
# garden entities.
def _stt01_017_validate(state: State, owner, src) -> jax.Array:
  opp = (owner + 1) % 2
  return _zone_count(state, opp, Zone.GARDEN) >= 1


def _stt01_017_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return ~already_selected(state, scope_is_cost, tp, ti)


def _stt01_017_effects(state: State) -> State:
  sel = state.ab_eff_selected.astype(jnp.int32)
  for k in range(2):
    tp, ti, has = _eff_target(state, k)
    state = deal_effect_damage(state, tp, ti, 1, do=has & (k < sel))
  return state


register(
    "STT01-017",
    effects=_stt01_017_effects,
    validate=_stt01_017_validate,
    target_validator=_stt01_017_target,
)


# AZK01-042 (spell, Main): deal 3/2/1 damage to three different enemy garden
# entities (in selection order).
def _azk01_042_validate(state: State, owner, src) -> jax.Array:
  opp = (owner + 1) % 2
  return _zone_count(state, opp, Zone.GARDEN) >= 3


def _azk01_042_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return ~already_selected(state, scope_is_cost, tp, ti)


def _azk01_042_effects(state: State) -> State:
  sel = state.ab_eff_selected.astype(jnp.int32)
  for k, dmg in enumerate((3, 2, 1)):
    tp, ti, has = _eff_target(state, k)
    state = deal_effect_damage(state, tp, ti, dmg, do=has & (k < sel))
  return state


register(
    "AZK01-042",
    effects=_azk01_042_effects,
    validate=_azk01_042_validate,
    target_validator=_azk01_042_target,
)


# STT02-015 (spell, Response): return an entity with cost <= 3 in any garden
# to its owner's hand.
def _stt02_015_valid_target(state: State, tp, ti) -> jax.Array:
  return (_card_type(state, tp, ti) == CardType.ENTITY) & (
      _ikz_cost(state, tp, ti) <= 3
  )


def _stt02_015_validate(state: State, owner, src) -> jax.Array:
  n = state.zone.shape[1]
  idx = jnp.arange(n)
  ok = jnp.zeros((), jnp.bool_)
  for p in (0, 1):
    valid = jax.vmap(lambda i: _stt02_015_valid_target(state, p, i))(idx)
    ok = ok | jnp.any((state.zone[p] == Zone.GARDEN) & valid)
  return ok


def _stt02_015_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return _stt02_015_valid_target(state, tp, ti)


def _stt02_015_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  return return_to_hand(state, tp, ti, do=has)


register(
    "STT02-015",
    effects=_stt02_015_effects,
    validate=_stt02_015_validate,
    target_validator=_stt02_015_target,
)


# STT03-016 (spell, Main): destroy all enemy garden entities with HP <= 2.
def _stt03_016_validate(state: State, owner, src) -> jax.Array:
  opp = (owner + 1) % 2
  return jnp.any((state.zone[opp] == Zone.GARDEN) & (state.cur_hp[opp] <= 2))


def _stt03_016_effects(state: State) -> State:
  owner, _ = _ctx(state)
  opp = (owner + 1) % 2
  order, in_garden = garden_seq_order(state, opp)
  marked = in_garden & (state.cur_hp[opp] <= 2)  # snapshot before destroys
  for k in range(5):
    inst = order[k]
    state = destroy_card(state, opp, inst, do=marked[inst])
  return state


register("STT03-016", effects=_stt03_016_effects, validate=_stt03_016_validate)


# AZK01-066 (spell, Main): deal 2 damage to every leader and garden entity.
def _azk01_066_effects(state: State) -> State:
  for p in (0, 1):
    leader = _leader_of(state, p)
    state = deal_effect_damage(
        state, p, jnp.maximum(leader, 0), 2, do=leader >= 0
    )
    order, in_garden = garden_seq_order(state, p)
    is_entity = jax.vmap(lambda i: _card_type(state, p, i) == CardType.ENTITY)(
        jnp.arange(state.zone.shape[1])
    )
    marked = in_garden & is_entity  # snapshot
    for k in range(5):
      inst = order[k]
      state = deal_effect_damage(state, p, inst, 2, do=marked[inst])
  return state


register("AZK01-066", effects=_azk01_066_effects)


# ---------------------------------------------------------------------------
# Wave F: cost-target abilities (discard / bounce costs)
# ---------------------------------------------------------------------------

# STT01-007: On Play (optional); discard 1: draw 1.
def _stt01_007_validate(state: State, owner, src) -> jax.Array:
  return (_zone_count(state, owner, Zone.HAND) >= 1) & (
      _zone_count(state, owner, Zone.DECK) >= 1
  )


def _stt01_007_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  _, src = _ctx(state)
  return ~scope_is_cost | (ti != src)


def _stt01_007_costs(state: State) -> State:
  from azuki_jax.engine.helpers import discard

  tp, ti, has = _cost_target(state, 0)
  return discard(state, tp, ti, do=has)


def _stt01_007_effects(state: State) -> State:
  owner, _ = _ctx(state)
  return draw_with_deckout(state, owner, 1, True)


register(
    "STT01-007",
    effects=_stt01_007_effects,
    costs=_stt01_007_costs,
    validate=_stt01_007_validate,
    target_validator=_stt01_007_target,
)


# STT02-016 (spell, Response): discard 1: a leader or enemy garden entity gets
# -2 attack until end of turn.
def _stt02_016_validate(state: State, owner, src) -> jax.Array:
  # opponent leader always exists; only the hand requirement can fail
  return _zone_count(state, owner, Zone.HAND) >= 1


def _stt02_016_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  _, src = _ctx(state)
  return ~scope_is_cost | (ti != src)


def _stt02_016_costs(state: State) -> State:
  from azuki_jax.engine.helpers import discard

  tp, ti, has = _cost_target(state, 0)
  return discard(state, tp, ti, do=has)


def _stt02_016_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  return apply_attack_modifier(state, tp, ti, -2, expires_eot=True, do=has)


register(
    "STT02-016",
    effects=_stt02_016_effects,
    costs=_stt02_016_costs,
    validate=_stt02_016_validate,
    target_validator=_stt02_016_target,
)


# AZK01-022: On Play (optional); discard 1: return an entity with cost <= 2 in
# any garden to its owner's hand.
def _azk01_022_valid_bounce(state: State, tp, ti) -> jax.Array:
  return (_card_type(state, tp, ti) == CardType.ENTITY) & (
      _ikz_cost(state, tp, ti) <= 2
  )


def _azk01_022_validate(state: State, owner, src) -> jax.Array:
  n = state.zone.shape[1]
  idx = jnp.arange(n)
  any_target = jnp.zeros((), jnp.bool_)
  for p in (0, 1):
    valid = jax.vmap(lambda i: _azk01_022_valid_bounce(state, p, i))(idx)
    any_target = any_target | jnp.any((state.zone[p] == Zone.GARDEN) & valid)
  return (_zone_count(state, owner, Zone.HAND) >= 1) & any_target


def _azk01_022_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  _, src = _ctx(state)
  return jnp.where(
      scope_is_cost, ti != src, _azk01_022_valid_bounce(state, tp, ti)
  )


def _azk01_022_costs(state: State) -> State:
  from azuki_jax.engine.helpers import discard

  tp, ti, has = _cost_target(state, 0)
  return discard(state, tp, ti, do=has)


def _azk01_022_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  return return_to_hand(state, tp, ti, do=has)


register(
    "AZK01-022",
    effects=_azk01_022_effects,
    costs=_azk01_022_costs,
    validate=_azk01_022_validate,
    target_validator=_azk01_022_target,
)


# AZK01-029 (spell, Response): discard 2: a leader or garden entity gets -3
# attack until end of turn.
def _azk01_029_validate(state: State, owner, src) -> jax.Array:
  n = state.zone.shape[1]
  others = (state.zone[owner] == Zone.HAND) & (jnp.arange(n) != src)
  return jnp.sum(others) >= 2


def _azk01_029_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  _, src = _ctx(state)
  cost_ok = (ti != src) & ~already_selected(state, scope_is_cost, tp, ti)
  return ~scope_is_cost | cost_ok


def _azk01_029_costs(state: State) -> State:
  from azuki_jax.engine.helpers import discard

  sel = state.ab_cost_selected.astype(jnp.int32)
  for k in range(2):
    tp, ti, has = _cost_target(state, k)
    state = discard(state, tp, ti, do=has & (k < sel))
  return state


def _azk01_029_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  return apply_attack_modifier(state, tp, ti, -3, expires_eot=True, do=has)


register(
    "AZK01-029",
    effects=_azk01_029_effects,
    costs=_azk01_029_costs,
    validate=_azk01_029_validate,
    target_validator=_azk01_029_target,
)


# STT02-009: On Play (optional); return an entity with cost >= 2 in your
# garden to hand: return up to 1 entity with cost <= 2 in the opponent's
# garden to its owner's hand.
def _stt02_009_validate(state: State, owner, src) -> jax.Array:
  n = state.zone.shape[1]
  idx = jnp.arange(n)
  cost_ok = jax.vmap(
      lambda i: (_card_type(state, owner, i) == CardType.ENTITY)
      & (_ikz_cost(state, owner, i) >= 2)
  )(idx)
  return jnp.any((state.zone[owner] == Zone.GARDEN) & cost_ok)


def _stt02_009_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  is_entity = _card_type(state, tp, ti) == CardType.ENTITY
  return jnp.where(
      scope_is_cost,
      is_entity & (_ikz_cost(state, tp, ti) >= 2),
      is_entity & (_ikz_cost(state, tp, ti) <= 2),
  )


def _stt02_009_costs(state: State) -> State:
  tp, ti, has = _cost_target(state, 0)
  return return_to_hand(state, tp, ti, do=has)


def _stt02_009_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  return return_to_hand(state, tp, ti, do=has)


register(
    "STT02-009",
    effects=_stt02_009_effects,
    costs=_stt02_009_costs,
    validate=_stt02_009_validate,
    target_validator=_stt02_009_target,
)


# ---------------------------------------------------------------------------
# Wave G: main-activated self-cost abilities + end-of-turn
# ---------------------------------------------------------------------------

# STT03-004: Main; sacrifice this card: heal 1 to your leader.
def _stt03_004_costs(state: State) -> State:
  owner, src = _ctx(state)
  return sacrifice_card(state, owner, src, do=True)


def _stt03_004_effects(state: State) -> State:
  owner, _ = _ctx(state)
  return heal_leader(state, owner, 1, True)


register("STT03-004", effects=_stt03_004_effects, costs=_stt03_004_costs)


# AZK01-105: Main (garden); sacrifice this card: deal damage equal to its HP
# to an enemy leader or garden entity.
def _azk01_105_validate(state: State, owner, src) -> jax.Array:
  return _in_garden(state, owner, src)


def _azk01_105_costs(state: State) -> State:
  owner, src = _ctx(state)
  damage = state.cur_hp[owner, src].astype(jnp.int16)
  state = state._replace(ab_scratch=state.ab_scratch.at[0].set(damage))
  return sacrifice_card(state, owner, src, do=True)


def _azk01_105_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  damage = state.ab_scratch[0].astype(jnp.int16)
  return deal_effect_damage(
      state, tp, ti, damage, do=has & picked & (damage > 0)
  )


register(
    "AZK01-105",
    effects=_azk01_105_effects,
    costs=_azk01_105_costs,
    validate=_azk01_105_validate,
)


# AZK01-011: End of Turn (garden); if this card is untapped, discard it (and
# its weapons).
def _azk01_011_validate(state: State, owner, src) -> jax.Array:
  return _in_garden(state, owner, src)


def _azk01_011_effects(state: State) -> State:
  from azuki_jax.engine.helpers import discard_equipped_weapons

  owner, src = _ctx(state)
  untapped = ~state.tapped[owner, src]
  state = discard_equipped_weapons(state, owner, src, do=untapped)
  return destroy_card(state, owner, src, do=untapped)


register("AZK01-011", effects=_azk01_011_effects, validate=_azk01_011_validate)


# STT04-015 (spell, Main): deal 1 damage to your leader: deal 2 damage to the
# enemy leader.
def _stt04_015_costs(state: State) -> State:
  owner, _ = _ctx(state)
  leader = _leader_of(state, owner)
  return deal_effect_damage(
      state, owner, jnp.maximum(leader, 0), 1, do=leader >= 0
  )


def _stt04_015_effects(state: State) -> State:
  owner, _ = _ctx(state)
  opp = (owner + 1) % 2
  leader = _leader_of(state, opp)
  return deal_effect_damage(
      state, opp, jnp.maximum(leader, 0), 2, do=leader >= 0
  )


register("STT04-015", effects=_stt04_015_effects, costs=_stt04_015_costs)


# AZK01-065 (spell, Main): deal 3 damage to your leader: deal 5 damage to a
# leader or garden entity.
def _azk01_065_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  is_leader = state.zone[tp, ti] == Zone.LEADER
  return is_leader | (_card_type(state, tp, ti) == CardType.ENTITY)


def _azk01_065_costs(state: State) -> State:
  owner, _ = _ctx(state)
  leader = _leader_of(state, owner)
  return deal_effect_damage(
      state, owner, jnp.maximum(leader, 0), 3, do=leader >= 0
  )


def _azk01_065_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return deal_effect_damage(state, tp, ti, 5, do=has & picked)


register(
    "AZK01-065",
    effects=_azk01_065_effects,
    costs=_azk01_065_costs,
    target_validator=_azk01_065_target,
)


# ---------------------------------------------------------------------------
# Wave H: remaining self/cost abilities
# ---------------------------------------------------------------------------

# AZK01-068: On Play (alley); draw 1: discard 1 card from your hand.
def _azk01_068_validate(state: State, owner, src) -> jax.Array:
  in_alley = _zone_of(state, owner, src) == Zone.ALLEY
  return in_alley & (_zone_count(state, owner, Zone.DECK) > 0)


def _azk01_068_costs(state: State) -> State:
  owner, _ = _ctx(state)
  return draw_with_deckout(state, owner, 1, True)


def _azk01_068_effects(state: State) -> State:
  from azuki_jax.engine.helpers import discard

  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return discard(state, tp, ti, do=has & picked)


register(
    "AZK01-068",
    effects=_azk01_068_effects,
    costs=_azk01_068_costs,
    validate=_azk01_068_validate,
)


# AZK01-070: Response (garden); tap this card, deal 1 damage to it: an enemy
# garden entity gets -1 attack until end of turn.
def _azk01_070_validate(state: State, owner, src) -> jax.Array:
  from azuki_jax.engine.helpers import can_tap

  return _in_garden(state, owner, src) & can_tap(
      state, owner, jnp.maximum(src, 0)
  )


def _azk01_070_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return _card_type(state, tp, ti) == CardType.ENTITY


def _azk01_070_costs(state: State) -> State:
  from azuki_jax.engine.helpers import tap

  owner, src = _ctx(state)
  state = tap(state, owner, src, do=True)
  return deal_effect_damage(state, owner, src, 1, do=True)


def _azk01_070_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return apply_attack_modifier(
      state, tp, ti, -1, expires_eot=True, do=has & picked
  )


register(
    "AZK01-070",
    effects=_azk01_070_effects,
    costs=_azk01_070_costs,
    validate=_azk01_070_validate,
    target_validator=_azk01_070_target,
)


# AZK01-058: After Attacking (optional, garden); sacrifice this card: a
# friendly garden entity or any leader gets +2 attack until end of turn.
def _azk01_058_validate(state: State, owner, src) -> jax.Array:
  return _in_garden(state, owner, src)


def _azk01_058_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  in_own_garden = (tp == owner) & (state.zone[tp, ti] == Zone.GARDEN)
  is_leader = state.zone[tp, ti] == Zone.LEADER
  return (in_own_garden & (_card_type(state, tp, ti) == CardType.ENTITY)) | is_leader


def _azk01_058_costs(state: State) -> State:
  owner, src = _ctx(state)
  return sacrifice_card(state, owner, src, do=True)


def _azk01_058_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return apply_attack_modifier(
      state, tp, ti, 2, expires_eot=True, do=has & picked
  )


register(
    "AZK01-058",
    effects=_azk01_058_effects,
    costs=_azk01_058_costs,
    validate=_azk01_058_validate,
    target_validator=_azk01_058_target,
)


# AZK01-008: On Play (optional); sacrifice this card: sacrifice an enemy
# garden entity with cost <= 3 (weapons discarded first on both).
def _azk01_008_valid_target(state: State, tp, ti) -> jax.Array:
  return (_card_type(state, tp, ti) == CardType.ENTITY) & (
      _ikz_cost(state, tp, ti) <= 3
  )


def _azk01_008_validate(state: State, owner, src) -> jax.Array:
  n = state.zone.shape[1]
  idx = jnp.arange(n)
  opp = (owner + 1) % 2
  ok = jax.vmap(lambda i: _azk01_008_valid_target(state, opp, i))(idx)
  return jnp.any((state.zone[opp] == Zone.GARDEN) & ok)


def _azk01_008_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return _azk01_008_valid_target(state, tp, ti)


def _azk01_008_costs(state: State) -> State:
  from azuki_jax.engine.helpers import discard_equipped_weapons

  owner, src = _ctx(state)
  state = discard_equipped_weapons(state, owner, src, do=True)
  return sacrifice_card(state, owner, src, do=True)


def _azk01_008_effects(state: State) -> State:
  from azuki_jax.engine.helpers import discard_equipped_weapons

  tp, ti, has = _eff_target(state, 0)
  state = discard_equipped_weapons(state, tp, ti, do=has)
  return sacrifice_card(state, tp, ti, do=has)


register(
    "AZK01-008",
    effects=_azk01_008_effects,
    costs=_azk01_008_costs,
    validate=_azk01_008_validate,
    target_validator=_azk01_008_target,
)


# STT03-011: On Play (optional, garden); destroy an enemy garden entity with
# BASE health <= 2.
def _stt03_011_base_hp_ok(state: State, tp, ti) -> jax.Array:
  def_id = state.def_id[tp, jnp.maximum(ti, 0)]
  base = jnp.where(def_id >= 0, _np(cards.BASE_HP)[jnp.maximum(def_id, 0)], 127)
  has_stats = jnp.where(
      def_id >= 0, _np(cards.HAS_BASE_STATS)[jnp.maximum(def_id, 0)], False
  )
  return has_stats & (base <= 2)


def _stt03_011_validate(state: State, owner, src) -> jax.Array:
  n = state.zone.shape[1]
  idx = jnp.arange(n)
  opp = (owner + 1) % 2
  ok = jax.vmap(lambda i: _stt03_011_base_hp_ok(state, opp, i))(idx)
  return _in_garden(state, owner, src) & jnp.any(
      (state.zone[opp] == Zone.GARDEN) & ok
  )


def _stt03_011_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return (_card_type(state, tp, ti) == CardType.ENTITY) & _stt03_011_base_hp_ok(
      state, tp, ti
  )


def _stt03_011_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return destroy_card(state, tp, ti, do=has & picked)


register(
    "STT03-011",
    effects=_stt03_011_effects,
    validate=_stt03_011_validate,
    target_validator=_stt03_011_target,
)


# STT04-004: On Play (optional); sacrifice this card: deal 1 damage to any
# garden entity.
def _stt04_004_validate(state: State, owner, src) -> jax.Array:
  return (_zone_count(state, 0, Zone.GARDEN) > 0) | (
      _zone_count(state, 1, Zone.GARDEN) > 0
  )


def _stt04_004_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return _card_type(state, tp, ti) == CardType.ENTITY


def _stt04_004_costs(state: State) -> State:
  owner, src = _ctx(state)
  return sacrifice_card(state, owner, src, do=True)


def _stt04_004_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return deal_effect_damage(state, tp, ti, 1, do=has & picked)


register(
    "STT04-004",
    effects=_stt04_004_effects,
    costs=_stt04_004_costs,
    validate=_stt04_004_validate,
    target_validator=_stt04_004_target,
)


# AZK01-009 (spell, Main): a garden entity with cost <= 4 gains Charge until
# end of turn.
def _azk01_009_valid_target(state: State, tp, ti) -> jax.Array:
  return (_card_type(state, tp, ti) == CardType.ENTITY) & (
      _ikz_cost(state, tp, ti) <= 4
  )


def _azk01_009_validate(state: State, owner, src) -> jax.Array:
  n = state.zone.shape[1]
  idx = jnp.arange(n)
  ok = jnp.zeros((), jnp.bool_)
  for p in (0, 1):
    valid = jax.vmap(lambda i: _azk01_009_valid_target(state, p, i))(idx)
    ok = ok | jnp.any((state.zone[p] == Zone.GARDEN) & valid)
  return ok


def _azk01_009_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return _azk01_009_valid_target(state, tp, ti)


def _azk01_009_effects(state: State) -> State:
  from azuki_jax.engine.helpers import GRANT_PHASE_END

  tp, ti, has = _eff_target(state, 0)
  return apply_charge_grant(state, tp, ti, GRANT_PHASE_END, 1, has)


register(
    "AZK01-009",
    effects=_azk01_009_effects,
    validate=_azk01_009_validate,
    target_validator=_azk01_009_target,
)


# AZK01-117 (spell, Main): deal 2 damage to your leader: a garden entity with
# cost <= 5 gains Charge until end of turn.
def _azk01_117_validate(state: State, owner, src) -> jax.Array:
  leader = _leader_of(state, owner)
  n = state.zone.shape[1]
  idx = jnp.arange(n)
  ok = jnp.zeros((), jnp.bool_)
  for p in (0, 1):
    cost_ok = jax.vmap(lambda i: _ikz_cost(state, p, i) <= 5)(idx)
    ok = ok | jnp.any((state.zone[p] == Zone.GARDEN) & cost_ok)
  return (leader >= 0) & ok


def _azk01_117_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return (_card_type(state, tp, ti) == CardType.ENTITY) & (
      _ikz_cost(state, tp, ti) <= 5
  )


def _azk01_117_costs(state: State) -> State:
  owner, _ = _ctx(state)
  leader = _leader_of(state, owner)
  return deal_effect_damage(
      state, owner, jnp.maximum(leader, 0), 2, do=leader >= 0
  )


def _azk01_117_effects(state: State) -> State:
  from azuki_jax.engine.helpers import GRANT_PHASE_END

  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return apply_charge_grant(state, tp, ti, GRANT_PHASE_END, 1, has & picked)


register(
    "AZK01-117",
    effects=_azk01_117_effects,
    costs=_azk01_117_costs,
    validate=_azk01_117_validate,
    target_validator=_azk01_117_target,
)


# STT02-011: Main (garden); sacrifice this card: a friendly garden entity
# cannot take effect damage until the start of your next turn (immune 2).
def _stt02_011_validate(state: State, owner, src) -> jax.Array:
  n = state.zone.shape[1]
  others = (state.zone[owner] == Zone.GARDEN) & (jnp.arange(n) != src)
  return _in_garden(state, owner, src) & jnp.any(others)


def _stt02_011_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  _, src = _ctx(state)
  return ti != src


def _stt02_011_costs(state: State) -> State:
  owner, src = _ctx(state)
  return sacrifice_card(state, owner, src, do=True)


def _stt02_011_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  return apply_effect_immune(state, tp, ti, 2, do=has)


register(
    "STT02-011",
    effects=_stt02_011_effects,
    costs=_stt02_011_costs,
    validate=_stt02_011_validate,
    target_validator=_stt02_011_target,
)


# STT01-005: Main (alley); sacrifice this card: draw 3, then discard 2.
def _stt01_005_validate(state: State, owner, src) -> jax.Array:
  in_alley = _zone_of(state, owner, src) == Zone.ALLEY
  return in_alley & (_zone_count(state, owner, Zone.DECK) >= 1)


def _stt01_005_costs(state: State) -> State:
  owner, src = _ctx(state)
  state = sacrifice_card(state, owner, src, do=True)
  return draw_with_deckout(state, owner, 3, True)


def _stt01_005_effects(state: State) -> State:
  # C allows selecting the same hand card twice; the second sacrifice_card is
  # a flecs no-op (already a child of discard) — guard to avoid re-appending.
  sel = state.ab_eff_selected.astype(jnp.int32)
  for k in range(2):
    tp, ti, has = _eff_target(state, k)
    not_discarded = state.zone[tp, ti] != Zone.DISCARD
    state = sacrifice_card(state, tp, ti, do=has & (k < sel) & not_discarded)
  return state


register(
    "STT01-005",
    effects=_stt01_005_effects,
    costs=_stt01_005_costs,
    validate=_stt01_005_validate,
)


# ---------------------------------------------------------------------------
# Wave I: scratch-state / multi-step abilities
# ---------------------------------------------------------------------------

_EARTH = 3  # CARD_ELEMENT_EARTH


def _element_of(state: State, p, inst):
  def_id = state.def_id[p, jnp.maximum(inst, 0)]
  return jnp.where(def_id >= 0, _np(cards.ELEMENT)[jnp.maximum(def_id, 0)], -1)


# AZK01-103: Main (garden, untapped); tap this card and sacrifice another
# untapped EARTH entity in your garden: deal damage equal to its HP (max 5) to
# a leader; if its HP was >= 3, also draw 1.
def _azk01_103_earth_garden(state: State, owner, ti) -> jax.Array:
  return (
      (_card_type(state, owner, ti) == CardType.ENTITY)
      & (_element_of(state, owner, ti) == _EARTH)
      & (state.zone[owner, ti] == Zone.GARDEN)
  )


def _azk01_103_validate(state: State, owner, src) -> jax.Array:
  n = state.zone.shape[1]
  idx = jnp.arange(n)
  self_untapped = ~state.tapped[owner, jnp.maximum(src, 0)]
  earth = jax.vmap(lambda i: _azk01_103_earth_garden(state, owner, i))(idx)
  others = (idx != src) & earth & ~state.tapped[owner]
  return _in_garden(state, owner, src) & self_untapped & jnp.any(others)


def _azk01_103_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  _, src = _ctx(state)
  cost_ok = (
      (ti != src)
      & _azk01_103_earth_garden(state, owner, ti)
      & ~state.tapped[tp, ti]
  )
  return ~scope_is_cost | cost_ok


def _azk01_103_costs(state: State) -> State:
  from azuki_jax.engine.helpers import tap

  owner, src = _ctx(state)
  tp, ti, has = _cost_target(state, 0)
  state = tap(state, owner, src, do=has)
  hp = state.cur_hp[tp, ti].astype(jnp.int16)
  health = jnp.clip(hp, 0, 5)
  draw_after = hp >= 3
  state = state._replace(
      ab_scratch=state.ab_scratch.at[0]
      .set(jnp.where(has, health, state.ab_scratch[0]))
      .at[1]
      .set(jnp.where(has & draw_after, 1, 0).astype(jnp.int16))
  )
  return sacrifice_card(state, tp, ti, do=has)


def _azk01_103_effects(state: State) -> State:
  owner, _ = _ctx(state)
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  damage = state.ab_scratch[0].astype(jnp.int16)
  state = deal_effect_damage(
      state, tp, ti, damage, do=has & picked & (damage > 0)
  )
  return draw_with_deckout(
      state, owner, 1, (state.ab_scratch[1] != 0) & picked
  )


register(
    "AZK01-103",
    effects=_azk01_103_effects,
    costs=_azk01_103_costs,
    validate=_azk01_103_validate,
    target_validator=_azk01_103_target,
)


# AZK01-032 (spell, Main): return an entity with cost >= 2 in your garden to
# hand: return up to 1 enemy garden entity with cost <= 4 to its owner's hand.
def _azk01_032_validate(state: State, owner, src) -> jax.Array:
  n = state.zone.shape[1]
  idx = jnp.arange(n)
  ok = jax.vmap(
      lambda i: (_card_type(state, owner, i) == CardType.ENTITY)
      & (_ikz_cost(state, owner, i) >= 2)
  )(idx)
  return jnp.any((state.zone[owner] == Zone.GARDEN) & ok)


def _azk01_032_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  is_entity = _card_type(state, tp, ti) == CardType.ENTITY
  return jnp.where(
      scope_is_cost,
      is_entity & (_ikz_cost(state, tp, ti) >= 2),
      is_entity & (_ikz_cost(state, tp, ti) <= 4),
  )


def _azk01_032_costs(state: State) -> State:
  tp, ti, has = _cost_target(state, 0)
  return return_to_hand(state, tp, ti, do=has)


def _azk01_032_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  return return_to_hand(state, tp, ti, do=has)


register(
    "AZK01-032",
    effects=_azk01_032_effects,
    costs=_azk01_032_costs,
    validate=_azk01_032_validate,
    target_validator=_azk01_032_target,
)


# STT04-016 (spell, Main): deal 1 damage to a friendly garden entity: deal 2
# damage to up to 1 enemy leader or garden entity.
def _stt04_016_validate(state: State, owner, src) -> jax.Array:
  return _zone_count(state, owner, Zone.GARDEN) > 0


def _stt04_016_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  cost_ok = _card_type(state, tp, ti) == CardType.ENTITY
  return ~scope_is_cost | cost_ok


def _stt04_016_costs(state: State) -> State:
  tp, ti, has = _cost_target(state, 0)
  picked = state.ab_cost_selected > 0
  return deal_effect_damage(state, tp, ti, 1, do=has & picked)


def _stt04_016_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return deal_effect_damage(state, tp, ti, 2, do=has & picked)


register(
    "STT04-016",
    effects=_stt04_016_effects,
    costs=_stt04_016_costs,
    validate=_stt04_016_validate,
    target_validator=_stt04_016_target,
)


# STT04-017 (spell, Main): sacrifice 1..5 friendly garden entities, then deal
# that many damage to one leader-or-garden target.
def _stt04_017_validate(state: State, owner, src) -> jax.Array:
  return _zone_count(state, owner, Zone.GARDEN) > 0


def _stt04_017_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  # cost: friendly garden entity not already selected; effect: any garden
  # entity or leader (runtime's type enumeration already guarantees zones)
  cost_ok = (
      (tp == owner)
      & (_zone_of(state, tp, ti) == Zone.GARDEN)
      & ~already_selected(state, jnp.bool_(True), tp, ti)
  )
  return jnp.where(scope_is_cost, cost_ok, True)


def _stt04_017_costs(state: State) -> State:
  sel = state.ab_cost_selected.astype(jnp.int16)
  state = state._replace(
      ab_scratch=state.ab_scratch.at[0].set(sel).at[1].set(jnp.int16(0))
  )
  for k in range(5):
    tp = jnp.maximum(state.ab_cost_target_players[k].astype(jnp.int32), 0)
    ti = jnp.maximum(state.ab_cost_targets[k].astype(jnp.int32), 0)
    do = (k < state.ab_cost_selected) & (state.ab_cost_targets[k] >= 0)
    state = sacrifice_card(state, tp, ti, do)
  return state


def _stt04_017_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  damage = state.ab_scratch[0].astype(jnp.int16)
  return deal_effect_damage(
      state, tp, ti, damage, do=has & picked & (damage > 0)
  )


register(
    "STT04-017",
    effects=_stt04_017_effects,
    costs=_stt04_017_costs,
    validate=_stt04_017_validate,
    target_validator=_stt04_017_target,
)


# AZK01-020 (spell, Main + Response): two different friendly garden entities
# get +1 attack until EOT (Main) or +1 health until EOT (Response).
def _azk01_020_validate(state: State, owner, src) -> jax.Array:
  return _zone_count(state, owner, Zone.GARDEN) >= 2


def _azk01_020_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return ~already_selected(state, scope_is_cost, tp, ti)


def _azk01_020_effects(state: State) -> State:
  from azuki_jax.constants import Phase

  is_response = state.phase == Phase.RESPONSE_WINDOW
  sel = state.ab_eff_selected.astype(jnp.int32)
  for k in range(2):
    tp, ti, has = _eff_target(state, k)
    do = has & (k < sel)
    state = apply_health_modifier(
        state, tp, ti, 1, expires_eot=True, do=do & is_response
    )
    state = apply_attack_modifier(
        state, tp, ti, 1, expires_eot=True, do=do & ~is_response
    )
  return state


register(
    "AZK01-020",
    effects=_azk01_020_effects,
    validate=_azk01_020_validate,
    target_validator=_azk01_020_target,
)


# ---------------------------------------------------------------------------
# Wave J: mass / deck-manipulation abilities + weapons
# ---------------------------------------------------------------------------

# AZK01-087 (spell, Main): put up to 2 enemy garden entities with combined
# cost <= 5 on the bottom of their owner's deck.
def _azk01_087_validate(state: State, owner, src) -> jax.Array:
  opp = (owner + 1) % 2
  return _zone_count(state, opp, Zone.GARDEN) > 0


def _azk01_087_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  from azuki_jax.constants import MAX_ABILITY_SELECTION

  ok = (
      (_card_type(state, tp, ti) == CardType.ENTITY)
      & (_ikz_cost(state, tp, ti) <= 5)
      & ~already_selected(state, scope_is_cost, tp, ti)
  )
  # combined cost of already-selected effect targets + this target <= 5
  k = jnp.arange(MAX_ABILITY_SELECTION)
  sel = state.ab_eff_selected.astype(jnp.int32)
  sel_p = jnp.maximum(state.ab_eff_target_players.astype(jnp.int32), 0)
  sel_i = jnp.maximum(state.ab_eff_targets.astype(jnp.int32), 0)
  costs = jax.vmap(lambda p, i: _ikz_cost(state, p, i))(sel_p, sel_i)
  total = jnp.sum(jnp.where(k < sel, costs.astype(jnp.int32), 0))
  return ok & (total + _ikz_cost(state, tp, ti).astype(jnp.int32) <= 5)


def _azk01_087_effects(state: State) -> State:
  sel = state.ab_eff_selected.astype(jnp.int32)
  for k in range(2):
    tp, ti, has = _eff_target(state, k)
    state = bottom_deck_from_play(state, tp, ti, do=has & (k < sel))
  return state


register(
    "AZK01-087",
    effects=_azk01_087_effects,
    validate=_azk01_087_validate,
    target_validator=_azk01_087_target,
)


# AZK01-028: On Play; discard your hand: return all OTHER garden entities
# (both players, insertion order p0 then p1) to their owners' hands.
def _azk01_028_costs(state: State) -> State:
  from azuki_jax.engine.helpers import batch_discard

  owner, _ = _ctx(state)
  in_hand = state.zone[owner] == Zone.HAND
  count = jnp.sum(in_hand, dtype=jnp.uint8)
  # discard pile receives the hand in hand order (C iterates ordered children)
  state = batch_discard(
      state, owner, in_hand, state.zpos[owner].astype(jnp.int32), do=True
  )
  # discard_card_internal increments discarded_cards_this_turn per hand card
  return state._replace(
      discarded_cards_turn=state.discarded_cards_turn.at[owner].add(count)
  )


def _azk01_028_effects(state: State) -> State:
  owner, src = _ctx(state)
  for p in (0, 1):
    order, in_garden = garden_seq_order(state, p)
    is_entity = jax.vmap(
        lambda i: _card_type(state, p, i) == CardType.ENTITY
    )(jnp.arange(state.zone.shape[1]))
    marked = in_garden & is_entity  # snapshot
    for k in range(5):
      inst = order[k]
      not_self = ~((p == owner) & (inst == src))
      state = return_to_hand(state, p, inst, do=marked[inst] & not_self)
  return state


register("AZK01-028", effects=_azk01_028_effects, costs=_azk01_028_costs)


# STT01-013 (weapon): On Play (optional); deal 1 damage to your leader: this
# weapon and its host each get +1 attack (direct stat bump, no buff record).
def _stt01_013_validate(state: State, owner, src) -> jax.Array:
  leader = _leader_of(state, owner)
  return state.cur_hp[owner, jnp.maximum(leader, 0)] >= 1


def _stt01_013_costs(state: State) -> State:
  owner, _ = _ctx(state)
  leader = _leader_of(state, owner)
  return deal_effect_damage(
      state, owner, jnp.maximum(leader, 0), 1, do=leader >= 0
  )


def _stt01_013_effects(state: State) -> State:
  owner, src = _ctx(state)
  host = jnp.maximum(state.attached_to[owner, src].astype(jnp.int32), 0)
  has_host = state.attached_to[owner, src] >= 0
  new_weapon_atk = (state.cur_atk[owner, src].astype(jnp.int16) + 1).astype(jnp.int8)
  state = state._replace(
      cur_atk=state.cur_atk.at[owner, src].set(new_weapon_atk)
  )
  new_host_atk = (state.cur_atk[owner, host].astype(jnp.int16) + 1).astype(jnp.int8)
  return state._replace(
      cur_atk=state.cur_atk.at[owner, host].set(
          jnp.where(has_host, new_host_atk, state.cur_atk[owner, host])
      )
  )


register(
    "STT01-013",
    effects=_stt01_013_effects,
    costs=_stt01_013_costs,
    validate=_stt01_013_validate,
)


# STT01-016 (weapon): When Attacking; if equipped to a (Raizan) card, deal 1
# damage to all entities in the opponent's garden.
def _stt01_016_validate(state: State, owner, src) -> jax.Array:
  from azuki_jax.engine.helpers import is_effect_immune

  host = state.attached_to[owner, jnp.maximum(src, 0)].astype(jnp.int32)
  has_host = host >= 0
  raizan = _has_subtype(state, owner, jnp.maximum(host, 0), "Raizan")
  n = state.zone.shape[1]
  idx = jnp.arange(n)
  opp = (owner + 1) % 2
  non_immune = ~jax.vmap(lambda i: is_effect_immune(state, opp, i))(idx)
  any_target = jnp.any((state.zone[opp] == Zone.GARDEN) & non_immune)
  return has_host & raizan & any_target


def _stt01_016_effects(state: State) -> State:
  owner, _ = _ctx(state)
  opp = (owner + 1) % 2
  order, in_garden = garden_seq_order(state, opp)
  marked = in_garden  # snapshot
  for k in range(5):
    inst = order[k]
    state = deal_effect_damage(state, opp, inst, 1, do=marked[inst])
  return state


register("STT01-016", effects=_stt01_016_effects, validate=_stt01_016_validate)


# STT04-014: On Play; if your leader is a (Scorchweaver): deal 1 damage to ALL
# garden entities (both players), then your OTHER garden entities get +1
# attack until end of turn.
def _stt04_014_validate(state: State, owner, src) -> jax.Array:
  leader = _leader_of(state, owner)
  return (leader >= 0) & _has_subtype(
      state, owner, jnp.maximum(leader, 0), "Scorchweaver"
  )


def _stt04_014_effects(state: State) -> State:
  owner, src = _ctx(state)
  # damage pass: snapshot both gardens (p0 then p1, insertion order)
  for p in (0, 1):
    order, in_garden = garden_seq_order(state, p)
    marked = in_garden
    for k in range(5):
      inst = order[k]
      state = deal_effect_damage(state, p, inst, 1, do=marked[inst])
  # buff pass: fresh snapshot of own garden survivors
  order, in_garden = garden_seq_order(state, owner)
  is_entity = jax.vmap(
      lambda i: _card_type(state, owner, i) == CardType.ENTITY
  )(jnp.arange(state.zone.shape[1]))
  marked = in_garden & is_entity
  for k in range(5):
    inst = order[k]
    state = apply_attack_modifier(
        state, owner, inst, 1, expires_eot=True,
        do=marked[inst] & (inst != src),
    )
  return state


register("STT04-014", effects=_stt04_014_effects, validate=_stt04_014_validate)
