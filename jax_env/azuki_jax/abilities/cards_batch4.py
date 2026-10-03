"""Card ability ports, batch 4 — FINAL tranche (registered on import).

Scope (22): the takes-damage cards AZK01-059/061/062 + STT04-007/009, every
leader (STT01/02/03/04-001, AZK01-119/121/123/125), every gate
(STT01/02/03/04-002, AZK01-120/122/124/126) and STT02-017. Each card mirrors
src/abilities/cards/<code>.c exactly.

Engine support added with this batch:
- DamageTracker fields last_dmg_from_effect / dmg_src_keys / dmg_src_count
  (state.py) written by triggers.record_damage_event, reset per turn in
  phases.start_of_turn (C resets lazily on the turn_number change).
- STT03-001 Bobu latch (state.bobu_expires_turn) consumed by the hardcoded
  destroy observer in engine/helpers.{discard,batch_discard}
  (card_utils.c maybe_trigger_bobu_state).
- AZK01-120 ReequipOrigin (state.reequip_prev_host) + the reequip branch in
  selection.{can_select_to_equip,process_selection_to_equip}.

Gate-portal scratch protocol (mirrors C AbilityScratchState):
  ab_scratch[2] == 1 (GATE_PORTAL): [0] = portaled instance, [1] = slot
  ab_scratch[2] == 2 (DISCARD_SELECTION, STT01-002): [0] = max weapon cost
  ab_scratch[2] == 3 (SACRIFICE_VALUE, AZK01-124): [0] = stored damage
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from azuki_jax import cards
from azuki_jax.abilities import selection
from azuki_jax.abilities.cards_impl import (
    _cost_target,
    _ctx,
    _eff_target,
    apply_attack_modifier,
    apply_charge_grant,
    apply_health_modifier,
    apply_timed_tag_grant,
    deal_effect_damage,
    garden_seq_order,
    register,
    return_to_hand,
    sacrifice_card,
)
from azuki_jax.constants import CardType, Zone
from azuki_jax.state import State


def _np(table):
  return jnp.asarray(table)


def _zone_count(state: State, p, zone):
  return jnp.sum(state.zone[p] == zone, dtype=jnp.int32)


def _leader_of(state: State, p):
  from azuki_jax.engine.helpers import leader_instance

  return leader_instance(state, p)


def _card_type_row(state: State, p):
  d = state.def_id[p]
  return jnp.where(d >= 0, _np(cards.TYPE)[jnp.maximum(d, 0)], -1)


def _cost_row(state: State, p):
  """(has_cost, cost) per instance (C IKZCost component presence + value)."""
  d = state.def_id[p]
  valid = d >= 0
  sd = jnp.maximum(d, 0)
  return valid & _np(cards.HAS_IKZ_COST)[sd], _np(cards.IKZ_COST)[sd]


def _entity_row(state: State, p):
  return _card_type_row(state, p) == CardType.ENTITY


def _card_type(state: State, p, inst):
  return _card_type_row(state, p)[jnp.maximum(inst, 0)]


def _weapon_count_row(state: State, p):
  """(N,) number of weapons attached to each instance (has_equipped_weapon)."""
  n = state.zone.shape[1]
  att = (state.zone[p] == Zone.ATTACHED) & (state.attached_to[p] >= 0)
  host = jnp.clip(state.attached_to[p].astype(jnp.int32), 0, n - 1)
  return jnp.zeros((n,), jnp.int32).at[host].add(att.astype(jnp.int32))


# ---------------------------------------------------------------------------
# Gate-portal scratch accessors (gate_power_from_ctx)
# ---------------------------------------------------------------------------

def _portal_active(state: State) -> jax.Array:
  return state.ab_scratch[2] == 1


def _portaled_inst(state: State) -> jax.Array:
  return jnp.clip(
      state.ab_scratch[0].astype(jnp.int32), 0, state.zone.shape[1] - 1
  )


def _gate_power(state: State, owner) -> jax.Array:
  """GatePoints of the portaled card; 0 when no GATE_PORTAL scratch."""
  inst = _portaled_inst(state)
  d = state.def_id[owner, inst]
  pts = jnp.where(d >= 0, _np(cards.GATE_POINTS)[jnp.maximum(d, 0)], 0)
  return jnp.where(_portal_active(state), pts, 0).astype(jnp.int32)


# ===========================================================================
# Takes-damage cards
# ===========================================================================

# AZK01-059 "Spice": [Once/Turn] whenever this card takes damage, another
# entity in your garden gets +1 attack until end of turn.
def _azk01_059_validate(state: State, owner, src) -> jax.Array:
  n = state.zone.shape[1]
  others = (
      (state.zone[owner] == Zone.GARDEN)
      & (jnp.arange(n) != src)
      & _entity_row(state, owner)
  )
  return jnp.any(others)


def _azk01_059_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  _, src = _ctx(state)
  return (
      (ti != src)
      & (_card_type(state, tp, ti) == CardType.ENTITY)
      & (state.zone[tp, ti] == Zone.GARDEN)
  )


def _azk01_059_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return apply_attack_modifier(
      state, tp, ti, 1, expires_eot=True, do=has & picked
  )


register(
    "AZK01-059",
    effects=_azk01_059_effects,
    validate=_azk01_059_validate,
    target_validator=_azk01_059_target,
)


# AZK01-061 "Firebrand Renji": [Once/Turn] when this card has taken damage
# from 3 different sources this turn, deal up to 3 damage to a leader or
# garden entity.
def _azk01_061_validate(state: State, owner, src) -> jax.Array:
  return state.dmg_src_count[owner, jnp.maximum(src, 0)].astype(jnp.int32) >= 3


def _azk01_061_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return deal_effect_damage(state, tp, ti, 3, do=has & picked)


register("AZK01-061", effects=_azk01_061_effects, validate=_azk01_061_validate)


# AZK01-062 "Pekiro": [Once/Turn] whenever this card would take effect
# damage, you may redirect that damage to another entity in any garden.
# deal_effect_damage defers the damage into the redirect queue and queues this
# ability; apply consumes the entry and re-deals from the ORIGINAL source.
def _azk01_062_validate(state: State, owner, src) -> jax.Array:
  k8 = jnp.arange(8)
  return jnp.any(
      (k8 < state.redirect_count)
      & (state.redirect_tgt_player.astype(jnp.int32) == owner)
      & (state.redirect_tgt_inst.astype(jnp.int32) == jnp.maximum(src, 0))
  )


def _azk01_062_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  _, src = _ctx(state)
  # "another entity in any garden": exclude source only when same player AND
  # same instance (ti/src are per-player indices; an enemy can share src).
  return (
      ((tp != owner) | (ti != src))
      & (_card_type(state, tp, ti) == CardType.ENTITY)
      & (state.zone[tp, ti] == Zone.GARDEN)
  )


def _azk01_062_consume(state: State):
  """azk_consume_pending_damage_redirect: pop the FIRST entry whose original
  target is the source card, shifting later entries forward."""
  owner, src = _ctx(state)
  k8 = jnp.arange(8)
  match = (
      (k8 < state.redirect_count)
      & (state.redirect_tgt_player.astype(jnp.int32) == owner)
      & (state.redirect_tgt_inst.astype(jnp.int32) == src)
  )
  has = match.any()
  idx = jnp.argmax(match)
  sp = state.redirect_src_player[idx].astype(jnp.int32)
  si = state.redirect_src_inst[idx].astype(jnp.int32)
  dmg = state.redirect_damage[idx].astype(jnp.int16)

  def shift(arr, fill):
    rolled = jnp.roll(arr, -1)
    out = jnp.where((k8 >= idx) & has, rolled, arr)
    return out.at[7].set(jnp.where(has, jnp.asarray(fill, arr.dtype), out[7]))

  state = state._replace(
      redirect_src_player=shift(state.redirect_src_player, -1),
      redirect_src_inst=shift(state.redirect_src_inst, -1),
      redirect_tgt_player=shift(state.redirect_tgt_player, -1),
      redirect_tgt_inst=shift(state.redirect_tgt_inst, -1),
      redirect_damage=shift(state.redirect_damage, 0),
      redirect_count=jnp.where(
          has, state.redirect_count - 1, state.redirect_count
      ).astype(jnp.int8),
  )
  return state, has, sp, si, dmg


def _azk01_062_effects(state: State) -> State:
  owner, src = _ctx(state)
  state, has, sp, si, dmg = _azk01_062_consume(state)
  tp, ti, sel_has = _eff_target(state, 0)
  picked = sel_has & (state.ab_eff_selected > 0)
  rp = jnp.where(picked, tp, owner)
  ri = jnp.where(picked, ti, src)
  # resolved == original target -> deal WITHOUT redirect (the damage lands on
  # Pekiro itself); a different target re-deals WITH redirect (may chain to
  # another Pekiro). Source = the redirect entry's original source.
  same = (rp == owner) & (ri == src)
  return deal_effect_damage(
      state, rp, ri, dmg, do=has, allow_redirect=~same,
      src_player=sp, src_inst=si,
  )


register(
    "AZK01-062",
    effects=_azk01_062_effects,
    validate=_azk01_062_validate,
    target_validator=_azk01_062_target,
)


# STT04-007 "Enraged Howler": [Once/Turn] whenever this entity takes damage
# (garden or alley), it gets +1 attack until end of turn.
def _stt04_007_validate(state: State, owner, src) -> jax.Array:
  safe = jnp.maximum(src, 0)
  z = state.zone[owner, safe]
  return state.took_damage_turn[owner, safe] & (
      (z == Zone.GARDEN) | (z == Zone.ALLEY)
  )


def _stt04_007_effects(state: State) -> State:
  owner, src = _ctx(state)
  return apply_attack_modifier(state, owner, src, 1, expires_eot=True, do=True)


register("STT04-007", effects=_stt04_007_effects, validate=_stt04_007_validate)


# STT04-009 "Cinderwake Ritualist": [Garden][Once/Turn] whenever this card
# takes EFFECT damage, you may deal that much damage (capped at 2) to another
# leader or garden entity.
def _stt04_009_validate(state: State, owner, src) -> jax.Array:
  safe = jnp.maximum(src, 0)
  return (
      (state.zone[owner, safe] == Zone.GARDEN)
      & state.took_damage_turn[owner, safe]
      & state.last_dmg_from_effect[owner, safe]
  )


def _stt04_009_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  _, src = _ctx(state)
  # "another" entity: source is owner's instance src; an enemy can share that
  # per-player index, so exclude only when same player AND same instance.
  return (tp != owner) | (ti != src)


def _stt04_009_effects(state: State) -> State:
  owner, src = _ctx(state)
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  dmg = jnp.minimum(state.last_dmg_taken[owner, src].astype(jnp.int16), 2)
  return deal_effect_damage(state, tp, ti, dmg, do=has & picked & (dmg > 0))


register(
    "STT04-009",
    effects=_stt04_009_effects,
    validate=_stt04_009_validate,
    target_validator=_stt04_009_target,
)


# ===========================================================================
# Leaders
# ===========================================================================

# STT01-001: [Main][Once/Turn] Pay 1 IKZ: a friendly garden entity equipped
# with a weapon and on cooldown gains Charge until end of turn.
def _stt01_001_validate(state: State, owner, src) -> jax.Array:
  ok = (
      (state.zone[owner] == Zone.GARDEN)
      & (_weapon_count_row(state, owner) > 0)
      & (state.cooldown[owner] != 0)
  )
  return jnp.any(ok)


def _stt01_001_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return (_weapon_count_row(state, tp)[jnp.maximum(ti, 0)] > 0) & (
      state.cooldown[tp, ti] != 0
  )


def _stt01_001_effects(state: State) -> State:
  from azuki_jax.engine.helpers import GRANT_PHASE_END

  tp, ti, has = _eff_target(state, 0)
  return apply_charge_grant(state, tp, ti, GRANT_PHASE_END, 1, has)


register(
    "STT01-001",
    effects=_stt01_001_effects,
    validate=_stt01_001_validate,
    target_validator=_stt01_001_target,
)


# STT02-001 "Shao": [Response][Once/Turn] Pay 1 IKZ: an enemy leader or
# garden entity gets -1 attack until end of turn.
def _stt02_001_validate(state: State, owner, src) -> jax.Array:
  opp = (owner + 1) % 2
  return (_zone_count(state, opp, Zone.GARDEN) > 0) | (
      _leader_of(state, opp) >= 0
  )


def _stt02_001_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  return apply_attack_modifier(state, tp, ti, -1, expires_eot=True, do=has)


register(
    "STT02-001", effects=_stt02_001_effects, validate=_stt02_001_validate
)


# STT03-001 "Bobu": [Main][Once/Turn] Pay 1 IKZ: until the start of your next
# turn, the first time an Earth entity of yours is destroyed/sacrificed from
# garden or alley, heal 1 to your leader (latch consumed by
# engine/helpers.bobu_destroy_heal).
def _stt03_001_effects(state: State) -> State:
  owner, _ = _ctx(state)
  return state._replace(
      bobu_expires_turn=state.bobu_expires_turn.at[owner].set(
          (state.turn_number + 2).astype(jnp.int16)
      )
  )


register("STT03-001", effects=_stt03_001_effects)


# STT04-001 "Zero": [Main][Once/Turn] deal 1 damage to this leader: deal 1
# damage to a friendly garden/alley entity, then (if it is still a friendly
# garden/alley entity) it gets +1 attack until end of turn.
def _stt04_001_validate(state: State, owner, src) -> jax.Array:
  return (_zone_count(state, owner, Zone.GARDEN) > 0) | (
      _zone_count(state, owner, Zone.ALLEY) > 0
  )


def _stt04_001_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return _card_type(state, tp, ti) == CardType.ENTITY


def _stt04_001_costs(state: State) -> State:
  owner, src = _ctx(state)
  return deal_effect_damage(state, owner, src, 1, do=True)


def _stt04_001_effects(state: State) -> State:
  owner, _ = _ctx(state)
  tp, ti, has = _eff_target(state, 0)
  state = deal_effect_damage(state, tp, ti, 1, do=has)
  z = state.zone[tp, ti]
  still_friendly = (
      (tp == owner)
      & ((z == Zone.GARDEN) | (z == Zone.ALLEY))
      & (_card_type(state, tp, ti) == CardType.ENTITY)
  )
  return apply_attack_modifier(
      state, tp, ti, 1, expires_eot=True, do=has & still_friendly
  )


register(
    "STT04-001",
    effects=_stt04_001_effects,
    costs=_stt04_001_costs,
    validate=_stt04_001_validate,
    target_validator=_stt04_001_target,
)


# AZK01-119 "Piko of Thousand Blades": [Main][Once/Turn] Pay 3 IKZ: a friendly
# equipped garden entity gets +1 attack per weapon in your discard (cap +3)
# until end of turn.
def _azk01_119_validate(state: State, owner, src) -> jax.Array:
  ok = (state.zone[owner] == Zone.GARDEN) & (
      _weapon_count_row(state, owner) > 0
  )
  return jnp.any(ok)


def _azk01_119_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return (
      (tp == owner)
      & (state.zone[tp, ti] == Zone.GARDEN)
      & (_card_type(state, tp, ti) == CardType.ENTITY)
      & (_weapon_count_row(state, tp)[jnp.maximum(ti, 0)] > 0)
  )


def _azk01_119_effects(state: State) -> State:
  owner, _ = _ctx(state)
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  weapons = jnp.sum(
      (state.zone[owner] == Zone.DISCARD)
      & (_card_type_row(state, owner) == CardType.WEAPON),
      dtype=jnp.int32,
  )
  buff = jnp.minimum(weapons, 3)
  return apply_attack_modifier(
      state, tp, ti, buff, expires_eot=True, do=has & picked & (buff > 0)
  )


register(
    "AZK01-119",
    effects=_azk01_119_effects,
    validate=_azk01_119_validate,
    target_validator=_azk01_119_target,
)


# AZK01-121 "Kagoro of the Burnt Path": [Main][Once/Turn] Pay 1 IKZ: this
# leader gets +1 attack per entity you played this turn (cap +2) until EOT.
def _azk01_121_validate(state: State, owner, src) -> jax.Array:
  total = (
      state.entities_played_garden_turn[owner].astype(jnp.int32)
      + state.entities_played_alley_turn[owner].astype(jnp.int32)
  )
  return total > 0


def _azk01_121_effects(state: State) -> State:
  owner, src = _ctx(state)
  total = (
      state.entities_played_garden_turn[owner].astype(jnp.int32)
      + state.entities_played_alley_turn[owner].astype(jnp.int32)
  )
  buff = jnp.minimum(total, 2)
  return apply_attack_modifier(
      state, owner, src, buff, expires_eot=True, do=buff > 0
  )


register("AZK01-121", effects=_azk01_121_effects, validate=_azk01_121_validate)


# AZK01-123 "Goro Graveloth": [Main][Once/Turn] Pay 1 IKZ: an entity in your
# garden gets +1 health until end of turn.
def _azk01_123_validate(state: State, owner, src) -> jax.Array:
  return _zone_count(state, owner, Zone.GARDEN) > 0


def _azk01_123_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return _card_type(state, tp, ti) == CardType.ENTITY


def _azk01_123_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return apply_health_modifier(
      state, tp, ti, 1, expires_eot=True, do=has & picked
  )


register(
    "AZK01-123",
    effects=_azk01_123_effects,
    validate=_azk01_123_validate,
    target_validator=_azk01_123_target,
)


# AZK01-125 "Benzai the Sly": [Main][Response][Once/Turn] Pay 1 IKZ: if you
# discarded a card this turn, your next card this turn costs 2 less.
def _azk01_125_validate(state: State, owner, src) -> jax.Array:
  return state.discarded_cards_turn[owner].astype(jnp.int32) > 0


def _azk01_125_effects(state: State) -> State:
  owner, _ = _ctx(state)
  return state._replace(
      next_play_cost_reduction=state.next_play_cost_reduction.at[owner].add(
          jnp.int8(2)
      )
  )


register("AZK01-125", effects=_azk01_125_effects, validate=_azk01_125_validate)


# ===========================================================================
# Gates (all AOnGatePortal; begun inline by apply.apply_gate_portal)
# ===========================================================================

# STT01-002 "Surge": you may equip a weapon with cost <= the portaled card's
# gate points from your discard pile.
def _stt01_002_ocp(state: State) -> State:
  owner, _ = _ctx(state)
  power = _gate_power(state, owner)
  has_cost, cost = _cost_row(state, owner)
  row = (
      (_card_type_row(state, owner) == CardType.WEAPON)
      & has_cost
      & (cost <= power)
      & (power > 0)  # C returns early (phase NONE) when gate points == 0
  )
  state = selection.move_matching_zone_to_selection(
      state, Zone.DISCARD, row, 1, do=True
  )
  # C overwrites the scratch with DISCARD_SELECTION{max_cost} for the
  # selection validator
  moved = state.ab_sel_count > 0
  return state._replace(
      ab_scratch=state.ab_scratch.at[0]
      .set(jnp.where(moved, power.astype(jnp.int16), state.ab_scratch[0]))
      .at[2]
      .set(jnp.where(moved, jnp.int16(2), state.ab_scratch[2]))
  )


def _stt01_002_sel(state: State, owner, inst) -> jax.Array:
  i = jnp.maximum(inst, 0)
  kind_ok = state.ab_scratch[2] == 2
  max_cost = state.ab_scratch[0].astype(jnp.int32)
  has_cost, cost = _cost_row(state, owner)
  return (
      kind_ok
      & (_card_type_row(state, owner)[i] == CardType.WEAPON)
      & has_cost[i]
      & (cost[i].astype(jnp.int32) <= max_cost)
  )


register(
    "STT01-002",
    on_cost_paid=_stt01_002_ocp,
    selection_target_validator=_stt01_002_sel,
    on_selection_complete=lambda s: selection.return_remaining_to_discard(
        s, True
    ),
)


# STT02-002 "Hydromancy": untap up to gate-points tapped IKZ-area cards (zone
# order; cooldown untouched — zone_util.c untap_n_ikz_cards).
def _stt02_002_effects(state: State) -> State:
  owner, _ = _ctx(state)
  power = _gate_power(state, owner)
  tapped = (state.zone[owner] == Zone.IKZ_AREA) & state.tapped[owner]
  zpos = state.zpos[owner].astype(jnp.int32)
  rank = jnp.sum((zpos[None, :] < zpos[:, None]) & tapped[None, :], axis=1)
  untap = tapped & (rank < power)
  return state._replace(
      tapped=state.tapped.at[owner].set(
          jnp.where(untap, False, state.tapped[owner])
      )
  )


register("STT02-002", effects=_stt02_002_effects)


# STT03-002 "Stonehaven Gate": you may give a friendly garden entity with
# base health <= gate points (and without Defender) Defender until the start
# of your next turn (timed grant, START phase, 2 ticks).
def _stt03_002_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  from azuki_jax.engine.helpers import has_defender_kw

  power = _gate_power(state, owner)
  d = state.def_id[tp, jnp.maximum(ti, 0)]
  base = jnp.where(d >= 0, _np(cards.BASE_HP)[jnp.maximum(d, 0)], 127)
  has_stats = jnp.where(
      d >= 0, _np(cards.HAS_BASE_STATS)[jnp.maximum(d, 0)], False
  )
  return (
      (_card_type(state, tp, ti) == CardType.ENTITY)
      & ~has_defender_kw(state, tp, ti)
      & has_stats
      & (base.astype(jnp.int32) <= power)
  )


def _stt03_002_effects(state: State) -> State:
  from azuki_jax.engine.helpers import GRANT_PHASE_START, TAG_DEFENDER

  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  state, _ = apply_timed_tag_grant(
      state, tp, ti, TAG_DEFENDER, GRANT_PHASE_START, 2, has & picked
  )
  return state


register(
    "STT03-002",
    effects=_stt03_002_effects,
    target_validator=_stt03_002_target,
)


# STT04-002 "Ragefire Gate": you may give a friendly garden entity that took
# damage this turn +attack equal to gate points until end of turn.
def _stt04_002_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  return (_card_type(state, tp, ti) == CardType.ENTITY) & state.took_damage_turn[
      tp, ti
  ]


def _stt04_002_effects(state: State) -> State:
  owner, _ = _ctx(state)
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  power = _gate_power(state, owner)
  return apply_attack_modifier(
      state, tp, ti, power, expires_eot=True, do=has & picked & (power > 0)
  )


register(
    "STT04-002",
    effects=_stt04_002_effects,
    target_validator=_stt04_002_target,
)


# AZK01-120 "Stormchain Gate": you may re-equip a weapon with cost <= gate
# points to a DIFFERENT friendly leader or garden entity.
def _azk01_120_candidates(state: State, owner):
  """(mask, order_key) over owner's instances: valid reequip weapons (leader
  weapons first by attach order, then garden hosts by board_seq)."""
  n = state.zone.shape[1]
  power = _gate_power(state, owner)
  attached = (state.zone[owner] == Zone.ATTACHED) & (state.attached_to[owner] >= 0)
  host = jnp.clip(state.attached_to[owner].astype(jnp.int32), 0, n - 1)
  host_zone = state.zone[owner][host]
  host_is_leader = host_zone == Zone.LEADER
  host_ok = host_is_leader | (host_zone == Zone.GARDEN)
  has_cost, cost = _cost_row(state, owner)
  is_weapon = _card_type_row(state, owner) == CardType.WEAPON
  # another host option: any garden card or the leader, != current host. The
  # leader always exists, so only leader-hosted weapons need a garden card.
  garden_count = _zone_count(state, owner, Zone.GARDEN)
  other_host = jnp.where(host_is_leader, garden_count >= 1, True)
  mask = (
      attached & host_ok & is_weapon & has_cost
      & (cost.astype(jnp.int32) <= power) & (power > 0) & other_host
  )
  host_seq = state.board_seq[owner][host].astype(jnp.int32)
  key = (
      jnp.where(host_is_leader, 0, (1 << 16) + host_seq * 16)
      + state.zpos[owner].astype(jnp.int32)
  )
  return mask, key


def _azk01_120_validate(state: State, owner, src) -> jax.Array:
  mask, _ = _azk01_120_candidates(state, owner)
  return jnp.any(mask)


def _azk01_120_ocp(state: State) -> State:
  """detach_weapon_to_selection for every candidate + init selection state."""
  from azuki_jax.constants import MAX_ABILITY_SELECTION, AbilityPhase
  from azuki_jax.engine.helpers import _rank_by_key
  from azuki_jax.zones import zone_count

  owner, _ = _ctx(state)
  n = state.zone.shape[1]
  idx = jnp.arange(n)
  mask, key = _azk01_120_candidates(state, owner)
  host = jnp.clip(state.attached_to[owner].astype(jnp.int32), 0, n - 1)
  rank = _rank_by_key(mask, key)
  take = mask & (rank < selection.MAX_MOVED)
  count = jnp.sum(take, dtype=jnp.int32)

  # host stat fix: atk -= sum of detached weapon atks (clamped at 0 — equals
  # C's sequential per-weapon clamp for non-negative weapon attacks)
  contrib = jnp.where(take, state.cur_atk[owner].astype(jnp.int16), 0)
  sums = jnp.zeros((n,), jnp.int16).at[host].add(contrib)
  host_has = jnp.zeros((n,), jnp.int32).at[host].add(take.astype(jnp.int32)) > 0
  new_atk = jnp.maximum(
      state.cur_atk[owner].astype(jnp.int16) - sums, 0
  ).astype(jnp.int8)
  state = state._replace(
      cur_atk=state.cur_atk.at[owner].set(
          jnp.where(host_has, new_atk, state.cur_atk[owner])
      )
  )
  # remove_weapon_combat_modifier_if_any (AZK01-018 on a leader host)
  is_018 = state.def_id[owner] == cards.CODE_TO_ID["AZK01-018"]
  host_is_leader = state.zone[owner][host] == Zone.LEADER
  undo = jnp.zeros((n,), jnp.int8).at[host].add(
      jnp.where(take & is_018 & host_is_leader, 1, 0).astype(jnp.int8)
  )
  state = state._replace(cmb_in_perm=state.cmb_in_perm.at[owner].add(undo))
  sel_base = zone_count(state.zone[owner], Zone.SELECTION)
  new_zone = jnp.where(take, jnp.int8(Zone.SELECTION), state.zone[owner])
  new_zpos = jnp.where(
      take, (sel_base + rank).astype(jnp.int8), state.zpos[owner]
  )
  sel_cards = jnp.full(state.ab_sel_cards.shape, -1, jnp.int8)
  sel_cards = sel_cards.at[
      jnp.where(take, rank, state.ab_sel_cards.shape[0])
  ].set(idx.astype(jnp.int8), mode="drop")

  phase = jnp.where(
      count > 0,
      jnp.int8(AbilityPhase.SELECTION_PICK),
      jnp.int8(AbilityPhase.NONE),
  )
  return state._replace(
      zone=state.zone.at[owner].set(new_zone),
      zpos=state.zpos.at[owner].set(new_zpos),
      attached_to=state.attached_to.at[owner].set(
          jnp.where(take, jnp.int8(-1), state.attached_to[owner])
      ),
      reequip_prev_host=state.reequip_prev_host.at[owner].set(
          jnp.where(take, host.astype(jnp.int8), state.reequip_prev_host[owner])
      ),
      ab_sel_cards=sel_cards,
      ab_sel_count=count.astype(jnp.int8),
      ab_sel_picked=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_sel_picked_count=jnp.int8(0),
      ab_sel_pick_max=jnp.int8(1),
      ab_phase=phase,
  )


def _azk01_120_sel(state: State, owner, inst) -> jax.Array:
  """is_valid_reequip_weapon for a selection-zone weapon: weapon + cost <=
  gate points (its 'current host' is the selection zone, so the other-host
  check passes whenever any host exists — the leader always does)."""
  i = jnp.maximum(inst, 0)
  power = _gate_power(state, owner)
  has_cost, cost = _cost_row(state, owner)
  return (
      (_card_type_row(state, owner)[i] == CardType.WEAPON)
      & has_cost[i]
      & (cost[i].astype(jnp.int32) <= power)
      & (power > 0)
  )


def _return_weapon_to_host(state: State, owner, weapon, do) -> State:
  """return_selection_weapon_to_host: reattach to ReequipOrigin.previous_host
  with the attack bonus + combat modifier; no triggers. (The C invalid-host
  branch — discard the weapon — is unreachable: card entities stay alive.)"""
  from azuki_jax.engine.helpers import weapons_of

  prev = state.reequip_prev_host[owner, weapon].astype(jnp.int32)
  do = jnp.asarray(do) & (prev >= 0)
  safe_prev = jnp.maximum(prev, 0)
  state = selection._selection_remove(state, owner, weapon, do)
  wcount = jnp.sum(weapons_of(state, owner, safe_prev), dtype=jnp.int32)
  state = state._replace(
      zone=state.zone.at[owner, weapon].set(
          jnp.where(do, jnp.int8(Zone.ATTACHED), state.zone[owner, weapon])
      ),
      zpos=state.zpos.at[owner, weapon].set(
          jnp.where(do, wcount.astype(jnp.int8), state.zpos[owner, weapon])
      ),
      attached_to=state.attached_to.at[owner, weapon].set(
          jnp.where(do, prev.astype(jnp.int8), state.attached_to[owner, weapon])
      ),
  )
  watk = state.cur_atk[owner, weapon].astype(jnp.int16)
  new_atk = jnp.maximum(
      state.cur_atk[owner, safe_prev].astype(jnp.int16) + watk, 0
  ).astype(jnp.int8)
  state = state._replace(
      cur_atk=state.cur_atk.at[owner, safe_prev].set(
          jnp.where(do, new_atk, state.cur_atk[owner, safe_prev])
      )
  )
  is_018 = state.def_id[owner, weapon] == cards.CODE_TO_ID["AZK01-018"]
  leader_host = state.zone[owner, safe_prev] == Zone.LEADER
  state = state._replace(
      cmb_in_perm=state.cmb_in_perm.at[owner, safe_prev].add(
          jnp.where(do & is_018 & leader_host, -1, 0).astype(jnp.int8)
      ),
      reequip_prev_host=state.reequip_prev_host.at[owner, weapon].set(
          jnp.where(do, jnp.int8(-1), state.reequip_prev_host[owner, weapon])
      ),
  )
  return state


def _azk01_120_complete(state: State) -> State:
  owner, _ = _ctx(state)
  for k in range(selection.MAX_MOVED):
    inst = state.ab_sel_cards[k]
    safe = jnp.maximum(inst.astype(jnp.int32), 0)
    has = (inst >= 0) & (state.zone[owner, safe] == Zone.SELECTION)
    state = _return_weapon_to_host(state, owner, safe, has)
  return state


register(
    "AZK01-120",
    validate=_azk01_120_validate,
    on_cost_paid=_azk01_120_ocp,
    selection_target_validator=_azk01_120_sel,
    on_selection_complete=_azk01_120_complete,
)


# AZK01-122 "Rushfire Gate": you may play an entity with cost <= gate points
# from hand into the garden or alley; it gains Charge while in play.
def _azk01_122_row(state: State, owner, power):
  has_cost, cost = _cost_row(state, owner)
  return (
      (_card_type_row(state, owner) == CardType.ENTITY)
      & has_cost
      & (cost.astype(jnp.int32) <= power)
  )


def _azk01_122_validate(state: State, owner, src) -> jax.Array:
  power = _gate_power(state, owner)
  row = _azk01_122_row(state, owner, power)
  return (power > 0) & jnp.any((state.zone[owner] == Zone.HAND) & row)


def _azk01_122_ocp(state: State) -> State:
  owner, _ = _ctx(state)
  power = _gate_power(state, owner)
  return selection.move_matching_zone_to_selection(
      state, Zone.HAND, _azk01_122_row(state, owner, power), 1, do=True
  )


def _azk01_122_sel(state: State, owner, inst) -> jax.Array:
  power = _gate_power(state, owner)
  return (power > 0) & _azk01_122_row(state, owner, power)[jnp.maximum(inst, 0)]


def _azk01_122_complete(state: State) -> State:
  from azuki_jax.engine.helpers import GRANT_PHASE_NONE, attr_force_tapped

  owner, _ = _ctx(state)
  picked = state.ab_sel_picked[0].astype(jnp.int32)
  has = (state.ab_sel_picked_count > 0) & (state.ab_sel_picked[0] >= 0)
  safe = jnp.maximum(picked, 0)
  # permanent Charge grant (TAG_GRANT_TICK_NONE, -1) + TapState fix-up
  state = apply_charge_grant(state, owner, safe, GRANT_PHASE_NONE, -1, has)
  force = attr_force_tapped(state, owner, safe)
  state = state._replace(
      tapped=state.tapped.at[owner, safe].set(
          jnp.where(has, force | state.tapped[owner, safe],
                    state.tapped[owner, safe])
      ),
      cooldown=state.cooldown.at[owner, safe].set(
          jnp.where(has, 0, state.cooldown[owner, safe]).astype(jnp.uint8)
      ),
  )
  return selection.return_remaining_to_hand(state, True)


register(
    "AZK01-122",
    validate=_azk01_122_validate,
    on_cost_paid=_azk01_122_ocp,
    selection_target_validator=_azk01_122_sel,
    on_selection_complete=_azk01_122_complete,
)


# AZK01-124 "Gate of Devotion": you may sacrifice another untapped garden
# entity with cost <= gate points; if you do, you may deal damage equal to
# its health to an enemy garden entity.
def _azk01_124_cost_row(state: State, owner):
  power = _gate_power(state, owner)
  portaled = _portaled_inst(state)
  n = state.zone.shape[1]
  has_cost, cost = _cost_row(state, owner)
  return (
      (jnp.arange(n) != portaled)
      & _entity_row(state, owner)
      & ~state.tapped[owner]
      & has_cost
      & (cost.astype(jnp.int32) <= power)
      & (power > 0)
  )


def _azk01_124_validate(state: State, owner, src) -> jax.Array:
  row = _azk01_124_cost_row(state, owner)
  return jnp.any((state.zone[owner] == Zone.GARDEN) & row)


def _azk01_124_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  cost_ok = _azk01_124_cost_row(state, owner)[jnp.maximum(ti, 0)]
  effect_ok = _card_type(state, tp, ti) == CardType.ENTITY
  return jnp.where(scope_is_cost, cost_ok, effect_ok)


def _azk01_124_costs(state: State) -> State:
  tp, ti, has = _cost_target(state, 0)
  hp = state.cur_hp[tp, ti].astype(jnp.int16)
  dmg = jnp.maximum(hp, 0)
  # scratch -> SACRIFICE_VALUE{damage} (overwrites the GATE_PORTAL scratch,
  # exactly as C does)
  state = state._replace(
      ab_scratch=state.ab_scratch.at[0]
      .set(jnp.where(has, dmg, state.ab_scratch[0]).astype(jnp.int16))
      .at[2]
      .set(jnp.where(has, jnp.int16(3), state.ab_scratch[2]))
  )
  return sacrifice_card(state, tp, ti, do=has)


def _azk01_124_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  ok = (state.ab_scratch[2] == 3) & (state.ab_scratch[0] > 0)
  return deal_effect_damage(
      state, tp, ti, state.ab_scratch[0].astype(jnp.int16),
      do=ok & has & picked,
  )


register(
    "AZK01-124",
    effects=_azk01_124_effects,
    costs=_azk01_124_costs,
    validate=_azk01_124_validate,
    target_validator=_azk01_124_target,
)


# AZK01-126 "Gate of Echoed Waves": return a spell with cost <= gate points
# from your discard pile to your hand.
def _azk01_126_row(state: State, owner, power):
  has_cost, cost = _cost_row(state, owner)
  return (
      (_card_type_row(state, owner) == CardType.SPELL)
      & has_cost
      & (cost.astype(jnp.int32) <= power)
  )


def _azk01_126_validate(state: State, owner, src) -> jax.Array:
  power = _gate_power(state, owner)
  row = _azk01_126_row(state, owner, power)
  return (power > 0) & jnp.any((state.zone[owner] == Zone.DISCARD) & row)


def _azk01_126_ocp(state: State) -> State:
  owner, _ = _ctx(state)
  power = _gate_power(state, owner)
  return selection.move_matching_zone_to_selection(
      state, Zone.DISCARD, _azk01_126_row(state, owner, power), 1, do=True
  )


def _azk01_126_sel(state: State, owner, inst) -> jax.Array:
  power = _gate_power(state, owner)
  return (power > 0) & _azk01_126_row(state, owner, power)[jnp.maximum(inst, 0)]


def _azk01_126_complete(state: State) -> State:
  state = selection.move_picked_to_hand(state, do=True)
  return selection.return_remaining_to_discard(state, True)


register(
    "AZK01-126",
    validate=_azk01_126_validate,
    on_cost_paid=_azk01_126_ocp,
    selection_target_validator=_azk01_126_sel,
    on_selection_complete=_azk01_126_complete,
    selection_complete_if_still=True,
)


# ===========================================================================
# STT02-017 "Shao's Perseverance" (spell, Main): if your leader is a Shao,
# return ALL entities with cost <= 4 in the opponent's garden to hand.
# ===========================================================================

def _stt02_017_validate(state: State, owner, src) -> jax.Array:
  leader = _leader_of(state, owner)
  d = state.def_id[owner, jnp.maximum(leader, 0)]
  col = _np(cards.SUBTYPE_MATRIX[:, cards.subtype_index("Shao")])
  return (leader >= 0) & jnp.where(d >= 0, col[jnp.maximum(d, 0)], False)


def _stt02_017_effects(state: State) -> State:
  owner, _ = _ctx(state)
  opp = (owner + 1) % 2
  order, in_garden = garden_seq_order(state, opp)
  has_cost, cost = _cost_row(state, opp)
  marked = (
      in_garden & _entity_row(state, opp) & has_cost & (cost <= 4)
  )  # snapshot before the bounces (C collects first)
  for k in range(5):
    inst = order[k]
    state = return_to_hand(state, opp, inst, do=marked[inst])
  return state


register(
    "STT02-017", effects=_stt02_017_effects, validate=_stt02_017_validate
)
