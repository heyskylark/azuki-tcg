"""Card ability ports, batch 3 (registered into cards_impl on import).

Selection-zone cards: reveal-top-N flows, hand/discard-to-selection flows,
and on_cost_paid hooks that override the ctx effect min/max. Each card
mirrors src/abilities/cards/<code>.c exactly (validate, cost/effect/selection
target validators, apply_costs, on_cost_paid, on_selection_complete,
apply_effects).

Scope (23): AZK01-003/015/016/017/021/024/031/033/041/045/056/069/084/086/
092/097/098/111, STT01-004, STT02-003/013, STT03-006, STT04-005.

Notes:
- No card in this batch sets selection_to_equip_is_reequip, so ReequipOrigin
  tracking is NOT implemented yet (azk_can_select_to_equip's reequip branch
  stays dead).
- Hand/discard-to-selection flows are bounded at selection.MAX_MOVED (8)
  matches; C caps at MAX_SELECTION_ZONE_SIZE (50).
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
    deal_effect_damage,
    draw_with_deckout,
    heal_leader,
    register,
    return_to_hand,
    sacrifice_card,
)
from azuki_jax.constants import (
    MAX_ABILITY_SELECTION,
    TOKEN_INSTANCE,
    AbilityPhase,
    CardType,
    Zone,
)
from azuki_jax.state import State


def _np(table):
  return jnp.asarray(table)


def _zone_count(state: State, p, zone):
  return jnp.sum(state.zone[p] == zone, dtype=jnp.int32)


def _def_of(state: State, p, inst):
  return state.def_id[p, jnp.maximum(inst, 0)]


def _card_type_row(state: State, p):
  d = state.def_id[p]
  return jnp.where(d >= 0, _np(cards.TYPE)[jnp.maximum(d, 0)], -1)


def _cost_row(state: State, p):
  """(has_cost, cost) per instance (C: IKZCost component presence + value)."""
  d = state.def_id[p]
  valid = d >= 0
  sd = jnp.maximum(d, 0)
  return valid & _np(cards.HAS_IKZ_COST)[sd], _np(cards.IKZ_COST)[sd]


def _subtype_row(state: State, p, name: str):
  d = state.def_id[p]
  col = _np(cards.SUBTYPE_MATRIX[:, cards.subtype_index(name)])
  return jnp.where(d >= 0, col[jnp.maximum(d, 0)], False)


def _element_row(state: State, p):
  d = state.def_id[p]
  return jnp.where(d >= 0, _np(cards.ELEMENT)[jnp.maximum(d, 0)], -1)


def _has_subtype(state: State, p, inst, name: str):
  return _subtype_row(state, p, name)[jnp.maximum(inst, 0)]


def _leader_of(state: State, p):
  from azuki_jax.engine.helpers import leader_instance

  return leader_instance(state, p)


# C CardElement values
_NORMAL, _LIGHTNING, _WATER, _EARTH, _FIRE = 0, 1, 2, 3, 4


def _reveal_hook(n: int, pick: int):
  """azk_setup_reveal_top_cards_selection(n, pick, <selection validator>):
  phase set only when at least one card was revealed (else stays NONE)."""

  def hook(state: State) -> State:
    state = selection.reveal_top_into_selection(state, n, pick, do=True)
    revealed = state.ab_sel_count > 0
    return selection.enter_selection_phase(state, do=revealed)

  return hook


def _reveal_complete(state: State) -> State:
  """move picked to hand + azk_begin_bottom_deck_for_remaining_selection."""
  state = selection.move_picked_to_hand(state, do=True)
  return selection.begin_bottom_deck_for_remaining(state, do=True)


def _sel_subtype(name: str, exclude: str | None = None):
  def validator(state: State, owner, inst) -> jax.Array:
    ok = _has_subtype(state, owner, inst, name)
    if exclude is not None:
      ok = ok & (_def_of(state, owner, inst) != cards.CODE_TO_ID[exclude])
    return ok

  return validator


# ---------------------------------------------------------------------------
# Reveal-top-N -> pick to hand -> bottom deck the rest
# ---------------------------------------------------------------------------

# AZK01-003: On Play; look at top 5, reveal up to 1 Black Jade card other
# than Black Jade Courier (itself) and add it to hand, bottom deck the rest.
register(
    "AZK01-003",
    on_cost_paid=_reveal_hook(5, 1),
    on_selection_complete=_reveal_complete,
    selection_target_validator=_sel_subtype("BlackJade", exclude="AZK01-003"),
)


# AZK01-021: On Play; top 5, reveal up to 1 Driftward card to hand.
register(
    "AZK01-021",
    on_cost_paid=_reveal_hook(5, 1),
    on_selection_complete=_reveal_complete,
    selection_target_validator=_sel_subtype("Driftward"),
)


# AZK01-031 (spell, Main): top 3, reveal up to 1 Water card to hand; the rest
# can be top-decked OR bottom-decked in any order (can_topdeck_selection).
def _azk01_031_sel(state: State, owner, inst) -> jax.Array:
  return _element_row(state, owner)[jnp.maximum(inst, 0)] == _WATER


register(
    "AZK01-031",
    on_cost_paid=_reveal_hook(3, 1),
    on_selection_complete=_reveal_complete,
    selection_target_validator=_azk01_031_sel,
)


# AZK01-033: On Play; top 5, reveal up to 1 Steelborn card to hand.
register(
    "AZK01-033",
    on_cost_paid=_reveal_hook(5, 1),
    on_selection_complete=_reveal_complete,
    selection_target_validator=_sel_subtype("Steelborn"),
)


# AZK01-045: On Play; top 5, reveal up to 1 Obsidian card to hand.
register(
    "AZK01-045",
    on_cost_paid=_reveal_hook(5, 1),
    on_selection_complete=_reveal_complete,
    selection_target_validator=_sel_subtype("Obsidian"),
)


# AZK01-056: On Play; top 5, reveal up to 1 Scorchweaver card to hand.
register(
    "AZK01-056",
    on_cost_paid=_reveal_hook(5, 1),
    on_selection_complete=_reveal_complete,
    selection_target_validator=_sel_subtype("Scorchweaver"),
)


# AZK01-069: On Play; top 5, reveal up to 1 Beanz card to hand.
register(
    "AZK01-069",
    on_cost_paid=_reveal_hook(5, 1),
    on_selection_complete=_reveal_complete,
    selection_target_validator=_sel_subtype("Beanz"),
)


# STT04-005: On Play; top 5, reveal up to 1 Pyreskin card to hand.
register(
    "STT04-005",
    on_cost_paid=_reveal_hook(5, 1),
    on_selection_complete=_reveal_complete,
    selection_target_validator=_sel_subtype("Pyreskin"),
)


# STT02-003: On Play; top 5, reveal up to 1 Watercrafting card to hand.
register(
    "STT02-003",
    on_cost_paid=_reveal_hook(5, 1),
    on_selection_complete=_reveal_complete,
    selection_target_validator=_sel_subtype("Watercrafting"),
)


def _water_le2(state: State, owner, inst) -> jax.Array:
  has_cost, cost = _cost_row(state, owner)
  i = jnp.maximum(inst, 0)
  return has_cost[i] & (cost[i] <= 2) & (
      _element_row(state, owner)[i] == _WATER
  )


# STT02-013: On Play (deck >= 3); top 3, reveal up to 1 water card of cost
# <= 2; add to hand OR play it to the alley if it is an entity.
def _stt02_013_validate(state: State, owner, src) -> jax.Array:
  return _zone_count(state, owner, Zone.DECK) >= 3


register(
    "STT02-013",
    validate=_stt02_013_validate,
    on_cost_paid=_reveal_hook(3, 1),
    on_selection_complete=_reveal_complete,
    selection_target_validator=_water_le2,
    selection_complete_if_still=True,
)


# AZK01-092 (spell, Main): top 5, reveal up to 1 water card of cost <= 2;
# add to hand, play to garden/alley (entity) or equip (weapon).
register(
    "AZK01-092",
    on_cost_paid=_reveal_hook(5, 1),
    on_selection_complete=_reveal_complete,
    selection_target_validator=_water_le2,
    selection_complete_if_still=True,
)


# STT01-004: On Play (optional); cost: discard a weapon from hand; top 5,
# reveal up to 1 weapon card to hand, bottom deck the rest.
def _stt01_004_validate(state: State, owner, src) -> jax.Array:
  in_hand = state.zone[owner] == Zone.HAND
  return jnp.any(in_hand & (_card_type_row(state, owner) == CardType.WEAPON))


def _stt01_004_cost_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  ok = (tp == owner) & (state.zone[tp, ti] == Zone.HAND) & (
      _card_type_row(state, owner)[ti] == CardType.WEAPON
  )
  return jnp.where(scope_is_cost, ok, True)


def _stt01_004_costs(state: State) -> State:
  tp, ti, has = _cost_target(state, 0)
  return sacrifice_card(state, tp, ti, do=has)


def _stt01_004_sel(state: State, owner, inst) -> jax.Array:
  return _card_type_row(state, owner)[jnp.maximum(inst, 0)] == CardType.WEAPON


register(
    "STT01-004",
    validate=_stt01_004_validate,
    target_validator=_stt01_004_cost_target,
    costs=_stt01_004_costs,
    on_cost_paid=_reveal_hook(5, 1),
    on_selection_complete=_reveal_complete,
    selection_target_validator=_stt01_004_sel,
)


# ---------------------------------------------------------------------------
# on_cost_paid -> EFFECT_SELECTION hooks (ctx min/max overrides)
# ---------------------------------------------------------------------------

def _leader_element(state: State, p) -> jax.Array:
  leader = _leader_of(state, p)
  d = _def_of(state, p, leader)
  # C get_card_element defaults to NORMAL when the Element component is absent
  return jnp.where((leader >= 0) & (d >= 0), _np(cards.ELEMENT)[jnp.maximum(d, 0)],
                   _NORMAL)


def _first_tapped_ikz(state: State, p) -> jax.Array:
  """find_first_tapped_ikz: the (held) IKZ token first, then ikz_area cards
  in zone order."""
  token = (state.zone[p, TOKEN_INSTANCE] == Zone.TOKEN) & state.tapped[
      p, TOKEN_INSTANCE
  ]
  in_area = (state.zone[p] == Zone.IKZ_AREA) & state.tapped[p]
  key = jnp.where(in_area, state.zpos[p].astype(jnp.int32), 1 << 20)
  area_inst = jnp.where(in_area.any(), jnp.argmin(key), -1)
  return jnp.where(token, TOKEN_INSTANCE, area_inst).astype(jnp.int32)


# AZK01-015: On Play; effect depends on your leader's element. WATER: untap
# a tapped IKZ. EARTH: heal your leader 2. LIGHTNING: this card gains Charge
# until end of turn. FIRE: deal 2 effect damage to a leader (select 1).
def _azk01_015_validate(state: State, owner, src) -> jax.Array:
  from azuki_jax.engine.helpers import has_charge

  leader = _leader_of(state, owner)
  d = _def_of(state, owner, leader)
  base_hp = jnp.where(d >= 0, _np(cards.BASE_HP)[jnp.maximum(d, 0)], 0).astype(
      jnp.int16
  )
  elem = _leader_element(state, owner)
  ok = jnp.where(
      elem == _WATER,
      _first_tapped_ikz(state, owner) >= 0,
      jnp.where(
          elem == _EARTH,
          state.cur_hp[owner, jnp.maximum(leader, 0)].astype(jnp.int16) < base_hp,
          jnp.where(
              elem == _LIGHTNING,
              ~has_charge(state, owner, jnp.maximum(src, 0)),
              elem == _FIRE,
          ),
      ),
  )
  return (leader >= 0) & ok


def _azk01_015_eff_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  ok = (_leader_element(state, owner) == _FIRE) & (
      state.zone[tp, ti] == Zone.LEADER
  )
  return jnp.where(scope_is_cost, True, ok)


def _azk01_015_apply(state: State, do) -> State:
  from azuki_jax.engine.helpers import GRANT_PHASE_END

  do = jnp.asarray(do)
  owner, src = _ctx(state)
  leader = _leader_of(state, owner)
  do = do & (leader >= 0)
  elem = _leader_element(state, owner)

  # WATER: untap the first tapped IKZ (TapState {false, false})
  ikz = _first_tapped_ikz(state, owner)
  untap = do & (elem == _WATER) & (ikz >= 0)
  safe_ikz = jnp.maximum(ikz, 0)
  state = state._replace(
      tapped=state.tapped.at[owner, safe_ikz].set(
          jnp.where(untap, False, state.tapped[owner, safe_ikz])
      ),
      cooldown=state.cooldown.at[owner, safe_ikz].set(
          jnp.where(untap, 0, state.cooldown[owner, safe_ikz])
      ),
  )

  # EARTH: heal leader up to 2
  state = heal_leader(state, owner, 2, do & (elem == _EARTH))

  # LIGHTNING: Charge until end of turn
  state = apply_charge_grant(
      state, owner, src, GRANT_PHASE_END, 1, do & (elem == _LIGHTNING)
  )

  # FIRE: 2 effect damage to the selected leader
  tp, ti, has = _eff_target(state, 0)
  return deal_effect_damage(
      state, tp, ti, 2, do & (elem == _FIRE) & has
  )


def _azk01_015_on_cost_paid(state: State) -> State:
  owner, _ = _ctx(state)
  fire = _leader_element(state, owner) == _FIRE
  state = state._replace(
      ab_eff_min=jnp.where(fire, jnp.int8(1), state.ab_eff_min),
      ab_eff_max=jnp.where(fire, jnp.int8(1), state.ab_eff_max),
      ab_phase=jnp.where(
          fire, jnp.int8(AbilityPhase.EFFECT_SELECTION), state.ab_phase
      ),
  )
  return _azk01_015_apply(state, do=~fire)


register(
    "AZK01-015",
    validate=_azk01_015_validate,
    target_validator=_azk01_015_eff_target,
    on_cost_paid=_azk01_015_on_cost_paid,
    effects=lambda state: _azk01_015_apply(state, True),
)


# AZK01-016 (spell, Main): draw 2 (cost), then discard exactly 2 from hand.
def _azk01_016_validate(state: State, owner, src) -> jax.Array:
  hand_count = _zone_count(state, owner, Zone.HAND)
  src_in_hand = state.zone[owner, jnp.maximum(src, 0)] == Zone.HAND
  hand_after = hand_count - src_in_hand.astype(jnp.int32)
  deck = _zone_count(state, owner, Zone.DECK)
  available = hand_after + jnp.where(deck >= 2, 2, 1)
  return (deck >= 1) & (available >= 2)


def _azk01_016_eff_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  from azuki_jax.abilities.cards_impl import already_selected

  _, src = _ctx(state)
  ok = (
      (tp == owner)
      & (state.zone[tp, ti] == Zone.HAND)
      & (ti != src)
      & ~already_selected(state, jnp.bool_(False), tp, ti)
  )
  return jnp.where(scope_is_cost, True, ok)


def _azk01_016_costs(state: State) -> State:
  owner, _ = _ctx(state)
  return draw_with_deckout(state, owner, 2, True)


def _azk01_016_on_cost_paid(state: State) -> State:
  go = state.winner == -1
  return state._replace(
      ab_eff_min=jnp.where(go, jnp.int8(2), state.ab_eff_min),
      ab_eff_max=jnp.where(go, jnp.int8(2), state.ab_eff_max),
      ab_phase=jnp.where(
          go, jnp.int8(AbilityPhase.EFFECT_SELECTION), state.ab_phase
      ),
  )


def _azk01_016_effects(state: State) -> State:
  from azuki_jax.engine.helpers import discard

  for k in range(2):
    tp, ti, has = _eff_target(state, k)
    state = discard(state, tp, ti, do=has & (k < state.ab_eff_selected))
  return state


register(
    "AZK01-016",
    validate=_azk01_016_validate,
    target_validator=_azk01_016_eff_target,
    costs=_azk01_016_costs,
    on_cost_paid=_azk01_016_on_cost_paid,
    effects=_azk01_016_effects,
)


# AZK01-017 (spell, Main): deal 1 effect damage to up to 1 garden entity
# and/or up to 1 leader (one target per "class").
def _azk01_017_classes(state: State) -> jax.Array:
  any_garden = (
      jnp.any(state.zone[0] == Zone.GARDEN) | jnp.any(state.zone[1] == Zone.GARDEN)
  )
  leaders = jnp.any(state.zone[0] == Zone.LEADER) | jnp.any(
      state.zone[1] == Zone.LEADER
  )
  return any_garden.astype(jnp.int32) + leaders.astype(jnp.int32)


def _azk01_017_validate(state: State, owner, src) -> jax.Array:
  return _azk01_017_classes(state) > 0


def _azk01_017_eff_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  target_is_leader = state.zone[tp, ti] == Zone.LEADER
  in_garden = state.zone[tp, ti] == Zone.GARDEN
  base = target_is_leader | in_garden

  k = jnp.arange(MAX_ABILITY_SELECTION)
  sel_inst = state.ab_eff_targets.astype(jnp.int32)
  sel_p = state.ab_eff_target_players.astype(jnp.int32)
  active = (k < state.ab_eff_selected) & (sel_inst >= 0)
  same = active & (sel_inst == ti) & (sel_p == tp)
  sel_is_leader = (
      state.zone[jnp.maximum(sel_p, 0), jnp.maximum(sel_inst, 0)] == Zone.LEADER
  )
  clash = active & (sel_is_leader == target_is_leader)
  ok = base & ~jnp.any(same | clash)
  return jnp.where(scope_is_cost, True, ok)


def _azk01_017_on_cost_paid(state: State) -> State:
  classes = _azk01_017_classes(state)
  go = classes > 0
  return state._replace(
      ab_eff_min=jnp.where(go, jnp.int8(0), state.ab_eff_min),
      ab_eff_max=jnp.where(go, classes, state.ab_eff_max).astype(jnp.int8),
      ab_phase=jnp.where(
          go, jnp.int8(AbilityPhase.EFFECT_SELECTION), state.ab_phase
      ),
  )


def _azk01_017_effects(state: State) -> State:
  for k in range(2):
    tp, ti, has = _eff_target(state, k)
    state = deal_effect_damage(
        state, tp, ti, 1, do=has & (k < state.ab_eff_selected)
    )
  return state


register(
    "AZK01-017",
    validate=_azk01_017_validate,
    target_validator=_azk01_017_eff_target,
    on_cost_paid=_azk01_017_on_cost_paid,
    effects=_azk01_017_effects,
)


# STT03-006: When Destroyed; draw 1, then discard 1 from hand (select 1).
def _stt03_006_eff_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  ok = (tp == owner) & (state.zone[tp, ti] == Zone.HAND)
  return jnp.where(scope_is_cost, True, ok)


def _stt03_006_on_cost_paid(state: State) -> State:
  owner, _ = _ctx(state)
  state = draw_with_deckout(state, owner, 1, True)
  go = (state.winner == -1) & (_zone_count(state, owner, Zone.HAND) > 0)
  return state._replace(
      ab_eff_min=jnp.where(go, jnp.int8(1), state.ab_eff_min),
      ab_eff_max=jnp.where(go, jnp.int8(1), state.ab_eff_max),
      ab_phase=jnp.where(
          go, jnp.int8(AbilityPhase.EFFECT_SELECTION), state.ab_phase
      ),
  )


def _stt03_006_effects(state: State) -> State:
  from azuki_jax.engine.helpers import discard

  tp, ti, has = _eff_target(state, 0)
  return discard(state, tp, ti, do=has & (state.ab_eff_selected > 0))


register(
    "STT03-006",
    target_validator=_stt03_006_eff_target,
    on_cost_paid=_stt03_006_on_cost_paid,
    effects=_stt03_006_effects,
)


# ---------------------------------------------------------------------------
# Hand -> selection flows
# ---------------------------------------------------------------------------

def _entity_le_row(state: State, p, max_cost: int):
  has_cost, cost = _cost_row(state, p)
  return (
      (_card_type_row(state, p) == CardType.ENTITY)
      & has_cost
      & (cost <= max_cost)
  )


def _weapon_le_row(state: State, p, max_cost: int):
  has_cost, cost = _cost_row(state, p)
  return (
      (_card_type_row(state, p) == CardType.WEAPON)
      & has_cost
      & (cost <= max_cost)
  )


# AZK01-024: On Play (optional); cost: return a friendly garden entity to
# hand; then play an entity of cost <= 2 from hand to the garden or alley.
def _azk01_024_validate(state: State, owner, src) -> jax.Array:
  return _zone_count(state, owner, Zone.GARDEN) > 0


def _azk01_024_cost_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  ok = (
      (tp == owner)
      & (state.zone[tp, ti] == Zone.GARDEN)
      & (_card_type_row(state, owner)[ti] == CardType.ENTITY)
  )
  return jnp.where(scope_is_cost, ok, True)


def _azk01_024_costs(state: State) -> State:
  tp, ti, has = _cost_target(state, 0)
  return return_to_hand(state, tp, ti, do=has)


def _azk01_024_on_cost_paid(state: State) -> State:
  owner, _ = _ctx(state)
  row = _entity_le_row(state, owner, 2)
  # C runs finish_cost_selection inside the readonly stage: apply_costs stays
  # DEFERRED (was_deferred is false there, ability_system.c:439-441), so the
  # entity the cost just returned to hand is not yet a hand child when
  # azk_move_matching_hand_cards_to_selection scans ordered children — it is
  # never eligible for the selection. JAX applies costs immediately, so
  # exclude the cost target explicitly.
  tp, ti, has = _cost_target(state, 0)
  n = state.zone.shape[1]
  row = row & ~((jnp.arange(n) == ti) & has & (tp == owner))
  return selection.move_matching_zone_to_selection(
      state, Zone.HAND, row, 1, do=True
  )


def _azk01_024_sel(state: State, owner, inst) -> jax.Array:
  return _entity_le_row(state, owner, 2)[jnp.maximum(inst, 0)]


register(
    "AZK01-024",
    validate=_azk01_024_validate,
    target_validator=_azk01_024_cost_target,
    costs=_azk01_024_costs,
    on_cost_paid=_azk01_024_on_cost_paid,
    on_selection_complete=lambda s: selection.return_remaining_to_hand(s, True),
    selection_target_validator=_azk01_024_sel,
)


# AZK01-098: On Play (optional, alley); cost: tap this card; play a weapon of
# cost <= 3 from hand and equip it to a friendly entity or leader.
def _azk01_098_validate(state: State, owner, src) -> jax.Array:
  from azuki_jax.engine.helpers import can_tap

  safe = jnp.maximum(src, 0)
  in_alley = state.zone[owner, safe] == Zone.ALLEY
  tappable = can_tap(state, owner, safe, ignore_cooldown=False)
  row = _weapon_le_row(state, owner, 3)
  has_weapon = jnp.any((state.zone[owner] == Zone.HAND) & row)
  return in_alley & tappable & has_weapon


def _azk01_098_costs(state: State) -> State:
  from azuki_jax.engine.helpers import tap

  owner, src = _ctx(state)
  return tap(state, owner, src, do=True)


def _azk01_098_on_cost_paid(state: State) -> State:
  owner, _ = _ctx(state)
  return selection.move_matching_zone_to_selection(
      state, Zone.HAND, _weapon_le_row(state, owner, 3), 1, do=True
  )


def _azk01_098_sel(state: State, owner, inst) -> jax.Array:
  return _weapon_le_row(state, owner, 3)[jnp.maximum(inst, 0)]


register(
    "AZK01-098",
    validate=_azk01_098_validate,
    costs=_azk01_098_costs,
    on_cost_paid=_azk01_098_on_cost_paid,
    on_selection_complete=lambda s: selection.return_remaining_to_hand(s, True),
    selection_target_validator=_azk01_098_sel,
)


# AZK01-111: Main (alley); cost: sacrifice this card; deal 2 effect damage to
# an enemy garden entity (if any), then you may play an entity of cost <= 2
# from hand to the garden.
def _azk01_111_validate(state: State, owner, src) -> jax.Array:
  safe = jnp.maximum(src, 0)
  in_alley = state.zone[owner, safe] == Zone.ALLEY
  opp = (owner + 1) % 2
  enemy_garden = _zone_count(state, opp, Zone.GARDEN) > 0
  has_hand_entity = jnp.any(
      (state.zone[owner] == Zone.HAND) & _entity_le_row(state, owner, 2)
  )
  return in_alley & (enemy_garden | has_hand_entity)


def _azk01_111_eff_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  opp = (owner + 1) % 2
  ok = (
      (tp == opp)
      & (state.zone[tp, ti] == Zone.GARDEN)
      & (_card_type_row(state, tp)[ti] == CardType.ENTITY)
  )
  return jnp.where(scope_is_cost, True, ok)


def _azk01_111_costs(state: State) -> State:
  owner, src = _ctx(state)
  return sacrifice_card(state, owner, src, do=True)


def _azk01_111_apply(state: State, do) -> State:
  do = jnp.asarray(do)
  owner, _ = _ctx(state)
  tp, ti, has = _eff_target(state, 0)
  state = deal_effect_damage(
      state, tp, ti, 2, do=do & has & (state.ab_eff_selected > 0)
  )
  return selection.move_matching_zone_to_selection(
      state, Zone.HAND, _entity_le_row(state, owner, 2), 1, do=do
  )


def _azk01_111_on_cost_paid(state: State) -> State:
  owner, _ = _ctx(state)
  opp = (owner + 1) % 2
  to_eff = _zone_count(state, opp, Zone.GARDEN) > 0
  state = state._replace(
      ab_phase=jnp.where(
          to_eff, jnp.int8(AbilityPhase.EFFECT_SELECTION), state.ab_phase
      )
  )
  return _azk01_111_apply(state, do=~to_eff)


def _azk01_111_sel(state: State, owner, inst) -> jax.Array:
  return _entity_le_row(state, owner, 2)[jnp.maximum(inst, 0)]


register(
    "AZK01-111",
    validate=_azk01_111_validate,
    target_validator=_azk01_111_eff_target,
    costs=_azk01_111_costs,
    on_cost_paid=_azk01_111_on_cost_paid,
    effects=lambda state: _azk01_111_apply(state, True),
    on_selection_complete=lambda s: selection.return_remaining_to_hand(s, True),
    selection_target_validator=_azk01_111_sel,
)


# ---------------------------------------------------------------------------
# Discard -> selection flows
# ---------------------------------------------------------------------------

# AZK01-041: Main (garden, once/turn, 1 IKZ); move weapons of cost <= 2 from
# your discard into the selection zone; equip up to 2 picks to this card.
def _azk01_041_validate(state: State, owner, src) -> jax.Array:
  safe = jnp.maximum(src, 0)
  in_garden = state.zone[owner, safe] == Zone.GARDEN
  row = _weapon_le_row(state, owner, 2)
  has = jnp.any((state.zone[owner] == Zone.DISCARD) & row)
  return in_garden & has


def _azk01_041_on_cost_paid(state: State) -> State:
  owner, _ = _ctx(state)
  row = _weapon_le_row(state, owner, 2)
  count = jnp.minimum(
      jnp.sum((state.zone[owner] == Zone.DISCARD) & row, dtype=jnp.int32),
      selection.MAX_MOVED,
  )
  return selection.move_matching_zone_to_selection(
      state, Zone.DISCARD, row, jnp.minimum(count, 2), do=True
  )


def _azk01_041_sel(state: State, owner, inst) -> jax.Array:
  return _weapon_le_row(state, owner, 2)[jnp.maximum(inst, 0)]


def _equip_to_source(state: State, weapon, do) -> State:
  """AZK01-041 equip_selection_weapon_to_source_card: attach + attack bonus +
  combat modifier + on-play/when-equipped triggers (no play counters)."""
  from azuki_jax.engine.helpers import weapons_of
  from azuki_jax.engine.triggers import queue_on_play, queue_when_equipped

  owner, src = _ctx(state)
  safe_w = jnp.maximum(weapon, 0)
  do = jnp.asarray(do) & (weapon >= 0) & (
      state.zone[owner, safe_w] == Zone.SELECTION
  )

  weapon_count = jnp.sum(weapons_of(state, owner, src), dtype=jnp.int32)
  state = selection._selection_remove(state, owner, safe_w, do)
  state = state._replace(
      zone=state.zone.at[owner, safe_w].set(
          jnp.where(do, jnp.int8(Zone.ATTACHED), state.zone[owner, safe_w])
      ),
      zpos=state.zpos.at[owner, safe_w].set(
          jnp.where(do, weapon_count.astype(jnp.int8), state.zpos[owner, safe_w])
      ),
      attached_to=state.attached_to.at[owner, safe_w].set(
          jnp.where(do, src.astype(jnp.int8), state.attached_to[owner, safe_w])
      ),
  )
  weapon_atk = state.cur_atk[owner, safe_w].astype(jnp.int16)
  new_atk = jnp.maximum(
      state.cur_atk[owner, src].astype(jnp.int16) + weapon_atk, 0
  ).astype(jnp.int8)
  state = state._replace(
      cur_atk=state.cur_atk.at[owner, src].set(
          jnp.where(do, new_atk, state.cur_atk[owner, src])
      )
  )
  # equipped combat modifier (AZK01-018: incoming -1, leaders only; the host
  # here is a garden entity so requires_leader always fails)
  is_018 = state.def_id[owner, safe_w] == cards.CODE_TO_ID["AZK01-018"]
  host_is_leader = state.zone[owner, src] == Zone.LEADER
  state = state._replace(
      cmb_in_perm=state.cmb_in_perm.at[owner, src].add(
          jnp.where(do & is_018 & host_is_leader, -1, 0).astype(jnp.int8)
      )
  )
  state = queue_on_play(state, owner, safe_w, do=do)
  state = queue_when_equipped(state, owner, safe_w, do=do)
  state = queue_when_equipped(state, owner, src, do=do)
  return state


def _azk01_041_complete(state: State) -> State:
  for k in range(2):  # pick_max <= 2
    inst = state.ab_sel_picked[k].astype(jnp.int32)
    state = _equip_to_source(state, inst, do=k < state.ab_sel_picked_count)
  return selection.return_remaining_to_discard(state, True)


register(
    "AZK01-041",
    validate=_azk01_041_validate,
    on_cost_paid=_azk01_041_on_cost_paid,
    on_selection_complete=_azk01_041_complete,
    selection_target_validator=_azk01_041_sel,
)


# AZK01-084 (spell, Main): move Normal entities of cost <= 6 from your
# discard into the selection zone; add 1 pick to hand, rest back to discard.
def _azk01_084_row(state: State, p):
  has_cost, cost = _cost_row(state, p)
  return (
      (_card_type_row(state, p) == CardType.ENTITY)
      & (_element_row(state, p) == _NORMAL)
      & has_cost
      & (cost <= 6)
  )


def _azk01_084_validate(state: State, owner, src) -> jax.Array:
  return jnp.any(
      (state.zone[owner] == Zone.DISCARD) & _azk01_084_row(state, owner)
  )


def _azk01_084_on_cost_paid(state: State) -> State:
  owner, _ = _ctx(state)
  return selection.move_matching_zone_to_selection(
      state, Zone.DISCARD, _azk01_084_row(state, owner), 1, do=True
  )


def _azk01_084_complete(state: State) -> State:
  state = selection.move_picked_to_hand(state, do=True)
  return selection.return_remaining_to_discard(state, True)


def _azk01_084_sel(state: State, owner, inst) -> jax.Array:
  return _azk01_084_row(state, owner)[jnp.maximum(inst, 0)]


register(
    "AZK01-084",
    validate=_azk01_084_validate,
    on_cost_paid=_azk01_084_on_cost_paid,
    on_selection_complete=_azk01_084_complete,
    selection_target_validator=_azk01_084_sel,
    selection_complete_if_still=True,
)


# AZK01-086 (spell, Main): move weapons from your discard into the selection
# zone; pick up to 5 — unpicked return to discard, your leader gets +1 attack
# per pick until end of turn, then phase = BOTTOM_DECK for the picks (which
# the CLEAR completion mode immediately clears: C quirk, the picks stay in
# the selection zone).
def _weapon_row(state: State, p):
  return _card_type_row(state, p) == CardType.WEAPON


def _azk01_086_validate(state: State, owner, src) -> jax.Array:
  return jnp.any((state.zone[owner] == Zone.DISCARD) & _weapon_row(state, owner))


def _azk01_086_on_cost_paid(state: State) -> State:
  owner, _ = _ctx(state)
  row = _weapon_row(state, owner)
  count = jnp.minimum(
      jnp.sum((state.zone[owner] == Zone.DISCARD) & row, dtype=jnp.int32),
      selection.MAX_MOVED,
  )
  return selection.move_matching_zone_to_selection(
      state, Zone.DISCARD, row, jnp.minimum(count, 5), do=True
  )


def _azk01_086_complete(state: State) -> State:
  owner, src = _ctx(state)
  picked = state.ab_sel_picked
  picked_count = jnp.minimum(state.ab_sel_picked_count.astype(jnp.int32), 5)

  # 1) unpicked back to discard
  state = selection.return_remaining_to_discard(state, True)

  # 2) re-init the selection state with the picked cards
  new_cards = jnp.full(state.ab_sel_cards.shape, -1, jnp.int8)
  for k in range(5):
    new_cards = new_cards.at[k].set(
        jnp.where(k < picked_count, picked[k], jnp.int8(-1))
    )
  state = state._replace(
      ab_sel_cards=new_cards,
      ab_sel_count=picked_count.astype(jnp.int8),
      ab_sel_picked=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_sel_picked_count=jnp.int8(0),
      ab_sel_pick_max=jnp.int8(0),
  )

  # 3) none picked -> phase NONE; else +picked leader attack (EOT) and
  # phase = BOTTOM_DECK
  none = picked_count == 0
  leader = _leader_of(state, owner)
  state = apply_attack_modifier(
      state, owner, jnp.maximum(leader, 0), picked_count,
      expires_eot=True, do=~none & (leader >= 0),
  )
  return state._replace(
      ab_phase=jnp.where(
          none, jnp.int8(AbilityPhase.NONE), jnp.int8(AbilityPhase.BOTTOM_DECK)
      )
  )


def _azk01_086_sel(state: State, owner, inst) -> jax.Array:
  return _weapon_row(state, owner)[jnp.maximum(inst, 0)]


register(
    "AZK01-086",
    validate=_azk01_086_validate,
    on_cost_paid=_azk01_086_on_cost_paid,
    on_selection_complete=_azk01_086_complete,
    selection_target_validator=_azk01_086_sel,
)


# AZK01-097: On Play; mill the top 5 into the selection zone; reveal up to 1
# weapon among them to hand, the rest go to the discard pile.
def _azk01_097_on_cost_paid(state: State) -> State:
  state = selection.reveal_top_into_selection(state, 5, 1, do=True)
  revealed = state.ab_sel_count > 0
  matching = selection.selection_matching_count(state)
  to_pick = revealed & (matching > 0)
  state = state._replace(
      ab_phase=jnp.where(
          to_pick, jnp.int8(AbilityPhase.SELECTION_PICK), state.ab_phase
      )
  )
  return selection.return_remaining_to_discard(
      state, do=revealed & (matching == 0)
  )


def _azk01_097_complete(state: State) -> State:
  state = selection.move_picked_to_hand(state, do=True)
  return selection.return_remaining_to_discard(state, True)


def _azk01_097_sel(state: State, owner, inst) -> jax.Array:
  return _weapon_row(state, owner)[jnp.maximum(inst, 0)]


register(
    "AZK01-097",
    on_cost_paid=_azk01_097_on_cost_paid,
    on_selection_complete=_azk01_097_complete,
    selection_target_validator=_azk01_097_sel,
    selection_complete_if_still=True,
)
