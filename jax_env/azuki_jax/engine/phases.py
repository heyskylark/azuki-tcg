"""Automatic phase systems + the engine tick ladder.

Mirrors src/systems/{start,end,combat_resolve}_phase.c, phase_gate.c and
azuki_engine.c (azk_engine_tick / azk_engine_requires_action).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from azuki_jax import cards
from azuki_jax.constants import (
    ALLEY_SIZE,
    GARDEN_SIZE,
    MAX_HAND_SIZE,
    TOKEN_INSTANCE,
    CardType,
    Phase,
    Zone,
)
from azuki_jax.engine import ikz
from azuki_jax.engine.helpers import (
    attr_force_tapped,
    discard,
    discard_equipped_weapons,
    godmode_in_play,
    has_defender_kw,
    has_infiltrate,
    in_play_zone,
    is_frozen,
    leader_instance,
    reset_entity_health,
    total_carapace,
    total_cmb_in,
    total_cmb_out,
)
from azuki_jax.engine.triggers import (
    has_queued,
    queue_after_attacking,
    queue_when_attacked_defender,
)
from azuki_jax.engine.validate import (
    can_play_from_hand_in_response,
    effective_play_cost,
    validate_attach_weapon,
    validate_play_entity,
)
from azuki_jax.state import State
from azuki_jax.zones import move_top_n, zone_count


def _np(table):
  return jnp.asarray(table)


def _arange(state):
  return jnp.arange(state.zone.shape[1])


# ---------------------------------------------------------------------------
# START_OF_TURN (no user action)
# ---------------------------------------------------------------------------

def _tick_start_statuses(state: State, p, do) -> State:
  """tick_status_effects_for_player: garden/alley/leader zones; frozen and
  effect-immune durations decrement (>0), removing the status at 0. Timed
  tag grants with START phase tick down (vanilla: none granted)."""
  z = state.zone[p]
  in_zones = ((z == Zone.GARDEN) | (z == Zone.ALLEY) | (z == Zone.LEADER)) & do

  frozen = state.frozen_dur[p]
  new_frozen = jnp.where(in_zones & (frozen > 0), frozen - 1, frozen)
  immune = state.effect_immune_dur[p]
  new_immune = jnp.where(in_zones & (immune > 0), immune - 1, immune)

  state = state._replace(
      frozen_dur=state.frozen_dur.at[p].set(new_frozen.astype(jnp.int8)),
      effect_immune_dur=state.effect_immune_dur.at[p].set(
          new_immune.astype(jnp.int8)
      ),
  )
  from azuki_jax.engine.helpers import GRANT_PHASE_START, tick_timed_grants

  return tick_timed_grants(state, p, GRANT_PHASE_START, do)


def _untap_all_for(state: State, p, do) -> State:
  """UntapAllCards over garden/alley/ikz_area/leader/gate.

  Shocked cards: duration > 1 decrements, else shocked removed; either way no
  untap this turn. AttrGardenForceTapped: cooldown clears, tapped persists."""
  z = state.zone[p]
  in_zones = (
      (z == Zone.GARDEN) | (z == Zone.ALLEY) | (z == Zone.IKZ_AREA)
      | (z == Zone.LEADER) | (z == Zone.GATE)
  ) & do
  shocked = state.shocked_dur[p]
  is_shocked = shocked != 0
  dec_shocked = jnp.where(
      in_zones & is_shocked & (shocked > 1), shocked - 1,
      jnp.where(in_zones & is_shocked, 0, shocked),
  )

  force_tapped = attr_force_tapped(state, p, _arange(state))
  untap = in_zones & ~is_shocked
  new_tapped = jnp.where(untap & ~force_tapped, False, state.tapped[p])
  new_cooldown = jnp.where(untap, 0, state.cooldown[p])

  return state._replace(
      shocked_dur=state.shocked_dur.at[p].set(dec_shocked.astype(jnp.int8)),
      tapped=state.tapped.at[p].set(new_tapped),
      cooldown=state.cooldown.at[p].set(new_cooldown.astype(jnp.uint8)),
  )


def start_of_turn(state: State, do=True) -> State:
  """StartPhase. Sets winner on deck-out draw; ends in MAIN/END_MATCH."""
  p = state.active_player
  do = jnp.asarray(do)

  turn = jnp.where(do, state.turn_number + 1, state.turn_number).astype(jnp.int16)
  state = state._replace(turn_number=turn)

  # DamageTracker per-turn reset (damage_util.c reset_damage_tracker_for_turn
  # runs lazily on the first access after gs->turn_number changes; clearing
  # every tracker at the increment is equivalent)
  n2 = state.took_damage_turn.shape
  state = state._replace(
      took_damage_turn=jnp.where(do, jnp.zeros(n2, jnp.bool_), state.took_damage_turn),
      dealt_damage_turn=jnp.where(do, jnp.zeros(n2, jnp.bool_), state.dealt_damage_turn),
      last_dmg_taken=jnp.where(do, jnp.zeros(n2, jnp.int8), state.last_dmg_taken),
      last_dmg_src_player=jnp.where(
          do, jnp.full(n2, -1, jnp.int8), state.last_dmg_src_player
      ),
      last_dmg_src_inst=jnp.where(
          do, jnp.full(n2, -1, jnp.int8), state.last_dmg_src_inst
      ),
      last_dmg_from_effect=jnp.where(
          do, jnp.zeros(n2, jnp.bool_), state.last_dmg_from_effect
      ),
      dmg_src_keys=jnp.where(
          do, jnp.full(state.dmg_src_keys.shape, -1, jnp.int16), state.dmg_src_keys
      ),
      dmg_src_count=jnp.where(
          do, jnp.zeros(n2, jnp.int8), state.dmg_src_count
      ),
  )

  for player in (0, 1):
    state = _tick_start_statuses(state, player, do)

  zeros = jnp.zeros(2, jnp.uint8)
  state = state._replace(
      entities_played_garden_turn=jnp.where(do, zeros, state.entities_played_garden_turn),
      entities_played_alley_turn=jnp.where(do, zeros, state.entities_played_alley_turn),
      cards_played_turn=jnp.where(do, zeros, state.cards_played_turn),
      discarded_cards_turn=jnp.where(do, zeros, state.discarded_cards_turn),
      returned_to_hand_turn=jnp.where(do, zeros, state.returned_to_hand_turn),
      next_play_cost_reduction=jnp.where(
          do, jnp.zeros(2, jnp.int8), state.next_play_cost_reduction
      ),
      once_per_turn_used=jnp.where(do, 0, state.once_per_turn_used).astype(jnp.uint8),
  )

  state = _untap_all_for(state, p, do)

  # Draw (skip on the very first turn); deck-out sets winner
  should_draw = state.turn_number > 1
  deck_count = zone_count(state.zone[p], Zone.DECK)
  deck_out = do & should_draw & (deck_count == 0)
  zone_row, zpos_row = state.zone[p], state.zpos[p]
  nzr, npr = move_top_n(zone_row, zpos_row, Zone.DECK, Zone.HAND, jnp.int32(1), 1)
  draw = do & should_draw & ~deck_out
  state = state._replace(
      zone=state.zone.at[p].set(jnp.where(draw, nzr, zone_row)),
      zpos=state.zpos.at[p].set(jnp.where(draw, npr, zpos_row)),
      winner=jnp.where(deck_out, (p + 1) % 2, state.winner).astype(jnp.int8),
  )

  # Grant IKZ: top of pile -> area
  zone_row, zpos_row = state.zone[p], state.zpos[p]
  nzr, npr = move_top_n(
      zone_row, zpos_row, Zone.IKZ_PILE, Zone.IKZ_AREA, jnp.int32(1), 1
  )
  grant = do & ~deck_out
  state = state._replace(
      zone=state.zone.at[p].set(jnp.where(grant, nzr, zone_row)),
      zpos=state.zpos.at[p].set(jnp.where(grant, npr, zpos_row)),
  )

  # start-of-turn / start-of-each-turn triggers (zones in C order: garden,
  # leader, alley; within slot zones the flecs ordered-children = insertion
  # order, i.e. board_seq ascending)
  from azuki_jax.engine.triggers import (
      TIMING_START_OF_EACH_TURN,
      TIMING_START_OF_TURN,
      queue_effect,
      queue_zone_by_seq,
  )

  def queue_player_zones(state, player, timing, ok):
    state = queue_zone_by_seq(state, player, Zone.GARDEN, timing, ok)
    leader = leader_instance(state, player)
    state = queue_effect(
        state, player, jnp.maximum(leader, 0), timing, ok & (leader >= 0)
    )
    return queue_zone_by_seq(state, player, Zone.ALLEY, timing, ok)

  state = queue_player_zones(state, p, TIMING_START_OF_TURN, do & ~deck_out)
  for player in (0, 1):
    state = queue_player_zones(
        state, player, TIMING_START_OF_EACH_TURN, do & ~deck_out
    )

  game_over = state.winner != -1
  new_phase = jnp.where(
      game_over, jnp.int8(Phase.END_MATCH), jnp.int8(Phase.MAIN)
  )
  return state._replace(phase=jnp.where(do, new_phase, state.phase))


# ---------------------------------------------------------------------------
# COMBAT_RESOLVE (no user action)
# ---------------------------------------------------------------------------

def combat_resolve(state: State, do=True) -> State:
  """HandleCombatResolution + resolve_combat."""
  do = jnp.asarray(do)
  p = state.active_player  # attacker (transition_to_combat_resolve restored it)
  opp = (p + 1) % 2

  attacker = jnp.maximum(state.combat_attacker.astype(jnp.int32), 0)
  defender = jnp.maximum(state.combat_defender.astype(jnp.int32), 0)
  have_combat = (state.combat_attacker >= 0) & (state.combat_defender >= 0)

  attacker_zone = state.zone[p, attacker]
  attacker_is_leader = attacker_zone == Zone.LEADER
  attacker_valid = have_combat & (
      attacker_is_leader | (attacker_zone == Zone.GARDEN)
  )
  defender_zone = state.zone[opp, defender]
  defender_is_leader = defender_zone == Zone.LEADER
  defender_valid = have_combat & (
      defender_is_leader | (defender_zone == Zone.GARDEN)
      | (defender_zone == Zone.ALLEY)
  )
  fizzled = do & (~attacker_valid | ~defender_valid)
  resolve = do & attacker_valid & defender_valid

  defender_frozen = is_frozen(state, opp, defender)
  deal = resolve & ~defender_frozen

  atk_stat_attacker = state.cur_atk[p, attacker].astype(jnp.int16)
  atk_stat_defender = state.cur_atk[opp, defender].astype(jnp.int16)

  dmg_to_attacker = jnp.clip(
      atk_stat_defender
      + total_cmb_out(state, opp, defender)
      + total_cmb_in(state, p, attacker)
      - total_carapace(state, p, attacker),
      0, 127,
  )
  dmg_to_defender = jnp.clip(
      atk_stat_attacker
      + total_cmb_out(state, p, attacker)
      + total_cmb_in(state, opp, defender)
      - total_carapace(state, opp, defender),
      0, 127,
  )

  attacker_godmode = godmode_in_play(state, p, attacker)
  defender_godmode = godmode_in_play(state, opp, defender)

  attacker_hp = state.cur_hp[p, attacker].astype(jnp.int16)
  defender_hp = state.cur_hp[opp, defender].astype(jnp.int16)
  new_attacker_hp = attacker_hp - dmg_to_attacker
  new_attacker_hp = jnp.where(
      attacker_godmode & (new_attacker_hp < 0), 0, new_attacker_hp
  )
  new_defender_hp = defender_hp - dmg_to_defender
  new_defender_hp = jnp.where(
      defender_godmode & (new_defender_hp < 0), 0, new_defender_hp
  )

  state = state._replace(
      cur_hp=state.cur_hp.at[p, attacker]
      .set(jnp.where(deal, new_attacker_hp.astype(jnp.int8), state.cur_hp[p, attacker]))
      .at[opp, defender]
      .set(jnp.where(deal, new_defender_hp.astype(jnp.int8), state.cur_hp[opp, defender])),
  )

  # AZK01-044 lightning kanabo (shock on damage dealt) — once per turn per
  # weapon. C passes the post-godmode damage actually TAKEN (prev - cur when
  # godmode clamped, raw damage otherwise).
  taken_by_attacker = jnp.where(
      attacker_godmode, attacker_hp - new_attacker_hp, dmg_to_attacker
  )
  taken_by_defender = jnp.where(
      defender_godmode, defender_hp - new_defender_hp, dmg_to_defender
  )
  state = _lightning_kanabo(state, p, attacker, opp, defender,
                            jnp.where(deal, taken_by_defender, 0))
  state = _lightning_kanabo(state, opp, defender, p, attacker,
                            jnp.where(deal, taken_by_attacker, 0))

  # azk_record_damage_event x2, C order: attacker-takes (source=defender,
  # queues AWhenTakesDamage then defender's AWhenDealsDamage), then
  # defender-takes (source=attacker)
  from azuki_jax.engine.triggers import record_damage_event

  state = record_damage_event(
      state, opp, defender, p, attacker, taken_by_attacker, deal,
      from_effect=False,
  )
  state = record_damage_event(
      state, p, attacker, opp, defender, taken_by_defender, deal,
      from_effect=False,
  )

  attacker_dead = deal & (state.cur_hp[p, attacker] <= 0) & ~attacker_godmode
  defender_dead = deal & (state.cur_hp[opp, defender] <= 0) & ~defender_godmode

  attacker_leader_defeated = attacker_dead & attacker_is_leader
  defender_leader_defeated = defender_dead & defender_is_leader

  # Combat deaths discard the card directly; attached weapons stay attached
  # to the dead host (C never detaches them here — they simply leave play).
  state = discard(state, p, attacker, do=attacker_dead & ~attacker_is_leader)
  state = discard(state, opp, defender, do=defender_dead & ~defender_is_leader)

  winner = state.winner
  winner = jnp.where(
      attacker_leader_defeated & defender_leader_defeated, 2,
      jnp.where(
          attacker_leader_defeated, (state.active_player + 1) % 2,
          jnp.where(defender_leader_defeated, state.active_player, winner),
      ),
  ).astype(jnp.int8)
  state = state._replace(winner=winner)

  # when-destroyed triggers for combat deaths
  from azuki_jax.engine.triggers import TIMING_WHEN_DESTROYED, queue_effect

  state = queue_effect(
      state, p, attacker, TIMING_WHEN_DESTROYED,
      attacker_dead & ~attacker_is_leader,
  )
  state = queue_effect(
      state, opp, defender, TIMING_WHEN_DESTROYED,
      defender_dead & ~defender_is_leader,
  )
  # NOTE: the Bobu destroy observer fires inside helpers.discard; Miharu
  # (STT03-012) / Kurai (STT04-013) observers stay unmodeled (outside the
  # training pool).

  # after-attacking triggers
  state = queue_after_attacking(state, p, attacker, do=resolve)

  state = state._replace(
      combat_attacker=jnp.where(do, jnp.int8(-1), state.combat_attacker),
      combat_defender=jnp.where(do, jnp.int8(-1), state.combat_defender),
      combat_defender_player=jnp.where(do, jnp.int8(-1), state.combat_defender_player),
      combat_intercepted=jnp.where(do, False, state.combat_intercepted),
  )

  game_over = state.winner != -1
  new_phase = jnp.where(game_over, jnp.int8(Phase.END_MATCH), jnp.int8(Phase.MAIN))
  return state._replace(
      phase=jnp.where(do | fizzled, new_phase, state.phase)
  )


def _lightning_kanabo(state: State, dp, dealer, rp, recipient, damage) -> State:
  """AZK01-044 weapons on dealer shock the recipient (once per turn EACH).

  combat_util.c trigger_lightning_kanabo_if_present iterates every attached
  AZK01-044 child: a weapon whose once-per-turn flag is unused marks it and
  applies the (idempotent, duration-1) shock."""
  from azuki_jax.engine.helpers import weapons_of

  mask = weapons_of(state, dp, dealer) & (
      state.def_id[dp] == cards.CODE_TO_ID["AZK01-044"]
  )
  unused = mask & ((state.once_per_turn_used[dp] & 1) == 0)
  mark = unused & (damage > 0)
  fire = mark.any()
  state = state._replace(
      once_per_turn_used=state.once_per_turn_used.at[dp].set(
          jnp.where(mark, state.once_per_turn_used[dp] | 1,
                    state.once_per_turn_used[dp])
      ),
      shocked_dur=state.shocked_dur.at[rp, recipient].set(
          jnp.where(fire, 1, state.shocked_dur[rp, recipient]).astype(jnp.int8)
      ),
  )
  return state


# ---------------------------------------------------------------------------
# END_TURN (no user action)
# ---------------------------------------------------------------------------

def end_turn(state: State, do=True) -> State:
  """HandleEndPhase (after EOT abilities queued+resolved). Vectorized."""
  from azuki_jax.engine.helpers import batch_detach_weapons, batch_discard

  do = jnp.asarray(do)
  p = state.active_player
  nxt = (p + 1) % 2
  n = state.zone.shape[1]

  # 1) sacrifice-at-EOT cards in garden+alley (ending player first, garden
  # then alley, slot order) — weapons first, then the card itself
  for player in (p, nxt):
    z = state.zone[player]
    marked = (
        ((z == Zone.GARDEN) | (z == Zone.ALLEY))
        & state.sacrifice_eot[player]
    )
    host_order = (
        jnp.where(z == Zone.ALLEY, 8, 0) + state.zpos[player].astype(jnp.int32)
    )
    state = batch_detach_weapons(state, player, marked, host_order, do=do)
    z = state.zone[player]
    marked = (
        ((z == Zone.GARDEN) | (z == Zone.ALLEY)) & state.sacrifice_eot[player]
    )
    order = jnp.where(z == Zone.ALLEY, 8, 0) + state.zpos[player].astype(jnp.int32)
    state = batch_discard(state, player, marked, order, do=do)

  # 2) IKZ token: delete if tapped or expires_eot
  for player in (0, 1):
    token_zone = state.zone[player, TOKEN_INSTANCE]
    present = token_zone == Zone.TOKEN
    spent = state.tapped[player, TOKEN_INSTANCE]
    kill = do & present & (spent | state.ikz_token_expires_eot[player])
    state = state._replace(
        zone=state.zone.at[player, TOKEN_INSTANCE].set(
            jnp.where(kill, jnp.int8(Zone.ABSENT), token_zone)
        ),
        tapped=state.tapped.at[player, TOKEN_INSTANCE].set(
            jnp.where(kill, False, state.tapped[player, TOKEN_INSTANCE])
        ),
    )

  # 3) reset entity health + expire EOT modifiers (vectorized per player)
  for player in (p, nxt):
    z = state.zone[player]
    in_play = (z == Zone.GARDEN) | (z == Zone.ALLEY)
    def_id = state.def_id[player]
    base_hp = jnp.where(def_id >= 0, _np(cards.BASE_HP)[def_id], 0).astype(jnp.int16)
    # recalculate_health_from_buffs sums ALL HealthBuff pairs, which
    # includes passive-aura pairs (state.passive_hp mirrors their sum).
    healed = jnp.clip(
        base_hp
        + state.hp_buff_perm[player].astype(jnp.int16)
        + state.hp_buff_eot[player].astype(jnp.int16)
        + state.passive_hp[player].astype(jnp.int16),
        -128, 127,
    ).astype(jnp.int8)
    state = state._replace(
        cur_hp=state.cur_hp.at[player].set(
            jnp.where(do & in_play, healed, state.cur_hp[player])
        )
    )

    in_mod = in_play | (z == Zone.LEADER)
    # C (end_phase.c) expires eot ATTACK modifiers only in the garden + leader
    # zones, NOT the alley (only those zones attack) — unlike combat-damage /
    # carapace eot mods, which it DOES expire in the alley too. So an eot atk
    # buff on an alley entity persists (e.g. STT04-001 buffing a friendly alley
    # entity). Use the garden+leader filter for cur_atk / atk_buff_eot only.
    atk_eot_zones = (z == Zone.GARDEN) | (z == Zone.LEADER)
    new_atk = jnp.maximum(
        state.cur_atk[player].astype(jnp.int16)
        - state.atk_buff_eot[player].astype(jnp.int16),
        0,
    ).astype(jnp.int8)
    hp_zones = (z == Zone.GARDEN) | (z == Zone.LEADER)
    new_hp = (
        state.cur_hp[player].astype(jnp.int16)
        - state.hp_buff_eot[player].astype(jnp.int16)
    ).astype(jnp.int8)
    state = state._replace(
        cur_atk=state.cur_atk.at[player].set(
            jnp.where(do & atk_eot_zones, new_atk, state.cur_atk[player])
        ),
        atk_buff_eot=state.atk_buff_eot.at[player].set(
            jnp.where(do & atk_eot_zones, 0, state.atk_buff_eot[player]).astype(jnp.int8)
        ),
        cmb_in_eot=state.cmb_in_eot.at[player].set(
            jnp.where(do & in_mod, 0, state.cmb_in_eot[player]).astype(jnp.int8)
        ),
        cmb_out_eot=state.cmb_out_eot.at[player].set(
            jnp.where(do & in_mod, 0, state.cmb_out_eot[player]).astype(jnp.int8)
        ),
        carapace_eot=state.carapace_eot.at[player].set(
            jnp.where(do & in_mod, 0, state.carapace_eot[player]).astype(jnp.int8)
        ),
        cur_hp=state.cur_hp.at[player].set(
            jnp.where(do & hp_zones, new_hp, state.cur_hp[player])
        ),
        hp_buff_eot=state.hp_buff_eot.at[player].set(
            jnp.where(do & hp_zones, 0, state.hp_buff_eot[player]).astype(jnp.int8)
        ),
    )
    # tick_end_of_turn_effects_for_player: END-phase timed tag grants
    from azuki_jax.engine.helpers import GRANT_PHASE_END, tick_timed_grants

    state = tick_timed_grants(state, player, GRANT_PHASE_END, do)

  # 4) discard ALL equipped weapons (garden slot order, alley, then leader —
  # per player, ending player first)
  for player in (p, nxt):
    z = state.zone[player]
    hosts = (z == Zone.GARDEN) | (z == Zone.ALLEY) | (z == Zone.LEADER)
    host_order = (
        jnp.where(z == Zone.ALLEY, 8, 0)
        + jnp.where(z == Zone.LEADER, 16, 0)
        + state.zpos[player].astype(jnp.int32)
    )
    state = batch_detach_weapons(state, player, hosts, host_order, do=do)

  return state._replace(
      eot_abilities_queued=jnp.where(do, False, state.eot_abilities_queued),
      phase=jnp.where(do, jnp.int8(Phase.START_OF_TURN), state.phase),
      active_player=jnp.where(do, nxt, state.active_player).astype(jnp.int8),
  )


# ---------------------------------------------------------------------------
# Response-window helpers + phase gate + requires_action
# ---------------------------------------------------------------------------

def defender_can_respond(state: State, defender_p) -> jax.Array:
  """player_util.c defender_can_respond, vanilla scope (response spells /
  response-from-hand cards / response abilities / declare-defender)."""
  from azuki_jax.abilities.tables import HAS_ABILITY, TIMING_IS_RESPONSE

  available = ikz.count_tappable(state, defender_p, True)
  n = state.zone.shape[1]
  idx = jnp.arange(n)

  in_hand = state.zone[defender_p] == Zone.HAND
  def_ids = state.def_id[defender_p]
  valid = def_ids >= 0
  is_spell = jnp.where(valid, _np(cards.TYPE)[def_ids] == CardType.SPELL, False)
  resp_spell = (
      in_hand & is_spell
      & jnp.where(valid, _np(TIMING_IS_RESPONSE)[def_ids], False)
      & jnp.where(valid, _np(HAS_ABILITY)[def_ids], False)
  )
  costs = jax.vmap(lambda i: effective_play_cost(state, defender_p, i))(idx)
  spell_playable = jnp.any(resp_spell & (costs <= available))

  # response-playable entities: any placement validates (probe with the
  # preview active player = defender)
  preview = state._replace(active_player=jnp.asarray(defender_p, jnp.int8),
                           phase=jnp.int8(Phase.RESPONSE_WINDOW))
  hand_count = zone_count(state.zone[defender_p], Zone.HAND)

  def probe_entity(hand_index):
    def per_token(use_token):
      garden = jax.vmap(
          lambda slot: validate_play_entity(
              preview, Zone.GARDEN, hand_index, slot, use_token
          )
      )(jnp.arange(GARDEN_SIZE))
      alley = jax.vmap(
          lambda slot: validate_play_entity(
              preview, Zone.ALLEY, hand_index, slot, use_token
          )
      )(jnp.arange(ALLEY_SIZE))
      return garden.any() | alley.any()

    return per_token(False) | per_token(True)

  entity_playable = jnp.any(
      jax.vmap(probe_entity)(jnp.arange(MAX_HAND_SIZE))
      & (jnp.arange(MAX_HAND_SIZE) < hand_count)
  )

  def probe_weapon(hand_index):
    def per_token(use_token):
      return jax.vmap(
          lambda target: validate_attach_weapon(
              preview, hand_index, target, use_token
          )
      )(jnp.arange(GARDEN_SIZE + 1)).any()

    return per_token(False) | per_token(True)

  weapon_playable = jnp.any(
      jax.vmap(probe_weapon)(jnp.arange(MAX_HAND_SIZE))
      & (jnp.arange(MAX_HAND_SIZE) < hand_count)
  )

  # response abilities on board cards (garden/alley/leader) — ability layer
  from azuki_jax.abilities.effects import response_ability_available

  ability_available = response_ability_available(state, defender_p, available)

  # declare defender
  attacker_p = (defender_p + 1) % 2
  attacker = state.combat_attacker
  can_declare = (
      ~state.combat_intercepted
      & (attacker >= 0)
      & ~has_infiltrate(
          state, attacker_p, jnp.maximum(attacker.astype(jnp.int32), 0)
      )
      & jnp.any(
          (state.zone[defender_p] == Zone.GARDEN)
          & has_defender_kw(state, defender_p, idx)
          & ~state.tapped[defender_p]
      )
  )

  return (
      spell_playable | entity_playable | weapon_playable | ability_available
      | can_declare
  )


def transition_to_combat_resolve(state: State, do=True) -> State:
  """azk_transition_to_combat_resolve: restore attacker as active player,
  queue defender's when-attacked, phase = COMBAT_RESOLVE."""
  do = jnp.asarray(do)
  attacker_p = jnp.where(
      state.combat_defender_player >= 0,
      (state.combat_defender_player + 1) % 2,
      state.active_player,
  ).astype(jnp.int8)
  have_attacker = state.combat_attacker >= 0
  state = state._replace(
      active_player=jnp.where(do & have_attacker, attacker_p, state.active_player)
  )
  state = queue_when_attacked_defender(state, do=do)
  return state._replace(
      phase=jnp.where(do, jnp.int8(Phase.COMBAT_RESOLVE), state.phase)
  )


def phase_gate(state: State) -> State:
  """PhaseGate auto-transitions (MAIN with pending combat -> response/combat)."""
  in_ability = state.ab_phase != 0
  pending_combat = (
      (state.phase == Phase.MAIN)
      & (state.combat_attacker >= 0)
      & ~has_queued(state)
      & ~in_ability
  )
  defender_p = (state.active_player + 1) % 2
  can_respond = defender_can_respond(state, defender_p)

  go_response = pending_combat & can_respond
  go_combat = pending_combat & ~can_respond

  state = state._replace(
      combat_defender_player=jnp.where(
          pending_combat, defender_p.astype(jnp.int8), state.combat_defender_player
      )
  )
  state2 = state._replace(
      phase=jnp.int8(Phase.RESPONSE_WINDOW),
      active_player=defender_p.astype(jnp.int8),
  )
  state3 = transition_to_combat_resolve(state, do=go_combat)
  out = jax.tree.map(lambda a, b: jnp.where(go_response, a, b), state2, state3)
  return jax.tree.map(lambda a, b: jnp.where(pending_combat, a, b), out, state)


def requires_action(state: State) -> jax.Array:
  """azk_engine_requires_action (an active ability phase always requires
  user input, regardless of the game phase)."""
  in_ability = state.ab_phase != 0
  user_phase = (
      (state.phase == Phase.PREGAME_MULLIGAN)
      | (state.phase == Phase.MAIN)
      | (state.phase == Phase.RESPONSE_WINDOW)
      | in_ability
  )
  queued = has_queued(state)

  pending_combat_main = (
      (state.phase == Phase.MAIN) & (state.combat_attacker >= 0)
      & ~queued & ~in_ability
  )
  dead_response = (
      (state.phase == Phase.RESPONSE_WINDOW)
      & ~queued & ~in_ability
      & ~defender_can_respond(state, state.active_player)
  )
  return (
      user_phase
      & ~(queued & ~in_ability)
      & ~pending_combat_main
      & ~dead_response
      & (state.winner == -1)
  )
