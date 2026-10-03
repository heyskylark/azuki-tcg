"""Triggered-effect queue (TriggeredEffectQueue, max 16).

Vanilla scope: only cards marked IMPLEMENTED in the ability layer enqueue; a
card with an unimplemented ability raises a coverage flag in the state via
`unimplemented_hit` (checked by tests) — for vanilla decks nothing queues.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from azuki_jax.abilities import tables
from azuki_jax.constants import MAX_TRIGGERED_EFFECTS
from azuki_jax.state import State

# timing ids mirror include/abilities/ability_system.h TIMING_TAG_*
TIMING_ON_PLAY = 0
TIMING_START_OF_TURN = 1
TIMING_START_OF_EACH_TURN = 2
TIMING_END_OF_TURN = 3
TIMING_WHEN_EQUIPPING = 4
TIMING_WHEN_EQUIPPED = 5
TIMING_WHEN_ATTACKING = 6
TIMING_WHEN_ATTACKED = 7
TIMING_WHEN_RETURNED_TO_HAND = 8
TIMING_ON_GATE_PORTAL = 9
TIMING_AFTER_ATTACKING = 10
TIMING_WHEN_TAKES_DAMAGE = 11
TIMING_WHEN_DEALS_DAMAGE = 12
TIMING_WHEN_DESTROYED = 13
TIMING_WHEN_SACRIFICED = 14
TIMING_WHEN_ENTERS_GARDEN = 15

_TIMING_TABLE = {
    TIMING_ON_PLAY: tables.TIMING_ON_PLAY,
    TIMING_WHEN_ATTACKING: tables.TIMING_WHEN_ATTACKING,
    TIMING_WHEN_ATTACKED: tables.TIMING_WHEN_ATTACKED,
    TIMING_AFTER_ATTACKING: tables.TIMING_AFTER_ATTACKING,
    TIMING_ON_GATE_PORTAL: tables.TIMING_ON_GATE_PORTAL,
    TIMING_WHEN_EQUIPPED: tables.TIMING_WHEN_EQUIPPED,
    TIMING_END_OF_TURN: tables.TIMING_END_OF_TURN,
    TIMING_START_OF_TURN: tables.TIMING_START_OF_TURN,
    TIMING_START_OF_EACH_TURN: tables.TIMING_START_OF_EACH_TURN,
    TIMING_WHEN_ENTERS_GARDEN: tables.TIMING_WHEN_ENTERS_GARDEN,
    TIMING_WHEN_DESTROYED: tables.TIMING_WHEN_DESTROYED,
    TIMING_WHEN_RETURNED_TO_HAND: tables.TIMING_WHEN_RETURNED_TO_HAND,
    TIMING_WHEN_TAKES_DAMAGE: tables.TIMING[:, tables.TIMING_INDEX["AWhenTakesDamage"]],
    TIMING_WHEN_DEALS_DAMAGE: tables.TIMING[:, tables.TIMING_INDEX["AWhenDealsDamage"]],
}


def has_queued(state: State) -> jax.Array:
  return state.trig_count > 0


def record_damage_event(state: State, sp, si, tp, ti, damage, do,
                        from_effect=True) -> State:
  """azk_record_damage_event: tracker updates + takes/deals-damage triggers.

  (sp, si) = source card (sp < 0 => none), (tp, ti) = target. Fires only when
  damage > 0. Order matches C: target tracker + AWhenTakesDamage queue, then
  source tracker + AWhenDealsDamage queue. from_effect mirrors the C param
  (effect damage True, combat damage False)."""
  do = jnp.asarray(do) & (damage > 0)
  tp_i = jnp.maximum(tp, 0)
  ti_i = jnp.maximum(ti, 0)
  dmg8 = jnp.clip(jnp.asarray(damage, jnp.int16), -128, 127).astype(jnp.int8)
  state = state._replace(
      took_damage_turn=state.took_damage_turn.at[tp_i, ti_i].set(
          jnp.where(do, True, state.took_damage_turn[tp_i, ti_i])
      ),
      last_dmg_taken=state.last_dmg_taken.at[tp_i, ti_i].set(
          jnp.where(do, dmg8, state.last_dmg_taken[tp_i, ti_i])
      ),
      last_dmg_src_player=state.last_dmg_src_player.at[tp_i, ti_i].set(
          jnp.where(do, jnp.asarray(sp, jnp.int8),
                    state.last_dmg_src_player[tp_i, ti_i])
      ),
      last_dmg_src_inst=state.last_dmg_src_inst.at[tp_i, ti_i].set(
          jnp.where(do, jnp.asarray(si, jnp.int8),
                    state.last_dmg_src_inst[tp_i, ti_i])
      ),
      last_dmg_from_effect=state.last_dmg_from_effect.at[tp_i, ti_i].set(
          jnp.where(do, jnp.asarray(from_effect),
                    state.last_dmg_from_effect[tp_i, ti_i])
      ),
  )
  # distinct-source set (DamageTracker.tracked_sources, max 8; C also counts
  # the null source — entity 0 — as one distinct entry)
  key = jnp.where(
      jnp.asarray(sp) >= 0,
      jnp.asarray(sp, jnp.int32) * 256 + jnp.asarray(si, jnp.int32),
      30000,
  ).astype(jnp.int16)
  row_keys = state.dmg_src_keys[tp_i, ti_i]
  cnt = state.dmg_src_count[tp_i, ti_i].astype(jnp.int32)
  seen = jnp.any((jnp.arange(8) < cnt) & (row_keys == key))
  add = do & ~seen & (cnt < 8)
  slot = jnp.clip(cnt, 0, 7)
  state = state._replace(
      dmg_src_keys=state.dmg_src_keys.at[tp_i, ti_i, slot].set(
          jnp.where(add, key, state.dmg_src_keys[tp_i, ti_i, slot])
      ),
      dmg_src_count=state.dmg_src_count.at[tp_i, ti_i].set(
          jnp.where(add, cnt + 1, cnt).astype(jnp.int8)
      ),
  )
  state = queue_effect(state, tp_i, ti_i, TIMING_WHEN_TAKES_DAMAGE, do)

  has_src = jnp.asarray(sp) >= 0
  sp_i = jnp.maximum(sp, 0)
  si_i = jnp.maximum(si, 0)
  state = state._replace(
      dealt_damage_turn=state.dealt_damage_turn.at[sp_i, si_i].set(
          jnp.where(do & has_src, True, state.dealt_damage_turn[sp_i, si_i])
      )
  )
  return queue_effect(state, sp_i, si_i, TIMING_WHEN_DEALS_DAMAGE, do & has_src)


def queue_effect(state: State, p, inst, timing: int, do) -> State:
  """azk_queue_triggered_effect: append (source, owner, timing)."""
  def_id = state.def_id[p, inst]
  timed = jnp.where(
      def_id >= 0, jnp.asarray(_TIMING_TABLE[timing])[def_id], False
  )
  do = jnp.asarray(do) & timed & (state.trig_count < MAX_TRIGGERED_EFFECTS)
  idx = jnp.clip(state.trig_count, 0, MAX_TRIGGERED_EFFECTS - 1)
  return state._replace(
      trig_source=state.trig_source.at[idx].set(
          jnp.where(do, jnp.asarray(inst, jnp.int8), state.trig_source[idx])
      ),
      trig_owner=state.trig_owner.at[idx].set(
          jnp.where(do, jnp.asarray(p, jnp.int8), state.trig_owner[idx])
      ),
      trig_timing=state.trig_timing.at[idx].set(
          jnp.where(do, jnp.int8(timing), state.trig_timing[idx])
      ),
      trig_count=jnp.where(do, state.trig_count + 1, state.trig_count).astype(
          jnp.int8
      ),
  )


def pop_effect(state: State) -> tuple[State, jax.Array, jax.Array, jax.Array]:
  """Dequeue head; shift remaining entries forward."""
  src = state.trig_source[0]
  owner = state.trig_owner[0]
  timing = state.trig_timing[0]
  state = state._replace(
      trig_source=jnp.roll(state.trig_source, -1).at[-1].set(-1),
      trig_owner=jnp.roll(state.trig_owner, -1).at[-1].set(-1),
      trig_timing=jnp.roll(state.trig_timing, -1).at[-1].set(-1),
      trig_count=jnp.maximum(state.trig_count - 1, 0).astype(jnp.int8),
  )
  return state, src, owner, timing


def queue_zone_by_seq(state: State, p, zone, timing: int, do=True) -> State:
  """queue_timing_abilities_in_zone for a slot zone: iterate cards in flecs
  ordered-children (insertion) order = board_seq ascending."""
  do = jnp.asarray(do)
  in_zone = state.zone[p] == zone
  key = jnp.where(in_zone, state.board_seq[p].astype(jnp.int32), 1 << 20)
  order = jnp.argsort(key)
  for k in range(5):
    inst = order[k]
    state = queue_effect(state, p, inst, timing, do & in_zone[inst])
  return state


def queue_on_play(state: State, p, inst, do=True) -> State:
  return queue_effect(state, p, inst, TIMING_ON_PLAY, do)


def queue_enter_garden(state: State, p, inst, do=True) -> State:
  return queue_effect(state, p, inst, TIMING_WHEN_ENTERS_GARDEN, do)


def queue_gate_portal(state: State, p, gate_inst, portaled, do=True) -> State:
  # scratch: remember portaled card for the gate ability
  state = state._replace(
      ab_scratch=state.ab_scratch.at[0].set(
          jnp.where(jnp.asarray(do), portaled.astype(jnp.int16), state.ab_scratch[0])
      )
  )
  return queue_effect(state, p, gate_inst, TIMING_ON_GATE_PORTAL, do)


def queue_when_equipped(state: State, p, inst, do=True) -> State:
  return queue_effect(state, p, inst, TIMING_WHEN_EQUIPPED, do)


def queue_when_attacking_chain(state: State, p, attacker, do=True) -> State:
  """Attacker + its attached weapons (in attach order)."""
  state = queue_effect(state, p, attacker, TIMING_WHEN_ATTACKING, do)
  from azuki_jax.engine.helpers import weapons_of
  from azuki_jax.constants import MAX_ATTACHED_WEAPONS

  mask = weapons_of(state, p, attacker)
  for k in range(MAX_ATTACHED_WEAPONS):
    at_order = mask & (state.zpos[p] == k)
    weapon = jnp.where(at_order.any(), jnp.argmax(at_order), -1)
    state = queue_effect(
        state, p, jnp.maximum(weapon, 0), TIMING_WHEN_ATTACKING,
        jnp.asarray(do) & (weapon >= 0),
    )
  # AZK01-034 'Kira' redirect: defender-side alley card queues when-attacked
  # at declare time if the defending card is in the defender's garden.
  opp = (p + 1) % 2
  defender = state.combat_defender
  defender_in_garden = (defender >= 0) & (
      state.zone[opp, jnp.maximum(defender.astype(jnp.int32), 0)] == 4
  )
  n = state.zone.shape[1]
  from azuki_jax import cards as _cards

  kira_mask = (state.zone[opp] == 5) & (
      state.def_id[opp] == _cards.CODE_TO_ID["AZK01-034"]
  )
  # alley slot order
  for slot in range(5):
    at_slot = kira_mask & (state.zpos[opp] == slot)
    kira = jnp.where(at_slot.any(), jnp.argmax(at_slot), -1)
    state = queue_effect(
        state, opp, jnp.maximum(kira, 0), TIMING_WHEN_ATTACKED,
        jnp.asarray(do) & defender_in_garden & (kira >= 0),
    )
  del n
  return state


def queue_when_attacked_defender(state: State, do=True) -> State:
  """azk_queue_current_defender_when_attacked."""
  defender = state.combat_defender
  dp = state.combat_defender_player
  ok = (defender >= 0) & (dp >= 0)
  return queue_effect(
      state,
      jnp.maximum(dp.astype(jnp.int32), 0),
      jnp.maximum(defender.astype(jnp.int32), 0),
      TIMING_WHEN_ATTACKED,
      jnp.asarray(do) & ok,
  )


def queue_after_attacking(state: State, p, attacker, do=True) -> State:
  return queue_effect(state, p, attacker, TIMING_AFTER_ATTACKING, do)
