"""Per-card ability implementations + dispatch tables.

Each implemented card provides JAX equivalents of its C hooks; dispatch is via
jax.lax.switch over IMPL_INDEX[def_id] (slot 0 = no-op). The "ctx" is the
State's ab_* block (source/owner/selected targets).

Implemented so far (first tranche, all in the training pool):
  STT02-007  On Play: draw 1 (deck-out check)
  AZK01-004  When Attacking: this card +1 attack until end of turn
  AZK01-005  On Play: deal 1 effect damage to up to 1 enemy garden entity
  AZK01-006  Main (once/turn): return this card (garden) to hand
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from azuki_jax import cards
from azuki_jax.abilities import tables
from azuki_jax.constants import Zone
from azuki_jax.state import State


def _np(table):
  return jnp.asarray(table)


# ---------------------------------------------------------------------------
# Effect primitives (mirror C utils)
# ---------------------------------------------------------------------------

def draw_with_deckout(state: State, p, n, do) -> State:
  """draw_cards_with_deckout_check (deck_utils.c:137): draw min(n, deck)
  cards — partial draws DO happen — and the player loses when the deck cannot
  cover the request OR is drained exactly (remaining == 0 after the last
  draw). Start-of-turn draws use the empty-deck-only rule instead."""
  from azuki_jax.zones import move_top_n, zone_count

  do = jnp.asarray(do)
  n = jnp.asarray(n)
  deck_count = zone_count(state.zone[p], Zone.DECK)
  deck_out = do & (deck_count <= n)
  take = jnp.minimum(n, deck_count)
  zone_row, zpos_row = state.zone[p], state.zpos[p]
  nzr, npr = move_top_n(zone_row, zpos_row, Zone.DECK, Zone.HAND, take, 3)
  apply = do & (take > 0)
  return state._replace(
      zone=state.zone.at[p].set(jnp.where(apply, nzr, zone_row)),
      zpos=state.zpos.at[p].set(jnp.where(apply, npr, zpos_row)),
      winner=jnp.where(deck_out, (p + 1) % 2, state.winner).astype(jnp.int8),
  )


def apply_attack_modifier(state: State, p, inst, modifier, expires_eot, do) -> State:
  """status_util apply_attack_modifier: clamp at 0, store ACTUAL delta."""
  cur = state.cur_atk[p, inst].astype(jnp.int16)
  new_atk = jnp.maximum(cur + modifier, 0)
  actual = (new_atk - cur).astype(jnp.int8)
  do = jnp.asarray(do)
  state = state._replace(
      cur_atk=state.cur_atk.at[p, inst].set(
          jnp.where(do, new_atk.astype(jnp.int8), state.cur_atk[p, inst])
      )
  )
  if expires_eot:
    return state._replace(
        atk_buff_eot=state.atk_buff_eot.at[p, inst].add(jnp.where(do, actual, 0))
    )
  return state._replace(
      atk_buff_perm=state.atk_buff_perm.at[p, inst].add(jnp.where(do, actual, 0))
  )


def deal_effect_damage(state: State, tp, ti, damage, do,
                       allow_redirect=True, src_player=None,
                       src_inst=None) -> State:
  """damage_util deal_effect_damage_from_source_internal: Pekiro redirect
  pre-check, immune block, carapace, godmode clamp, damage record + takes/
  deals-damage triggers, leader defeat, entity discard. Source = the active
  ability's source card (C current_damage_source) unless (src_player,
  src_inst) override it (deal_effect_damage_from_source — the AZK01-062
  redirect re-deal keeps the ORIGINAL source; src_player < 0 = no source)."""
  from azuki_jax.engine.helpers import (
      discard,
      godmode_in_play,
      is_effect_immune,
      total_carapace,
  )
  from azuki_jax.engine.triggers import (
      TIMING_WHEN_DESTROYED,
      TIMING_WHEN_TAKES_DAMAGE,
      queue_effect,
      record_damage_event,
  )

  do = jnp.asarray(do)
  if src_player is None:
    src_p = jnp.where(
        state.ab_source >= 0, state.ab_owner, jnp.int8(-1)
    ).astype(jnp.int32)
    src_i = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  else:
    src_p = jnp.asarray(src_player, jnp.int32)
    src_i = jnp.maximum(jnp.asarray(src_inst, jnp.int32), 0)

  # maybe_queue_pekiro_redirect: target is an IMPLEMENTED AZK01-062,
  # damage > 0, no pending redirect for it -> defer the damage, push the
  # redirect entry, queue its takes-damage ability
  target_def = state.def_id[tp, ti]
  is_pekiro = (target_def == cards.CODE_TO_ID["AZK01-062"]) & jnp.where(
      target_def >= 0, _np(tables.IMPLEMENTED)[jnp.maximum(target_def, 0)], False
  )
  k8 = jnp.arange(8)
  already_pending = jnp.any(
      (k8 < state.redirect_count)
      & (state.redirect_tgt_player == jnp.asarray(tp, jnp.int8))
      & (state.redirect_tgt_inst == jnp.asarray(ti, jnp.int8))
  )
  # AZK01-062 is [Once/Turn]: C only defers when its takes-damage ability can
  # still be queued. Once the redirect fired this turn, maybe_queue_pekiro_
  # redirect finds no queueable ability and undoes the deferral -> the damage
  # lands directly. Mirror that: don't defer when the once-per-turn is used.
  once_used = (state.once_per_turn_used[tp, ti] & 1) != 0
  redirect = (
      do & jnp.asarray(allow_redirect) & is_pekiro
      & (jnp.asarray(damage, jnp.int16) > 0) & ~already_pending & ~once_used
      & (state.redirect_count < 8)  # MAX_PENDING_DAMAGE_REDIRECTS
  )
  slot = jnp.clip(state.redirect_count.astype(jnp.int32), 0, 7)
  state = state._replace(
      redirect_src_player=state.redirect_src_player.at[slot].set(
          jnp.where(redirect, src_p.astype(jnp.int8), state.redirect_src_player[slot])
      ),
      redirect_src_inst=state.redirect_src_inst.at[slot].set(
          jnp.where(redirect, src_i.astype(jnp.int8), state.redirect_src_inst[slot])
      ),
      redirect_tgt_player=state.redirect_tgt_player.at[slot].set(
          jnp.where(redirect, jnp.asarray(tp, jnp.int8), state.redirect_tgt_player[slot])
      ),
      redirect_tgt_inst=state.redirect_tgt_inst.at[slot].set(
          jnp.where(redirect, jnp.asarray(ti, jnp.int8), state.redirect_tgt_inst[slot])
      ),
      redirect_damage=state.redirect_damage.at[slot].set(
          jnp.where(redirect, jnp.asarray(damage, jnp.int8), state.redirect_damage[slot])
      ),
      redirect_count=jnp.where(
          redirect, state.redirect_count + 1, state.redirect_count
      ).astype(jnp.int8),
  )
  state = queue_effect(state, tp, ti, TIMING_WHEN_TAKES_DAMAGE, redirect)
  do = do & ~redirect

  immune = is_effect_immune(state, tp, ti)
  adjusted = jnp.asarray(damage, jnp.int16) - total_carapace(state, tp, ti)
  apply = do & ~immune & (adjusted > 0)

  godmode = godmode_in_play(state, tp, ti)
  prev_hp = state.cur_hp[tp, ti].astype(jnp.int16)
  next_hp = prev_hp - adjusted
  next_hp = jnp.where(godmode & (next_hp < 0), 0, next_hp)
  state = state._replace(
      cur_hp=state.cur_hp.at[tp, ti].set(
          jnp.where(apply, next_hp.astype(jnp.int8), state.cur_hp[tp, ti])
      ),
  )
  actual = prev_hp - jnp.where(apply, next_hp, prev_hp)
  state = record_damage_event(state, src_p, src_i, tp, ti, actual, apply)

  dead = apply & (state.cur_hp[tp, ti] <= 0) & ~godmode
  is_leader = state.zone[tp, ti] == Zone.LEADER
  # leader defeated by effect damage: owner loses
  state = state._replace(
      winner=jnp.where(dead & is_leader, (tp + 1) % 2, state.winner).astype(jnp.int8)
  )
  kill = dead & ~is_leader
  state = queue_effect(state, tp, ti, TIMING_WHEN_DESTROYED, kill)
  state = discard(state, tp, ti, do=kill)
  return state


def heal_leader(state: State, p, max_heal, do) -> State:
  """card_utils heal_leader_in_zone: heal = min(base - cur, max_heal), >= 0."""
  from azuki_jax.engine.helpers import leader_instance

  leader = leader_instance(state, p)
  ok = jnp.asarray(do) & (leader >= 0)
  li = jnp.maximum(leader, 0)
  def_id = state.def_id[p, li]
  base = jnp.where(def_id >= 0, _np(cards.BASE_HP)[def_id], 0).astype(jnp.int16)
  cur = state.cur_hp[p, li].astype(jnp.int16)
  heal = jnp.clip(base - cur, 0, jnp.asarray(max_heal, jnp.int16))
  return state._replace(
      cur_hp=state.cur_hp.at[p, li].set(
          jnp.where(ok, (cur + heal).astype(jnp.int8), state.cur_hp[p, li])
      )
  )


def mill_with_deckout(state: State, p, n, do, max_n: int = 5) -> State:
  """deck_utils mill_cards_with_deckout_check: discard top min(n, deck) cards
  (top first; discard_card resets state, no triggers from deck); deck-out loss
  iff deck was non-empty and fully consumed (empty deck mills nothing, no
  loss)."""
  from azuki_jax.engine.helpers import batch_discard
  from azuki_jax.zones import zone_count

  del max_n
  do = jnp.asarray(do)
  n = jnp.asarray(n, jnp.int32)
  deck_count = zone_count(state.zone[p], Zone.DECK)
  loss = do & (deck_count > 0) & (n >= deck_count)
  # top n cards = highest zpos; discard order top-first
  zpos = state.zpos[p].astype(jnp.int32)
  in_deck = state.zone[p] == Zone.DECK
  mask = in_deck & (zpos >= deck_count - n)
  order_key = deck_count - 1 - zpos  # top card first
  state = batch_discard(state, p, mask, order_key, do=do)
  return state._replace(
      winner=jnp.where(loss, (p + 1) % 2, state.winner).astype(jnp.int8)
  )


def apply_frozen(state: State, p, inst, duration, do) -> State:
  """status_util apply_frozen: duration is assigned (not stacked)."""
  return state._replace(
      frozen_dur=state.frozen_dur.at[p, inst].set(
          jnp.where(jnp.asarray(do), jnp.asarray(duration, jnp.int8),
                    state.frozen_dur[p, inst])
      )
  )


def apply_shocked(state: State, p, inst, duration, do) -> State:
  """status_util apply_shocked: duration assigned."""
  return state._replace(
      shocked_dur=state.shocked_dur.at[p, inst].set(
          jnp.where(jnp.asarray(do), jnp.asarray(duration, jnp.int8),
                    state.shocked_dur[p, inst])
      )
  )


def apply_effect_immune(state: State, p, inst, duration, do) -> State:
  """status_util apply_effect_immune: never downgrade permanent (-1)."""
  cur = state.effect_immune_dur[p, inst]
  new = jnp.where(cur == -1, cur, jnp.asarray(duration, jnp.int8))
  return state._replace(
      effect_immune_dur=state.effect_immune_dur.at[p, inst].set(
          jnp.where(jnp.asarray(do), new, cur)
      )
  )


def apply_health_modifier(state: State, p, inst, modifier, expires_eot, do) -> State:
  """status_util apply_health_modifier: UNclamped; stores requested value."""
  do = jnp.asarray(do)
  mod = jnp.asarray(modifier, jnp.int16)
  new_hp = (state.cur_hp[p, inst].astype(jnp.int16) + mod).astype(jnp.int8)
  state = state._replace(
      cur_hp=state.cur_hp.at[p, inst].set(
          jnp.where(do, new_hp, state.cur_hp[p, inst])
      )
  )
  delta = jnp.where(do, mod.astype(jnp.int8), 0)
  if expires_eot:
    return state._replace(hp_buff_eot=state.hp_buff_eot.at[p, inst].add(delta))
  return state._replace(hp_buff_perm=state.hp_buff_perm.at[p, inst].add(delta))


def apply_timed_tag_grant(state: State, p, inst, tag: int, phase: int, ticks,
                          do):
  """status_util apply_timed_tag_grant: dedupe identical (tag, phase, ticks)
  records, append at first empty slot, raise the keyword flag. Returns
  (state, applied)."""
  from azuki_jax.engine.helpers import GRANT_FLAG_FIELDS

  do = jnp.asarray(do)
  ticks = jnp.asarray(ticks, jnp.int8)
  tags_row = state.timed_tag[p, inst]
  dup = jnp.any(
      (tags_row == tag)
      & (state.timed_phase[p, inst] == phase)
      & (state.timed_ticks[p, inst] == ticks)
  )
  empty = tags_row == 0
  slot = jnp.argmax(empty)
  can_add = empty.any()
  add = do & ~dup & can_add
  state = state._replace(
      timed_tag=state.timed_tag.at[p, inst, slot].set(
          jnp.where(add, jnp.uint8(tag), state.timed_tag[p, inst, slot])
      ),
      timed_ticks=state.timed_ticks.at[p, inst, slot].set(
          jnp.where(add, ticks, state.timed_ticks[p, inst, slot])
      ),
      timed_phase=state.timed_phase.at[p, inst, slot].set(
          jnp.where(add, jnp.uint8(phase), state.timed_phase[p, inst, slot])
      ),
  )
  applied = do & (dup | can_add)
  field = GRANT_FLAG_FIELDS[tag]
  arr = getattr(state, field)
  state = state._replace(
      **{field: arr.at[p, inst].set(jnp.where(applied, True, arr[p, inst]))}
  )
  return state, applied


def apply_charge_grant(state: State, p, inst, phase: int, ticks, do) -> State:
  """status_util apply_charge_grant: timed Charge grant + cooldown clear."""
  from azuki_jax.engine.helpers import TAG_CHARGE, has_charge

  do = jnp.asarray(do)
  had = has_charge(state, p, inst)
  state, applied = apply_timed_tag_grant(state, p, inst, TAG_CHARGE, phase,
                                         ticks, do)
  clear = (applied | (do & had)) & (state.cooldown[p, inst] != 0)
  return state._replace(
      cooldown=state.cooldown.at[p, inst].set(
          jnp.where(clear, 0, state.cooldown[p, inst])
      )
  )


def destroy_card(state: State, p, inst, do) -> State:
  """card_utils discard_card (DESTROY reason): godmode check, when-destroyed
  trigger when leaving play, then discard. Weapons stay attached (C does not
  detach here)."""
  from azuki_jax.engine.helpers import discard, godmode_in_play
  from azuki_jax.engine.triggers import TIMING_WHEN_DESTROYED, queue_effect

  do = jnp.asarray(do) & ~godmode_in_play(state, p, inst)
  z = state.zone[p, inst]
  from_play = (
      (z == Zone.GARDEN) | (z == Zone.ALLEY) | (z == Zone.LEADER)
      | (z == Zone.ATTACHED)
  )
  state = queue_effect(state, p, inst, TIMING_WHEN_DESTROYED, do & from_play)
  return discard(state, p, inst, do=do)


def sacrifice_card(state: State, p, inst, do) -> State:
  """card_utils sacrifice_card: discard without when-destroyed (when-sacrificed
  abilities are not in the ported set)."""
  from azuki_jax.engine.helpers import discard

  return discard(state, p, inst, do=do)


def bottom_deck_from_play(state: State, p, inst, do) -> State:
  """deck_utils add_card_to_bottom_of_deck: reparent to deck zpos 0 (shift the
  rest up). C does NOT reset stats/taps/statuses here."""
  do = jnp.asarray(do)
  from_garden = state.zone[p, inst] == Zone.GARDEN
  from_alley = state.zone[p, inst] == Zone.ALLEY
  in_deck = state.zone[p] == Zone.DECK
  zpos_row = jnp.where(do & in_deck, state.zpos[p] + 1, state.zpos[p])
  zpos_row = zpos_row.at[inst].set(
      jnp.where(do, 0, zpos_row[inst]).astype(zpos_row.dtype)
  )
  zone_row = state.zone[p].at[inst].set(
      jnp.where(do, jnp.int8(Zone.DECK), state.zone[p, inst])
  )
  state = state._replace(
      zone=state.zone.at[p].set(zone_row),
      zpos=state.zpos.at[p].set(zpos_row),
  )
  # garden REMOVE event (STT02-012 observers fire on any reparent out of the
  # garden, including the AZK01-087 bottom-deck)
  from azuki_jax.engine.helpers import passive_zone_event, stt02_012_garden_event

  state = passive_zone_event(state, p, Zone.GARDEN, inst, False,
                             do=do & from_garden)
  state = passive_zone_event(state, p, Zone.ALLEY, inst, False,
                             do=do & from_alley)
  return stt02_012_garden_event(state, p, True, do & from_garden, inst)


def ikz_grant_tapped(state: State, p, do) -> State:
  """STT03-009 ramp: move top of IKZ pile to IKZ area, tapped."""
  from azuki_jax.engine.helpers import tap
  from azuki_jax.zones import move_top_n, top_instance

  do = jnp.asarray(do)
  zone_row, zpos_row = state.zone[p], state.zpos[p]
  top = top_instance(zone_row, zpos_row, Zone.IKZ_PILE)
  ok = do & (top >= 0)
  nzr, npr = move_top_n(
      zone_row, zpos_row, Zone.IKZ_PILE, Zone.IKZ_AREA, jnp.int32(1), 1
  )
  state = state._replace(
      zone=state.zone.at[p].set(jnp.where(ok, nzr, zone_row)),
      zpos=state.zpos.at[p].set(jnp.where(ok, npr, zpos_row)),
  )
  return tap(state, p, jnp.maximum(top, 0), do=ok)


def garden_seq_order(state: State, p, zone=Zone.GARDEN):
  """Instance ids of zone cards in insertion order (board_seq asc), padded.

  Mirrors ecs_get_ordered_children iteration for slot zones."""
  in_zone = state.zone[p] == zone
  key = jnp.where(in_zone, state.board_seq[p].astype(jnp.int32), 1 << 20)
  order = jnp.argsort(key)
  return order, in_zone


def return_to_hand(state: State, p, inst, do) -> State:
  """card_utils return_card_to_hand (from play): discard weapons, reset
  stats/taps/statuses, append to hand, counters; for returns FROM PLAY, queue
  AWhenReturnedToHand for the returned card itself and then for the garden
  observers of both players (azk_trigger_return_to_hand_observers)."""
  from azuki_jax.engine.helpers import (
      clear_temporary_state,
      discard_equipped_weapons,
      godmode_in_play,
  )
  from azuki_jax.zones import zone_count

  do = jnp.asarray(do) & ~godmode_in_play(state, p, inst)
  z = state.zone[p, inst]
  from_play = (z == Zone.GARDEN) | (z == Zone.ALLEY)

  state = discard_equipped_weapons(state, p, inst, do=do)

  def_id = state.def_id[p, inst]
  base_atk = jnp.where(def_id >= 0, _np(cards.BASE_ATK)[def_id], 0).astype(jnp.int8)
  base_hp = jnp.where(def_id >= 0, _np(cards.BASE_HP)[def_id], 0).astype(jnp.int8)
  hand_count = zone_count(state.zone[p], Zone.HAND)

  def set2(arr, value):
    return arr.at[p, inst].set(jnp.where(do, value, arr[p, inst]))

  state = state._replace(
      tapped=set2(state.tapped, False),
      cooldown=set2(state.cooldown, 0),
      cur_atk=set2(state.cur_atk, base_atk),
      cur_hp=set2(state.cur_hp, base_hp),
      atk_buff_perm=set2(state.atk_buff_perm, 0),
      atk_buff_eot=set2(state.atk_buff_eot, 0),
      hp_buff_perm=set2(state.hp_buff_perm, 0),
      hp_buff_eot=set2(state.hp_buff_eot, 0),
      zone=set2(state.zone, jnp.int8(Zone.HAND)),
      zpos=set2(state.zpos, hand_count.astype(jnp.int8)),
  )
  state = clear_temporary_state(state, p, inst, do=do)

  # garden REMOVE latch event (STT02-012 observers)
  from azuki_jax.engine.helpers import passive_zone_event, stt02_012_garden_event

  state = passive_zone_event(state, p, Zone.GARDEN, inst, False,
                             do=do & (z == Zone.GARDEN))
  state = passive_zone_event(state, p, Zone.ALLEY, inst, False,
                             do=do & (z == Zone.ALLEY))
  state = stt02_012_garden_event(state, p, True, do & (z == Zone.GARDEN), inst)

  # --- AWhenReturnedToHand triggers (card_utils.c return_card_to_hand) ---
  from azuki_jax.engine.helpers import can_tap
  from azuki_jax.engine.triggers import TIMING_WHEN_RETURNED_TO_HAND, queue_effect

  bounced = do & from_play
  # 1) the returned card's own when-returned abilities (queued after the move;
  #    timing filter inside queue_effect mirrors collect_card_timed_abilities)
  state = queue_effect(state, p, inst, TIMING_WHEN_RETURNED_TO_HAND, bounced)
  # 2) per-turn counter (C increments before the observer scan)
  state = state._replace(
      returned_to_hand_turn=state.returned_to_hand_turn.at[p].add(
          bounced.astype(jnp.uint8)
      )
  )
  # 3) observer scan: both players' gardens in insertion order; C checks
  #    def->validate at QUEUE time here. The only registered card with this
  #    timing is STT02-010 (garden + untapped[cooldown ignored] + deck > 0) —
  #    its validate is inlined; any future card with the timing queues
  #    unfiltered and surfaces via the unimplemented-coverage counter.
  stt02_010 = cards.CODE_TO_ID["STT02-010"]
  for gp in (0, 1):
    order, in_garden = garden_seq_order(state, gp)
    deck_ok = zone_count(state.zone[gp], Zone.DECK) > 0
    for k in range(5):
      gi = order[k]
      is_010 = state.def_id[gp, gi] == stt02_010
      validate_ok = ~is_010 | (
          can_tap(state, gp, gi, ignore_cooldown=True) & deck_ok
      )
      state = queue_effect(
          state, gp, gi, TIMING_WHEN_RETURNED_TO_HAND,
          bounced & in_garden[gi] & validate_ok,
      )
  return state


# ---------------------------------------------------------------------------
# ctx accessors
# ---------------------------------------------------------------------------

def _ctx(state: State):
  owner = jnp.maximum(state.ab_owner.astype(jnp.int32), 0)
  src = jnp.maximum(state.ab_source.astype(jnp.int32), 0)
  return owner, src


def _eff_target(state: State, k=0):
  tp = jnp.maximum(state.ab_eff_target_players[k].astype(jnp.int32), 0)
  ti = jnp.maximum(state.ab_eff_targets[k].astype(jnp.int32), 0)
  has = state.ab_eff_targets[k] >= 0
  return tp, ti, has


def _cost_target(state: State, k=0):
  tp = jnp.maximum(state.ab_cost_target_players[k].astype(jnp.int32), 0)
  ti = jnp.maximum(state.ab_cost_targets[k].astype(jnp.int32), 0)
  has = state.ab_cost_targets[k] >= 0
  return tp, ti, has


def already_selected(state: State, scope_is_cost, tp, ti):
  """C card hooks' duplicate-target rejection over ctx selections."""
  from azuki_jax.constants import MAX_ABILITY_SELECTION

  sel = jnp.where(
      scope_is_cost, state.ab_cost_selected, state.ab_eff_selected
  ).astype(jnp.int32)
  targets = jnp.where(scope_is_cost, state.ab_cost_targets, state.ab_eff_targets)
  players = jnp.where(
      scope_is_cost, state.ab_cost_target_players, state.ab_eff_target_players
  )
  k = jnp.arange(MAX_ABILITY_SELECTION)
  return jnp.any(
      (k < sel)
      & (targets.astype(jnp.int32) == ti)
      & (players.astype(jnp.int32) == tp)
  )


# ---------------------------------------------------------------------------
# Card hooks
# ---------------------------------------------------------------------------

def _noop(state: State) -> State:
  return state


def _stt02_007_effects(state: State) -> State:
  owner, _ = _ctx(state)
  return draw_with_deckout(state, owner, 1, True)


def _azk01_004_effects(state: State) -> State:
  owner, src = _ctx(state)
  return apply_attack_modifier(state, owner, src, 1, expires_eot=True, do=True)


def _azk01_005_effects(state: State) -> State:
  tp, ti, has = _eff_target(state, 0)
  picked = state.ab_eff_selected > 0
  return deal_effect_damage(state, tp, ti, 1, do=has & picked)


def _azk01_006_effects(state: State) -> State:
  owner, src = _ctx(state)
  return return_to_hand(state, owner, src, do=True)


# tranche-1 validate()/target-validator hooks
def _azk01_005_validate(state: State, owner, src) -> jax.Array:
  from azuki_jax.engine.helpers import is_effect_immune

  n = state.zone.shape[1]
  idx = jnp.arange(n)
  opp = (owner + 1) % 2
  enemy_garden = state.zone[opp] == Zone.GARDEN
  non_immune = ~jax.vmap(lambda i: is_effect_immune(state, opp, i))(idx)
  return jnp.any(enemy_garden & non_immune)


def _azk01_006_validate(state: State, owner, src) -> jax.Array:
  return state.zone[owner, jnp.maximum(src, 0)] == Zone.GARDEN


def _azk01_005_target(state: State, scope_is_cost, owner, tp, ti) -> jax.Array:
  from azuki_jax.engine.helpers import is_effect_immune

  opp = (owner + 1) % 2
  return scope_is_cost | (
      (tp == opp)
      & (state.zone[tp, ti] == Zone.GARDEN)
      & ~is_effect_immune(state, tp, ti)
  )


# ---------------------------------------------------------------------------
# Registry + dispatch tables
# ---------------------------------------------------------------------------

_HOOKS: dict[str, dict] = {}


def register(code: str, *, effects=None, costs=None, validate=None,
             target_validator=None, on_cost_paid=None,
             on_selection_complete=None, selection_target_validator=None,
             selection_complete_if_still=False) -> None:
  """Register a card's JAX hooks (one call per card, from any module).

  selection_complete_if_still: the card's C on_selection_complete uses
  azk_move_picked_selection_cards_to_hand_if_still_in_selection. Under C's
  deferred ops a completing TO_GARDEN/TO_ALLEY/TO_EQUIP pick still has the
  stale selection parent when that check runs, so the hook bounces the picked
  card to hand (overriding the placement at flush)."""
  if code in _HOOKS:
    raise ValueError(f"{code} already registered")
  _HOOKS[code] = {
      "effects": effects,
      "costs": costs,
      "validate": validate,
      "target_validator": target_validator,
      "on_cost_paid": on_cost_paid,
      "on_selection_complete": on_selection_complete,
      "selection_target_validator": selection_target_validator,
      "selection_complete_if_still": selection_complete_if_still,
  }


register("STT02-007", effects=_stt02_007_effects)
register("AZK01-004", effects=_azk01_004_effects)
register(
    "AZK01-005",
    effects=_azk01_005_effects,
    validate=_azk01_005_validate,
    target_validator=_azk01_005_target,
)
register("AZK01-006", effects=_azk01_006_effects, validate=_azk01_006_validate)

# per-set card modules register themselves on import (add new sets here)
from azuki_jax.abilities import cards_batch1  # noqa: E402,F401  (registers)
from azuki_jax.abilities import cards_batch2  # noqa: E402,F401  (registers)
from azuki_jax.abilities import cards_batch3  # noqa: E402,F401  (registers)
from azuki_jax.abilities import cards_batch4  # noqa: E402,F401  (registers)
from azuki_jax.abilities import cards_batch_passives  # noqa: E402,F401  (registers)

_TRUE_VALIDATE = lambda state, owner, src: jnp.asarray(True)  # noqa: E731
_TRUE_TARGET = lambda state, scope, owner, tp, ti: jnp.asarray(True)  # noqa: E731

_TRUE_SEL_TARGET = lambda state, owner, inst: jnp.asarray(True)  # noqa: E731

IMPL_INDEX = np.zeros(cards.CARD_DEF_COUNT, np.int32)  # 0 = not implemented
HAS_APPLY_COSTS = np.zeros(cards.CARD_DEF_COUNT, np.bool_)
HAS_ON_COST_PAID = np.zeros(cards.CARD_DEF_COUNT, np.bool_)
SEL_COMPLETE_IF_STILL = np.zeros(cards.CARD_DEF_COUNT, np.bool_)
_EFFECT_FNS = [_noop]
_COST_FNS = [_noop]
_VALIDATE_FNS = [_TRUE_VALIDATE]
_TARGET_FNS = [_TRUE_TARGET]
_ON_COST_PAID_FNS = [_noop]
_ON_SEL_COMPLETE_FNS = [_noop]
_SEL_TARGET_FNS = [_TRUE_SEL_TARGET]
for _slot, (_code, _hooks) in enumerate(sorted(_HOOKS.items()), start=1):
  _def_id = cards.CODE_TO_ID[_code]
  IMPL_INDEX[_def_id] = _slot
  tables.IMPLEMENTED[_def_id] = True
  HAS_APPLY_COSTS[_def_id] = _hooks.get("costs") is not None
  HAS_ON_COST_PAID[_def_id] = _hooks.get("on_cost_paid") is not None
  SEL_COMPLETE_IF_STILL[_def_id] = bool(
      _hooks.get("selection_complete_if_still")
  )
  _EFFECT_FNS.append(_hooks.get("effects") or _noop)
  _COST_FNS.append(_hooks.get("costs") or _noop)
  _VALIDATE_FNS.append(_hooks.get("validate") or _TRUE_VALIDATE)
  _TARGET_FNS.append(_hooks.get("target_validator") or _TRUE_TARGET)
  _ON_COST_PAID_FNS.append(_hooks.get("on_cost_paid") or _noop)
  _ON_SEL_COMPLETE_FNS.append(_hooks.get("on_selection_complete") or _noop)
  _SEL_TARGET_FNS.append(
      _hooks.get("selection_target_validator") or _TRUE_SEL_TARGET
  )


def _slot_of(state: State, def_id):
  return jnp.where(def_id >= 0, _np(IMPL_INDEX)[jnp.maximum(def_id, 0)], 0)


def dispatch_apply_effects(state: State) -> State:
  owner, src = _ctx(state)
  return jax.lax.switch(
      _slot_of(state, state.def_id[owner, src]), _EFFECT_FNS, state
  )


def dispatch_apply_costs(state: State) -> State:
  owner, src = _ctx(state)
  return jax.lax.switch(
      _slot_of(state, state.def_id[owner, src]), _COST_FNS, state
  )


def validate_card(state: State, def_id, owner, src) -> jax.Array:
  """def->validate dispatch; True when the card has no validate hook."""
  return jax.lax.switch(
      _slot_of(state, def_id),
      [lambda s, o, c, fn=fn: fn(s, o, c) for fn in _VALIDATE_FNS],
      state, owner, src,
  )


def target_validator(state: State, def_id, scope_is_cost, owner, tp, ti) -> jax.Array:
  """validate_cost_target / validate_effect_target dispatch."""
  return jax.lax.switch(
      _slot_of(state, def_id),
      [lambda s, sc, o, p, i, fn=fn: fn(s, sc, o, p, i) for fn in _TARGET_FNS],
      state, scope_is_cost, owner, tp, ti,
  )


def dispatch_on_cost_paid(state: State) -> State:
  owner, src = _ctx(state)
  return jax.lax.switch(
      _slot_of(state, state.def_id[owner, src]), _ON_COST_PAID_FNS, state
  )


def dispatch_on_selection_complete(state: State) -> State:
  owner, src = _ctx(state)
  return jax.lax.switch(
      _slot_of(state, state.def_id[owner, src]), _ON_SEL_COMPLETE_FNS, state
  )


def selection_target_validator(state: State, def_id, owner, inst) -> jax.Array:
  """validate_selection_target dispatch (True when the card has none)."""
  return jax.lax.switch(
      _slot_of(state, def_id),
      [lambda s, o, i, fn=fn: fn(s, o, i) for fn in _SEL_TARGET_FNS],
      state, owner, inst,
  )
