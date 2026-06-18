"""User-action application (mulligan/main/response systems).

Mirrors src/systems/{mulligan,main,response}_phase.c + zone_util summon /
weapon_util attach / combat_util attack. Vanilla scope: ability triggers are
queued via the trigger table, but only implemented cards enqueue (others are
asserted away by deck choice during early verification).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from azuki_jax import cards
from azuki_jax.constants import (
    GARDEN_SIZE,
    INITIAL_DRAW_COUNT,
    MAX_DECK_SIZE,
    Act,
    Phase,
    Zone,
)
from azuki_jax.engine import ikz
from azuki_jax.engine.helpers import (
    attr_force_tapped,
    card_at_slot,
    gate_instance,
    has_charge,
    hand_instance,
    leader_instance,
    discard,
    tap,
)
from azuki_jax.engine.validate import effective_play_cost
from azuki_jax.rng import fisher_yates
from azuki_jax.state import State
from azuki_jax.zones import (
    list_zone_order,
    move_top_n,
    zone_count,
)


def _np(table):
  return jnp.asarray(table)


# ---------------------------------------------------------------------------
# Mulligan
# ---------------------------------------------------------------------------

def apply_mulligan(state: State, action_type) -> State:
  """HandleMulliganAction with the engine's deferred-ECS semantics:

  Both zone moves and the shuffle read the PRE-ACTION deck snapshot, then
  flush in order. Net effect of MULLIGAN_SHUFFLE:
  - new hand = old deck's top 7 (top first)
  - new deck = the surviving 36 old-deck cards in their order under a
    Fisher-Yates of the full 43-card snapshot (42 RNG draws), with the old
    hand appended (reversed: h6..h0) at the deck top.
  NOOP keeps everything. Then advance the mulligan counter."""
  p = state.active_player
  do_shuffle = action_type == Act.MULLIGAN_SHUFFLE

  zone_row, zpos_row = state.zone[p], state.zpos[p]
  deck_count = jnp.sum(zone_row == Zone.DECK, dtype=jnp.int32)

  # snapshot order: order[k] = instance with deck zpos k
  order = list_zone_order(zone_row, zpos_row, Zone.DECK, MAX_DECK_SIZE)
  shuffled, rng = fisher_yates(order, deck_count, state.rng_state)

  n = zone_row.shape[0]
  idx = jnp.arange(n)
  # per-INSTANCE position within the shuffled snapshot
  pos_in_shuffle = jnp.full((n,), n, jnp.int32)
  pos_in_shuffle = pos_in_shuffle.at[shuffled].set(
      jnp.arange(MAX_DECK_SIZE), mode="drop"
  )

  in_deck = zone_row == Zone.DECK
  in_hand = zone_row == Zone.HAND
  old_zpos = zpos_row.astype(jnp.int32)

  drawn = in_deck & (old_zpos >= deck_count - INITIAL_DRAW_COUNT)
  survivor = in_deck & ~drawn

  # survivors: zpos = rank of their shuffled position among survivors
  inst_pos = jnp.where(survivor, pos_in_shuffle[idx], n + old_zpos)
  surv_rank = jnp.sum(
      (inst_pos[None, :] < inst_pos[:, None]) & survivor[None, :], axis=1
  ).astype(jnp.int32)

  survivor_count = deck_count - INITIAL_DRAW_COUNT
  new_zpos = old_zpos
  new_zone = zone_row
  # drawn cards -> hand, top first: zpos = (deck_count-1) - old_zpos
  new_zpos = jnp.where(drawn, (deck_count - 1) - old_zpos, new_zpos)
  new_zone = jnp.where(drawn, jnp.int8(Zone.HAND), new_zone)
  # survivors keep DECK with shuffled rank
  new_zpos = jnp.where(survivor, surv_rank, new_zpos)
  # old hand -> deck top, reversed: zpos = survivor_count + (6 - hand_zpos)
  new_zpos = jnp.where(
      in_hand, survivor_count + (INITIAL_DRAW_COUNT - 1) - old_zpos, new_zpos
  )
  new_zone = jnp.where(in_hand, jnp.int8(Zone.DECK), new_zone)

  zone_row = jnp.where(do_shuffle, new_zone, zone_row)
  zpos_row = jnp.where(do_shuffle, new_zpos.astype(zpos_row.dtype), zpos_row)
  rng = jnp.where(do_shuffle, rng, state.rng_state)

  state = state._replace(
      zone=state.zone.at[p].set(zone_row),
      zpos=state.zpos.at[p].set(zpos_row),
      rng_state=rng,
  )

  # handle_phase_transition
  done = state.mulligan_done + 1
  finished = done >= 2
  return state._replace(
      mulligan_done=jnp.where(finished, 0, done).astype(jnp.uint8),
      active_player=jnp.where(
          finished, state.starting_player, (state.active_player + 1) % 2
      ).astype(jnp.int8),
      phase=jnp.where(
          finished, jnp.int8(Phase.START_OF_TURN), state.phase
      ),
  )


# ---------------------------------------------------------------------------
# Plays
# ---------------------------------------------------------------------------

def _enter_board_slot(state: State, p, inst, zone, slot, do) -> State:
  """insert_card_into_zone_index: displace when full, set slot, garden tap
  rules (cooldown unless Charge; AttrGardenForceTapped enters tapped)."""
  old_zone = state.zone[p, inst]
  from_garden = old_zone == Zone.GARDEN
  from_alley = old_zone == Zone.ALLEY
  displaced = card_at_slot(state, p, zone, slot)
  full = zone_count(state.zone[p], zone) >= GARDEN_SIZE
  do_displace = do & (displaced >= 0) & full
  state = discard(
      state, p, jnp.maximum(displaced, 0),
      reason_replacement=True, ignore_godmode=True, do=do_displace,
  )

  # remove from hand/alley (list zone compaction happens for HAND; ALLEY is a
  # slot zone — clearing is implicit by moving the card)
  from azuki_jax.engine.helpers import _detach_from_location

  state = _detach_from_location(state, p, inst, do)

  zone_arr = state.zone.at[p, inst].set(
      jnp.where(do, jnp.int8(zone), state.zone[p, inst])
  )
  zpos_arr = state.zpos.at[p, inst].set(
      jnp.where(do, jnp.asarray(slot, jnp.int8), state.zpos[p, inst])
  )
  seq_arr = state.board_seq.at[p, inst].set(
      jnp.where(do, state.seq_counter, state.board_seq[p, inst])
  )
  state = state._replace(
      zone=zone_arr,
      zpos=zpos_arr,
      board_seq=seq_arr,
      seq_counter=(state.seq_counter + jnp.where(do, 1, 0)).astype(jnp.int16),
  )

  is_garden = jnp.asarray(zone, jnp.int8) == jnp.int8(Zone.GARDEN)
  enters_tapped = attr_force_tapped(state, p, inst)
  new_tapped = jnp.where(
      do & is_garden,
      enters_tapped | state.tapped[p, inst],
      state.tapped[p, inst],
  )
  new_cooldown = jnp.where(
      do & is_garden,
      (~has_charge(state, p, inst)).astype(jnp.uint8),
      state.cooldown[p, inst],
  )
  state = state._replace(
      tapped=state.tapped.at[p, inst].set(new_tapped),
      cooldown=state.cooldown.at[p, inst].set(new_cooldown),
  )

  # STT03-013 gains Taunt on garden entry
  is_stt03_013 = state.def_id[p, inst] == cards.CODE_TO_ID["STT03-013"]
  grant = state.grant_taunt.at[p, inst].set(
      jnp.where(do & is_garden & is_stt03_013, True, state.grant_taunt[p, inst])
  )
  state = state._replace(grant_taunt=grant)

  # Watched zone events fire after the displacement's removal event, as in C.
  from azuki_jax.engine.helpers import (
      MAX_PASSIVE_BUFF_QUEUE,
      passive_zone_event,
      self_passive_event_count,
      stt02_012_garden_event,
  )

  state = passive_zone_event(state, p, Zone.GARDEN, inst, False,
                             do=do & from_garden)
  state = passive_zone_event(state, p, Zone.ALLEY, inst, False,
                             do=do & from_alley)
  count_before_add = state.passive_queue_count.astype(jnp.int16)
  add_self_count = self_passive_event_count(state, p, zone)
  state = passive_zone_event(state, p, zone, inst, True,
                             do=do & ((zone == Zone.GARDEN) | (zone == Zone.ALLEY)))

  # C's garden-add observers put STT02-012 before the final self-passive
  # observer. Replay the STT02 event against that queue count, then restore the
  # final occupancy as if both the STT02 event and the self-passive batch ran.
  adjusted_count = jnp.minimum(
      count_before_add + jnp.maximum(add_self_count - 1, 0),
      MAX_PASSIVE_BUFF_QUEUE,
  ).astype(jnp.uint8)
  stt_input = state._replace(
      passive_queue_count=jnp.where(do & is_garden, adjusted_count,
                                    state.passive_queue_count)
  )
  stt_state = stt02_012_garden_event(
      stt_input, p, False, do & is_garden, inst
  )
  stt_queued = (
      stt_state.passive_queue_count.astype(jnp.int16)
      > adjusted_count.astype(jnp.int16)
  ).astype(jnp.int16)
  final_count = jnp.minimum(
      count_before_add + add_self_count + stt_queued,
      MAX_PASSIVE_BUFF_QUEUE,
  ).astype(jnp.uint8)
  return stt_state._replace(
      passive_queue_count=jnp.where(do & is_garden, final_count,
                                    stt_state.passive_queue_count)
  )


def apply_play_entity(state: State, placement_zone, hand_index, slot,
                      use_token, do=True) -> State:
  p = state.active_player
  inst = hand_instance(state, p, hand_index)
  safe = jnp.maximum(inst, 0)
  do = jnp.asarray(do) & (inst >= 0)

  cost = effective_play_cost(state, p, safe)
  state = ikz.pay(state, p, cost, use_token, do=do)
  state = _enter_board_slot(state, p, safe, placement_zone, slot, do)

  is_garden = jnp.asarray(placement_zone, jnp.int8) == jnp.int8(Zone.GARDEN)
  state = state._replace(
      entities_played_garden_turn=state.entities_played_garden_turn.at[p].add(
          (do & is_garden).astype(jnp.uint8)
      ),
      entities_played_alley_turn=state.entities_played_alley_turn.at[p].add(
          (do & ~is_garden).astype(jnp.uint8)
      ),
      cards_played_turn=state.cards_played_turn.at[p].add(do.astype(jnp.uint8)),
      next_play_cost_reduction=state.next_play_cost_reduction.at[p].set(
          jnp.where(do, 0, state.next_play_cost_reduction[p])
      ),
  )
  from azuki_jax.engine.triggers import queue_on_play, queue_enter_garden

  state = queue_enter_garden(state, p, safe, do=do & is_garden)
  state = queue_on_play(state, p, safe, do=do)
  return state


def apply_gate_portal(state: State, alley_index, garden_index, do=True) -> State:
  """gate_card_into_garden: place, tap gate, queue enter-garden, then BEGIN
  the gate's portal ability immediately (C runs it inline with scratch —
  before the queued enter-garden trigger processes)."""
  from azuki_jax.abilities import cards_impl, runtime, tables as ab_tables
  from azuki_jax.engine.triggers import queue_enter_garden

  p = state.active_player
  alley_card = card_at_slot(state, p, Zone.ALLEY, alley_index)
  safe = jnp.maximum(alley_card, 0)
  do = jnp.asarray(do) & (alley_card >= 0)

  # The gate (a permanent fixture) and the portaled instance are known before
  # placement. C inserts the portaled card via a DEFERRED flecs op, so the
  # gate portal ability validate runs against the PRE-placement garden (the
  # displaced slot occupant is still present, the portaled card not yet in).
  # Mirror that: set the scratch + compute card_ok BEFORE _enter_board_slot.
  gate = gate_instance(state, p)
  safe_gate = jnp.maximum(gate, 0)
  gate_def = state.def_id[p, safe_gate]
  has_gate_ability = (gate >= 0) & (gate_def >= 0) & jnp.where(
      gate_def >= 0, _np(ab_tables.HAS_ABILITY)[jnp.maximum(gate_def, 0)], False
  )
  implemented = jnp.where(
      gate_def >= 0, _np(ab_tables.IMPLEMENTED)[jnp.maximum(gate_def, 0)], False
  )
  begin = do & has_gate_ability & implemented
  state = state._replace(
      ab_scratch=state.ab_scratch.at[0]
      .set(jnp.where(begin, safe.astype(jnp.int16), state.ab_scratch[0]))
      .at[1]
      .set(jnp.where(begin, jnp.asarray(garden_index, jnp.int16), state.ab_scratch[1]))
      .at[2]
      .set(jnp.where(begin, jnp.int16(1), state.ab_scratch[2])),  # kind=GATE_PORTAL
  )
  # validate against the PRE-placement garden (matches C's deferred insert)
  card_ok = cards_impl.validate_card(state, gate_def, p, safe_gate)
  count_state = state._replace(
      ab_source=safe_gate.astype(jnp.int8),
      ab_owner=p.astype(jnp.int8),
  )
  pre_effect_available = runtime.count_targets(
      count_state, gate_def, jnp.bool_(False), p.astype(jnp.int32)
  )

  # coverage counter for unported gates
  state = state._replace(
      ab_scratch=state.ab_scratch.at[3].add(
          jnp.where(do & has_gate_ability & ~implemented, 1, 0).astype(jnp.int16)
      )
  )

  # now place the portaled card (displacing the slot occupant when full), tap
  # the gate, and queue its enter-garden trigger. begin_ability then runs on
  # the POST-placement garden; with card_ok already decided against the
  # pre-placement state, the confirmation is entered iff C would, and the
  # cost selection re-enumerates post-placement (fizzling if the now-displaced
  # occupant was the only valid sacrifice).
  state = _enter_board_slot(state, p, safe, Zone.GARDEN, garden_index, do)
  state = tap(state, p, safe_gate, do=do & (gate >= 0))
  state = queue_enter_garden(state, p, safe, do=do)

  # validate failure restores the saved (clear) ctx in C — drop the scratch so
  # it cannot leak into a later ability (ability_system.c:1711-1714)
  drop = begin & ~card_ok
  state = state._replace(
      ab_scratch=state.ab_scratch.at[0:3].set(
          jnp.where(drop, jnp.zeros(3, jnp.int16), state.ab_scratch[0:3])
      )
  )
  return runtime.begin_ability(
      state, p, safe_gate.astype(jnp.int32), runtime.BEGIN_GATE_PORTAL,
      begin & card_ok,
      effect_available_override=pre_effect_available,
  )


def apply_attach_weapon(state: State, hand_index, entity_index, use_token,
                        do=True) -> State:
  p = state.active_player
  weapon = hand_instance(state, p, hand_index)
  safe_weapon = jnp.maximum(weapon, 0)
  do = jnp.asarray(do) & (weapon >= 0)

  target_is_leader = entity_index == GARDEN_SIZE
  leader = leader_instance(state, p)
  garden_target = card_at_slot(state, p, Zone.GARDEN, entity_index)
  target = jnp.where(target_is_leader, leader, garden_target)
  safe_target = jnp.maximum(target, 0)
  do = do & (target >= 0)

  # remove from hand, set ATTACHED with attach order = current weapon count
  from azuki_jax.engine.helpers import _detach_from_location, weapons_of

  weapon_count = jnp.sum(weapons_of(state, p, safe_target), dtype=jnp.int32)
  state = _detach_from_location(state, p, safe_weapon, do)
  state = state._replace(
      zone=state.zone.at[p, safe_weapon].set(
          jnp.where(do, jnp.int8(Zone.ATTACHED), state.zone[p, safe_weapon])
      ),
      zpos=state.zpos.at[p, safe_weapon].set(
          jnp.where(do, weapon_count.astype(jnp.int8), state.zpos[p, safe_weapon])
      ),
      attached_to=state.attached_to.at[p, safe_weapon].set(
          jnp.where(do, safe_target.astype(jnp.int8), state.attached_to[p, safe_weapon])
      ),
  )

  # apply_weapon_attack_bonus: host atk += weapon cur_atk (>= 0)
  weapon_atk = state.cur_atk[p, safe_weapon].astype(jnp.int16)
  host_atk = state.cur_atk[p, safe_target].astype(jnp.int16)
  new_atk = jnp.maximum(host_atk + weapon_atk, 0).astype(jnp.int8)
  state = state._replace(
      cur_atk=state.cur_atk.at[p, safe_target].set(
          jnp.where(do, new_atk, state.cur_atk[p, safe_target])
      )
  )

  # AZK01-018 equipped combat modifier (incoming -1, leaders only)
  is_018 = state.def_id[p, safe_weapon] == cards.CODE_TO_ID["AZK01-018"]
  apply_mod = do & is_018 & target_is_leader
  state = state._replace(
      cmb_in_perm=state.cmb_in_perm.at[p, safe_target].add(
          jnp.where(apply_mod, -1, 0).astype(jnp.int8)
      )
  )

  cost = effective_play_cost(state, p, safe_weapon)
  state = ikz.pay(state, p, cost, use_token, do=do)

  state = state._replace(
      cards_played_turn=state.cards_played_turn.at[p].add(do.astype(jnp.uint8)),
      next_play_cost_reduction=state.next_play_cost_reduction.at[p].set(
          jnp.where(do, 0, state.next_play_cost_reduction[p])
      ),
  )

  from azuki_jax.engine.triggers import queue_on_play, queue_when_equipped

  state = queue_on_play(state, p, safe_weapon, do=do)
  state = queue_when_equipped(state, p, safe_weapon, do=do)
  state = queue_when_equipped(state, p, safe_target, do=do)
  return state


def apply_attack(state: State, attacker_index, defender_index, do=True) -> State:
  """combat_util.c attack(): set combat state, tap attacker, queue
  when-attacking triggers (attacker + weapons) + AZK01-034 redirect."""
  p = state.active_player
  opp = (p + 1) % 2

  attacker_is_leader = attacker_index == GARDEN_SIZE
  attacker = jnp.where(
      attacker_is_leader,
      leader_instance(state, p),
      card_at_slot(state, p, Zone.GARDEN, attacker_index),
  )
  safe_attacker = jnp.maximum(attacker, 0)

  defender_is_leader = defender_index == GARDEN_SIZE
  is_garden_target = defender_index < GARDEN_SIZE
  defender = jnp.where(
      defender_is_leader,
      leader_instance(state, opp),
      jnp.where(
          is_garden_target,
          card_at_slot(state, opp, Zone.GARDEN, defender_index),
          card_at_slot(state, opp, Zone.ALLEY, defender_index - (GARDEN_SIZE + 1)),
      ),
  )
  safe_defender = jnp.maximum(defender, 0)
  do = jnp.asarray(do) & (attacker >= 0) & (defender >= 0)

  state = tap(state, p, safe_attacker, do=do)
  state = state._replace(
      combat_attacker=jnp.where(do, safe_attacker.astype(jnp.int8), state.combat_attacker),
      combat_defender=jnp.where(do, safe_defender.astype(jnp.int8), state.combat_defender),
      combat_defender_player=jnp.where(do, jnp.int8(opp), state.combat_defender_player),
      combat_intercepted=jnp.where(do, False, state.combat_intercepted),
      combat_attacker_is_leader=jnp.where(
          do, attacker_is_leader, state.combat_attacker_is_leader
      ),
  )

  from azuki_jax.engine.triggers import queue_when_attacking_chain

  state = queue_when_attacking_chain(state, p, safe_attacker, do=do)
  # NOTE: phase transition out of MAIN happens in the phase gate (auto loop).
  return state


def apply_declare_defender(state: State, garden_index, do=True) -> State:
  p = state.active_player
  card = card_at_slot(state, p, Zone.GARDEN, garden_index)
  safe = jnp.maximum(card, 0)
  do = jnp.asarray(do) & (card >= 0)
  state = tap(state, p, safe, do=do)  # taps even on cooldown
  return state._replace(
      combat_defender=jnp.where(do, safe.astype(jnp.int8), state.combat_defender),
      combat_intercepted=jnp.where(do, True, state.combat_intercepted),
  )


def _ability_cost_targets_sufficient(state: State, p, inst) -> jax.Array:
  """azk_trigger_*_ability precondition: cost_req.min == 0 or enough valid
  cost targets exist (counted with the source/owner ctx pre-set, as C passes
  the source card explicitly to the validators)."""
  from azuki_jax.abilities import runtime, tables as ab_tables

  def_id = state.def_id[p, inst]
  cost_min = jnp.where(
      def_id >= 0, _np(ab_tables.COST_MIN)[jnp.maximum(def_id, 0)], 0
  )
  probe = state._replace(
      ab_source=jnp.asarray(inst, jnp.int8),
      ab_owner=jnp.asarray(p, jnp.int8),
  )
  avail = runtime.count_targets(
      probe, def_id, jnp.bool_(True), jnp.asarray(p, jnp.int32)
  )
  return (cost_min == 0) | (avail >= cost_min)


def apply_play_spell(state: State, hand_index, use_token, do=True) -> State:
  """handle_play_spell_from_hand: pay, move to discard, counters, trigger.

  azk_trigger_spell_ability refuses to begin when the cost requirement has
  fewer valid targets than min (spell is consumed with no effect)."""
  from azuki_jax.abilities import runtime
  from azuki_jax.engine.helpers import discard

  p = state.active_player
  inst = hand_instance(state, p, hand_index)
  safe = jnp.maximum(inst, 0)
  do = jnp.asarray(do) & (inst >= 0)

  cost = effective_play_cost(state, p, safe)
  state = ikz.pay(state, p, cost, use_token, do=do)
  state = discard(state, p, safe, do=do)
  state = state._replace(
      cards_played_turn=state.cards_played_turn.at[p].add(do.astype(jnp.uint8)),
      next_play_cost_reduction=state.next_play_cost_reduction.at[p].set(
          jnp.where(do, 0, state.next_play_cost_reduction[p])
      ),
  )
  can_begin = _ability_cost_targets_sufficient(state, p, safe)
  return runtime.begin_ability(
      state, p, safe, runtime.BEGIN_SPELL, do & can_begin
  )


def apply_activate_garden_or_leader(state: State, slot, use_token, do=True) -> State:
  """handle_activate_garden_or_leader_ability: pay ability ikz cost, begin."""
  from azuki_jax.abilities import runtime, tables as ab_tables
  from azuki_jax.engine.helpers import leader_instance as _leader

  p = state.active_player
  inst = jnp.where(
      slot == GARDEN_SIZE,
      _leader(state, p),
      card_at_slot(state, p, Zone.GARDEN, slot),
  )
  safe = jnp.maximum(inst, 0)
  do = jnp.asarray(do) & (inst >= 0)

  def_id = state.def_id[p, safe]
  cost = jnp.where(
      def_id >= 0, _np(ab_tables.ABILITY_IKZ_COST)[jnp.maximum(def_id, 0)], 0
  )
  state = ikz.pay(state, p, cost, use_token, do=do)

  in_response = state.phase == Phase.RESPONSE_WINDOW
  kind = jnp.where(in_response, runtime.BEGIN_RESPONSE, runtime.BEGIN_MAIN)
  can_begin = _ability_cost_targets_sufficient(state, p, safe)
  return runtime.begin_ability(state, p, safe, kind, do & can_begin)


def apply_activate_alley(state: State, slot, do=True) -> State:
  """handle_activate_alley_ability (no ikz payment in C handler path —
  alley validator doesn't fetch payment)."""
  from azuki_jax.abilities import runtime

  p = state.active_player
  inst = card_at_slot(state, p, Zone.ALLEY, slot)
  safe = jnp.maximum(inst, 0)
  do = jnp.asarray(do) & (inst >= 0)
  in_response = state.phase == Phase.RESPONSE_WINDOW
  kind = jnp.where(in_response, runtime.BEGIN_RESPONSE, runtime.BEGIN_MAIN)
  can_begin = _ability_cost_targets_sufficient(state, p, safe)
  return runtime.begin_ability(state, p, safe, kind, do & can_begin)


def apply_ability_action(state: State, s1, s2, use_token, is_spell, is_garden,
                         is_alley) -> State:
  """Combined spell-play / garden-or-leader / alley ability activation.

  The three handler pre-paths (payment, spell discard, counters) are merged
  with disjoint predicates so begin_ability — and its card-hook dispatch — is
  instantiated once. Mirrors handle_play_spell_from_hand /
  handle_activate_*_ability + azk_trigger_{spell,main,leader_response}."""
  from azuki_jax.abilities import runtime, tables as ab_tables
  from azuki_jax.engine.helpers import discard, leader_instance as _leader

  p = state.active_player

  # --- spell pre-path ---
  spell_inst = hand_instance(state, p, s1)
  spell_do = is_spell & (spell_inst >= 0)
  safe_spell = jnp.maximum(spell_inst, 0)
  spell_cost = effective_play_cost(state, p, safe_spell)

  # --- garden/leader pre-path ---
  g_inst = jnp.where(
      s1 == GARDEN_SIZE,
      _leader(state, p),
      card_at_slot(state, p, Zone.GARDEN, s1),
  )
  g_do = is_garden & (g_inst >= 0)
  safe_g = jnp.maximum(g_inst, 0)
  g_def = state.def_id[p, safe_g]
  g_cost = jnp.where(
      g_def >= 0, _np(ab_tables.ABILITY_IKZ_COST)[jnp.maximum(g_def, 0)], 0
  )

  # --- alley pre-path (no payment) ---
  a_inst = card_at_slot(state, p, Zone.ALLEY, s2)
  a_do = is_alley & (a_inst >= 0)

  # single payment covering spell play cost / garden ability ikz cost
  pay_amount = jnp.where(spell_do, spell_cost, jnp.where(g_do, g_cost, 0))
  state = ikz.pay(state, p, pay_amount, use_token, do=spell_do | g_do)

  # spell-only: discard + play counters
  state = discard(state, p, safe_spell, do=spell_do)
  state = state._replace(
      cards_played_turn=state.cards_played_turn.at[p].add(
          spell_do.astype(jnp.uint8)
      ),
      next_play_cost_reduction=state.next_play_cost_reduction.at[p].set(
          jnp.where(spell_do, 0, state.next_play_cost_reduction[p])
      ),
  )

  src = jnp.where(
      is_spell, safe_spell, jnp.where(is_garden, safe_g, jnp.maximum(a_inst, 0))
  )
  do = spell_do | g_do | a_do
  in_response = state.phase == Phase.RESPONSE_WINDOW
  kind = jnp.where(
      is_spell,
      runtime.BEGIN_SPELL,
      jnp.where(in_response, runtime.BEGIN_RESPONSE, runtime.BEGIN_MAIN),
  )
  can_begin = _ability_cost_targets_sufficient(state, p, src)
  return runtime.begin_ability(state, p, src, kind, do & can_begin)


def apply_noop_main(state: State, do=True) -> State:
  """NOOP in MAIN = end turn."""
  return state._replace(
      phase=jnp.where(jnp.asarray(do), jnp.int8(Phase.END_TURN), state.phase)
  )


def apply_noop_response(state: State, do=True) -> State:
  """NOOP in RESPONSE = pass; transition to combat resolve."""
  from azuki_jax.engine.phases import transition_to_combat_resolve

  return transition_to_combat_resolve(state, do=jnp.asarray(do))
