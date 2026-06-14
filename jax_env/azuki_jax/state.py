"""Environment state pytree: fixed-size arrays only (paper checklist B.1).

Per-card-instance arrays have shape (2, NUM_INSTANCES) = (player, instance).
Instances 0..61 are the 62 deck cards in C expansion order; instance 62 is
the player's IKZ token. Location = (zone, zpos): list zones use compacting
positions (deck top = highest zpos, flecs ordered-children semantics),
garden/alley use slot indices, ATTACHED uses weapon order on the host.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from azuki_jax.constants import (
    MAX_ABILITY_SELECTION,
    MAX_PLAYERS,
    MAX_SELECTION_ZONE_SIZE,
    MAX_TRIGGERED_EFFECTS,
    NUM_INSTANCES,
    RECENT_ACTION_HISTORY_LEN,
    Phase,
    Zone,
)


class State(NamedTuple):
  # --- per-card-instance (2, NUM_INSTANCES) ---
  def_id: jax.Array          # int16, -1 = unused slot
  zone: jax.Array            # int8 (Zone)
  zpos: jax.Array            # int8
  board_seq: jax.Array       # int16 garden/alley entry order (flecs children)
  attached_to: jax.Array     # int8 host instance (own=weapon), -1 = none
  tapped: jax.Array          # bool
  cooldown: jax.Array        # uint8
  cur_atk: jax.Array         # int8
  cur_hp: jax.Array          # int8
  atk_buff_perm: jax.Array   # int8 (one-shot grants, permanent)
  atk_buff_eot: jax.Array    # int8 (one-shot grants expiring at end of turn)
  hp_buff_perm: jax.Array    # int8
  hp_buff_eot: jax.Array     # int8
  passive_atk: jax.Array     # int8 passive-aura atk currently APPLIED (abilities/passives.py)
  passive_hp: jax.Array      # int8 passive-aura hp currently APPLIED
  carapace_perm: jax.Array   # int8
  carapace_eot: jax.Array    # int8
  cmb_in_perm: jax.Array     # int8 incoming combat damage modifier
  cmb_in_eot: jax.Array      # int8
  cmb_out_perm: jax.Array    # int8 outgoing combat damage modifier
  cmb_out_eot: jax.Array     # int8
  frozen_dur: jax.Array      # int8 (-1 permanent, 0 none, >0 ticks)
  shocked_dur: jax.Array     # int8
  effect_immune_dur: jax.Array  # int8
  grant_charge: jax.Array    # bool (granted, separate from inherent table flag)
  grant_defender: jax.Array  # bool
  grant_infiltrate: jax.Array  # bool
  grant_taunt: jax.Array     # bool
  grant_rooted: jax.Array    # bool
  grant_godmode: jax.Array   # bool
  timed_tag: jax.Array       # uint8 (2, N, MAX_TIMED_TAG_GRANTS) tag id, 0=empty
  timed_ticks: jax.Array     # int8  (2, N, MAX_TIMED_TAG_GRANTS)
  timed_phase: jax.Array     # uint8 (2, N, MAX_TIMED_TAG_GRANTS) tick phase
  sacrifice_eot: jax.Array   # bool
  once_per_turn_used: jax.Array  # uint8 bitmask over ability slots
  took_damage_turn: jax.Array  # bool
  dealt_damage_turn: jax.Array  # bool
  last_dmg_taken: jax.Array    # int8 (this turn)
  last_dmg_src_player: jax.Array  # int8 (-1 none)
  last_dmg_src_inst: jax.Array    # int8 (-1 none)
  last_dmg_from_effect: jax.Array  # bool (DamageTracker.last_taken_from_effect)
  # DamageTracker.tracked_sources: distinct damage sources this turn, encoded
  # sp*256+si (30000 = C's null source entity 0); -1 = empty slot
  dmg_src_keys: jax.Array      # int16 (2, N, 8)
  dmg_src_count: jax.Array     # int8 (2, N)

  # STT03-001 Bobu latch (C STT03BobuState on the leader card): heal 1 the
  # first time an own earth entity is destroyed/sacrificed from garden/alley
  # while turn_number < expires_turn (0 = inactive)
  bobu_expires_turn: jax.Array  # int16 (2,)

  # AZK01-120 ReequipOrigin: previous host of a weapon detached into the
  # selection zone (-1 = none)
  reequip_prev_host: jax.Array  # int8 (2, N)

  # STT02-012 event latch: C evaluates its aura per garden add/remove EVENT
  # with an off-by-one on removals (flecs post-removal recount minus one);
  # the result persists until the next garden event.
  stt02_012_latch: jax.Array  # bool (2, N)

  # pending damage redirects (AZK01-062 Pekiro), 8 slots
  # (C MAX_PENDING_DAMAGE_REDIRECTS)
  redirect_src_player: jax.Array  # int8 (8,)
  redirect_src_inst: jax.Array    # int8 (8,)
  redirect_tgt_player: jax.Array  # int8 (8,) original target
  redirect_tgt_inst: jax.Array    # int8 (8,)
  redirect_damage: jax.Array      # int8 (8,)
  redirect_count: jax.Array       # int8

  # --- player-level (2,) ---
  ikz_token_expires_eot: jax.Array  # bool
  entities_played_garden_turn: jax.Array  # uint8
  entities_played_alley_turn: jax.Array   # uint8
  cards_played_turn: jax.Array            # uint8
  discarded_cards_turn: jax.Array         # uint8
  returned_to_hand_turn: jax.Array        # uint8
  next_play_cost_reduction: jax.Array     # int8

  # --- game-level scalars ---
  phase: jax.Array            # int8 (Phase)
  active_player: jax.Array    # int8
  starting_player: jax.Array  # int8
  turn_number: jax.Array      # int16
  mulligan_done: jax.Array    # uint8
  winner: jax.Array           # int8 (-1 ongoing, 0/1, 2 draw)
  rng_state: jax.Array        # uint32
  eot_abilities_queued: jax.Array  # bool

  # combat
  combat_attacker: jax.Array       # int8 instance (-1 none); owner = active
  combat_defender: jax.Array       # int8 instance (-1 none)
  combat_defender_player: jax.Array  # int8
  combat_intercepted: jax.Array    # bool
  combat_attacker_is_leader: jax.Array  # bool

  # ability FSM (mirrors AbilityContext)
  ab_phase: jax.Array          # int8 (AbilityPhase)
  ab_source: jax.Array         # int8 instance, -1 none
  ab_owner: jax.Array          # int8 player
  ab_slot: jax.Array           # int8 registry slot (0=primary, 1.. additional)
  ab_is_optional: jax.Array    # bool
  ab_costs_applied: jax.Array  # bool
  ab_saved_active: jax.Array   # int8
  ab_restores_active: jax.Array  # bool
  ab_cost_selected: jax.Array  # int8 count
  ab_cost_max: jax.Array       # int8 context cost.max_allowed (begin-clamped)
  ab_cost_targets: jax.Array   # int8 (MAX_ABILITY_SELECTION,) instance ids
  ab_cost_target_players: jax.Array  # int8 (MAX_ABILITY_SELECTION,)
  ab_eff_selected: jax.Array   # int8
  ab_eff_min: jax.Array        # int8 context effect.min_required (hook-mutable)
  ab_eff_max: jax.Array        # int8 context effect.max_allowed (clamped)
  ab_eff_targets: jax.Array    # int8 (MAX_ABILITY_SELECTION,)
  ab_eff_target_players: jax.Array  # int8 (MAX_ABILITY_SELECTION,)
  ab_sel_cards: jax.Array      # int8 (MAX_SELECTION_ZONE_SIZE,) instance ids
  ab_sel_count: jax.Array      # int8
  ab_sel_picked: jax.Array     # int8 (MAX_ABILITY_SELECTION,)
  ab_sel_picked_count: jax.Array  # int8
  ab_sel_pick_max: jax.Array   # int8
  ab_scratch: jax.Array        # int16 (4,) transient values

  # triggered effect queue (ring buffer semantics: head always 0, shift on pop)
  trig_source: jax.Array       # int8 (MAX_TRIGGERED_EFFECTS,) instance
  trig_owner: jax.Array        # int8 (MAX_TRIGGERED_EFFECTS,)
  trig_slot: jax.Array         # int8 (MAX_TRIGGERED_EFFECTS,) registry slot
  trig_timing: jax.Array       # int8 (MAX_TRIGGERED_EFFECTS,)
  trig_count: jax.Array        # int8

  # recent actions (2, RECENT_ACTION_HISTORY_LEN, 6): valid,primary,s1,s2,s3,was_noop
  recent_actions: jax.Array    # int16

  # episode bookkeeping (tcg.h env layer)
  tick_guard: jax.Array        # int32 (auto-resolve loop bound, transient)
  tick: jax.Array              # int32
  time_weight: jax.Array       # float32
  last_phi: jax.Array          # float32 (2,)
  last_leader_ratio: jax.Array  # float32 (2,)
  last_garden_attack: jax.Array  # float32 (2,)
  has_last_snapshot: jax.Array  # bool
  episode_returns: jax.Array   # float32 (2,)

  seq_counter: jax.Array       # int16 monotonic board-entry counter

  # env-level episode RNG / deck selection
  env_seed: jax.Array          # uint32
  starter_rng_state: jax.Array  # uint32
  deck_rng_state: jax.Array    # uint32
  current_deck_indices: jax.Array  # int16 (2,)
  completed_episodes: jax.Array  # int32


def _zeros(shape, dtype):
  return jnp.zeros(shape, dtype=dtype)


def empty_state() -> State:
  """All-zero state template (shapes/dtypes only; reset fills real values)."""
  n = (MAX_PLAYERS, NUM_INSTANCES)
  return State(
      def_id=jnp.full(n, -1, jnp.int16),
      zone=jnp.full(n, int(Zone.ABSENT), jnp.int8),
      zpos=_zeros(n, jnp.int8),
      board_seq=_zeros(n, jnp.int16),
      attached_to=jnp.full(n, -1, jnp.int8),
      tapped=_zeros(n, jnp.bool_),
      cooldown=_zeros(n, jnp.uint8),
      cur_atk=_zeros(n, jnp.int8),
      cur_hp=_zeros(n, jnp.int8),
      atk_buff_perm=_zeros(n, jnp.int8),
      atk_buff_eot=_zeros(n, jnp.int8),
      hp_buff_perm=_zeros(n, jnp.int8),
      hp_buff_eot=_zeros(n, jnp.int8),
      passive_atk=_zeros(n, jnp.int8),
      passive_hp=_zeros(n, jnp.int8),
      carapace_perm=_zeros(n, jnp.int8),
      carapace_eot=_zeros(n, jnp.int8),
      cmb_in_perm=_zeros(n, jnp.int8),
      cmb_in_eot=_zeros(n, jnp.int8),
      cmb_out_perm=_zeros(n, jnp.int8),
      cmb_out_eot=_zeros(n, jnp.int8),
      frozen_dur=_zeros(n, jnp.int8),
      shocked_dur=_zeros(n, jnp.int8),
      effect_immune_dur=_zeros(n, jnp.int8),
      grant_charge=_zeros(n, jnp.bool_),
      grant_defender=_zeros(n, jnp.bool_),
      grant_infiltrate=_zeros(n, jnp.bool_),
      grant_taunt=_zeros(n, jnp.bool_),
      grant_rooted=_zeros(n, jnp.bool_),
      grant_godmode=_zeros(n, jnp.bool_),
      timed_tag=_zeros((*n, 8), jnp.uint8),
      timed_ticks=_zeros((*n, 8), jnp.int8),
      timed_phase=_zeros((*n, 8), jnp.uint8),
      sacrifice_eot=_zeros(n, jnp.bool_),
      once_per_turn_used=_zeros(n, jnp.uint8),
      took_damage_turn=_zeros(n, jnp.bool_),
      dealt_damage_turn=_zeros(n, jnp.bool_),
      last_dmg_taken=_zeros(n, jnp.int8),
      last_dmg_src_player=jnp.full(n, -1, jnp.int8),
      last_dmg_src_inst=jnp.full(n, -1, jnp.int8),
      last_dmg_from_effect=_zeros(n, jnp.bool_),
      dmg_src_keys=jnp.full((*n, 8), -1, jnp.int16),
      dmg_src_count=_zeros(n, jnp.int8),
      bobu_expires_turn=_zeros((MAX_PLAYERS,), jnp.int16),
      reequip_prev_host=jnp.full(n, -1, jnp.int8),
      stt02_012_latch=_zeros(n, jnp.bool_),
      redirect_src_player=jnp.full((8,), -1, jnp.int8),
      redirect_src_inst=jnp.full((8,), -1, jnp.int8),
      redirect_tgt_player=jnp.full((8,), -1, jnp.int8),
      redirect_tgt_inst=jnp.full((8,), -1, jnp.int8),
      redirect_damage=_zeros((8,), jnp.int8),
      redirect_count=jnp.int8(0),
      ikz_token_expires_eot=_zeros((MAX_PLAYERS,), jnp.bool_),
      entities_played_garden_turn=_zeros((MAX_PLAYERS,), jnp.uint8),
      entities_played_alley_turn=_zeros((MAX_PLAYERS,), jnp.uint8),
      cards_played_turn=_zeros((MAX_PLAYERS,), jnp.uint8),
      discarded_cards_turn=_zeros((MAX_PLAYERS,), jnp.uint8),
      returned_to_hand_turn=_zeros((MAX_PLAYERS,), jnp.uint8),
      next_play_cost_reduction=_zeros((MAX_PLAYERS,), jnp.int8),
      phase=jnp.int8(Phase.PREGAME_MULLIGAN),
      active_player=jnp.int8(0),
      starting_player=jnp.int8(0),
      turn_number=jnp.int16(0),
      mulligan_done=jnp.uint8(0),
      winner=jnp.int8(-1),
      rng_state=jnp.uint32(0),
      eot_abilities_queued=jnp.bool_(False),
      combat_attacker=jnp.int8(-1),
      combat_defender=jnp.int8(-1),
      combat_defender_player=jnp.int8(-1),
      combat_intercepted=jnp.bool_(False),
      combat_attacker_is_leader=jnp.bool_(False),
      ab_phase=jnp.int8(0),
      ab_source=jnp.int8(-1),
      ab_owner=jnp.int8(-1),
      ab_slot=jnp.int8(0),
      ab_is_optional=jnp.bool_(False),
      ab_costs_applied=jnp.bool_(False),
      ab_saved_active=jnp.int8(-1),
      ab_restores_active=jnp.bool_(False),
      ab_cost_selected=jnp.int8(0),
      ab_cost_max=jnp.int8(0),
      ab_cost_targets=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_cost_target_players=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_eff_selected=jnp.int8(0),
      ab_eff_min=jnp.int8(0),
      ab_eff_max=jnp.int8(0),
      ab_eff_targets=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_eff_target_players=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_sel_cards=jnp.full((MAX_SELECTION_ZONE_SIZE,), -1, jnp.int8),
      ab_sel_count=jnp.int8(0),
      ab_sel_picked=jnp.full((MAX_ABILITY_SELECTION,), -1, jnp.int8),
      ab_sel_picked_count=jnp.int8(0),
      ab_sel_pick_max=jnp.int8(0),
      ab_scratch=_zeros((4,), jnp.int16),
      trig_source=jnp.full((MAX_TRIGGERED_EFFECTS,), -1, jnp.int8),
      trig_owner=jnp.full((MAX_TRIGGERED_EFFECTS,), -1, jnp.int8),
      trig_slot=_zeros((MAX_TRIGGERED_EFFECTS,), jnp.int8),
      trig_timing=jnp.full((MAX_TRIGGERED_EFFECTS,), -1, jnp.int8),
      trig_count=jnp.int8(0),
      recent_actions=_zeros((MAX_PLAYERS, RECENT_ACTION_HISTORY_LEN, 6), jnp.int16),
      tick_guard=jnp.int32(0),
      tick=jnp.int32(0),
      time_weight=jnp.float32(1.0),
      last_phi=_zeros((MAX_PLAYERS,), jnp.float32),
      last_leader_ratio=_zeros((MAX_PLAYERS,), jnp.float32),
      last_garden_attack=_zeros((MAX_PLAYERS,), jnp.float32),
      has_last_snapshot=jnp.bool_(False),
      episode_returns=_zeros((MAX_PLAYERS,), jnp.float32),
      seq_counter=jnp.int16(0),
      env_seed=jnp.uint32(0),
      starter_rng_state=jnp.uint32(0),
      deck_rng_state=jnp.uint32(0),
      current_deck_indices=jnp.full((MAX_PLAYERS,), -1, jnp.int16),
      completed_episodes=jnp.int32(0),
  )
