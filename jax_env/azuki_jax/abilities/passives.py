"""Passive-observer layer: stat auras recomputed from the current board.

Replaces the C observer/passive-buff-queue machinery (per-card observers in
src/abilities/cards/*.c that call azk_queue_passive_buff_update, drained by
status_util.c azk_process_passive_buff_queue at azk_engine_tick step 2) with
a recompute: each ported passive card registers a contribution function that
derives its aura amounts for every instance from the CURRENT board;
`recompute_passives` diffs the summed totals against the amounts currently
applied (state.passive_atk / state.passive_hp) and applies the delta the way
the C queue drain would.

Equivalence argument (verified empirically against the C engine, see
tests/test_l3_passives.py and the report):
- C observers fire on the zone/attach events that are exactly the inputs of
  each card's condition, and each event re-evaluates the condition on the
  POST-event board (flecs children/ordered-children are already updated when
  EcsOnRemove callbacks run — probed: azk01_073 grant after a garden death,
  stt01_011 removal when the last copy dies). The queue keeps the LAST update
  per (target, source), so the drained state equals condition(current board)
  — i.e. a recompute.
- `azk_engine_requires_action` returns false while passive buffs are pending
  (azuki_engine.c:280), so C never exposes a decision point with a stale
  aura; recomputing at the end of apply_user_action and of every micro_tick
  reproduces that invariant.
- Processing is deferred while an ability is resolving (azk_engine_tick:
  "Defer passive queue processing until the active ability fully resolves"),
  and no tick runs after the game ends — hence the (ab_phase == 0) &
  (winner == -1) gate.

Clamp/remove bookkeeping (status_util.c):
- apply_attack_modifier clamps cur_atk at 0 and stores the post-clamp ACTUAL
  in the pair; remove_attack_modifier subtracts the stored value, clamping
  the result at 0 and dropping the pair entirely. All ported contributions
  are non-negative, so the apply-side clamp never bites and the per-source
  pair bookkeeping always equals the contribution total: we set
  passive_atk := total and clamp only the cur_atk update (both directions),
  which reproduces both C paths exactly. (A future NEGATIVE aura would need
  the store-the-clamped-actual variant — see report.)
- apply/remove_health_modifier are UNclamped; only the REMOVAL path checks
  for death (azk_process_passive_buff_queue is_removal branch): a non-leader
  whose hp drops to 0 or below is discard_card'ed (godmode prevents it).
  No ported hp aura targets a leader (all are self-buffs on entities), so the
  leader-defeat branch is unreachable and not modeled.
- If the buffed entity is a weapon (zone ATTACHED), the queue drain also
  propagates the atk delta to the host's cur_atk, clamped at 0
  (status_util.c:1333-1352) — mirrored here via a scatter-add per host.

Out-of-play targets: C resets CurStats on discard/return-to-hand but leaves
AttackBuff/HealthBuff pairs on the card; the stale pair is lazily removed by
the next observer event (subtracting from the already-reset stats, which is
never observable) — except when no event fires before the card re-enters
play (possible only via discard-retrieval effects). We zero passive_* for
out-of-play instances without touching their stats, which matches the
common lazily-cleaned path; the stale-pair re-entry corner is documented as
a known divergence in the report.

Death chains: a passive-removal death changes the board, which can ENABLE
other auras (removing a non-matching card can only turn all-of-garden
conditions on, never off, so chained deaths cannot occur) — C drains the
chain in the same queue pass (the queue is appended to while iterating);
mirrored by a second static round applied only when round 1 killed
something (see recompute_passives for the termination proof).
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from azuki_jax import cards
from azuki_jax.constants import MAX_PLAYERS, NUM_INSTANCES, Zone
from azuki_jax.state import State

# (def_id, fn) — fn(state) -> (atk (2, N) int16, hp (2, N) int16): the card's
# aura contribution to EVERY instance given the current board. Each fn does
# its own source-in-play gating (mirroring its C observer's checks).
PASSIVE_FNS: list[tuple[int, Callable[[State], tuple[jax.Array, jax.Array]]]] = []


def register_passive(code: str, fn) -> None:
  PASSIVE_FNS.append((cards.CODE_TO_ID[code], fn))


def _np(table):
  return jnp.asarray(table)


def zero_contrib() -> tuple[jax.Array, jax.Array]:
  z = jnp.zeros((MAX_PLAYERS, NUM_INSTANCES), jnp.int16)
  return z, z


def _in_play(state: State) -> jax.Array:
  z = state.zone
  return (
      (z == Zone.GARDEN) | (z == Zone.ALLEY) | (z == Zone.LEADER)
      | (z == Zone.ATTACHED)
  )


def _godmode_rows(state: State) -> jax.Array:
  def_id = state.def_id
  inherent = jnp.where(
      def_id >= 0, _np(cards.INHERENT_GODMODE)[jnp.maximum(def_id, 0)], False
  )
  return (inherent | state.grant_godmode) & _in_play(state)


def _totals(state: State) -> tuple[jax.Array, jax.Array]:
  atk, hp = zero_contrib()
  for _, fn in PASSIVE_FNS:
    a, h = fn(state)
    atk = atk + a
    hp = hp + h
  return atk, hp


def _round(state: State) -> tuple[State, jax.Array]:
  """One C queue drain: apply deltas, then passive-removal deaths."""
  from azuki_jax.engine.helpers import batch_discard

  total_atk, total_hp = _totals(state)
  in_play = _in_play(state)

  delta_atk = total_atk - state.passive_atk.astype(jnp.int16)
  delta_hp = total_hp - state.passive_hp.astype(jnp.int16)
  # C modifier-pair apply/remove mutates CurStats even when the card is hidden
  # in hand/discard; only death handling below is gated to in-play cards.
  stat_delta_atk = delta_atk
  stat_delta_hp = delta_hp

  # --- attack: clamp at 0 on change (apply_attack_modifier /
  # remove_attack_modifier both clamp) ---
  cur_atk = state.cur_atk.astype(jnp.int16)
  new_atk = jnp.where(
      stat_delta_atk != 0,
      jnp.maximum(cur_atk + stat_delta_atk, 0),
      cur_atk,
  )

  # --- weapon -> host propagation (azk_process_passive_buff_queue): the
  # actually-applied weapon atk delta is added to the host's cur_atk, clamped
  # at 0. All same-host weapon deltas share one sign (same aura condition),
  # so one batched clamp == C's sequential per-entry clamps. ---
  n = state.zone.shape[1]
  attached = (state.zone == Zone.ATTACHED) & (state.attached_to >= 0)
  wdelta = jnp.where(attached, stat_delta_atk, 0)
  host_idx = jnp.clip(state.attached_to.astype(jnp.int32), 0, n - 1)
  host_delta = jax.vmap(
      lambda d, h: jnp.zeros((n,), jnp.int16).at[h].add(d)
  )(wdelta, host_idx)
  new_atk = jnp.where(host_delta != 0, jnp.maximum(new_atk + host_delta, 0), new_atk)

  # --- health: unclamped (apply/remove_health_modifier) ---
  new_hp = state.cur_hp.astype(jnp.int16) + stat_delta_hp

  state = state._replace(
      cur_atk=new_atk.astype(jnp.int8),
      cur_hp=new_hp.astype(jnp.int8),
      passive_atk=total_atk.astype(jnp.int8),
      passive_hp=total_hp.astype(jnp.int8),
  )

  # --- death on hp-buff REMOVAL only (C checks death only in the is_removal
  # branch of the queue drain). Leaders: unreachable (self-buff entities
  # only). Godmode prevents the discard (C discard_card godmode guard);
  # the negative hp then simply persists, as in C. When-destroyed triggers:
  # no current hp-passive card has the AWhenDestroyED timing, so C's
  # maybe_queue_self_leave_play_trigger is a guaranteed no-op here (the
  # bobu/miharu/kurai destroy observers are not modeled engine-wide, see
  # phases.py combat note). Revisit if an hp-passive gains a destroy timing.
  died = (
      in_play
      & (delta_hp < 0)
      & (new_hp <= 0)
      & ~_godmode_rows(state)
      & (state.zone != Zone.LEADER)
  )
  order_key = jnp.arange(n, dtype=jnp.int32)  # C queues in instance order
  for p in range(MAX_PLAYERS):
    state = batch_discard(state, p, died[p], order_key, do=died[p].any())
  return state, died.any()


def _where_state(pred, a: State, b: State) -> State:
  return jax.tree.map(lambda x, y: jnp.where(pred, x, y), a, b)


def _two_rounds(state: State) -> State:
  out1, killed = _round(state)
  out2, _ = _round(out1)
  return _where_state(killed, out2, out1)


def recompute_passives(state: State) -> State:
  """Sync passive auras with the board (C tick-ladder step 2 equivalent).

  Skipped while an ability FSM is resolving and after game over, mirroring
  azk_engine_tick's gating; also skipped while no registered source is in
  play and no residual bookkeeping needs cleanup. The lax.cond keeps the
  body in an XLA subcomputation — both inlining it flat into micro_tick/
  apply_user_action and an inner while_loop fixpoint made whole-module XLA
  optimization pathologically slow. Keep this shape.

  Two static rounds replace the C drain-to-fixpoint: round-1 deaths are the
  only board changes a round can make, and a death (removing a garden card)
  can only turn the all-of-garden conditions ON (all() over fewer elements;
  073's non-empty clause holds while the surviving source itself sits in the
  garden) and never adds/removes weapons in play or in the discard — so
  round 2 carries only death-enabled APPLIES plus atk removals for weapons
  whose host died (no hp removals => no further deaths), and round 3 would
  be a no-op. Revisit if a future passive can be DISABLED by a removal."""
  import numpy as np

  from azuki_jax.abilities import cards_impl  # noqa: F401  (forces card registration)

  if not PASSIVE_FNS:
    return state
  source_table = np.zeros(cards.CARD_DEF_COUNT, np.bool_)
  for def_id, _ in PASSIVE_FNS:
    source_table[def_id] = True
  is_source = jnp.where(
      state.def_id >= 0, _np(source_table)[jnp.maximum(state.def_id, 0)], False
  )
  relevant = (
      jnp.any(is_source & _in_play(state))
      | (state.passive_queue_count != 0)
      | jnp.any(state.stt02_012_event_pending)
      | jnp.any(state.passive_latched_atk != 0)
      | jnp.any(state.passive_latched_hp != 0)
      | jnp.any(state.passive_atk != 0)
      | jnp.any(state.passive_hp != 0)
  )
  gate = (state.ab_phase == 0) & (state.winner == -1)
  drain = gate & relevant
  out = jax.lax.cond(drain, _two_rounds, lambda s: s, state)
  return out._replace(
      stt02_012_event_pending=jnp.where(
          drain,
          jnp.zeros_like(out.stt02_012_event_pending),
          out.stt02_012_event_pending,
      ),
      passive_queue_count=jnp.where(
          drain,
          jnp.asarray(0, out.passive_queue_count.dtype),
          out.passive_queue_count,
      ),
  )
