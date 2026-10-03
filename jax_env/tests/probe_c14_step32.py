"""Probe c14 step-32: which _attack_entity_response_fast_mask component fired.

Replays the c14 fullpool case (pool 14 v 15, seed 7014) to step 32 exactly like
verify_vector_fullpool.py, then dumps the has_response components of
_attack_entity_response_fast_mask plus the defender's hand/IKZ state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "build/python/src", REPO / "python/src", REPO / "jax_env", REPO / "jax_env/tests"):
  if str(entry) not in sys.path:
    sys.path.insert(0, str(entry))

from conftest import CRef  # noqa: E402
from test_l3_abilities_batch4 import DRIVER_TYPES_4  # noqa: E402
from training_deck_pool import load_training_deck_pool  # noqa: E402
from azk_puffer.jax_vector import JaxVecEnv, SEND  # noqa: E402
from azuki_jax.env import init_state_with_decks  # noqa: E402
from azuki_jax.engine.step import stabilize  # noqa: E402
from azuki_jax.setup import deck_tables_from_card_lists  # noqa: E402
from azuki_jax import cards  # noqa: E402

LEFT, RIGHT, SEED, STOP = 14, 15, 7014, 32


def main() -> None:
  pool = [
      list(deck)
      for deck in load_training_deck_pool(str(REPO / ".codex/docs/azuki_tcg_decks_final.json"))
  ]
  cref = CRef(SEED, deck_pool=None)
  cref.reset_with_decks(SEED, pool[LEFT], pool[RIGHT])
  rng = np.random.default_rng(SEED)
  tables = deck_tables_from_card_lists(pool[LEFT], pool[RIGHT])
  state = stabilize(init_state_with_decks(SEED, tables))

  batched = jax.tree.map(lambda x: jnp.stack([x]), state)
  env = JaxVecEnv(1, pool, seed=0)
  obs, legal, count = env._observe_fn(batched)
  env._states = batched
  env._terms = jnp.zeros((1, 2), jnp.bool_)
  env._truncs = jnp.zeros((1, 2), jnp.bool_)
  rewards = jnp.zeros((1, 2), jnp.float32)
  env._pending = (obs, rewards, env._terms, env._truncs, legal, count, batched.active_player)

  final_action = None
  for step_index in range(STOP + 1):
    active = cref.active_player
    c_rows = cref.legal_actions(active)
    if not c_rows:
      print(f"case ended before step {step_index}")
      return
    driver_rows = [row for row in c_rows if row[0] in DRIVER_TYPES_4]
    action = driver_rows[int(rng.integers(0, len(driver_rows)))]
    actions = np.zeros((1, 2, 4), np.int32)
    actions[0, 0] = action
    actions[0, 1] = action
    if step_index == STOP:
      final_action = action
      break
    cref.step(actions[0, active])
    env.flag = SEND
    env.send(actions.reshape(2, 4))
    if step_index % 10 == 0:
      print(f"  step {step_index} done", flush=True)

  print(f"stopped before sending step {STOP} action={final_action}")
  states = env._states
  rows = np.arange(1)
  active = np.asarray(states.active_player).astype(np.int32)
  opp = (active + 1) % 2
  print(f"active={active} opp={opp} phase={np.asarray(states.phase)}")

  zone_host = np.asarray(states.zone)
  def_host = np.asarray(states.def_id)
  tapped_host = np.asarray(states.tapped)
  def_zone = zone_host[rows, opp]
  def_defs = def_host[rows, opp]
  def_tapped = tapped_host[rows, opp]
  def_safe_defs = np.maximum(def_defs, 0)

  names = np.array([cards.CARD_CODES[i] for i in range(len(cards.CARD_CODES))])

  in_hand = def_zone == env._zone_hand
  print("\ndefender hand:", [(int(i), names[d]) for i, d in zip(np.flatnonzero(in_hand[0]), def_defs[0][in_hand[0]])])
  ikz_untapped = (def_zone == env._zone_ikz_area) & ~def_tapped
  ikz_tapped = (def_zone == env._zone_ikz_area) & def_tapped
  print("defender ikz untapped:", [names[d] for d in def_defs[0][ikz_untapped[0]]],
        "tapped:", [names[d] for d in def_defs[0][ikz_tapped[0]]])

  token_ready = (
      (def_zone[:, env._token_instance] == env._zone_token)
      & ~def_tapped[:, env._token_instance]
  )
  payment_sources = np.sum((def_zone == env._zone_ikz_area) & ~def_tapped, axis=1)
  payment_sources += np.sum(
      (def_zone == env._zone_garden) & ~def_tapped & env._counts_as_ikz[def_safe_defs],
      axis=1,
  )
  payment_sources += token_ready.astype(np.int32)
  print(f"token_ready={token_ready} payment_sources={payment_sources}")

  next_reduction = np.asarray(states.next_play_cost_reduction)[rows, opp].astype(np.int32)
  response_cost = np.maximum(
      env._ikz_cost[def_safe_defs].astype(np.int32) - next_reduction[:, None], 0
  )
  response_spell = (
      in_hand
      & (env._card_type[def_safe_defs] == env._card_type_spell)
      & env._timing_is_response[def_safe_defs]
      & env._has_ability[def_safe_defs]
      & (response_cost <= payment_sources[:, None])
  )
  response_from_hand = (
      in_hand
      & env._response_play_from_hand[def_safe_defs]
      & (response_cost <= payment_sources[:, None])
  )
  response_board = (
      ((def_zone == env._zone_garden) | (def_zone == env._zone_alley) | (def_zone == env._zone_leader))
      & env._timing_is_response[def_safe_defs]
      & env._has_ability[def_safe_defs]
      & (np.asarray(states.frozen_dur)[rows, opp] == 0)
      & (
          ~env._once_per_turn[def_safe_defs]
          | ((np.asarray(states.once_per_turn_used)[rows, opp] & 1) == 0)
      )
      & (env._ability_ikz_cost[def_safe_defs].astype(np.int32) <= payment_sources[:, None])
  )
  defender_cards = (
      (def_zone == env._zone_garden)
      & (env._inherent_defender[def_safe_defs] | np.asarray(states.grant_defender)[rows, opp])
      & ~def_tapped
  )
  for label, comp in (
      ("response_spell", response_spell),
      ("response_from_hand", response_from_hand),
      ("response_board", response_board),
      ("defender_cards", defender_cards),
  ):
    hits = np.flatnonzero(comp[0])
    print(f"{label}: {[(int(i), str(names[def_defs[0, i]])) for i in hits]}")

  print("\nC legal actions for defender preview:")
  print("  (C is pre-attack; C's own window decision comes from defender_can_respond)")
  cref.close()


if __name__ == "__main__":
  main()
