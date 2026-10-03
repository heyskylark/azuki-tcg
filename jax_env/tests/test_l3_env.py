"""L3: env-level rollout equivalence — rewards/terminals/truncations vs C.

Drives the full azuki_jax.step.step (auto-reset semantics disabled by feeding
prev flags) against the C binding on vanilla decks, comparing per-step rewards
(shaped PBRS / terminal) to float tolerance, plus terminal flags, over
multiple complete episodes.
"""
from __future__ import annotations

import numpy as np
import pytest

from test_l2_vanilla import (
    DRIVER_TYPES,
    MASK_COMPARE_TYPES,
    VANILLA_DECK,
    c_semantic_view,
    filtered_mask,
    jax_semantic_view,
)


@pytest.mark.parametrize("seed", [5, 77])
def test_env_rollout_rewards_match(seed, make_cref):
  import jax
  import jax.numpy as jnp

  from azuki_jax.engine.step import stabilize
  from azuki_jax.env import init_state_with_decks
  from azuki_jax.masks import build_mask
  from azuki_jax.setup import deck_tables_from_card_lists
  from azuki_jax.step import step as env_step

  tables = deck_tables_from_card_lists(VANILLA_DECK, VANILLA_DECK)

  @jax.jit
  def jstep(state, actions, prev_term, prev_trunc):
    return env_step(state, actions, prev_term, prev_trunc, tables)

  jit_mask = jax.jit(build_mask)

  rng = np.random.default_rng(seed)
  episodes = 0
  episode_seed = seed
  cref = make_cref(seed, deck_pool=None)

  while episodes < 3:
    cref.reset_with_decks(episode_seed, VANILLA_DECK, VANILLA_DECK)
    state = stabilize(init_state_with_decks(episode_seed, tables))
    prev_term = jnp.zeros(2, bool)
    prev_trunc = jnp.zeros(2, bool)

    for step_index in range(600):
      cview = c_semantic_view(cref)
      jview = jax_semantic_view(state)
      for key, cval in cview.items():
        if key == "winner":
          continue
        assert jview[key] == cval, (
            f"ep {episodes} step {step_index}: {key}\nC  ={cval}\nJAX={jview[key]}"
        )

      active = cview["active"]
      c_rows = filtered_mask(cref.legal_actions(active), MASK_COMPARE_TYPES)
      legal, count, _ = jit_mask(state)
      j_rows = filtered_mask(
          [tuple(int(x) for x in row) for row in np.asarray(legal)[: int(count)]],
          MASK_COMPARE_TYPES,
      )
      assert j_rows == c_rows, f"ep {episodes} step {step_index}: mask mismatch"

      driver_rows = filtered_mask(c_rows, DRIVER_TYPES)
      action = driver_rows[int(rng.integers(0, len(driver_rows)))]

      cref.step(np.asarray(action, np.int32))
      actions = jnp.asarray([action, action], jnp.int32)
      state, rewards, terms, truncs = jstep(state, actions, prev_term, prev_trunc)

      c_rewards = cref.rewards()
      j_rewards = np.asarray(rewards)
      np.testing.assert_allclose(
          j_rewards, c_rewards, rtol=0, atol=2e-5,
          err_msg=f"ep {episodes} step {step_index}: rewards mismatch",
      )

      c_term, c_trunc = cref.dones()
      assert bool(terms.all()) == c_term, (
          f"ep {episodes} step {step_index}: terminal mismatch"
      )
      assert bool(truncs.all()) == c_trunc

      if c_term or c_trunc:
        episodes += 1
        episode_seed = int(rng.integers(0, 2**31 - 1))
        break
    else:
      raise AssertionError("episode did not finish within 600 steps")
