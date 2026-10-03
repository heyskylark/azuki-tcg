"""Privileged drafted-deck exposure (deck_building_privileged_decks).

With the flag on, critic_privileged self/opponent deck lists must carry the
drafted pick-order compositions — identically on the native C path and the
legacy wrapper — during the draft and at battle start. With the flag off the
lists stay sanitized (covered by the existing parity test's assertions).

Run: PYTHONPATH=build/python/src:python/src pytest \
    python/src/test_deckbuild_priv_decks.py -q
"""

from __future__ import annotations

import os

os.environ.setdefault("AZK_DEBUG_FORCE_GATE_DEF_IDS", "3,161")

import numpy as np

from azk_native import AzukiNativeEnv, NATIVE_DECKBUILD_OBS_DTYPE
from deck_building import DeckBuildingParallelEnv
from tcg_parallel import AzukiTCGParallel
from training_deck_pool import load_training_deck_pool

FORCED_GATES = (3, 161)


def _make_legacy(pool):
  env = AzukiTCGParallel(seed=123, deck_pool=pool)
  wrapper = DeckBuildingParallelEnv(env, deck_pool=pool, seed=123, privileged_decks=True)
  calls = {"n": 0}

  def forced_gate():
    value = FORCED_GATES[calls["n"] % 2]
    calls["n"] += 1
    return int(value)

  wrapper._sample_gate_def_id = forced_gate
  return wrapper


def _priv_decks(view_row):
  cp = view_row["critic_privileged"]
  return (
    np.asarray(cp["self_deck"]["card_def_id"], dtype=np.int16),
    np.asarray(cp["opponent_deck"]["card_def_id"], dtype=np.int16),
  )


def test_privileged_decks_native_matches_legacy():
  pool = load_training_deck_pool()
  legacy = _make_legacy(pool)
  legacy_obs, _ = legacy.reset(seed=99)

  native = AzukiNativeEnv(
    num_envs=1, deck_pool=pool, seed=7, deck_building=True,
    deck_building_privileged_decks=True,
  )
  native.reset(seed=7)
  view = native.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(native.num_agents)

  rng = np.random.default_rng(2024)
  steps = 0
  saw_nonempty_opponent = False
  while legacy._building:
    active = legacy._active_player_index
    for agent in range(2):
      n_self, n_opp = _priv_decks(view[agent])
      l_cp = legacy_obs[agent]["critic_privileged"]
      l_self = np.asarray([c["card_def_id"] for c in l_cp["self_deck"]], dtype=np.int16)
      l_opp = np.asarray([c["card_def_id"] for c in l_cp["opponent_deck"]], dtype=np.int16)
      np.testing.assert_array_equal(n_self, l_self, err_msg=f"step {steps} agent {agent} self")
      np.testing.assert_array_equal(n_opp, l_opp, err_msg=f"step {steps} agent {agent} opp")
      if (n_opp >= 0).any():
        saw_nonempty_opponent = True

    ctx = legacy_obs[active]["deck_context"]
    count = int(ctx["candidate_count"])
    pick = int(rng.integers(0, count))
    action = np.array([3, pick, 0, 0], dtype=np.int32)
    legacy_obs, _, _, _, _ = legacy.step({active: action})
    native.actions[:] = 0
    native.actions[active] = action
    native.step()
    steps += 1
    assert steps < 210

  assert saw_nonempty_opponent, "opponent picks never appeared during draft"
  # battle start: both full 50-card compositions visible, identical across paths
  for agent in range(2):
    n_self, n_opp = _priv_decks(view[agent])
    l_cp = legacy_obs[agent]["critic_privileged"]
    l_self = np.asarray([c["card_def_id"] for c in l_cp["self_deck"]], dtype=np.int16)
    l_opp = np.asarray([c["card_def_id"] for c in l_cp["opponent_deck"]], dtype=np.int16)
    np.testing.assert_array_equal(n_self, l_self)
    np.testing.assert_array_equal(n_opp, l_opp)
    assert (n_self >= 0).sum() == 50
    assert (n_opp >= 0).sum() == 50

  native.close()
  legacy.close()
