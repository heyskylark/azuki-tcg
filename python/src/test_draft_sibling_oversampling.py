"""Sibling-matchup oversampling knob (draft_same_element_matchup_prob).

With prob 1.0 every episode must pit a gate against its same-element sibling
(different gate card, same candidate pool) on both the native C draft path
and the legacy DeckBuildingParallelEnv. With the knob off, cross-element
matchups must still occur.

Run: PYTHONPATH=build/python/src:python/src pytest \
    python/src/test_draft_sibling_oversampling.py -q
"""

from __future__ import annotations

import os

import numpy as np

from azk_native import AzukiNativeEnv, NATIVE_DECKBUILD_OBS_DTYPE
from deck_building import DeckBuildingParallelEnv, build_deck_build_catalog
from tcg_parallel import AzukiTCGParallel
from training_deck_pool import load_training_deck_pool

NUM_ENVS = 32


def _native_gate_pairs(prob: float, seed: int) -> list[tuple[int, int]]:
  # Forced gates (set at import by the parity test module) bypass the sampler;
  # pop for the duration of this env's episodes, then restore.
  saved = os.environ.pop("AZK_DEBUG_FORCE_GATE_DEF_IDS", None)
  try:
    pool = load_training_deck_pool()
    env = AzukiNativeEnv(
      num_envs=NUM_ENVS,
      deck_pool=pool,
      seed=seed,
      deck_building=True,
      draft_same_element_matchup_prob=prob,
    )
    env.reset(seed=seed)
    view = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(env.num_agents)
    pairs = [
      (
        int(view[2 * i]["deck_context"]["gate_card_def_id"]),
        int(view[2 * i + 1]["deck_context"]["gate_card_def_id"]),
      )
      for i in range(NUM_ENVS)
    ]
    env.close()
    return pairs
  finally:
    if saved is not None:
      os.environ["AZK_DEBUG_FORCE_GATE_DEF_IDS"] = saved


def _element_of():
  records = build_deck_build_catalog(load_training_deck_pool()).records_by_def_id
  return {def_id: rec.element for def_id, rec in records.items()}


def test_native_prob_one_forces_sibling_matchups():
  element = _element_of()
  pairs = _native_gate_pairs(1.0, seed=11)
  assert len(pairs) == NUM_ENVS
  for g0, g1 in pairs:
    assert g0 >= 0 and g1 >= 0
    assert element[g0] == element[g1], (g0, g1)
    assert g0 != g1, "sibling override must produce the partner gate, not a mirror"


def test_native_prob_zero_keeps_cross_element_matchups():
  element = _element_of()
  pairs = _native_gate_pairs(0.0, seed=23)
  cross = sum(1 for g0, g1 in pairs if element[g0] != element[g1])
  # P(all 32 same-element) ~ 0.25^32 under the population sampler.
  assert cross > 0


def test_legacy_prob_one_forces_sibling_matchups():
  pool = load_training_deck_pool()
  element = _element_of()
  base = AzukiTCGParallel(seed=3, deck_pool=pool)
  wrapper = DeckBuildingParallelEnv(
    base, deck_pool=pool, seed=3, same_element_matchup_prob=1.0
  )
  try:
    for seed in range(5):
      wrapper.reset(seed=seed)
      g0, g1 = (int(state.gate_card_def_id) for state in wrapper._states)
      assert element[g0] == element[g1], (seed, g0, g1)
      assert g0 != g1
  finally:
    wrapper.close()
