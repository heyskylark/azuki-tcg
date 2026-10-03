"""Element-specialist knob (env.learner_element) and opponent-lineage guard.

Run: PYTHONPATH=build/python/src:python/src pytest python/tests/test_specialist_element.py -q
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from policy.tcg_distribution import TCGLegalActionDistribution
from policy.v2.tcg_sampler import tcg_argmax_logits
from specialist_ensemble import SpecialistEnsemble

from azk_native import AzukiNativeEnv, NATIVE_DECKBUILD_OBS_DTYPE
from deck_building import GATE_CARD_TYPE, LEADER_CARD_TYPE, build_deck_build_catalog
import league_training
import specialist
from specialist import LEARNER_ELEMENT_CODES, assert_recent_lineage, gate_element_codes
from training_deck_pool import load_training_deck_pool

NUM_ENVS = 48
WATER = LEARNER_ELEMENT_CODES["water"]


def _pool_groups():
  pool = load_training_deck_pool()
  catalog = build_deck_build_catalog(pool)
  by_context = defaultdict(list)
  for deck_index, deck in enumerate(pool):
    gate = leader = -1
    for card_code, _ in deck:
      record = catalog.records_by_code[card_code]
      if record.card_type == GATE_CARD_TYPE:
        gate = record.card_def_id
      elif record.card_type == LEADER_CARD_TYPE:
        leader = record.card_def_id
    by_context[(gate, leader)].append(deck_index)
  groups = tuple(tuple(indices) for _, indices in sorted(by_context.items()) if len(indices) >= 2)
  water_decks = tuple(
    index
    for (gate, _), indices in sorted(by_context.items())
    if len(indices) >= 2 and gate_element_codes(np.array([gate]))[0] == WATER
    for index in indices
  )
  assert groups and water_decks
  return pool, groups, water_decks


def _env(element: str, *, uniform: bool, prebuilt_probability: float, sibling_prob: float, seed: int):
  pool, groups, water_decks = _pool_groups()
  return AzukiNativeEnv(
    num_envs=NUM_ENVS,
    seed=seed,
    deck_pool=pool,
    deck_building=True,
    draft_uniform_assignment=uniform,
    draft_same_element_matchup_prob=sibling_prob,
    prebuilt_deck_groups=groups,
    prebuilt_probability=prebuilt_probability,
    learner_element=element,
    learner_prebuilt_deck_indices=water_decks if element != "none" else None,
  )


def _seat_state(env):
  rows = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(NUM_ENVS, 2)
  gates = rows["deck_context"]["gate_card_def_id"].astype(np.int64)
  prebuilt = (rows["deck_context"]["mode"] == 0).all(axis=1)
  return gate_element_codes(gates), prebuilt


@pytest.mark.parametrize("uniform", [True, False])
@pytest.mark.parametrize("sibling_prob", [0.0, 1.0])
def test_learner_element_forces_flagged_seats_on_prebuilt_and_draft_paths(uniform, sibling_prob):
  env = _env("water", uniform=uniform, prebuilt_probability=0.5, sibling_prob=sibling_prob, seed=5)
  rng = np.random.default_rng(17)
  seen = defaultdict(int)
  try:
    for episode in range(8):
      # Mirror (1,1), learner seat 0, learner seat 1, and no learner seat.
      mask = rng.integers(0, 2, size=(NUM_ENVS, 2), dtype=np.uint8)
      env.learner_seat_mask[:] = mask.reshape(-1)
      env.reset(seed=1000 + episode)
      elements, prebuilt = _seat_state(env)
      flagged = mask.astype(bool)
      assert np.all(elements[flagged] == WATER)
      for path, rows in (("prebuilt", prebuilt), ("draft", ~prebuilt)):
        for seat in range(2):
          seen[(path, seat)] += int(np.count_nonzero(flagged[rows, seat]))
          seen[(path, "free_non_water")] += int(
            np.count_nonzero(~flagged[rows, seat] & (elements[rows, seat] != WATER))
          )
      if sibling_prob == 1.0:
        draft_free_seat0 = ~prebuilt & ~flagged[:, 0] & flagged[:, 1]
        # The sibling roll pairs the free seat with the learner's element.
        assert np.all(elements[draft_free_seat0, 0] == WATER)
  finally:
    env.close()
  for path in ("prebuilt", "draft"):
    assert seen[(path, 0)] > 0 and seen[(path, 1)] > 0
    if sibling_prob == 0.0:
      assert seen[(path, "free_non_water")] > 0


def _rollout_digest(element: str, mask_value: int, *, steps: int = 400) -> str:
  env = _env(element, uniform=True, prebuilt_probability=0.5, sibling_prob=0.35, seed=29)
  env.learner_seat_mask[:] = mask_value
  rng = np.random.default_rng(3)
  digest = hashlib.sha256()
  try:
    env.reset(seed=29)
    for _ in range(steps):
      rows = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)
      masks = rows["action_mask"]
      count = masks["legal_action_count"].astype(np.int64)
      pick = (rng.random(count.size) * np.maximum(count, 1)).astype(np.int64)
      take = np.arange(count.size)
      actions = np.stack(
        [masks[name][take, pick] for name in ("legal_primary", "legal_sub1", "legal_sub2", "legal_sub3")],
        axis=1,
      ).astype(env.actions.dtype)
      env.actions[:] = actions.reshape(env.actions.shape)
      env.step()
      for buffer in (env.observations, env.rewards, env.terminals, env.truncations):
        digest.update(np.ascontiguousarray(buffer).tobytes())
  finally:
    env.close()
  return digest.hexdigest()


def test_unflagged_seats_leave_every_rng_stream_identical_to_knob_off():
  baseline = _rollout_digest("none", 1)
  assert _rollout_digest("water", 0) == baseline
  assert _rollout_digest("water", 1) != baseline


def test_publish_flags_learner_seat_and_both_seats_in_self_play():
  mask = np.full(8, 7, dtype=np.uint8)
  trainer = SimpleNamespace(
    _agents_per_env=2,
    _env_learner_seat=np.array([0, 1, 1, 0], dtype=np.int32),
    _env_use_latest=np.array([False, False, True, True]),
    _learner_seat_mask=mask,
    _env_episode_unclassified=np.zeros(4, dtype=bool),
  )
  league_training.LeaguePuffeRL._publish_learner_seats(trainer, np.array([0, 1, 2], dtype=np.int32))
  assert mask.tolist() == [1, 0, 0, 1, 1, 1, 7, 7]


def _fake_checkpoint(tmp_path, name: str, global_step: int):
  path = tmp_path / name
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(name.encode())
  (tmp_path / f"{name}.meta.json").write_text(json.dumps({"global_step": global_step, "update": 1}))
  return path


def test_lineage_guard_rejects_legacy_1b_checkpoints(tmp_path, monkeypatch):
  recent = _fake_checkpoint(tmp_path, "model_azuki_local_008223.pt", 76_363_000)
  assert_recent_lineage([recent])
  by_step = _fake_checkpoint(tmp_path, "model_azuki_local_021000.pt", 194_576_077)
  by_path = _fake_checkpoint(tmp_path, "evaluation_opponents/p060000/model.pt", 1)
  by_hash = _fake_checkpoint(tmp_path, "renamed.pt", 1)
  digest = hashlib.sha256(by_hash.read_bytes()).hexdigest()
  monkeypatch.setitem(specialist.LEGACY_CHECKPOINT_SHA256, digest, "p065105")
  for legacy in (by_step, by_path, by_hash):
    with pytest.raises(ValueError, match="Legacy August 1B-run"):
      assert_recent_lineage([recent, legacy])


def test_lineage_guard_allows_long_specialist_runs_but_not_elsewhere(tmp_path, monkeypatch):
  runs = tmp_path / "train-specialist" / "runs"
  monkeypatch.setattr(specialist, "SPECIALIST_RUNS_ROOT", runs)
  continued = _fake_checkpoint(runs, "earth_c1/artifacts/a/model_azuki_local_020000.pt", 206_000_000)
  assert_recent_lineage([continued])
  elsewhere = _fake_checkpoint(tmp_path / "other", "model_azuki_local_020000.pt", 206_000_000)
  with pytest.raises(ValueError, match="Legacy August 1B-run"):
    assert_recent_lineage([elsewhere])
  banned_hash = _fake_checkpoint(runs, "earth_c1/artifacts/b/renamed.pt", 206_000_000)
  monkeypatch.setitem(specialist.LEGACY_CHECKPOINT_SHA256, hashlib.sha256(banned_hash.read_bytes()).hexdigest(), "p044000")
  with pytest.raises(ValueError, match="Legacy August 1B-run"):
    assert_recent_lineage([banned_hash])


class _ConstantPolicy(torch.nn.Module):
  """Member stub: fixed action, adds `bump` to the recurrent state it receives."""

  hidden_size = 3

  def __init__(self, action: int, bump: float):
    super().__init__()
    self.action = action
    self.bump = bump

  def forward_eval(self, observations, state):
    batch = observations.shape[0]
    state["lstm_h"] = state["lstm_h"] + self.bump
    state["lstm_c"] = state["lstm_c"] + self.bump
    return TCGLegalActionDistribution(
      legal_action_logits=torch.tensor([[0.0, 1.0]]).repeat(batch, 1),
      legal_actions=torch.tensor([[[9, 9, 9, 9], [self.action, 0, 0, 0]]]).repeat(batch, 1, 1),
      legal_action_count=torch.full((batch,), 2),
    ), None


def test_ensemble_routes_each_row_to_its_seat_element_and_keeps_states_apart():
  pool, groups, water_decks = _pool_groups()
  catalog = build_deck_build_catalog(pool)
  gate_by_element = {}
  for record in catalog.records_by_def_id.values():
    if record.card_type == GATE_CARD_TYPE:
      gate_by_element.setdefault(str(record.element).lower(), record.card_def_id)
  rows = np.zeros(4, dtype=NATIVE_DECKBUILD_OBS_DTYPE)
  rows["deck_context"]["gate_card_def_id"] = [
    gate_by_element["water"], gate_by_element["fire"], gate_by_element["water"], gate_by_element["earth"],
  ]
  observations = torch.from_numpy(rows.view(np.uint8).reshape(4, -1).copy())
  ensemble = SpecialistEnsemble(
    {"water": ("water", _ConstantPolicy(2, 10.0)), "fire": ("fire", _ConstantPolicy(5, 20.0))},
    ("fallback", _ConstantPolicy(7, 30.0)),
  )
  state = {"lstm_h": torch.arange(4.0)[:, None].repeat(1, 3), "lstm_c": torch.zeros(4, 3)}
  logits, _ = ensemble.forward_eval(observations, state)
  assert tcg_argmax_logits(logits)[:, 0].tolist() == [2, 5, 2, 7]
  assert state["lstm_h"][:, 0].tolist() == [10.0, 21.0, 12.0, 33.0]
  assert dict(ensemble.routed_rows) == {"water->water": 2, "fire->fire": 1, "earth->fallback": 1}


def test_evaluation_reset_can_fix_both_seat_decks():
  pool, groups, _ = _pool_groups()
  env = AzukiNativeEnv(num_envs=1, seed=3, deck_pool=pool, deck_building=True, evaluation_mode=True)
  try:
    env.reset_evaluation_games([{
      "env_index": 0, "seed": 77, "reference_seat": 1, "reference_deck_index": groups[0][0],
      "other_deck_index": groups[-1][0],
    }])
    rows = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)
    assert list(rows["deck_context"]["mode"]) == [0, 0]
    assert [env.draft_snapshot(seat)["main_count"] for seat in range(2)] == [50, 50]
    env.force_evaluation_truncations([0])
    record = env.drain_evaluation_records()[0]
    assert record["prebuilt"] is True
    assert record["prebuilt_deck_indices"] == [groups[-1][0], groups[0][0]]
  finally:
    env.close()
