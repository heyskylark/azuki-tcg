from __future__ import annotations

import contextlib
import unittest
from unittest.mock import patch

import numpy as np
from policy.tcg_distribution import TCGLegalActionDistribution
import torch

from league_training import (
  compute_frozen_matchup_ratio,
  LeagueConfig,
  LeaguePuffeRL,
  compute_league_active,
  compute_learner_row_mask,
  compute_trainable_row_mask,
)


class LeagueTrainingUtilsTests(unittest.TestCase):
  def test_compute_learner_row_mask(self):
    env_ids = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    env_learner_seat = np.asarray([0, 1, 0], dtype=np.int32)
    mask = compute_learner_row_mask(
      env_ids,
      agents_per_env=2,
      env_learner_seat=env_learner_seat,
    )
    expected = np.asarray([True, False, False, True, True, False], dtype=np.bool_)
    self.assertTrue(np.array_equal(mask, expected))

  def test_compute_trainable_row_mask_marks_latest_policy_rows(self):
    env_ids = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    env_learner_seat = np.asarray([0, 1, 0], dtype=np.int32)
    env_use_latest = np.asarray([False, True, False], dtype=np.bool_)
    mask = compute_trainable_row_mask(
      env_ids,
      agents_per_env=2,
      env_learner_seat=env_learner_seat,
      env_use_latest=env_use_latest,
      league_active=True,
    )
    expected = np.asarray([True, False, True, True, True, False], dtype=np.bool_)
    self.assertTrue(np.array_equal(mask, expected))

  def test_compute_trainable_row_mask_inactive_league_trains_all_rows(self):
    env_ids = np.asarray([0, 1, 2, 3], dtype=np.int64)
    env_learner_seat = np.asarray([1, 0], dtype=np.int32)
    env_use_latest = np.asarray([False, False], dtype=np.bool_)
    mask = compute_trainable_row_mask(
      env_ids,
      agents_per_env=2,
      env_learner_seat=env_learner_seat,
      env_use_latest=env_use_latest,
      league_active=False,
    )
    self.assertTrue(np.array_equal(mask, np.ones(4, dtype=np.bool_)))

  def test_compute_frozen_matchup_ratio_two_player_mapping(self):
    self.assertAlmostEqual(
      compute_frozen_matchup_ratio(frozen_row_ratio=0.0, agents_per_env=2),
      0.0,
    )
    self.assertAlmostEqual(
      compute_frozen_matchup_ratio(frozen_row_ratio=0.2, agents_per_env=2),
      0.4,
    )
    self.assertAlmostEqual(
      compute_frozen_matchup_ratio(frozen_row_ratio=0.5, agents_per_env=2),
      1.0,
    )

  def test_compute_league_active_threshold(self):
    self.assertTrue(compute_league_active(global_step=0, activate_after_steps=0))
    self.assertFalse(compute_league_active(global_step=79_999_999, activate_after_steps=80_000_000))
    self.assertTrue(compute_league_active(global_step=80_000_000, activate_after_steps=80_000_000))
    self.assertTrue(compute_league_active(global_step=1, activate_after_steps=-1))

  def test_zero_done_states_uses_row_indices(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._use_rnn = True
    trainer.total_agents = 6
    trainer.config = {"device": "cpu"}
    trainer.opponent_policies = [object()]
    trainer._learner_lstm_h = torch.ones((6, 3), dtype=torch.float32)
    trainer._learner_lstm_c = torch.ones((6, 3), dtype=torch.float32)
    trainer._opp_lstm_h = [torch.ones((6, 3), dtype=torch.float32)]
    trainer._opp_lstm_c = [torch.ones((6, 3), dtype=torch.float32)]

    LeaguePuffeRL._zero_done_states(trainer, np.asarray([1, 4, 4, -1, 42], dtype=np.int64))

    for state in [trainer._learner_lstm_h, trainer._learner_lstm_c, trainer._opp_lstm_h[0], trainer._opp_lstm_c[0]]:
      self.assertTrue(torch.equal(state[1], torch.zeros(3)))
      self.assertTrue(torch.equal(state[4], torch.zeros(3)))
      self.assertTrue(torch.equal(state[0], torch.ones(3)))
      self.assertTrue(torch.equal(state[2], torch.ones(3)))

  def test_infer_actions_latest_opponent_rows_are_trainable(self):
    class _FakePolicy:
      def __init__(self, label: str, value: float):
        self.label = label
        self.value = value

    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer.config = {"device": "cpu"}
    trainer.league_cfg = LeagueConfig(enabled=True, latest_ratio=1.0, activate_after_steps=0)
    trainer.global_step = 0
    trainer._agents_per_env = 2
    trainer._env_learner_seat = np.asarray([0], dtype=np.int32)
    trainer._env_use_latest = np.asarray([True], dtype=np.bool_)
    trainer._use_rnn = False
    trainer._learner_lstm_h = None
    trainer._learner_lstm_c = None
    trainer._opp_lstm_h = []
    trainer._opp_lstm_c = []
    trainer.policy = _FakePolicy("latest", 1.0)
    trainer.opponent_policies = []
    trainer.vecenv = type("VecEnv", (), {"single_action_space": type("Space", (), {"shape": (4,)})()})()
    trainer.amp_context = contextlib.nullcontext()
    trainer._split_value_heads_enabled = lambda: False
    trainer._safe_forward_eval = lambda model, obs, state: (
      {"label": model.label, "batch": obs.shape[0]},
      torch.full((obs.shape[0], 1), model.value, dtype=torch.float32),
    )

    def fake_sample_logits(logits, action=None):
      batch = int(logits["batch"])
      if logits["label"] == "latest":
        logprob = 0.7
        action_value = 1
      else:  # pragma: no cover - defensive
        logprob = -0.7
        action_value = 2
      actions = torch.full((batch, 4), action_value, dtype=torch.int32)
      logprobs = torch.full((batch,), logprob, dtype=torch.float32)
      entropy = torch.zeros((batch,), dtype=torch.float32)
      return actions, logprobs, entropy

    with patch("league_training.azk_pytorch.sample_logits", side_effect=fake_sample_logits):
      (
        _actions,
        logprobs,
        _values,
        _terminal_values,
        _shaped_values,
        trainable_rows,
        learner_rows,
        _env_indices,
      ) = LeaguePuffeRL._infer_actions(
        trainer,
        torch.zeros((2, 3), dtype=torch.float32),
        torch.ones((2,), dtype=torch.bool),
        np.asarray([0, 1], dtype=np.int64),
      )

    self.assertTrue(np.array_equal(learner_rows, np.asarray([True, False], dtype=np.bool_)))
    self.assertTrue(np.array_equal(trainable_rows, np.asarray([True, True], dtype=np.bool_)))
    self.assertTrue(torch.allclose(logprobs, torch.full((2,), 0.7, dtype=torch.float32)))

  def test_infer_actions_frozen_opponent_rows_stay_frozen(self):
    class _FakePolicy:
      def __init__(self, label: str, value: float):
        self.label = label
        self.value = value

    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer.config = {"device": "cpu"}
    trainer.league_cfg = LeagueConfig(enabled=True, frozen_ratio=0.2, latest_ratio=None, activate_after_steps=0)
    trainer.global_step = 0
    trainer._agents_per_env = 2
    trainer._env_learner_seat = np.asarray([0], dtype=np.int32)
    trainer._env_use_latest = np.asarray([False], dtype=np.bool_)
    trainer._env_opp_policy = np.asarray([0], dtype=np.int32)
    trainer._use_rnn = False
    trainer._learner_lstm_h = None
    trainer._learner_lstm_c = None
    trainer._opp_lstm_h = []
    trainer._opp_lstm_c = []
    trainer.policy = _FakePolicy("latest", 1.0)
    trainer.opponent_policies = [_FakePolicy("frozen", -1.0)]
    trainer.vecenv = type("VecEnv", (), {"single_action_space": type("Space", (), {"shape": (4,)})()})()
    trainer.amp_context = contextlib.nullcontext()
    trainer._split_value_heads_enabled = lambda: False
    trainer._safe_forward_eval = lambda model, obs, state: (
      {"label": model.label, "batch": obs.shape[0]},
      torch.full((obs.shape[0], 1), model.value, dtype=torch.float32),
    )

    def fake_sample_logits(logits, action=None):
      batch = int(logits["batch"])
      if logits["label"] == "latest":
        logprob = 0.7
        action_value = 1
      else:
        logprob = -0.7
        action_value = 2
      actions = torch.full((batch, 4), action_value, dtype=torch.int32)
      logprobs = torch.full((batch,), logprob, dtype=torch.float32)
      entropy = torch.zeros((batch,), dtype=torch.float32)
      return actions, logprobs, entropy

    with patch("league_training.azk_pytorch.sample_logits", side_effect=fake_sample_logits):
      (
        actions,
        logprobs,
        _values,
        _terminal_values,
        _shaped_values,
        trainable_rows,
        learner_rows,
        _env_indices,
      ) = LeaguePuffeRL._infer_actions(
        trainer,
        torch.zeros((2, 3), dtype=torch.float32),
        torch.ones((2,), dtype=torch.bool),
        np.asarray([0, 1], dtype=np.int64),
      )

    self.assertTrue(np.array_equal(learner_rows, np.asarray([True, False], dtype=np.bool_)))
    self.assertTrue(np.array_equal(trainable_rows, np.asarray([True, False], dtype=np.bool_)))
    self.assertTrue(torch.all(actions[0] == 1))
    self.assertTrue(torch.all(actions[1] == 2))
    self.assertTrue(torch.allclose(logprobs, torch.tensor([0.7, 0.0], dtype=torch.float32)))

  def test_safe_forward_eval_slices_single_row_legal_distribution(self):
    class _FakeModel:
      def forward_eval(self, obs, state):
        return (
          TCGLegalActionDistribution(
            legal_action_logits=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            legal_actions=torch.tensor(
              [
                [[1, 0, 0, 0], [2, 0, 0, 0]],
                [[3, 0, 0, 0], [4, 0, 0, 0]],
              ],
              dtype=torch.int64,
            ),
            legal_action_count=torch.tensor([2, 2], dtype=torch.int64),
          ),
          torch.tensor([[5.0], [6.0]], dtype=torch.float32),
        )

    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._use_rnn = False
    trainer.amp_context = contextlib.nullcontext()

    logits, values = LeaguePuffeRL._safe_forward_eval(
      trainer,
      _FakeModel(),
      torch.zeros((1, 3), dtype=torch.float32),
      {"mask": torch.ones((1,), dtype=torch.bool)},
    )

    self.assertEqual(tuple(logits.legal_action_logits.shape), (1, 2))
    self.assertEqual(tuple(logits.legal_actions.shape), (1, 2, 4))
    self.assertEqual(tuple(logits.legal_action_count.shape), (1,))
    self.assertEqual(tuple(values.shape), (1, 1))



class FrozenWindowSamplingTests(unittest.TestCase):
  def _make_trainer(self, *, pool_size: int, window_epochs: int, k: int = 1, epoch: int = 0):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer.league_cfg = LeagueConfig(
      enabled=True,
      frozen_ratio=0.5,
      randomize_learner_seat=False,
      frozen_window_epochs=window_epochs,
      max_distinct_frozen=k,
    )
    trainer._rng = np.random.default_rng(0)
    trainer.opponent_policies = [object() for _ in range(pool_size)]
    trainer.epoch = epoch
    trainer.global_step = 10_000
    trainer._agents_per_env = 2
    trainer._num_envs_total = 64
    trainer._env_learner_seat = np.zeros(64, dtype=np.int32)
    trainer._env_opp_policy = np.zeros(64, dtype=np.int32)
    trainer._env_use_latest = np.ones(64, dtype=np.bool_)
    trainer._window_policy_ids = None
    trainer._window_index = -1
    return trainer

  def test_window_restricts_assignments_to_k_ids(self):
    trainer = self._make_trainer(pool_size=6, window_epochs=8, k=1)
    trainer._refresh_frozen_window()
    trainer._resample_matchups(np.arange(64, dtype=np.int32))
    distinct = set(np.unique(trainer._env_opp_policy).tolist())
    self.assertEqual(len(distinct), 1)
    self.assertTrue(all(0 <= i < 6 for i in distinct))

  def test_window_redraws_on_epoch_boundary(self):
    trainer = self._make_trainer(pool_size=6, window_epochs=4, k=1, epoch=0)
    trainer._refresh_frozen_window()
    first = trainer._window_policy_ids.copy()
    trainer.epoch = 3
    trainer._refresh_frozen_window()
    np.testing.assert_array_equal(first, trainer._window_policy_ids)
    seen = {int(first[0])}
    for boundary_epoch in (4, 8, 12, 16, 20, 24):
      trainer.epoch = boundary_epoch
      trainer._refresh_frozen_window()
      seen.add(int(trainer._window_policy_ids[0]))
    self.assertGreater(len(seen), 1)

  def test_disabled_window_uses_whole_pool(self):
    trainer = self._make_trainer(pool_size=6, window_epochs=0)
    trainer._refresh_frozen_window()
    self.assertIsNone(trainer._window_policy_ids)
    trainer._resample_matchups(np.arange(64, dtype=np.int32))
    distinct = set(np.unique(trainer._env_opp_policy).tolist())
    self.assertGreater(len(distinct), 2)

  def test_window_survives_pool_shrink(self):
    trainer = self._make_trainer(pool_size=6, window_epochs=8, k=2)
    trainer._refresh_frozen_window()
    trainer.opponent_policies = [object()]  # pool shrank below old ids
    trainer._refresh_frozen_window()
    self.assertTrue(int(trainer._window_policy_ids.max()) < 1)

if __name__ == "__main__":
  unittest.main()
