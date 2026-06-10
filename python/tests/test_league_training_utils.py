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


if __name__ == "__main__":
  unittest.main()
