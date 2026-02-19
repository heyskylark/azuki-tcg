from __future__ import annotations

import unittest

import numpy as np
import torch

from league_training import LeaguePuffeRL, compute_league_active, compute_learner_row_mask


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


if __name__ == "__main__":
  unittest.main()
