from __future__ import annotations

import unittest

import numpy as np

from league_training import compute_learner_row_mask


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


if __name__ == "__main__":
  unittest.main()
