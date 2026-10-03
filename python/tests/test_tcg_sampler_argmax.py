from __future__ import annotations

import unittest

import torch

from policy.tcg_distribution import TCGLegalActionDistribution
from policy.v2.tcg_sampler import tcg_argmax_logits


class TCGSamplerArgmaxTests(unittest.TestCase):
  def test_legal_row_argmax_masks_padding_and_breaks_ties_by_first_row(self):
    distribution = TCGLegalActionDistribution(
      legal_action_logits=torch.tensor(
        [[1.0, 4.0, 4.0, 100.0], [3.0, 2.0, 99.0, 99.0]], dtype=torch.float32
      ),
      legal_actions=torch.tensor(
        [
          [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
          [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7]],
        ],
        dtype=torch.int64,
      ),
      legal_action_count=torch.tensor([3, 2], dtype=torch.int64),
    )

    actions = tcg_argmax_logits(distribution)

    self.assertTrue(torch.equal(actions[0], torch.tensor([1, 1, 1, 1])))
    self.assertTrue(torch.equal(actions[1], torch.tensor([4, 4, 4, 4])))


if __name__ == "__main__":
  unittest.main()
