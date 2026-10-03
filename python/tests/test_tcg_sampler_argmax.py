from __future__ import annotations

import unittest

import torch

from policy.tcg_distribution import TCGLegalActionDistribution
from policy.v2.tcg_sampler import (
  legal_action_logprob_components,
  legal_action_row_distribution,
  legal_action_kl,
  tcg_argmax_logits,
  tcg_sample_logits,
)


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

  def test_legal_row_logprob_components_reconstruct_flat_action_probability(self):
    distribution = TCGLegalActionDistribution(
      legal_action_logits=torch.zeros((1, 4), dtype=torch.float32),
      legal_actions=torch.tensor(
        [[
          [0, 0, 0, 0],
          [0, 1, 0, 0],
          [1, 0, 0, 0],
          [1, 0, 1, 0],
        ]],
        dtype=torch.int64,
      ),
      legal_action_count=torch.tensor([4], dtype=torch.int64),
    )
    action = torch.tensor([[1, 0, 1, 0]], dtype=torch.int64)

    _, total_logprob, _ = tcg_sample_logits(distribution, action=action)
    components = legal_action_logprob_components(distribution, action)

    torch.testing.assert_close(
      components,
      torch.tensor([[-0.6931472, 0.0, -0.6931472, 0.0]]),
    )
    torch.testing.assert_close(components.sum(dim=-1), total_logprob)

  def test_duplicate_legal_rows_share_one_environment_action_probability(self):
    duplicated_action = torch.tensor([[6, 1, 2, 0]], dtype=torch.int64)
    distribution = TCGLegalActionDistribution(
      legal_action_logits=torch.zeros((1, 3), dtype=torch.float32),
      legal_actions=torch.tensor(
        [[[6, 1, 2, 0], [6, 1, 2, 0], [0, 0, 0, 0]]],
        dtype=torch.int64,
      ),
      legal_action_count=torch.tensor([3]),
    )

    _, total_logprob, _ = tcg_sample_logits(
      distribution,
      action=duplicated_action,
    )
    components = legal_action_logprob_components(distribution, duplicated_action)

    torch.testing.assert_close(total_logprob, torch.tensor([-0.4054651]))
    torch.testing.assert_close(components.sum(dim=-1), total_logprob)

  def test_legal_row_logprobs_remain_normalized_below_old_epsilon_floor(self):
    distribution = TCGLegalActionDistribution(
      legal_action_logits=torch.tensor([[0.0, -30.0, 100.0]], dtype=torch.float32),
      legal_actions=torch.tensor(
        [[[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]],
        dtype=torch.int64,
      ),
      legal_action_count=torch.tensor([2], dtype=torch.int64),
    )

    probs, log_probs, row_mask = legal_action_row_distribution(distribution)
    _, selected_logprob, _ = tcg_sample_logits(
      distribution,
      action=torch.tensor([[1, 0, 0, 0]], dtype=torch.int64),
    )

    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1))
    self.assertLess(float(selected_logprob.item()), -18.4206807)
    torch.testing.assert_close(selected_logprob, log_probs[:, 1])
    self.assertFalse(bool(row_mask[:, 2].item()))
    self.assertEqual(float(probs[:, 2].item()), 0.0)

  def test_exact_legal_action_kl_ignores_padding(self):
    legal_actions = torch.tensor(
      [[[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]],
      dtype=torch.int64,
    )
    old = TCGLegalActionDistribution(
      legal_action_logits=torch.tensor([[1.0986123, 0.0, 50.0]]),
      legal_actions=legal_actions,
      legal_action_count=torch.tensor([2]),
    )
    new = TCGLegalActionDistribution(
      legal_action_logits=torch.tensor([[0.0, 0.0, -50.0]]),
      legal_actions=legal_actions,
      legal_action_count=torch.tensor([2]),
    )

    exact_kl = legal_action_kl(old, new)

    torch.testing.assert_close(exact_kl, torch.tensor([0.13081204]))

  def test_exact_kl_ignores_probability_movement_between_duplicate_rows(self):
    legal_actions = torch.tensor(
      [[[6, 1, 2, 0], [6, 1, 2, 0], [0, 0, 0, 0]]],
      dtype=torch.int64,
    )
    old = TCGLegalActionDistribution(
      legal_action_logits=torch.tensor([[-0.22314355, -1.609438, 0.0]]),
      legal_actions=legal_actions,
      legal_action_count=torch.tensor([3]),
    )
    new = TCGLegalActionDistribution(
      legal_action_logits=torch.tensor([[-1.609438, -0.22314355, 0.0]]),
      legal_actions=legal_actions,
      legal_action_count=torch.tensor([3]),
    )

    torch.testing.assert_close(
      legal_action_kl(old, new),
      torch.zeros(1),
      atol=1e-7,
      rtol=0.0,
    )


if __name__ == "__main__":
  unittest.main()
