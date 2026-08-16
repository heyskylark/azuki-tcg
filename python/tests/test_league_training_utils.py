from __future__ import annotations

import contextlib
from collections import defaultdict
import unittest
from unittest.mock import patch

import numpy as np
from observation import DECKBUILD_OBSERVATION_CTYPE
from policy.tcg_distribution import TCGLegalActionDistribution
import torch

from azk_puffer.trainer import (
  PuffeRL,
  recombine_reward_components,
  terminal_win_labels_from_rewards,
  trainer_shaped_reward_multiplier,
)
from league_training import (
  clipped_terminal_policy_loss,
  compensate_frozen_matchup_ratio_for_reference,
  compute_battle_row_mask,
  compute_frozen_matchup_ratio,
  detect_reset_reference_seats,
  deterministic_draft_credit_pick,
  deterministic_draft_episode_priority,
  deterministic_draft_prefix_candidate,
  deterministic_draft_prefix_length,
  draft_episode_credit_update_due,
  DRAFT_TERMINAL_CREDIT_QUARTILES,
  LeagueConfig,
  LeaguePuffeRL,
  masked_tensor_mean,
  masked_tensor_mean_std,
  pad_complete_draft_episodes,
  pad_terminal_credit_records,
  parse_draft_prefix_distribution,
  take_terminal_credit_training_records,
  compute_league_active,
  compute_learner_row_mask,
  compute_trainable_row_mask,
  terminal_only_rewards,
)


class TerminalCreditTests(unittest.TestCase):
  def test_random_prefix_distribution_is_deterministic_and_has_expected_mean(self):
    lengths, probabilities = parse_draft_prefix_distribution(
      "0,1,2,4",
      "0.5,0.25,0.125,0.125",
    )
    first = [
      deterministic_draft_prefix_length(i, i % 2, 420053, lengths, probabilities)
      for i in range(10_000)
    ]
    second = [
      deterministic_draft_prefix_length(i, i % 2, 420053, lengths, probabilities)
      for i in range(10_000)
    ]

    self.assertEqual(first, second)
    self.assertLess(abs(float(np.mean(first)) - 1.0), 0.05)

  def test_random_prefix_candidate_is_legal_and_pick_specific(self):
    picks = [
      deterministic_draft_prefix_candidate(17, 1, pick, 73, 420053)
      for pick in range(1, 5)
    ]

    self.assertTrue(all(0 <= pick < 73 for pick in picks))
    self.assertGreater(len(set(picks)), 1)

  def test_random_prefix_overrides_only_supplied_live_policy_rows(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._draft_prefix_enabled = True
    trainer._draft_prefix_lengths = (4,)
    trainer._draft_prefix_probabilities = (1.0,)
    trainer._draft_prefix_seed = 420053
    trainer._draft_prefix_episode_lengths = []
    trainer._draft_prefix_forced_rows = 0
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([17], dtype=np.int64)
    trainer.amp_context = contextlib.nullcontext()

    def fake_sample_logits(_logits, action=None):
      return action, action[:, 1].float(), torch.zeros(action.shape[0])

    with patch(
      "league_training.azk_pytorch.sample_logits",
      side_effect=fake_sample_logits,
    ):
      actions, logprobs, forced = LeaguePuffeRL._apply_random_draft_prefix(
        trainer,
        logits=object(),
        actions=torch.zeros((1, 4), dtype=torch.long),
        logprobs=torch.zeros(1),
        policy_rows_np=np.asarray([0], dtype=np.int64),
        env_id_np=np.asarray([0, 1], dtype=np.int64),
        draft_rows_np=np.asarray([True, True]),
        draft_main_counts_np=np.asarray([0, 0], dtype=np.int16),
        draft_legal_counts_np=np.asarray([7, 7], dtype=np.int16),
      )

    self.assertEqual(int(actions[0, 0]), 3)
    self.assertTrue(0 <= int(actions[0, 1]) < 7)
    self.assertEqual(float(logprobs[0]), float(actions[0, 1]))
    self.assertTrue(bool(forced[0]))
    self.assertEqual(trainer._draft_prefix_forced_rows, 1)
    self.assertEqual(trainer._draft_prefix_episode_lengths, [4])

  def test_terminal_reward_labels_use_sign_and_ignore_nonterminal_rows(self):
    labels = terminal_win_labels_from_rewards(
      np.asarray([0, 1, 2, 3], dtype=np.int64),
      np.asarray([True, True, False, False], dtype=np.bool_),
      np.asarray([1.0, -1.0, 0.25, -0.25], dtype=np.float32),
      agents_per_env=2,
    )
    self.assertEqual(labels, {0: {0: 1.0, 1: 0.0}})

  def test_terminal_only_rewards_preserves_native_outcomes(self):
    rewards = terminal_only_rewards(
      np.asarray([0.25, -0.25, 1.0, -1.0], dtype=np.float32),
      np.asarray([False, False, True, True], dtype=np.bool_),
    )

    np.testing.assert_array_equal(rewards, [0.0, 0.0, 1.0, -1.0])
    labels = terminal_win_labels_from_rewards(
      np.asarray([0, 1, 0, 1], dtype=np.int64),
      np.asarray([False, False, True, True], dtype=np.bool_),
      rewards,
      agents_per_env=2,
    )
    self.assertEqual(labels, {0: {0: 1.0, 1: 0.0}})

  def test_direct_terminal_labels_stamp_episode_rows_and_advance_ids(self):
    trainer = PuffeRL.__new__(PuffeRL)
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([10, 11], dtype=np.int64)
    trainer._next_env_episode_id = 12
    trainer.win_prob_episode_ids = torch.tensor(
      [[10, 10], [10, 10], [11, 11], [11, 11]],
      dtype=torch.int64,
    )
    trainer.win_prob_agent_ids = torch.tensor(
      [[0, 0], [1, 1], [2, 2], [3, 3]],
      dtype=torch.int32,
    )
    trainer.win_prob_targets = torch.zeros((4, 2), dtype=torch.float32)
    trainer.win_prob_target_mask = torch.zeros((4, 2), dtype=torch.bool)

    finished = PuffeRL._assign_terminal_win_prob_targets(
      trainer,
      [{"aggregate": True}],
      np.asarray([0, 1, 2, 3], dtype=np.int64),
      np.ones(4, dtype=np.bool_),
      terminal_rewards=np.asarray([1.0, -1.0, -1.0, 1.0], dtype=np.float32),
      label_mask=np.ones(4, dtype=np.bool_),
    )

    np.testing.assert_array_equal(finished, [0, 1])
    self.assertTrue(bool(trainer.win_prob_target_mask.all()))
    torch.testing.assert_close(
      trainer.win_prob_targets,
      torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]),
    )
    np.testing.assert_array_equal(trainer._env_episode_ids, [12, 13])
    self.assertEqual(trainer._next_env_episode_id, 14)

  def test_direct_label_path_does_not_fall_back_on_truncation(self):
    trainer = PuffeRL.__new__(PuffeRL)
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([4], dtype=np.int64)
    trainer._next_env_episode_id = 5
    trainer.win_prob_episode_ids = torch.full((2, 1), 4, dtype=torch.int64)
    trainer.win_prob_agent_ids = torch.tensor([[0], [1]], dtype=torch.int32)
    trainer.win_prob_targets = torch.zeros((2, 1), dtype=torch.float32)
    trainer.win_prob_target_mask = torch.zeros((2, 1), dtype=torch.bool)

    PuffeRL._assign_terminal_win_prob_targets(
      trainer,
      [{0: {"win": 1.0}, 1: {"win": 0.0}}],
      np.asarray([0, 1], dtype=np.int64),
      np.ones(2, dtype=np.bool_),
      terminal_rewards=np.asarray([1.0, -1.0], dtype=np.float32),
      label_mask=np.zeros(2, dtype=np.bool_),
    )

    self.assertFalse(bool(trainer.win_prob_target_mask.any()))

  def test_clipped_terminal_policy_loss_matches_ppo_surrogate(self):
    old_logprobs = torch.zeros(2, dtype=torch.float32)
    new_logprobs = torch.log(torch.tensor([1.5, 0.5], dtype=torch.float32))
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)
    loss, ratio = clipped_terminal_policy_loss(
      new_logprobs,
      old_logprobs,
      advantages,
      clip_coef=0.2,
    )
    torch.testing.assert_close(ratio, torch.tensor([1.5, 0.5]))
    self.assertAlmostEqual(float(loss.item()), -0.2, places=6)

  def test_clipped_terminal_policy_loss_masks_fixed_batch_padding(self):
    old_logprobs = torch.zeros(3, dtype=torch.float32)
    new_logprobs = torch.log(torch.tensor([1.5, 0.5, 10.0], dtype=torch.float32))
    advantages = torch.tensor([1.0, -1.0, 100.0], dtype=torch.float32)
    loss, _ = clipped_terminal_policy_loss(
      new_logprobs,
      old_logprobs,
      advantages,
      clip_coef=0.2,
      mask=torch.tensor([True, True, False]),
    )
    self.assertAlmostEqual(float(loss.item()), -0.2, places=6)

  def test_draft_credit_sampler_is_deterministic_and_covers_each_quartile(self):
    for quartile, (low, high) in enumerate(DRAFT_TERMINAL_CREDIT_QUARTILES):
      picks = [
        deterministic_draft_credit_pick(episode, episode % 2, quartile, 420051)
        for episode in range(4096)
      ]
      self.assertTrue(all(low <= pick <= high for pick in picks))
      self.assertEqual(set(picks), set(range(low, high + 1)))
      self.assertEqual(
        picks,
        [
          deterministic_draft_credit_pick(episode, episode % 2, quartile, 420051)
          for episode in range(4096)
        ],
      )

  def test_draft_credit_padding_has_fixed_shape_and_valid_mask(self):
    records = [{"slot": 0}, {"slot": 1}, {"slot": 2}]
    padded, valid = pad_terminal_credit_records(records, 8)
    self.assertEqual(len(padded), 8)
    np.testing.assert_array_equal(
      valid,
      np.asarray([True, True, True, False, False, False, False, False]),
    )
    self.assertEqual(padded[:3], records)
    self.assertTrue(all(record is records[0] for record in padded[3:]))

  def test_draft_credit_training_defers_partial_batches_until_flush(self):
    records = [{"row": index} for index in range(700)]
    selected, remaining = take_terminal_credit_training_records(
      records,
      512,
      flush=False,
    )
    self.assertEqual(selected, records[:512])
    self.assertEqual(remaining, records[512:])

    selected, remaining = take_terminal_credit_training_records(
      remaining,
      512,
      flush=False,
    )
    self.assertEqual(selected, [])
    self.assertEqual(remaining, records[512:])

    selected, remaining = take_terminal_credit_training_records(
      remaining,
      512,
      flush=True,
    )
    self.assertEqual(selected, records[512:])
    self.assertEqual(remaining, [])

  def test_draft_credit_training_rejects_invalid_batch_size(self):
    with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
      take_terminal_credit_training_records([], 0, flush=False)

  def test_complete_draft_padding_is_episode_shaped(self):
    records = [{"episode": 1}, {"episode": 2}]
    padded, valid = pad_complete_draft_episodes(records, 4)
    self.assertEqual(len(padded), 4)
    np.testing.assert_array_equal(valid, [True, True, False, False])
    self.assertIs(padded[2], records[0])
    self.assertIs(padded[3], records[0])

  def test_masked_tensor_mean_excludes_padding(self):
    values = torch.tensor([[1.0, 3.0], [100.0, 100.0]])
    mask = torch.tensor([[True, True], [False, False]])
    self.assertEqual(float(masked_tensor_mean(values, mask)), 2.0)

  def test_masked_tensor_mean_std_matches_selected_rows_and_handles_empty(self):
    values = torch.tensor([[1.0, 100.0], [3.0, 5.0]])
    mask = torch.tensor([[True, False], [True, True]])
    mean, std, count = masked_tensor_mean_std(values, mask)
    selected = torch.tensor([1.0, 3.0, 5.0])
    torch.testing.assert_close(mean, selected.mean())
    torch.testing.assert_close(std, selected.std())
    self.assertEqual(float(count), 3.0)

    empty_mean, empty_std, empty_count = masked_tensor_mean_std(
      values,
      torch.zeros_like(mask),
    )
    self.assertEqual(float(empty_mean), 0.0)
    self.assertEqual(float(empty_std), 0.0)
    self.assertEqual(float(empty_count), 0.0)

  def test_full_episode_credit_decode_excludes_inactive_draft_seat(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._draft_episode_credit_enabled = True
    trainer._draft_episode_credit_layout = None
    trainer._draft_episode_credit_coef = 0.1
    trainer._draft_episode_credit_baseline_coef = 0.01
    trainer._draft_episode_credit_clip = 0.2
    trainer._draft_episode_credit_batch_drafts = 80
    trainer._draft_episode_credit_update_interval = 4
    trainer._draft_episode_credit_seed = 420052
    dtype = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
    observations = np.zeros(2, dtype=dtype)
    observations["deck_context"]["mode"] = 2
    observations["deck_context"]["main_count"] = 7
    observations["action_mask"]["legal_action_count"] = [3, 0]
    packed = torch.from_numpy(observations.view(np.uint8).reshape(2, -1).copy())

    eligible, modes, main_counts, _, _, legal_counts = (
      LeaguePuffeRL._decode_draft_episode_credit_rows(
        trainer,
        packed,
        np.asarray([True, True]),
      )
    )

    np.testing.assert_array_equal(eligible, [True, False])
    np.testing.assert_array_equal(modes, [2, 2])
    np.testing.assert_array_equal(main_counts, [7, 7])
    np.testing.assert_array_equal(legal_counts, [3, 0])

  def test_full_episode_credit_retains_all_main_picks_and_labels_terminal(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._draft_episode_credit_enabled = True
    trainer._draft_episode_credit_pending = {}
    trainer._draft_episode_credit_ready = []
    trainer._draft_episode_credit_captured = 0
    trainer._draft_episode_credit_labeled = 0
    trainer._draft_episode_credit_truncated = 0
    trainer._draft_episode_credit_draws = 0
    trainer._draft_episode_credit_decisive = 0
    trainer._draft_episode_credit_wins = 0
    trainer._draft_episode_credit_losses = 0
    trainer._draft_episode_credit_incomplete = 0
    trainer._draft_episode_credit_completed = 0
    trainer._draft_episode_credit_sample_dropped = 0
    trainer._draft_episode_credit_window_completed = 0
    trainer._draft_episode_credit_window_sample_dropped = 0
    trainer._draft_episode_credit_batch_drafts = 80
    trainer._draft_episode_credit_seed = 420052
    trainer._draft_episode_credit_capture_seconds = 0.0
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([17], dtype=np.int64)
    trainer.epoch = 5
    observations = torch.zeros((2, 4), dtype=torch.uint8)
    actions = torch.zeros((1, 4), dtype=torch.long)
    logprobs = torch.zeros(1, dtype=torch.float32)
    for main_count in range(50):
      observations[0, 0] = main_count
      LeaguePuffeRL._stash_draft_episode_credit_rows(
        trainer,
        observations_cpu=observations,
        policy_rows_np=np.asarray([0], dtype=np.int64),
        actions=actions,
        logprobs=logprobs,
        pre_lstm_h=torch.zeros((1, 3), dtype=torch.float32),
        pre_lstm_c=torch.zeros((1, 3), dtype=torch.float32),
        env_id_np=np.asarray([0, 1], dtype=np.int64),
        draft_rows_np=np.asarray([True, False]),
        draft_main_counts_np=np.asarray([main_count, 0], dtype=np.int16),
        draft_gate_ids_np=np.asarray([122, 126], dtype=np.int16),
        draft_leader_ids_np=np.asarray([121, 125], dtype=np.int16),
      )
    pending = trainer._draft_episode_credit_pending[(17, 0)]
    self.assertTrue(bool(pending["seen"].all()))
    self.assertEqual(trainer._draft_episode_credit_captured, 50)
    for field in (
      "observations",
      "actions",
      "old_logprobs",
      "lstm_h",
      "lstm_c",
      "actor_valid",
    ):
      self.assertTrue(all(row.device.type == "cpu" for row in pending[field]))
    self.assertEqual(
      [int(observation[0]) for observation in pending["observations"]],
      list(range(50)),
    )

    LeaguePuffeRL._finalize_draft_episode_credit_rows(
      trainer,
      np.asarray([0, 1], dtype=np.int64),
      np.asarray([True, True]),
      np.asarray([True, True]),
      np.asarray([1.0, -1.0], dtype=np.float32),
    )
    self.assertNotIn((17, 0), trainer._draft_episode_credit_pending)
    self.assertEqual(len(trainer._draft_episode_credit_ready), 1)
    self.assertEqual(trainer._draft_episode_credit_ready[0]["target"], 1.0)
    self.assertEqual(
      trainer._draft_episode_credit_ready[0]["observations"][:, 0].tolist(),
      list(range(50)),
    )
    self.assertEqual(trainer._draft_episode_credit_labeled, 50)
    self.assertEqual(trainer._draft_episode_credit_decisive, 1)
    self.assertEqual(trainer._draft_episode_credit_wins, 1)
    self.assertEqual(trainer._draft_episode_credit_losses, 0)

  def test_full_episode_credit_drops_truncations(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._draft_episode_credit_enabled = True
    trainer._draft_episode_credit_pending = {
      (4, 0): {
        "seen": np.ones(50, dtype=np.bool_),
      }
    }
    trainer._draft_episode_credit_ready = []
    trainer._draft_episode_credit_truncated = 0
    trainer._draft_episode_credit_incomplete = 0
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([4], dtype=np.int64)
    trainer.epoch = 3
    LeaguePuffeRL._finalize_draft_episode_credit_rows(
      trainer,
      np.asarray([0, 1], dtype=np.int64),
      np.asarray([True, True]),
      np.asarray([False, False]),
      np.asarray([0.0, 0.0], dtype=np.float32),
    )
    self.assertEqual(trainer._draft_episode_credit_ready, [])
    self.assertEqual(trainer._draft_episode_credit_truncated, 50)

  def test_full_episode_credit_keeps_true_terminal_draw_as_half_outcome(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._draft_episode_credit_enabled = True
    trainer._draft_episode_credit_pending = {
      (4, 0): {
        "episode_id": 4,
        "seat": 0,
        "gate_id": 122,
        "leader_id": 121,
        "seen": np.ones(50, dtype=np.bool_),
        "observations": [torch.zeros(4, dtype=torch.uint8) for _ in range(50)],
        "actions": [torch.zeros(4, dtype=torch.long) for _ in range(50)],
        "old_logprobs": [torch.zeros(()) for _ in range(50)],
        "lstm_h": [torch.zeros(3) for _ in range(50)],
        "lstm_c": [torch.zeros(3) for _ in range(50)],
        "actor_valid": [torch.ones((), dtype=torch.bool) for _ in range(50)],
        "capture_epochs": np.zeros(50, dtype=np.int32),
      }
    }
    trainer._draft_episode_credit_ready = []
    trainer._draft_episode_credit_truncated = 0
    trainer._draft_episode_credit_draws = 0
    trainer._draft_episode_credit_decisive = 0
    trainer._draft_episode_credit_wins = 0
    trainer._draft_episode_credit_losses = 0
    trainer._draft_episode_credit_incomplete = 0
    trainer._draft_episode_credit_labeled = 0
    trainer._draft_episode_credit_completed = 0
    trainer._draft_episode_credit_sample_dropped = 0
    trainer._draft_episode_credit_window_completed = 0
    trainer._draft_episode_credit_window_sample_dropped = 0
    trainer._draft_episode_credit_batch_drafts = 80
    trainer._draft_episode_credit_seed = 420052
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([4], dtype=np.int64)
    trainer.epoch = 3

    LeaguePuffeRL._finalize_draft_episode_credit_rows(
      trainer,
      np.asarray([0, 1], dtype=np.int64),
      np.asarray([True, True]),
      np.asarray([True, True]),
      np.asarray([0.0, 0.0], dtype=np.float32),
    )

    self.assertEqual(len(trainer._draft_episode_credit_ready), 1)
    self.assertEqual(trainer._draft_episode_credit_ready[0]["target"], 0.5)
    self.assertEqual(trainer._draft_episode_credit_draws, 1)
    self.assertEqual(trainer._draft_episode_credit_decisive, 0)
    self.assertEqual(trainer._draft_episode_credit_wins, 0)
    self.assertEqual(trainer._draft_episode_credit_losses, 0)
    self.assertEqual(trainer._draft_episode_credit_truncated, 0)

  def test_full_episode_credit_sample_is_bounded_and_timing_unbiased(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._draft_episode_credit_batch_drafts = 2
    trainer._draft_episode_credit_seed = 91
    trainer._draft_episode_credit_ready = []
    trainer._draft_episode_credit_completed = 0
    trainer._draft_episode_credit_sample_dropped = 0
    trainer._draft_episode_credit_window_completed = 0
    trainer._draft_episode_credit_window_sample_dropped = 0
    records = [
      {"episode_id": episode_id, "seat": episode_id % 2}
      for episode_id in range(7)
    ]

    for record in records:
      LeaguePuffeRL._offer_completed_draft_episode_credit(trainer, record)

    expected = sorted(
      records,
      key=lambda record: deterministic_draft_episode_priority(
        int(record["episode_id"]),
        int(record["seat"]),
        91,
      ),
    )[:2]
    self.assertEqual(
      {int(record["episode_id"]) for record in trainer._draft_episode_credit_ready},
      {int(record["episode_id"]) for record in expected},
    )
    self.assertEqual(trainer._draft_episode_credit_completed, 7)
    self.assertEqual(trainer._draft_episode_credit_sample_dropped, 5)

  def test_full_episode_credit_retains_sample_until_update_interval(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._draft_episode_credit_enabled = True
    trainer._draft_episode_credit_update_interval = 4
    trainer._draft_episode_credit_ready = [{"sample_priority": 1}]
    trainer.epoch = 10
    trainer.total_epochs = 20

    metrics = LeaguePuffeRL._train_draft_episode_credit(trainer)

    self.assertEqual(metrics["draft_episode_credit_examples"], 0.0)
    self.assertEqual(len(trainer._draft_episode_credit_ready), 1)

  def test_full_episode_credit_update_schedule_uses_post_train_epoch_and_flushes(self):
    self.assertFalse(draft_episode_credit_update_due(10, 20, 4))
    self.assertTrue(draft_episode_credit_update_due(11, 20, 4))
    self.assertTrue(draft_episode_credit_update_due(18, 19, 4))

  def test_full_episode_credit_masks_forced_actor_rows_but_trains_baseline(self):
    class FakePolicy(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.actor = torch.nn.Parameter(torch.zeros(50))
        self.baseline = torch.nn.Parameter(torch.tensor(0.0))

      def _cg_eager_forward_eval(self, observations, state):
        state["_azk_win_prob_logits"] = self.baseline.expand(observations.shape[0])
        return {"pick": observations[:, 0].long()}, torch.zeros(observations.shape[0])

      def forward_eval(self, observations, state):
        raise AssertionError("credit training must bypass rollout CUDA graphs")

    policy = FakePolicy()
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._draft_episode_credit_enabled = True
    trainer._draft_episode_credit_batch_drafts = 2
    trainer._draft_episode_credit_update_interval = 1
    trainer._draft_episode_credit_coef = 1.0
    trainer._draft_episode_credit_baseline_coef = 0.1
    trainer._draft_episode_credit_clip = 0.2
    trainer._draft_episode_credit_ready = [
      {
        "observations": torch.nn.functional.pad(
          torch.arange(50, dtype=torch.uint8).reshape(50, 1),
          (0, 3),
        ),
        "actions": torch.zeros((50, 4), dtype=torch.long),
        "old_logprobs": torch.zeros(50),
        "lstm_h": torch.zeros((50, 3)),
        "lstm_c": torch.zeros((50, 3)),
        "actor_valid": torch.arange(50) >= 4,
        "capture_epochs": np.arange(50, dtype=np.int32) * 0 + 2,
        "terminal_epoch": 8,
        "target": 1.0,
        "gate_id": 122,
        "leader_id": 121,
        "sample_priority": 1,
      }
    ]
    trainer._draft_episode_credit_window_completed = 1
    trainer._draft_episode_credit_window_sample_dropped = 0
    trainer.config = {
      "device": "cpu",
      "ent_coef": 0.0,
      "max_grad_norm": 10.0,
    }
    trainer.policy = policy
    trainer.optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    trainer.amp_context = contextlib.nullcontext()
    trainer.epoch = 10
    trainer.total_epochs = 20

    def fake_sample_logits(logits, action=None):
      picks = logits["pick"]
      return action, policy.actor[picks], torch.ones(picks.shape[0])

    with patch(
      "league_training.azk_pytorch.sample_logits",
      side_effect=fake_sample_logits,
    ):
      metrics = LeaguePuffeRL._train_draft_episode_credit(trainer)

    self.assertEqual(metrics["draft_episode_credit_episodes"], 1.0)
    self.assertEqual(metrics["draft_episode_credit_examples"], 46.0)
    self.assertEqual(metrics["draft_episode_credit_baseline_examples"], 50.0)
    self.assertEqual(metrics["draft_episode_credit_forced_rows"], 4.0)
    self.assertEqual(metrics["draft_episode_credit_fixed_rows"], 100.0)
    self.assertEqual(metrics["draft_episode_credit_q1_examples"], 9.0)
    self.assertEqual(metrics["draft_episode_credit_q4_examples"], 12.0)
    self.assertTrue(bool((policy.actor.detach()[:4] == 0.0).all()))
    self.assertTrue(bool((policy.actor.detach()[4:] > 0.0).all()))
    self.assertGreater(float(policy.baseline.detach()), 0.0)

  def test_draft_credit_stashes_leader_and_one_pick_per_quartile(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._draft_credit_enabled = True
    trainer._draft_credit_seed = 420051
    trainer._draft_credit_pending = {}
    trainer._draft_credit_captured = 0
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([17], dtype=np.int64)
    observations = torch.zeros((2, 4), dtype=torch.uint8)
    actions = torch.zeros((1, 4), dtype=torch.long)
    logprobs = torch.zeros(1, dtype=torch.float32)
    common = {
      "observations_cpu": observations,
      "policy_rows_np": np.asarray([0], dtype=np.int64),
      "actions": actions,
      "logprobs": logprobs,
      "pre_lstm_h": None,
      "pre_lstm_c": None,
      "env_id_np": np.asarray([0, 1], dtype=np.int64),
      "draft_rows_np": np.asarray([True, False]),
      "draft_gate_ids_np": np.asarray([122, 126], dtype=np.int16),
    }
    LeaguePuffeRL._stash_selected_draft_credit_rows(
      trainer,
      **common,
      draft_modes_np=np.asarray([1, 1], dtype=np.int32),
      draft_main_counts_np=np.asarray([0, 0], dtype=np.int16),
    )
    for main_count in range(50):
      LeaguePuffeRL._stash_selected_draft_credit_rows(
        trainer,
        **common,
        draft_modes_np=np.asarray([2, 2], dtype=np.int32),
        draft_main_counts_np=np.asarray([main_count, main_count], dtype=np.int16),
      )

    pending = trainer._draft_credit_pending[(17, 0)]
    self.assertEqual(set(pending["records"]), set(range(5)))
    self.assertEqual(trainer._draft_credit_captured, 5)
    self.assertEqual(
      [pending["records"][slot]["main_pick"] for slot in range(1, 5)],
      list(pending["sampled_picks"]),
    )

  def test_finalize_draft_credit_requires_exact_terminal_and_complete_sample(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._draft_credit_enabled = True
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([7], dtype=np.int64)
    trainer._draft_credit_pending = {
      (7, seat): {
        "records": {slot: {"slot": slot} for slot in range(5)},
      }
      for seat in range(2)
    }
    trainer._draft_credit_ready = []
    trainer._draft_credit_labeled = 0
    trainer._draft_credit_truncated = 0
    trainer._draft_credit_incomplete = 0
    trainer._draft_credit_incomplete_records = 0

    LeaguePuffeRL._finalize_draft_credit_rows(
      trainer,
      np.asarray([0, 1], dtype=np.int64),
      np.ones(2, dtype=np.bool_),
      np.ones(2, dtype=np.bool_),
      np.asarray([1.0, -1.0], dtype=np.float32),
    )

    self.assertEqual(trainer._draft_credit_pending, {})
    self.assertEqual(trainer._draft_credit_labeled, 10)
    self.assertEqual(
      [record["target"] for record in trainer._draft_credit_ready],
      [1.0] * 5 + [0.0] * 5,
    )

  def test_finalize_draft_credit_drops_truncations_and_partial_samples(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._draft_credit_enabled = True
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([9], dtype=np.int64)
    trainer._draft_credit_pending = {
      (9, 0): {"records": {slot: {"slot": slot} for slot in range(5)}},
      (9, 1): {"records": {slot: {"slot": slot} for slot in range(2)}},
    }
    trainer._draft_credit_ready = []
    trainer._draft_credit_labeled = 0
    trainer._draft_credit_truncated = 0
    trainer._draft_credit_incomplete = 0
    trainer._draft_credit_incomplete_records = 0

    LeaguePuffeRL._finalize_draft_credit_rows(
      trainer,
      np.asarray([0, 1], dtype=np.int64),
      np.ones(2, dtype=np.bool_),
      np.zeros(2, dtype=np.bool_),
      np.zeros(2, dtype=np.float32),
    )
    self.assertEqual(trainer._draft_credit_ready, [])
    self.assertEqual(trainer._draft_credit_truncated, 7)

    trainer._env_episode_ids[0] = 10
    trainer._draft_credit_pending = {
      (10, 0): {"records": {slot: {"slot": slot} for slot in range(2)}}
    }
    LeaguePuffeRL._finalize_draft_credit_rows(
      trainer,
      np.asarray([0, 1], dtype=np.int64),
      np.ones(2, dtype=np.bool_),
      np.ones(2, dtype=np.bool_),
      np.asarray([1.0, -1.0], dtype=np.float32),
    )
    self.assertEqual(trainer._draft_credit_ready, [])
    self.assertEqual(trainer._draft_credit_incomplete, 1)
    self.assertEqual(trainer._draft_credit_incomplete_records, 2)

  def test_finalize_leader_credit_uses_current_episode_before_id_advance(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._leader_credit_enabled = True
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([7], dtype=np.int64)
    trainer._leader_credit_pending = {
      (7, 0): {"seat": 0},
      (7, 1): {"seat": 1},
    }
    trainer._leader_credit_ready = []
    trainer._leader_credit_completed = 0

    LeaguePuffeRL._finalize_leader_credit_rows(
      trainer,
      np.asarray([0, 1], dtype=np.int64),
      np.ones(2, dtype=np.bool_),
      np.ones(2, dtype=np.bool_),
      np.asarray([1.0, -1.0], dtype=np.float32),
    )

    self.assertEqual(trainer._leader_credit_pending, {})
    self.assertEqual(trainer._leader_credit_completed, 2)
    self.assertEqual(
      [record["target"] for record in trainer._leader_credit_ready],
      [1.0, 0.0],
    )

  def test_finalize_leader_credit_drops_truncated_episode(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._leader_credit_enabled = True
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([9], dtype=np.int64)
    trainer._leader_credit_pending = {(9, 0): {"seat": 0}}
    trainer._leader_credit_ready = []
    trainer._leader_credit_completed = 0

    LeaguePuffeRL._finalize_leader_credit_rows(
      trainer,
      np.asarray([0, 1], dtype=np.int64),
      np.ones(2, dtype=np.bool_),
      np.zeros(2, dtype=np.bool_),
      np.asarray([0.0, 0.0], dtype=np.float32),
    )

    self.assertEqual(trainer._leader_credit_pending, {})
    self.assertEqual(trainer._leader_credit_ready, [])
    self.assertEqual(trainer._leader_credit_completed, 0)


class ShapedRewardAnnealTests(unittest.TestCase):
  def test_absolute_epoch_multiplier_has_hold_ramp_and_exact_zero_tail(self):
    values = [
      trainer_shaped_reward_multiplier(
        update,
        enabled=True,
        start_epoch=6000,
        end_epoch=6900,
      )
      for update in (5999, 6000, 6450, 6900, 7400)
    ]
    self.assertEqual(values[:2], [1.0, 1.0])
    self.assertAlmostEqual(values[2], 0.5)
    self.assertEqual(values[3:], [0.0, 0.0])

  def test_disabled_multiplier_ignores_unused_boundaries(self):
    self.assertEqual(
      trainer_shaped_reward_multiplier(
        100,
        enabled=False,
        start_epoch=0,
        end_epoch=0,
      ),
      1.0,
    )

  def test_recombine_scales_only_shaped_component(self):
    raw = torch.tensor([0.3, 1.0, -1.0])
    terminal = torch.tensor([0.0, 1.0, -1.0])
    shaped = torch.tensor([0.3, 0.4, -0.4])

    total, scaled = recombine_reward_components(raw, terminal, shaped, 0.5)

    torch.testing.assert_close(scaled, torch.tensor([0.15, 0.2, -0.2]))
    torch.testing.assert_close(total, torch.tensor([0.15, 1.0, -1.0]))

  def test_recombine_one_preserves_raw_reward_exactly(self):
    raw = torch.tensor([0.4, 1.0, -1.0])
    terminal = torch.tensor([0.0, 1.0, -1.0])
    shaped = torch.tensor([0.4, 0.3, -0.2])

    total, scaled = recombine_reward_components(raw, terminal, shaped, 1.0)

    self.assertIs(total, raw)
    self.assertIs(scaled, shaped)

  def test_draft_aux_uses_native_and_trainer_scales(self):
    trainer = PuffeRL.__new__(PuffeRL)
    trainer._draftaux_anneal = True
    trainer._draftaux_last_scale = 1.0
    trainer._trainer_shaped_reward_multiplier = 0.5
    trainer.stats = defaultdict(list, reward_shaping_scale=[0.15])

    self.assertAlmostEqual(PuffeRL._draftaux_aux_scale(trainer), 0.075)

  def test_prepare_uses_next_restored_absolute_epoch(self):
    trainer = PuffeRL.__new__(PuffeRL)
    trainer.epoch = 6449
    trainer.stats = defaultdict(list)
    trainer._trainer_shaped_reward_anneal_enabled = True
    trainer._trainer_shaped_reward_start_epoch = 6000
    trainer._trainer_shaped_reward_end_epoch = 6900

    multiplier = PuffeRL._prepare_trainer_shaped_reward_anneal(trainer)

    self.assertAlmostEqual(multiplier, 0.5)
    self.assertEqual(trainer._trainer_shaped_reward_update, 6450)
    self.assertEqual(trainer.stats['trainer_shaped_reward_update'], [6450.0])


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

  def test_reference_compensation_preserves_target_frozen_mix(self):
    for reference_probability in (0.0, 0.05, 0.10):
      base = compensate_frozen_matchup_ratio_for_reference(
        target_frozen_matchup_ratio=0.8,
        reference_matchup_probability=reference_probability,
      )
      realized = reference_probability + (1.0 - reference_probability) * base
      self.assertAlmostEqual(realized, 0.8)

  def test_reference_compensation_rejects_more_refs_than_frozen_matchups(self):
    with self.assertRaisesRegex(ValueError, "cannot exceed"):
      compensate_frozen_matchup_ratio_for_reference(
        target_frozen_matchup_ratio=0.2,
        reference_matchup_probability=0.3,
      )

  def test_detect_reset_reference_seats(self):
    env_ids = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    modes = np.asarray([1, 1, 0, 1, 2, 0], dtype=np.int32)
    resolved = detect_reset_reference_seats(
      env_ids,
      modes,
      pending_envs=np.asarray([True, True, False], dtype=np.bool_),
      agents_per_env=2,
    )
    self.assertEqual(resolved, {0: -1, 1: 0})

  def test_compute_battle_row_mask_requires_both_completed_decks(self):
    env_ids = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    modes = np.asarray([0, 0, 0, 2, 0, 1], dtype=np.int32)
    mask = compute_battle_row_mask(env_ids, modes, agents_per_env=2)
    np.testing.assert_array_equal(
      mask,
      np.asarray([True, True, False, False, False, False], dtype=np.bool_),
    )

  def test_align_reference_matchup_forces_frozen_reference_opponent(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._reference_opponent_only = True
    trainer._agents_per_env = 2
    trainer._env_reference_pending = np.asarray([True, True], dtype=np.bool_)
    trainer._env_is_reference = np.asarray([False, False], dtype=np.bool_)
    trainer._env_reference_seat = np.asarray([-1, -1], dtype=np.int8)
    trainer._env_learner_seat = np.asarray([0, 0], dtype=np.int32)
    trainer._env_use_latest = np.asarray([True, True], dtype=np.bool_)
    trainer.opponent_policies = [object()]

    LeaguePuffeRL._align_reference_matchups(
      trainer,
      np.asarray([0, 1, 2, 3], dtype=np.int64),
      np.asarray([1, 1, 0, 1], dtype=np.int32),
    )

    np.testing.assert_array_equal(trainer._env_reference_pending, [False, False])
    np.testing.assert_array_equal(trainer._env_is_reference, [False, True])
    np.testing.assert_array_equal(trainer._env_reference_seat, [-1, 0])
    np.testing.assert_array_equal(trainer._env_learner_seat, [0, 1])
    np.testing.assert_array_equal(trainer._env_use_latest, [True, False])

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
    trainer._draftaux_enabled = False
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
    trainer._draftaux_enabled = False
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

  def test_safe_forward_eval_accepts_single_row_legal_distribution(self):
    class _FakeModel:
      def forward_eval(self, obs, state):
        return (
          TCGLegalActionDistribution(
            legal_action_logits=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            legal_actions=torch.tensor(
              [
                [[1, 0, 0, 0], [2, 0, 0, 0]],
              ],
              dtype=torch.int64,
            ),
            legal_action_count=torch.tensor([2], dtype=torch.int64),
          ),
          torch.tensor([[5.0]], dtype=torch.float32),
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

  def test_pool_refresh_preserves_inflight_opponent_and_rnn_state(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    policy_a = torch.nn.Linear(1, 1)
    policy_b = torch.nn.Linear(1, 1)
    policy_c = torch.nn.Linear(1, 1)
    trainer.opponent_policies = [policy_a, policy_b]
    trainer.opponent_keys = ["a", "b"]
    trainer._env_use_latest = np.asarray([False, True], dtype=np.bool_)
    trainer._env_opp_policy = np.asarray([1, 0], dtype=np.int32)
    trainer._env_learner_seat = np.zeros(2, dtype=np.int32)
    trainer._num_envs_total = 2
    trainer._agents_per_env = 2
    trainer.total_agents = 4
    trainer._use_rnn = True
    trainer.policy = type("Policy", (), {"hidden_size": 3})()
    trainer.config = {"device": "cpu"}
    trainer._opp_lstm_h = [torch.full((4, 3), 1.0), torch.full((4, 3), 2.0)]
    trainer._opp_lstm_c = [torch.full((4, 3), 3.0), torch.full((4, 3), 4.0)]
    trainer._pfsp_wins = np.asarray([2.0, 5.0])
    trainer._pfsp_games = np.asarray([4.0, 8.0])
    trainer._pfsp_keys = ["a", "b"]
    trainer._pfsp_enabled = False
    trainer._pfsp_power = 2.0
    trainer.league_cfg = LeagueConfig(
      enabled=True,
      frozen_ratio=0.4,
      randomize_learner_seat=False,
      frozen_window_epochs=8,
      max_distinct_frozen=1,
    )
    trainer._rng = np.random.default_rng(0)
    trainer._window_index = 0
    trainer._window_policy_ids = np.asarray([1], dtype=np.int32)
    trainer.epoch = 1
    trainer.global_step = 100
    trainer.stats = defaultdict(list)

    LeaguePuffeRL.set_opponent_policies(
      trainer,
      [policy_a, policy_c],
      opponent_keys=["a", "c"],
    )

    self.assertEqual(trainer.opponent_keys, ["a", "c", "b"])
    self.assertTrue(np.array_equal(trainer._sampling_policy_ids, np.asarray([0, 1])))
    self.assertEqual(int(trainer._env_opp_policy[0]), 2)
    self.assertTrue(torch.equal(trainer._opp_lstm_h[2], torch.full((4, 3), 2.0)))
    self.assertTrue(torch.equal(trainer._opp_lstm_c[2], torch.full((4, 3), 4.0)))
    self.assertTrue(np.isin(trainer._window_policy_ids, trainer._sampling_policy_ids).all())

    trainer._env_use_latest[0] = True
    LeaguePuffeRL.set_opponent_policies(
      trainer,
      [policy_a, policy_c],
      opponent_keys=["a", "c"],
    )
    self.assertEqual(trainer.opponent_keys, ["a", "c"])



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
