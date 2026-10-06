from __future__ import annotations

import contextlib
from collections import defaultdict
import unittest
from unittest.mock import patch

import numpy as np
from observation import DECKBUILD_OBSERVATION_CTYPE
from action import ActionType
from policy.tcg_distribution import TCGLegalActionDistribution
import torch

from policy.v2.tcg_policy import TCGLSTM

from azk_puffer.models import LSTMWrapper
from azk_puffer.trainer import (
  PuffeRL,
  _compute_puff_advantage_fallback,
  compute_puff_advantage,
  terminal_win_labels_from_rewards,
)
from league_training import (
  clipped_terminal_policy_loss,
  compensate_frozen_matchup_ratio_for_reference,
  compute_battle_row_mask,
  compute_reference_fixed_actor_mask,
  compute_frozen_matchup_ratio,
  detect_reset_reference_seats,
  deterministic_draft_credit_pick,
  deterministic_draft_episode_priority,
  deterministic_draft_prefix_candidate,
  deterministic_draft_prefix_length,
  draft_episode_credit_update_due,
  draft_episode_credit_coefficient,
  draft_episode_credit_sampled_rows,
  DRAFT_TERMINAL_CREDIT_QUARTILES,
  LeagueConfig,
  LeaguePuffeRL,
  masked_tensor_mean,
  grouped_ppo_diagnostic_means,
  ppo_diagnostic_bucket_code,
  ppo_diagnostic_legal_count_labels,
  ppo_diagnostic_observation_layout,
  summarize_ppo_diagnostic_tensor,
  masked_tensor_mean_std,
  pad_complete_draft_episodes,
  pad_terminal_credit_records,
  parse_draft_prefix_distribution,
  take_terminal_credit_training_records,
  compute_league_active,
  compute_learner_row_mask,
  compute_trainable_row_mask,
  select_role_window,
  sampled_row_reward_scale,
  terminal_only_rewards,
)
from draft_normal_penalty import leader_normal_cost


class PpoDiagnosticsTests(unittest.TestCase):
  def test_distribution_summary_reports_quantiles_and_ignores_nonfinite_values(self):
    summary = summarize_ppo_diagnostic_tensor(
      torch.tensor([0.0, 1.0, 2.0, 3.0, float("nan"), float("inf")])
    )

    self.assertEqual(summary["mean"], 1.5)
    self.assertEqual(summary["p50"], 1.5)
    self.assertAlmostEqual(summary["p90"], 2.7, places=5)
    self.assertAlmostEqual(summary["p99"], 2.97, places=5)
    self.assertEqual(summary["max"], 3.0)

  def test_grouped_means_and_legal_count_buckets_are_stable(self):
    grouped = grouped_ppo_diagnostic_means(
      torch.tensor([1.0, 3.0, 5.0, 9.0]),
      torch.tensor([2, 2, 7, 7]),
    )
    labels = ppo_diagnostic_legal_count_labels(
      torch.tensor([1, 2, 4, 5, 8, 9, 256, 257])
    )

    self.assertEqual(grouped, {2: (2.0, 2), 7: (7.0, 2)})
    torch.testing.assert_close(
      labels,
      torch.tensor([0, 1, 1, 2, 2, 3, 7, 8]),
    )
    self.assertEqual(ppo_diagnostic_bucket_code("recent"), 1)
    self.assertEqual(ppo_diagnostic_bucket_code("promotion_panel"), 4)

  def test_packed_observation_context_decodes_both_sides(self):
    observations = np.zeros(2, dtype=DECKBUILD_OBSERVATION_CTYPE)
    observations["my_observation_data"]["gate"]["card_def_id"] = [11, 12]
    observations["my_observation_data"]["leader"]["card_def_id"] = [21, 22]
    observations["opponent_observation_data"]["gate"]["card_def_id"] = [31, 32]
    observations["opponent_observation_data"]["leader"]["card_def_id"] = [41, 42]
    packed = torch.from_numpy(
      observations.view(np.uint8).reshape(2, -1).copy()
    )
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._ppo_diagnostics_enabled = True
    trainer._ppo_diagnostic_layout = ppo_diagnostic_observation_layout(
      packed.shape[-1]
    )

    decoded = LeaguePuffeRL._decode_ppo_diagnostic_context(trainer, packed)

    self.assertIsNotNone(decoded)
    np.testing.assert_array_equal(decoded[0], [11, 12])
    np.testing.assert_array_equal(decoded[1], [21, 22])
    np.testing.assert_array_equal(decoded[2], [31, 32])
    np.testing.assert_array_equal(decoded[3], [41, 42])

  def test_finalizer_emits_tail_group_and_resampling_contract(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    losses = defaultdict(float)
    diagnostics = defaultdict(list)
    diagnostics["kl"].append(torch.tensor([0.0, 0.1, 0.2, 10.0]))
    diagnostics["mb_kl"].append(torch.tensor([0.0, 2.575]))
    diagnostics["kl_primary"].append(torch.tensor([0.0, 0.05, 0.1, 9.0]))
    diagnostics["opponent_bucket"].append(torch.tensor([0, 1, 2, 4]))
    diagnostics["gate"].append(torch.tensor([150, 150, 152, 152]))
    diagnostics["leader"].append(torch.tensor([149, 149, 151, 151]))
    diagnostics["opponent_gate"].append(torch.tensor([3, 20, 3, 20]))
    diagnostics["legal_count"].append(torch.tensor([1, 4, 32, 300]))
    diagnostics["chosen_primary"].append(torch.tensor([0, 1, 1, 6]))

    LeaguePuffeRL._finalize_ppo_diagnostics(
      trainer,
      losses,
      diagnostics,
      torch.tensor([3, 0, 1, 2], dtype=torch.int32),
    )

    self.assertEqual(losses["ppo_diag_selected_rows"], 4.0)
    self.assertEqual(losses["ppo_diag_kl_max"], 10.0)
    self.assertGreater(losses["ppo_diag_kl_top1pct_share"], 0.9)
    self.assertEqual(losses["ppo_diag_count_opponent_frozen_all"], 3.0)
    self.assertEqual(losses["ppo_diag_count_legal_count_257_plus"], 1.0)
    self.assertEqual(losses["ppo_diag_count_gate_matchup_150_vs_3"], 1.0)
    self.assertEqual(losses["ppo_diag_segment_sample_count_max"], 3.0)

  def test_finalizer_omits_kl_summaries_when_no_actor_rows_are_selected(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    losses = defaultdict(float)

    LeaguePuffeRL._finalize_ppo_diagnostics(
      trainer,
      losses,
      defaultdict(list),
      torch.zeros(4, dtype=torch.int32),
    )

    self.assertEqual(losses["ppo_diag_selected_rows"], 0.0)
    self.assertNotIn("ppo_diag_kl_mean", losses)
    self.assertNotIn("ppo_diag_kl_max", losses)
    self.assertNotIn("ppo_diag_mb_kl_mean", losses)
    self.assertNotIn("ppo_diag_kl_top1pct_share", losses)

  def test_normalization_advances_once_at_boundary_and_finishes_frozen(self):
    class FakeBasePolicy(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.scalar_normalizer = torch.nn.Identity()
        self.seen: list[tuple[int, bool]] = []

      def encode_observations(self, observations, state=None):
        self.seen.append(
          (int(observations.shape[0]), bool(self.scalar_normalizer.training))
        )
        return observations.float()

    base_policy = FakeBasePolicy()
    base_policy.eval()
    wrapper = type("Wrapper", (), {"policy": base_policy})()
    trainer = PuffeRL.__new__(PuffeRL)
    trainer.uncompiled_policy = wrapper
    trainer.scalar_normalizer = base_policy.scalar_normalizer
    trainer.observations = torch.zeros((3, 2, 4), dtype=torch.uint8)
    trainer.vecenv = type(
      "VecEnv",
      (),
      {"single_observation_space": type("Space", (), {"shape": (4,)})()},
    )()
    trainer.minibatch_size = 4
    trainer.config = {"device": "cpu"}
    trainer.amp_context = contextlib.nullcontext()
    trainer.stats = defaultdict(list)

    PuffeRL._advance_running_normalization(trainer)

    self.assertEqual(base_policy.seen, [(4, True), (2, True)])
    self.assertFalse(base_policy.training)
    self.assertFalse(trainer.scalar_normalizer.training)
    self.assertEqual(trainer.stats["normalization/update_rows"], [6.0])

  def test_lstm_training_replay_matches_stepwise_rollout_with_resets(self):
    class FakeEnv:
      single_observation_space = type("Space", (), {"shape": (2,)})()

    class FakePolicy(torch.nn.Module):
      is_continuous = False

      def encode_observations(self, observations, state=None):
        return observations.float()

      def decode_actions(self, hidden):
        return hidden, hidden.sum(dim=-1, keepdim=True)

    torch.manual_seed(7)
    wrapper = LSTMWrapper(
      FakeEnv(),
      FakePolicy(),
      input_size=2,
      hidden_size=3,
    )
    observations = torch.randn(2, 4, 2)
    initial_h = torch.randn(2, 3)
    initial_c = torch.randn(2, 3)
    resets = torch.tensor(
      [[False, True, False, False], [False, False, True, False]]
    )

    rollout_h = initial_h.clone()
    rollout_c = initial_c.clone()
    rollout_logits = []
    rollout_values = []
    for step in range(observations.shape[1]):
      if step > 0:
        keep = (~resets[:, step - 1]).float().unsqueeze(-1)
        rollout_h = rollout_h * keep
        rollout_c = rollout_c * keep
      state = {"lstm_h": rollout_h, "lstm_c": rollout_c}
      logits, values = wrapper.forward_eval(observations[:, step], state)
      rollout_h = state["lstm_h"]
      rollout_c = state["lstm_c"]
      rollout_logits.append(logits.clone())
      rollout_values.append(values.clone())

    replay_state = {
      "lstm_h": initial_h.unsqueeze(0),
      "lstm_c": initial_c.unsqueeze(0),
      "lstm_reset": resets,
    }
    replay_logits, replay_values = wrapper(observations, replay_state)

    torch.testing.assert_close(
      replay_logits.reshape(2, 4, 3),
      torch.stack(rollout_logits, dim=1),
    )
    torch.testing.assert_close(
      replay_values,
      torch.stack(rollout_values, dim=1).squeeze(-1),
    )


class LeagueRecurrentReplayTests(unittest.TestCase):
  def setUp(self):
    class FakeEnv:
      single_observation_space = type("Space", (), {"shape": (2,)})()

    class FakePolicy(torch.nn.Module):
      is_continuous = False

      def encode_observations(self, observations, state=None):
        return observations

      def decode_actions(self, hidden, action_context=None, state=None):
        return hidden, hidden.sum(dim=-1, keepdim=True)

    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(7)
      self.policy = TCGLSTM(FakeEnv(), FakePolicy(), input_size=2, hidden_size=3)
      self.observations = torch.randn(2, 4, 2)
      self.initial_h = torch.randn(2, 3)
      self.initial_c = torch.randn(2, 3)
    self.resets = torch.tensor(
      [[False, False, False, False], [False, True, False, False]]
    )
    self.trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)

  def test_later_losses_train_history_without_crossing_resets_or_seats(self):
    observations = self.observations.requires_grad_()
    initial_h = self.initial_h.requires_grad_()
    initial_c = self.initial_c.requires_grad_()
    logits, values, _ = self.trainer._forward_recurrent_sequence(
      self.policy, observations, initial_h, initial_c, self.resets
    )
    # Only the last decision has an actor loss; earlier rows must still learn.
    actor_loss = -logits.reshape(2, 4, 3)[0, -1].log_softmax(dim=-1)[0]
    actor_grad, h_grad, c_grad = torch.autograd.grad(
      actor_loss,
      (observations, initial_h, initial_c),
      allow_unused=True,
      retain_graph=True,
    )
    self.assertGreater(float(actor_grad[0, 0].norm()), 0.0)
    self.assertEqual(float(actor_grad[1].norm()), 0.0)
    self.assertIsNone(h_grad)
    self.assertIsNone(c_grad)

    critic_grad = torch.autograd.grad(values[1, -1], observations)[0]
    self.assertEqual(float(critic_grad[0].norm()), 0.0)
    self.assertEqual(float(critic_grad[1, :2].norm()), 0.0)
    self.assertGreater(float(critic_grad[1, 2].norm()), 0.0)
    self.assertGreater(float(critic_grad[1, -1].norm()), 0.0)

  def test_sequence_likelihoods_and_values_match_rollout_before_update(self):
    actions = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 2]])
    h, c = self.initial_h, self.initial_c
    rollout_logprobs = []
    rollout_values = []
    with torch.no_grad():
      for step in range(self.observations.shape[1]):
        if step:
          keep = (~self.resets[:, step - 1]).float().unsqueeze(-1)
          h, c = h * keep, c * keep
        state = {"lstm_h": h, "lstm_c": c}
        logits, values = self.policy.forward_eval(self.observations[:, step], state)
        h, c = state["lstm_h"], state["lstm_c"]
        rollout_logprobs.append(
          torch.distributions.Categorical(logits=logits).log_prob(actions[:, step])
        )
        rollout_values.append(values.squeeze(-1))

    replay_logits, replay_values, _ = self.trainer._forward_recurrent_sequence(
      self.policy, self.observations, self.initial_h, self.initial_c, self.resets
    )
    replay_logprobs = torch.distributions.Categorical(
      logits=replay_logits.reshape(2, 4, 3)
    ).log_prob(actions)
    torch.testing.assert_close(replay_logprobs, torch.stack(rollout_logprobs, dim=1))
    torch.testing.assert_close(replay_values, torch.stack(rollout_values, dim=1))

  def test_inference_keeps_global_agent_history_across_batches_and_resets(self):
    trainer = self.trainer
    trainer.config = {"device": "cpu"}
    trainer.policy = trainer.uncompiled_policy = self.policy
    trainer.opponent_policies = [self.policy]
    trainer.vecenv = type(
      "VecEnv", (), {"single_action_space": type("Space", (), {"shape": ()})()}
    )()
    trainer.league_cfg = LeagueConfig(enabled=True, activate_after_steps=0)
    trainer.global_step = 0
    trainer.total_agents = 6
    trainer._agents_per_env = 2
    trainer._env_learner_seat = np.zeros(3, dtype=np.int32)
    trainer._env_use_latest = np.array([True, False, False])
    trainer._env_opp_policy = np.zeros(3, dtype=np.int32)
    trainer._use_rnn = True
    trainer._draftaux_enabled = False
    trainer.amp_context = contextlib.nullcontext()
    trainer._learner_lstm_h = torch.zeros(6, 3)
    trainer._learner_lstm_c = torch.zeros(6, 3)
    trainer._opp_lstm_h = [torch.zeros(6, 3)]
    trainer._opp_lstm_c = [torch.zeros(6, 3)]
    reference = [
      {"lstm_h": torch.zeros(1, 3), "lstm_c": torch.zeros(1, 3)}
      for _ in range(6)
    ]

    # Worker batches reuse local rows 0/1 for different global game/seat IDs.
    batches = ([2, 3], [4, 5], [0, 1], [2, 3], [4, 5], [0, 1])
    with torch.random.fork_rng(devices=[]), torch.no_grad():
      for step, agents in enumerate(batches):
        if step == 3:
          trainer._zero_done_states(np.array([2, 3]))
          for agent in (2, 3):
            reference[agent] = {
              "lstm_h": torch.zeros(1, 3), "lstm_c": torch.zeros(1, 3)
            }
        observations = torch.tensor(
          [[agent + 0.1 * (step + 1), -0.2 * (step + 1)] for agent in agents]
        )
        expected_values = []
        for row, agent in enumerate(agents):
          _, value = self.policy.forward_eval(observations[row:row + 1], reference[agent])
          expected_values.append(value.squeeze())
        _, _, values, *_ = trainer._infer_actions(
          observations, torch.ones(2, dtype=torch.bool), np.array(agents)
        )
        with self.subTest(step=step, agents=agents):
          torch.testing.assert_close(values, torch.stack(expected_values))



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

  def test_strategic_prefix_resolves_card_identity_and_stops_after_four_picks(self):
    from policy.v2.tcg_sampler import tcg_sample_logits

    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._draft_prefix_enabled = True
    trainer._draft_prefix_lengths = (4,)
    trainer._draft_prefix_probabilities = (1.0,)
    trainer._draft_prefix_seed = 420053
    trainer._draft_prefix_episode_lengths = []
    trainer._draft_prefix_forced_rows = 0
    trainer._draft_prefix_pool = {(100, 200): ((10, 20, 30, 10),)}
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([17], dtype=np.int64)
    trainer.amp_context = contextlib.nullcontext()
    dtype = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
    deck_dtype, deck_offset = dtype.fields["deck_context"][:2]
    trainer._draft_episode_credit_layout = {
      "candidate_count_offset": deck_offset + deck_dtype.fields["candidate_count"][1],
      "candidate_offset": deck_offset + deck_dtype.fields["candidate_card_def_ids"][1],
    }
    packed = np.zeros(2, dtype=dtype)
    packed["deck_context"]["candidate_count"] = 3
    packed["deck_context"]["candidate_card_def_ids"][1, :3] = [30, 10, 20]
    observations = torch.from_numpy(packed.view(np.uint8).reshape(2, -1))
    distribution = TCGLegalActionDistribution(
      legal_action_logits=torch.zeros((1, 3)),
      legal_actions=torch.tensor([[[3, 0, 0, 0], [3, 1, 0, 0], [3, 2, 0, 0]]]),
      legal_action_count=torch.tensor([3]),
    )
    kwargs = {
      "logits": distribution,
      "actions": torch.tensor([[3, 0, 0, 0]]),
      "logprobs": torch.tensor([-1.0986123]),
      "policy_rows_np": np.asarray([1]),
      "env_id_np": np.asarray([0, 1]),
      "draft_rows_np": np.asarray([True, True]),
      "draft_legal_counts_np": np.asarray([3, 3]),
      "observations_cpu": observations,
      "draft_gate_ids_np": np.asarray([100, 100]),
      "draft_leader_ids_np": np.asarray([200, 200]),
    }
    with patch("league_training.azk_pytorch.sample_logits", new=tcg_sample_logits):
      selected = []
      for pick in range(4):
        actions, logprobs, forced = trainer._apply_random_draft_prefix(
          **kwargs, draft_main_counts_np=np.asarray([0, pick]),
        )
        selected.append(int(packed["deck_context"]["candidate_card_def_ids"][1, actions[0, 1]]))
        self.assertTrue(bool(forced[0]))
        torch.testing.assert_close(logprobs.exp(), torch.tensor([1 / 3]))
      self.assertEqual(selected, [10, 20, 30, 10])
      actions, _, forced = trainer._apply_random_draft_prefix(
        **kwargs, draft_main_counts_np=np.asarray([0, 4]),
      )
      self.assertFalse(bool(forced[0]))
      torch.testing.assert_close(actions, kwargs["actions"])
      packed["deck_context"]["candidate_card_def_ids"][1, 1] = 40
      with self.assertRaises(RuntimeError):
        trainer._apply_random_draft_prefix(
          **kwargs, draft_main_counts_np=np.asarray([0, 0]),
        )

  def test_terminal_reward_labels_use_sign_and_ignore_nonterminal_rows(self):
    labels = terminal_win_labels_from_rewards(
      np.asarray([0, 1, 2, 3], dtype=np.int64),
      np.asarray([True, True, False, False], dtype=np.bool_),
      np.asarray([5.0, -5.0, 0.25, -0.25], dtype=np.float32),
      agents_per_env=2,
    )
    self.assertEqual(labels, {0: {0: 1.0, 1: 0.0}})

  def test_terminal_only_rewards_preserves_native_outcomes(self):
    rewards = terminal_only_rewards(
      np.asarray([0.25, -0.25, 5.0, -5.0], dtype=np.float32),
      np.asarray([False, False, True, True], dtype=np.bool_),
    )

    np.testing.assert_array_equal(rewards, [0.0, 0.0, 5.0, -5.0])
    labels = terminal_win_labels_from_rewards(
      np.asarray([0, 1, 0, 1], dtype=np.int64),
      np.asarray([False, False, True, True], dtype=np.bool_),
      rewards,
      agents_per_env=2,
    )
    self.assertEqual(labels, {0: {0: 1.0, 1: 0.0}})

  def test_true_draw_label_is_independent_of_shaping(self):
    labels = terminal_win_labels_from_rewards(
      np.asarray([0, 1], dtype=np.int64),
      np.asarray([True, True], dtype=np.bool_),
      np.asarray([0.0, 1.0], dtype=np.float32),
      agents_per_env=2,
    )

    self.assertEqual(labels, {0: {0: 0.5, 1: 1.0}})

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
    dtype = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
    packed = np.zeros(2, dtype=dtype)
    deck_dtype, deck_offset = dtype.fields["deck_context"][:2]
    trainer._draft_normal_penalty_settings = {121: (0.08, 1.0)}
    trainer._draft_normal_card_ids = frozenset({158})
    trainer._draft_episode_credit_layout = {
      "candidate_offset": deck_offset + deck_dtype.fields["candidate_card_def_ids"][1],
      "candidate_count_offset": deck_offset + deck_dtype.fields["candidate_count"][1],
    }
    packed["deck_context"]["candidate_count"] = 1
    observations = torch.from_numpy(packed.view(np.uint8).reshape(2, -1))
    actions = torch.tensor([[int(ActionType.DECK_PICK_CARD), 0, 0, 0]])
    logprobs = torch.zeros(1, dtype=torch.float32)
    for main_count in range(50):
      observations[0, 0] = main_count
      # Include forced-prefix cards and the 50th selected card, not merely the
      # pre-pick deck snapshot (which has only 49 cards at the final decision).
      packed["deck_context"]["candidate_card_def_ids"][0, 0] = (
        158 if main_count < 4 or main_count == 49 else 10
      )
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
        forced_rows=torch.tensor([main_count < 4]),
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
    record = trainer._draft_episode_credit_ready[0]
    self.assertEqual(record["normal_count"], 5)
    self.assertGreater(
      leader_normal_cost(record["normal_count"], 121, trainer._draft_normal_penalty_settings), 0
    )

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


  def test_full_episode_credit_sampled_rows_ignore_trainable_row_fraction(self):
    self.assertEqual(draft_episode_credit_sampled_rows(0, 15_360), 15_360)
    self.assertEqual(
      draft_episode_credit_sampled_rows(976, 15_360),
      15_006_720,
    )
    with self.assertRaisesRegex(ValueError, "epoch"):
      draft_episode_credit_sampled_rows(-1, 15_360)
    with self.assertRaisesRegex(ValueError, "batch size"):
      draft_episode_credit_sampled_rows(0, 0)

  def test_full_episode_credit_coefficient_uses_sampled_row_boundaries(self):
    schedule = {
      "initial": 1.0,
      "final": 0.1,
      "anneal_start_rows": 0,
      "anneal_end_rows": 7_500_000,
    }
    self.assertEqual(draft_episode_credit_coefficient(0, **schedule), 1.0)
    self.assertAlmostEqual(
      draft_episode_credit_coefficient(3_750_000, **schedule),
      0.55,
    )
    self.assertEqual(
      draft_episode_credit_coefficient(7_500_000, **schedule),
      0.1,
    )
    self.assertEqual(
      draft_episode_credit_coefficient(45_000_000, **schedule),
      0.1,
    )
    with self.assertRaisesRegex(ValueError, "row interval"):
      draft_episode_credit_coefficient(
        0,
        initial=1.0,
        final=0.0,
        anneal_start_rows=0,
        anneal_end_rows=0,
      )

  def test_decomposed_reward_scale_uses_global_sampled_row_boundaries(self):
    schedule = {
      "initial": 1.0,
      "final": 0.05,
      "anneal_start_rows": 0,
      "anneal_end_rows": 7_503_360,
    }
    self.assertEqual(sampled_row_reward_scale(0, **schedule), 1.0)
    self.assertAlmostEqual(
      sampled_row_reward_scale(3_751_680, **schedule), 0.525
    )
    self.assertEqual(
      sampled_row_reward_scale(7_503_360, **schedule), 0.05
    )
    self.assertEqual(
      sampled_row_reward_scale(15_006_720, **schedule), 0.05
    )
    with self.assertRaisesRegex(ValueError, "row interval"):
      sampled_row_reward_scale(
        0,
        initial=1.0,
        final=0.0,
        anneal_start_rows=0,
        anneal_end_rows=0,
      )

  def test_full_episode_credit_excludes_cross_gate_record_before_labeling(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._draft_episode_credit_enabled = True
    trainer._draft_episode_credit_exclude_xgate = True
    trainer._draft_episode_credit_pending = {
      (17, 0): {
        "seen": np.ones(50, dtype=np.bool_),
        "gate_swapped": True,
        "original_gate_id": 122,
        "battle_gate_id": 124,
        "leader_id": 121,
        "seat": 0,
      }
    }
    trainer._draft_episode_credit_ready = []
    trainer._draft_episode_credit_xgate_excluded = 0
    trainer._draft_episode_credit_xgate_excluded_rows = 0
    trainer._draft_episode_credit_xgate_excluded_breakdown = defaultdict(int)
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([17], dtype=np.int64)
    trainer.epoch = 5

    LeaguePuffeRL._finalize_draft_episode_credit_rows(
      trainer,
      np.asarray([0, 1], dtype=np.int64),
      np.asarray([True, True]),
      np.asarray([True, True]),
      np.asarray([1.0, -1.0], dtype=np.float32),
    )

    self.assertEqual(trainer._draft_episode_credit_ready, [])
    self.assertEqual(trainer._draft_episode_credit_xgate_excluded, 1)
    self.assertEqual(trainer._draft_episode_credit_xgate_excluded_rows, 50)
    self.assertEqual(
      trainer._draft_episode_credit_xgate_excluded_breakdown[
        (122, 124, 121, 0, "win")
      ],
      1,
    )

  def test_full_episode_credit_marks_cross_gate_seat_and_balances_reservoir(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._draft_episode_credit_enabled = True
    trainer._draft_episode_credit_exclude_xgate = True
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([23], dtype=np.int64)
    trainer._draft_episode_credit_pending = {
      (23, 1): {
        "original_gate_id": 122,
        "battle_gate_id": 122,
        "gate_swapped": False,
      }
    }
    LeaguePuffeRL._mark_draft_episode_credit_cross_gate(
      trainer,
      np.asarray([1], dtype=np.int64),
      np.asarray([124], dtype=np.int32),
    )
    pending = trainer._draft_episode_credit_pending[(23, 1)]
    self.assertTrue(pending["gate_swapped"])
    self.assertEqual(pending["battle_gate_id"], 124)

    trainer._draft_episode_credit_ready = []
    trainer._draft_episode_credit_batch_drafts = 4
    trainer._draft_episode_credit_seed = 420052
    trainer._draft_episode_credit_completed = 0
    trainer._draft_episode_credit_window_completed = 0
    trainer._draft_episode_credit_sample_dropped = 0
    trainer._draft_episode_credit_window_sample_dropped = 0
    records = [
      {"episode_id": episode_id, "seat": 0, "gate_id": gate, "leader_id": leader}
      for episode_id, gate, leader in (
        (1, 122, 121),
        (2, 122, 121),
        (3, 122, 121),
        (4, 122, 121),
        (5, 124, 125),
        (6, 124, 125),
      )
    ]
    for record in records:
      LeaguePuffeRL._offer_completed_draft_episode_credit(trainer, record)
    context_counts = defaultdict(int)
    for record in trainer._draft_episode_credit_ready:
      context_counts[(record["gate_id"], record["leader_id"])] += 1
    self.assertEqual(sorted(context_counts.values()), [2, 2])

  def test_normal_penalty_changes_only_unforced_draft_actor_credit(self):
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

    baselines = []
    for normal_count, expected_sign in ((0, 1), (50, -1)):
      with self.subTest(normal_count=normal_count):
        policy = FakePolicy()
        trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
        trainer._draft_episode_credit_enabled = True
        trainer._draft_episode_credit_batch_drafts = 2
        trainer._draft_episode_credit_update_interval = 1
        trainer._draft_episode_credit_coef = 1.0
        trainer._draft_episode_credit_baseline_coef = 0.1
        trainer._draft_episode_credit_clip = 0.2
        trainer._draft_normal_penalty_settings = {121: (0.2, 1.0)}
        trainer._draft_normal_penalty_schedule = (1.0, 1.0, 0, 0)
        trainer._draft_episode_credit_ready = [
          {
            "observations": torch.nn.functional.pad(
              torch.arange(50, dtype=torch.uint8).reshape(50, 1), (0, 3)
            ),
            "actions": torch.zeros((50, 4), dtype=torch.long),
            "old_logprobs": torch.zeros(50),
            "lstm_h": torch.zeros((50, 3)),
            "lstm_c": torch.zeros((50, 3)),
            "actor_valid": torch.arange(50) >= 4,
            "capture_epochs": np.full(50, 2, dtype=np.int32),
            "terminal_epoch": 8,
            "target": 1.0,
            "gate_id": 122,
            "leader_id": 121,
            "normal_count": normal_count,
            "sample_priority": 1,
          }
        ]
        trainer._draft_episode_credit_window_completed = 1
        trainer._draft_episode_credit_window_sample_dropped = 0
        trainer.config = {
          "device": "cpu", "batch_size": 15_360, "ent_coef": 0.0, "max_grad_norm": 10.0,
        }
        trainer.policy = policy
        trainer.optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        trainer.amp_context = contextlib.nullcontext()
        trainer.epoch = 10
        trainer.total_epochs = 20

        def fake_sample_logits(logits, action=None):
          picks = logits["pick"]
          return action, policy.actor[picks], torch.ones(picks.shape[0])

        with patch("league_training.azk_pytorch.sample_logits", side_effect=fake_sample_logits):
          LeaguePuffeRL._train_draft_episode_credit(trainer)
        self.assertTrue(bool((policy.actor.detach()[:4] == 0.0).all()))
        self.assertTrue(bool((policy.actor.detach()[4:] * expected_sign > 0.0).all()))
        baselines.append(float(policy.baseline.detach()))
    self.assertGreater(baselines[0], 0.0)
    self.assertEqual(baselines[0], baselines[1])

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


class RewardAccountingTests(unittest.TestCase):
  def test_incoming_reward_alignment_matches_upstream_contract(self):
    values = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    rewards = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    terminals = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    ratio = torch.ones_like(values)
    result = _compute_puff_advantage_fallback(
      values, rewards, terminals, ratio, torch.zeros_like(values),
      1.0, 1.0, 1.0, 1.0,
    )

    torch.testing.assert_close(result, torch.tensor([[-7.0, -18.0, 13.0, 0.0]]))

  def test_truncation_bootstraps_final_value_but_stops_trace(self):
    values = torch.tensor([[10.0, 20.0, 30.0, 1000.0]])
    rewards = torch.tensor([[0.0, 1.0, 2.0, 999.0]])
    terminals = torch.zeros_like(values)
    truncations = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    result = compute_puff_advantage(
      values, rewards, terminals, torch.ones_like(values),
      torch.zeros_like(values), 1.0, 1.0, 1.0, 1.0,
      truncations=truncations,
    )

    torch.testing.assert_close(result[:, :2], torch.tensor([[23.0, 12.0]]))


class LeagueTrainingUtilsTests(unittest.TestCase):
  def test_hand_size_telemetry_uses_actual_count_and_decision_roles(self):
    observations = np.zeros(8, dtype=DECKBUILD_OBSERVATION_CTYPE)
    observations["action_mask"]["primary_action_mask"][:, 1] = True
    observations["action_mask"]["primary_action_mask"][2, 1:] = False
    observations["deck_context"]["mode"][3] = 2
    observations["my_observation_data"]["hand_count"] = [
      31,
      35,
      40,
      42,
      44,
      45,
      30,
      0,
    ]
    packed = torch.from_numpy(observations.view(np.uint8).reshape(8, -1))
    active = np.asarray([True, True, True, True, False, True, True, True])
    trainable = np.asarray([True, False, True, False, True, False, True, False])
    done = np.asarray([False, False, False, False, False, True, False, False])
    counts = np.zeros((2, 4), dtype=np.int64)
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._prebuilt_enabled = True
    trainer._prebuilt_obs_dtype = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
    trainer._prebuilt_initial_probability = 1.0
    trainer.prebuilt_battle_decisions = 0
    trainer.config = {}
    trainer.vecenv = type(
      "VecEnv",
      (),
      {"prebuilt_probability": np.zeros(1, dtype=np.float32)},
    )()

    trainer._advance_prebuilt_curriculum(
      packed,
      active,
      trainable,
      done,
      counts,
    )

    np.testing.assert_array_equal(
      counts,
      np.asarray(
        [
          [2, 1, 61, 31],
          [2, 1, 35, 35],
        ],
        dtype=np.int64,
      ),
    )
    self.assertEqual(trainer.prebuilt_battle_decisions, 2)

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

  def test_reference_fixed_actor_mask_preserves_only_battle_actions(self):
    env_ids = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    modes = np.asarray([0, 2, 2, 0, 0, 0], dtype=np.int32)
    mask = compute_reference_fixed_actor_mask(
      env_ids,
      modes,
      agents_per_env=2,
      env_is_reference=np.asarray([True, True, False], dtype=np.bool_),
      env_reference_seat=np.asarray([0, 1, -1], dtype=np.int8),
    )
    np.testing.assert_array_equal(
      mask,
      np.asarray([False, True, True, False, True, True], dtype=np.bool_),
    )

  def test_align_reference_matchup_forces_frozen_reference_opponent(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._reference_opponent_only = True
    trainer._reference_alignment_enabled = True
    trainer._reference_learner_fixed = False
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

  def test_align_reference_matchup_uses_latest_until_frozen_pool_exists(self):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._reference_alignment_enabled = True
    trainer._reference_opponent_only = False
    trainer._reference_learner_fixed = True
    trainer._agents_per_env = 2
    trainer._env_reference_pending = np.asarray([True], dtype=np.bool_)
    trainer._env_is_reference = np.asarray([False], dtype=np.bool_)
    trainer._env_reference_seat = np.asarray([-1], dtype=np.int8)
    trainer._env_learner_seat = np.asarray([1], dtype=np.int32)
    trainer._env_use_latest = np.asarray([True], dtype=np.bool_)
    trainer.opponent_policies = []

    LeaguePuffeRL._align_reference_matchups(
      trainer,
      np.asarray([0, 1], dtype=np.int64),
      np.asarray([0, 2], dtype=np.int32),
    )

    np.testing.assert_array_equal(trainer._env_is_reference, [True])
    np.testing.assert_array_equal(trainer._env_reference_seat, [0])
    np.testing.assert_array_equal(trainer._env_learner_seat, [0])
    np.testing.assert_array_equal(trainer._env_use_latest, [True])

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
        self.hidden_size = 3

    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer.config = {"device": "cpu"}
    trainer.league_cfg = LeagueConfig(enabled=True, latest_ratio=1.0, activate_after_steps=0)
    trainer.global_step = 0
    trainer._agents_per_env = 2
    trainer._env_learner_seat = np.asarray([0], dtype=np.int32)
    trainer._env_use_latest = np.asarray([True], dtype=np.bool_)
    trainer._use_rnn = True
    trainer._learner_lstm_h = torch.tensor(
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    trainer._learner_lstm_c = torch.tensor(
      [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    )
    trainer._opp_lstm_h = []
    trainer._opp_lstm_c = []
    trainer._draftaux_enabled = False
    trainer.policy = _FakePolicy("latest", 1.0)
    trainer.opponent_policies = []
    trainer.vecenv = type("VecEnv", (), {"single_action_space": type("Space", (), {"shape": (4,)})()})()
    trainer.amp_context = contextlib.nullcontext()
    trainer._split_value_heads_enabled = lambda: False
    def fake_forward(model, obs, state):
      initial_h = state["lstm_h"]
      initial_c = state["lstm_c"]
      state["lstm_h"] = initial_h + 100.0
      state["lstm_c"] = initial_c + 200.0
      return (
        {"label": model.label, "batch": obs.shape[0]},
        torch.full((obs.shape[0], 1), model.value, dtype=torch.float32),
      )

    trainer._safe_forward_eval = fake_forward

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
    initial_h, initial_c = trainer._pending_rollout_lstm_states
    torch.testing.assert_close(
      initial_h,
      torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    torch.testing.assert_close(
      initial_c,
      torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
    )

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
    trainer.opponent_buckets = ["old", "recent"]
    trainer.opponent_roles = ["history", "recent"]
    trainer._env_use_latest = np.asarray([False, True], dtype=np.bool_)
    trainer._env_opp_policy = np.asarray([1, 0], dtype=np.int32)
    trainer._env_opp_role = np.asarray(["recent", "latest"], dtype=object)
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
    trainer._window_policy_roles = ("recent",)
    trainer._role_sampling_floors = ()
    trainer._role_window_counts = {}
    trainer.epoch = 1
    trainer.global_step = 100
    trainer.stats = defaultdict(list)

    LeaguePuffeRL.set_opponent_policies(
      trainer,
      [policy_a, policy_c],
      opponent_keys=["a", "c"],
      opponent_buckets=["old", "recent"],
      opponent_roles=["history", "recent"],
    )

    self.assertEqual(trainer.opponent_keys, ["a", "c", "b"])
    self.assertEqual(trainer.opponent_roles, ["history", "recent", "recent"])
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
    trainer._env_opp_role = np.full(64, "latest", dtype=object)
    trainer._window_policy_ids = None
    trainer._window_index = -1
    trainer._window_policy_roles = ()
    trainer._role_sampling_floors = ()
    trainer._role_window_counts = {}
    trainer.opponent_roles = ["history"] * pool_size
    trainer._sampling_policy_ids = np.arange(pool_size, dtype=np.int32)
    trainer._pfsp_enabled = False
    trainer._pfsp_power = 2.0
    trainer._pfsp_wins = np.zeros(pool_size, dtype=np.float64)
    trainer._pfsp_games = np.zeros(pool_size, dtype=np.float64)
    trainer.stats = defaultdict(list)
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

  def test_temporal_role_floors_work_with_single_policy_windows(self):
    role_floors = (
      ("anchor", 0.15),
      ("recent", 0.25),
      ("hard", 0.25),
      ("distinct", 0.15),
      ("history", 0.20),
    )
    counts = {role: 0 for role, _ in role_floors}
    available = np.arange(4, dtype=np.int32)
    roles = ("anchor", "recent", "distinct", "history")
    weights = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    rng = np.random.default_rng(9)

    for _ in range(100):
      selected, selected_roles = select_role_window(
        available,
        roles,
        max_distinct=1,
        role_floors=role_floors,
        role_counts=counts,
        policy_weights=weights,
        rng=rng,
      )
      self.assertEqual(selected.size, 1)
      self.assertEqual(len(selected_roles), 1)

    self.assertEqual(sum(counts.values()), 100)
    for role, target in role_floors:
      self.assertLessEqual(abs(counts[role] / 100.0 - target), 0.01)

if __name__ == "__main__":
  unittest.main()
