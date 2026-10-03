from types import SimpleNamespace
import unittest

from play_selfplay_games import terminal_winner


class ReplayOutcomeTests(unittest.TestCase):
  def test_potential_settlement_cannot_reverse_winner(self):
    env = SimpleNamespace(
      _terminal_rewards=(1.0, -1.0),
      _rewards=(-0.75, 0.75),
    )
    self.assertEqual(terminal_winner(env, terminated=True, truncated=False), 0)

  def test_draw_is_not_decided_by_shaping(self):
    env = SimpleNamespace(
      _terminal_rewards=(0.0, 0.0),
      _rewards=(0.25, -0.25),
    )
    self.assertEqual(terminal_winner(env, terminated=True, truncated=False), -1)

  def test_truncation_cannot_reuse_an_outcome(self):
    env = SimpleNamespace(_terminal_rewards=(1.0, -1.0))
    self.assertEqual(terminal_winner(env, terminated=False, truncated=True), -1)
