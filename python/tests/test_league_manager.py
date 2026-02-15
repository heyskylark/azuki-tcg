from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from league_manager import LeagueManager, LeagueManagerConfig, parse_league_manager_config
from league_state import LeagueState, register_policy, save_league_state


class LeagueManagerTests(unittest.TestCase):
  def test_config_validation_rejects_bad_quick_interval(self):
    with self.assertRaises(ValueError):
      parse_league_manager_config(
        {"league": {"enable": True, "quick_eval_interval": 0}}
      )

  def test_min_candidate_epoch_gap(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      manager = LeagueManager(
        LeagueManagerConfig(
          enabled=True,
          state_path=str(root / "league_state.json"),
          min_candidate_epoch_gap=3,
        )
      )
      ckpt1 = root / "model_000001.pt"
      ckpt2 = root / "model_000002.pt"
      ckpt1.write_bytes(b"a")
      ckpt2.write_bytes(b"b")

      first = manager.maybe_add_checkpoint(ckpt1, epoch=1)
      second = manager.maybe_add_checkpoint(ckpt2, epoch=2)
      third = manager.maybe_add_checkpoint(ckpt2, epoch=4)

      self.assertIsNotNone(first)
      self.assertIsNone(second)
      self.assertIsNotNone(third)

  def test_state_recovery_from_backup(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      state_path = root / "league_state.json"
      ckpt = root / "model_000001.pt"
      ckpt.write_bytes(b"x")
      state = LeagueState()
      register_policy(state, checkpoint_path=ckpt, created_epoch=1, source="seed")
      save_league_state(state_path, state)
      save_league_state(state_path, state)  # creates .bak

      state_path.write_text("{broken-json", encoding="utf-8")
      manager = LeagueManager(LeagueManagerConfig(enabled=True, state_path=str(state_path)))
      events = [item.get("event") for item in manager.state.history if isinstance(item, dict)]
      self.assertIn("state_recovered_from_backup", events)


if __name__ == "__main__":
  unittest.main()
