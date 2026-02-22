from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from league_state import LeagueState, classify_and_prune, load_league_state, register_policy, save_league_state


class LeagueStateTests(unittest.TestCase):
  def test_register_and_roundtrip(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      state_path = root / "league_state.json"
      ckpt = root / "model_000001.pt"
      ckpt.write_bytes(b"x")

      state = LeagueState()
      entry = register_policy(
        state,
        checkpoint_path=ckpt,
        created_epoch=10,
        source="seed",
      )
      save_league_state(state_path, state)
      loaded = load_league_state(state_path)

      self.assertIn(entry.policy_id, loaded.policies)
      self.assertEqual(loaded.champion_policy_id, entry.policy_id)
      self.assertEqual(loaded.policies[entry.policy_id].created_epoch, 10)

  def test_classify_and_prune_keeps_champion(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      state = LeagueState()
      ids = []
      for idx in range(8):
        ckpt = root / f"model_{idx:06d}.pt"
        ckpt.write_bytes(b"x")
        entry = register_policy(
          state,
          checkpoint_path=ckpt,
          created_epoch=idx,
          source="checkpoint",
        )
        ids.append(entry.policy_id)
      state.champion_policy_id = ids[2]
      kept = classify_and_prune(state, keep_recent=2, keep_mid=2, keep_old=1)
      self.assertIn(ids[2], kept)
      self.assertTrue(state.policies[ids[2]].active)


if __name__ == "__main__":
  unittest.main()
