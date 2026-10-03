from __future__ import annotations

import unittest

from league_promotion import PromotionGameRecord
from promotion_ablation import _reorient_records, _select_panel


class PromotionAblationTests(unittest.TestCase):
  def test_reorientation_complements_score_and_swaps_policy_fields(self) -> None:
    source = PromotionGameRecord(
      game_id="game",
      block_id="block",
      phase="screen",
      opponent_id="policy_b",
      seed=7,
      candidate_seat=0,
      candidate_gate=10,
      opponent_gate=20,
      winner_seat=0,
      steps=50,
      end_reason="gameover",
    )

    inverse = _reorient_records(
      [source],
      source_candidate="policy_a",
      target_candidate="policy_b",
      opponent_id="policy_a",
    )[0]

    self.assertEqual(inverse.candidate_seat, 1)
    self.assertEqual(inverse.candidate_gate, 20)
    self.assertEqual(inverse.opponent_gate, 10)
    self.assertEqual(inverse.opponent_id, "policy_a")
    self.assertEqual(inverse.candidate_score, 0.0)

  def test_panel_selection_uses_anchor_recent_hardest_and_distance_roles(self) -> None:
    manifest = {
      "anchor_id": "anchor",
      "policies": [
        {"id": "anchor", "epoch": 10, "panel_eligible": False, "qualified_prior": True},
        {"id": "recent", "epoch": 30, "panel_eligible": True, "qualified_prior": True},
        {"id": "hard", "epoch": 20, "panel_eligible": True, "qualified_prior": False},
        {"id": "distinct", "epoch": 15, "panel_eligible": True, "qualified_prior": True},
        {"id": "other", "epoch": 5, "panel_eligible": True, "qualified_prior": False},
      ],
    }
    ids = [item["id"] for item in manifest["policies"]]
    matrix = {
      left: {right: 0.5 for right in ids}
      for left in ids
    }
    matrix["anchor"]["hard"] = 0.2
    matrix["anchor"]["other"] = 0.4
    matrix["distinct"]["other"] = 1.0

    panel = _select_panel(manifest, matrix)

    self.assertEqual([item["role"] for item in panel], [
      "production_anchor",
      "recent_quality",
      "hardest_retained",
      "historically_distinct",
    ])
    self.assertEqual([item["policy_id"] for item in panel[:3]], ["anchor", "recent", "hard"])
    self.assertEqual(panel[3]["policy_id"], "distinct")


if __name__ == "__main__":
  unittest.main()
