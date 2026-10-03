from __future__ import annotations

import json
import unittest

from training_deck_pool import (
  DEFAULT_TRAINING_DECK_POOL_PATH,
  EXPECTED_DECK_SIZE,
  load_training_deck_labels,
  load_training_deck_pool,
  resolve_training_deck_pool_path,
)


class TrainingDeckPoolTests(unittest.TestCase):
  def test_default_pool_uses_supported_regional_submissions(self):
    deck_pool = load_training_deck_pool()
    self.assertEqual(len(deck_pool), 237)
    self.assertNotEqual(deck_pool[0][0], ("STT01-001", 1))

    for deck in deck_pool:
      self.assertEqual(sum(quantity for _, quantity in deck), EXPECTED_DECK_SIZE)
      self.assertEqual(dict(deck).get("IKZ-001"), 10)

  def test_default_path_resolves_to_repo_artifact(self):
    self.assertEqual(
      resolve_training_deck_pool_path(),
      DEFAULT_TRAINING_DECK_POOL_PATH,
    )

  def test_default_labels_are_stable_submission_ids(self):
    labels = load_training_deck_labels()
    self.assertEqual(len(labels), 237)
    self.assertEqual(labels[0], "garden-arena-2026-08-15-submission-048")
    self.assertEqual(len(set(labels)), len(labels))

  def test_legacy_pool_keeps_starters_for_in_flight_run_compatibility(self):
    legacy_path = ".codex/docs/azuki_tcg_decks_final.json"
    deck_pool = load_training_deck_pool(legacy_path)
    labels = load_training_deck_labels(legacy_path)
    self.assertEqual(len(deck_pool), 18)
    self.assertEqual(deck_pool[0][0], ("STT01-001", 1))
    self.assertEqual(labels[:2], ("starter_raizan", "starter_shao"))

  def test_reference_panel_preserves_even_promotion_odd_holdout_contract(self):
    payload = json.loads(DEFAULT_TRAINING_DECK_POOL_PATH.read_text(encoding="utf-8"))
    self.assertEqual(
      payload["summary"]["promotion_reference_deck_indices"],
      list(range(0, 18, 2)),
    )
    self.assertEqual(
      payload["summary"]["holdout_reference_deck_indices"],
      list(range(1, 18, 2)),
    )
    roles = [deck.get("reference_role") for deck in payload["decks"][:18]]
    self.assertEqual(roles[0::2], ["promotion"] * 9)
    self.assertEqual(roles[1::2], ["holdout"] * 9)


if __name__ == "__main__":
  unittest.main()
