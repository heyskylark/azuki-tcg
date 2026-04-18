from __future__ import annotations

import unittest

from training_deck_pool import (
  DEFAULT_TRAINING_DECK_POOL_PATH,
  EXPECTED_DECK_SIZE,
  load_training_deck_labels,
  load_training_deck_pool,
  resolve_training_deck_pool_path,
)


class TrainingDeckPoolTests(unittest.TestCase):
  def test_default_pool_includes_starters_and_balanced_json_decks(self):
    deck_pool = load_training_deck_pool()
    self.assertEqual(len(deck_pool), 18)
    self.assertEqual(deck_pool[0][0], ("STT01-001", 1))
    self.assertEqual(deck_pool[1][0], ("STT02-001", 1))

    for deck in deck_pool:
      self.assertEqual(sum(quantity for _, quantity in deck), EXPECTED_DECK_SIZE)

  def test_default_path_resolves_to_repo_artifact(self):
    self.assertEqual(
      resolve_training_deck_pool_path(),
      DEFAULT_TRAINING_DECK_POOL_PATH,
    )

  def test_default_labels_cover_starters_and_final_pool(self):
    labels = load_training_deck_labels()
    self.assertEqual(len(labels), 18)
    self.assertEqual(labels[0], "starter_raizan")
    self.assertEqual(labels[1], "starter_shao")


if __name__ == "__main__":
  unittest.main()
