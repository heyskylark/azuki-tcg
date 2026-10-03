from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_DIR))

from build_prebuilt_deck_pool import DEFAULT_SOURCE, build_prebuilt_deck_pool
from prebuilt_deck_pool import (
  MIN_DECKS_PER_CONTEXT,
  REFERENCE_PANEL_SIZE,
  load_prebuilt_deck_groups,
)

TEST_CONTEXTS = (
  ("AZK01-120", "STT01-001"),
  ("AZK01-122", "STT04-001"),
  ("STT01-002", "STT01-001"),
  ("STT03-002", "AZK01-123"),
  ("STT04-002", "STT04-001"),
)


def _signature(deck: dict[str, object]) -> tuple[tuple[str, int], ...]:
  cards = deck["cards"]
  assert isinstance(cards, list)
  return tuple(sorted((str(card["card_id"]), int(card["quantity"])) for card in cards))


class PrebuiltDeckPoolTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls._temporary_directory = tempfile.TemporaryDirectory()
    source_payload = json.loads(DEFAULT_SOURCE.read_text(encoding="utf-8"))
    source_payload["decks"] = [
      *source_payload["decks"][:REFERENCE_PANEL_SIZE],
      *(
        deck
        for deck in source_payload["decks"][REFERENCE_PANEL_SIZE:]
        if (deck["gate_card_id"], deck["leader_card_id"]) in TEST_CONTEXTS
      ),
    ]
    cls.filtered_source = Path(cls._temporary_directory.name) / "filtered-source.json"
    cls.filtered_source.write_text(json.dumps(source_payload), encoding="utf-8")
    reference_source = Path(cls._temporary_directory.name) / "references.json"
    reference_source.write_text(
      json.dumps({"schema_version": 1, "decks": source_payload["decks"][:REFERENCE_PANEL_SIZE]}),
      encoding="utf-8",
    )
    with patch("build_prebuilt_deck_pool._expected_contexts", return_value=TEST_CONTEXTS):
      cls.payload = build_prebuilt_deck_pool(cls.filtered_source, (reference_source,))

  @classmethod
  def tearDownClass(cls) -> None:
    cls._temporary_directory.cleanup()

  def test_reintroduced_reference_signatures_remain_excluded(self):
    decks = self.payload["decks"]
    reference_signatures = {_signature(deck) for deck in decks[:REFERENCE_PANEL_SIZE]}
    training_signatures = {_signature(deck) for deck in decks[REFERENCE_PANEL_SIZE:]}
    self.assertTrue(reference_signatures.isdisjoint(training_signatures))
    self.assertGreaterEqual(
      self.payload["summary"]["excluded_reference_signature_row_count"],
      2 * REFERENCE_PANEL_SIZE,
    )

  def test_training_decks_are_distinct_and_groups_are_balanced(self):
    decks = self.payload["decks"]
    training_indices = self.payload["summary"]["prebuilt_training_deck_indices"]
    signatures = [_signature(decks[index]) for index in training_indices]
    self.assertEqual(len(signatures), len(set(signatures)))

    groups = self.payload["summary"]["prebuilt_deck_groups"]
    self.assertEqual(len(groups), len(TEST_CONTEXTS))
    contexts = [(group["gate_card_id"], group["leader_card_id"]) for group in groups]
    self.assertEqual(contexts, sorted(contexts))
    self.assertEqual(
      [index for group in groups for index in group["deck_indices"]],
      training_indices,
    )
    for group in groups:
      self.assertGreaterEqual(len(group["deck_indices"]), MIN_DECKS_PER_CONTEXT)
      self.assertEqual(group["deck_count"], len(group["deck_indices"]))
      for index in group["deck_indices"]:
        self.assertEqual(decks[index]["gate_card_id"], group["gate_card_id"])
        self.assertEqual(decks[index]["leader_card_id"], group["leader_card_id"])

  def test_runtime_loader_rejects_false_deck_declarations(self):
    payload = deepcopy(self.payload)
    payload["decks"][REFERENCE_PANEL_SIZE]["gate_card_id"] = "NOT-A-GATE"
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "prebuilt.json"
      path.write_text(json.dumps(payload), encoding="utf-8")
      with patch("prebuilt_deck_pool._expected_contexts", return_value=TEST_CONTEXTS):
        with self.assertRaises(ValueError):
          load_prebuilt_deck_groups(path)

  def test_primary_source_alone_rejects_incomplete_coverage(self):
    with self.assertRaises(ValueError):
      build_prebuilt_deck_pool(DEFAULT_SOURCE)

  def test_additional_decks_reject_unsupported_cards_and_false_declarations(self):
    base_deck = self.payload["decks"][REFERENCE_PANEL_SIZE]
    cases = []

    unsupported = deepcopy(base_deck)
    unsupported["cards"][0]["card_id"] = "NOT-A-CARD"
    cases.append((unsupported, "contains unsupported card NOT-A-CARD"))

    false_declaration = deepcopy(base_deck)
    false_declaration["leader_card_id"] = "NOT-A-LEADER"
    cases.append((false_declaration, "false leader declaration"))

    for deck, expected_error in cases:
      with self.subTest(expected_error=expected_error):
        with tempfile.TemporaryDirectory() as directory:
          path = Path(directory) / "additional.json"
          path.write_text(
            json.dumps({"schema_version": 1, "decks": [deck]}),
            encoding="utf-8",
          )
          with patch("build_prebuilt_deck_pool._expected_contexts", return_value=TEST_CONTEXTS):
            with self.assertRaises(ValueError):
              build_prebuilt_deck_pool(self.filtered_source, (path,))


if __name__ == "__main__":
  unittest.main()
