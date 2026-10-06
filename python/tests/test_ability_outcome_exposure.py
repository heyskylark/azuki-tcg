from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "train-ablation-1781126582/build_ability_outcome_exposure.py"
SPEC = importlib.util.spec_from_file_location("build_ability_outcome_exposure", BUILDER_PATH)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError(f"Cannot import {BUILDER_PATH}")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class AbilityOutcomeExposureTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.prefix_pool, cls.manifest = BUILDER.build_payloads()
    cls.source = json.loads(BUILDER.SOURCE.read_text(encoding="utf-8"))
    metadata = json.loads(BUILDER.METADATA.read_text(encoding="utf-8"))
    cls.records = {record["card_code"]: record for record in metadata["records"]}

  def test_schema_packages_are_legal_tournament_quartets(self) -> None:
    self.assertEqual(self.prefix_pool["schema_id"], "azuki.strategic_prefix_pool")
    self.assertEqual(self.prefix_pool["schema_version"], 1)
    self.assertEqual(len(self.prefix_pool["contexts"]), 16)
    source_decks = self.source["decks"]
    coverage = {item["context"]: item for item in self.manifest["context_coverage"]}

    for context, packages in self.prefix_pool["contexts"].items():
      gate_code, _ = context.split(":")
      element = self.records[gate_code]["element"]
      self.assertEqual(len(packages), 2)
      self.assertEqual(len({package["id"] for package in packages}), 2)
      provenance = {item["package_id"]: item for item in coverage[context]["packages"]}
      for package in packages:
        self.assertEqual(len(package["cards"]), 4)
        self.assertEqual(len(set(package["cards"])), 4)
        source = source_decks[provenance[package["id"]]["source_index"]]
        source_counts = Counter({card["card_id"]: card["quantity"] for card in source["cards"]})
        package_counts = Counter(package["cards"])
        for code, quantity in package_counts.items():
          record = self.records[code]
          self.assertIn(record["card_type"], {"ENTITY", "SPELL", "WEAPON"})
          self.assertIn(record["element"], {"NORMAL", element})
          self.assertLessEqual(quantity, source_counts[code])

  def test_no_selected_source_has_a_holdout_full_deck_signature(self) -> None:
    holdout = set(self.manifest["heldout_exclusion"]["signatures"])
    selected = {
      package["source_canonical_deck_signature"]
      for context in self.manifest["context_coverage"]
      for package in context["packages"]
    }
    self.assertTrue(holdout)
    self.assertTrue(self.manifest["heldout_exclusion"]["excluded_source_rows"])
    self.assertTrue(selected.isdisjoint(holdout))
    for index in self.source["summary"]["holdout_reference_deck_indices"]:
      self.assertIn(BUILDER.canonical_deck_signature(self.source["decks"][index]), holdout)

  def test_reproducible_balanced_complete_context_coverage(self) -> None:
    self.assertEqual(self.manifest["element_package_counts"], {
      "EARTH": 8,
      "FIRE": 8,
      "LIGHTNING": 8,
      "WATER": 8,
    })
    missing = {
      item["context"]
      for item in self.manifest["context_coverage"]
      if item["tournament_context_missing"]
    }
    self.assertEqual(missing, BUILDER.MISSING_TOURNAMENT_CONTEXTS)
    for item in self.manifest["context_coverage"]:
      for package in item["packages"]:
        if item["tournament_context_missing"]:
          self.assertEqual(
            package["context_assignment"],
            "constructed_adapted_same_element_tournament_package",
          )

    with tempfile.TemporaryDirectory() as temp:
      first = Path(temp) / "first"
      second = Path(temp) / "second"
      first_paths = BUILDER.build(first)
      second_paths = BUILDER.build(second)
      self.assertEqual(
        [hashlib.sha256(path.read_bytes()).hexdigest() for path in first_paths],
        [hashlib.sha256(path.read_bytes()).hexdigest() for path in second_paths],
      )


if __name__ == "__main__":
  unittest.main()
