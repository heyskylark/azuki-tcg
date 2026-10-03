#!/usr/bin/env python3
"""Build a signature-disjoint training corpus and evaluation split."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO_ROOT / ".codex" / "docs" / "azuki_garden_arena_2026-08-15_decks.json"
DEFAULT_TRAINING_CORPUS = (
  REPO_ROOT / ".codex" / "docs" / "azuki_garden_arena_2026-08-15_training_decks.json"
)
DEFAULT_MANIFEST = (
  REPO_ROOT / ".codex" / "docs" / "azuki_garden_arena_2026-08-15_reference_split.json"
)


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _portable(path: Path) -> str:
  resolved = path.resolve()
  try:
    return str(resolved.relative_to(REPO_ROOT))
  except ValueError:
    return str(resolved)


def _load_corpus(path: Path) -> tuple[dict[str, Any], bytes]:
  raw = path.read_bytes()
  payload = json.loads(raw)
  if not isinstance(payload, dict):
    raise ValueError(f"Corpus must be a JSON object: {path}")
  decks = payload.get("decks")
  if not isinstance(decks, list) or not decks:
    raise ValueError(f"Corpus must contain a non-empty decks list: {path}")
  return payload, raw


def _deck_hash(deck: object, index: int) -> str:
  if not isinstance(deck, dict):
    raise ValueError(f"Deck {index} must be an object")
  value = deck.get("content_sha256")
  if not isinstance(value, str) or len(value) != 64:
    raise ValueError(f"Deck {index} has no valid content_sha256")
  return value


def _element_counts(decks: list[object], indices: list[int]) -> dict[str, int]:
  counts: Counter[str] = Counter()
  for index in indices:
    deck = decks[index]
    if not isinstance(deck, dict) or not isinstance(deck.get("element"), str):
      raise ValueError(f"Deck {index} has no element")
    counts[str(deck["element"])] += 1
  return dict(sorted(counts.items()))


def build_split(
  corpus_path: Path,
  training_corpus_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
  corpus, corpus_raw = _load_corpus(corpus_path)
  decks = corpus["decks"]
  summary = corpus.get("summary")
  if not isinstance(summary, dict):
    raise ValueError("Corpus has no summary object")
  seed_indices = summary.get("holdout_reference_deck_indices")
  if not isinstance(seed_indices, list) or not seed_indices:
    raise ValueError("Corpus has no holdout reference seed indices")
  if any(not isinstance(index, int) or isinstance(index, bool) for index in seed_indices):
    raise ValueError("Holdout reference seed indices must be integers")
  if min(seed_indices) < 0 or max(seed_indices) >= len(decks):
    raise ValueError("Holdout reference seed index is outside the corpus")

  hashes = [_deck_hash(deck, index) for index, deck in enumerate(decks)]
  holdout_hashes = {hashes[index] for index in seed_indices}
  holdout_indices = [index for index, value in enumerate(hashes) if value in holdout_hashes]
  training_indices = [index for index, value in enumerate(hashes) if value not in holdout_hashes]
  training_hashes = {hashes[index] for index in training_indices}
  if training_hashes & holdout_hashes:
    raise AssertionError("Training and holdout content signatures overlap")
  if sorted(training_indices + holdout_indices) != list(range(len(decks))):
    raise AssertionError("Split does not partition the source corpus")

  training_decks = [decks[index] for index in training_indices]
  training_payload = {
    "schema_version": 1,
    "source": {
      "corpus_path": _portable(corpus_path),
      "corpus_sha256": _sha256(corpus_raw),
      "split_protocol": "legacy-holdout-signature-groups-v1",
    },
    "normalization_notes": [
      "Preserved source order and duplicate submissions for field-weighted sampling.",
      "Removed every row sharing a content signature with the regional holdout seed panel.",
      "Decks remain in the engine-required 62-card representation including IKZ-001 x10.",
    ],
    "summary": {
      "source_submission_count": len(decks),
      "training_submission_count": len(training_decks),
      "training_unique_signature_count": len(training_hashes),
      "heldout_submission_count": len(holdout_indices),
      "heldout_unique_signature_count": len(holdout_hashes),
      "training_element_distribution": _element_counts(decks, training_indices),
    },
    "decks": training_decks,
  }
  training_raw = (json.dumps(training_payload, indent=2) + "\n").encode()

  manifest = {
    "schema_version": 1,
    "protocol": "legacy-holdout-signature-groups-v1",
    "source_corpus": {
      "path": _portable(corpus_path),
      "sha256": _sha256(corpus_raw),
      "rows": len(decks),
      "unique_signatures": len(set(hashes)),
    },
    "training": {
      "corpus_path": _portable(training_corpus_path),
      "corpus_sha256": _sha256(training_raw),
      "source_indices": training_indices,
      "rows": len(training_indices),
      "unique_signatures": len(training_hashes),
      "content_sha256": sorted(training_hashes),
      "element_distribution": _element_counts(decks, training_indices),
    },
    "holdout": {
      "seed_source_indices": seed_indices,
      "source_indices": holdout_indices,
      "rows": len(holdout_indices),
      "unique_signatures": len(holdout_hashes),
      "content_sha256": sorted(holdout_hashes),
      "element_distribution": _element_counts(decks, holdout_indices),
    },
    "invariants": {
      "partitions_source_rows": True,
      "content_signatures_disjoint": True,
      "duplicates_preserved_within_splits": True,
    },
  }
  return training_payload, manifest


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
  parser.add_argument("--training-corpus", type=Path, default=DEFAULT_TRAINING_CORPUS)
  parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  args = parser.parse_args()

  training_payload, manifest = build_split(args.corpus, args.training_corpus)
  args.training_corpus.parent.mkdir(parents=True, exist_ok=True)
  args.manifest.parent.mkdir(parents=True, exist_ok=True)
  args.training_corpus.write_text(
    json.dumps(training_payload, indent=2) + "\n", encoding="utf-8"
  )
  args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  print(
    f"wrote {manifest['training']['rows']} training rows and "
    f"{manifest['holdout']['rows']} heldout rows: {args.manifest}"
  )


if __name__ == "__main__":
  main()
