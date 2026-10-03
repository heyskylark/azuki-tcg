#!/usr/bin/env python3
"""Build a signature-disjoint, context-balanced prebuilt training deck pool."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
  sys.path.insert(0, str(PYTHON_SRC))

from deck_building import (  # noqa: E402
  GATE_CARD_TYPE,
  IKZ_CARD_CODE,
  IKZ_CARD_COUNT,
  LEADER_CARD_TYPE,
  MAIN_CARD_TYPES,
  MAX_MAIN_COPIES,
  _load_policy_card_records,
  _validate_deck_shape,
)
from prebuilt_deck_pool import (  # noqa: E402
  EXPECTED_CONTEXT_COUNT,
  MIN_DECKS_PER_CONTEXT,
  REFERENCE_PANEL_SIZE,
)
from training_deck_pool import NativeDeck  # noqa: E402

DEFAULT_SOURCE = REPO_ROOT / ".codex" / "docs" / "azuki_garden_arena_2026-08-15_decks.json"
EXPECTED_TOTAL_CARDS = 62
EXPECTED_MAIN_CARDS = 50

DeckSignature = tuple[tuple[str, int], ...]
DeckContext = tuple[str, str]


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _portable(path: Path) -> str:
  resolved = path.expanduser().resolve()
  try:
    return str(resolved.relative_to(REPO_ROOT))
  except ValueError:
    return str(resolved)


def _load_pool(path: Path) -> tuple[dict[str, Any], bytes]:
  raw = path.expanduser().resolve().read_bytes()
  payload = json.loads(raw)
  if not isinstance(payload, dict):
    raise ValueError(f"Deck pool must contain a JSON object: {path}")
  if payload.get("schema_version") != 1:
    raise ValueError(f"Deck pool must use schema_version 1: {path}")
  decks = payload.get("decks")
  if not isinstance(decks, list) or not decks:
    raise ValueError(f"Deck pool must contain a non-empty decks list: {path}")
  return payload, raw


def _normalize_deck(
  deck: object,
  *,
  source_path: Path,
  source_index: int,
  records_by_code: dict[str, Any],
) -> tuple[dict[str, Any], NativeDeck, DeckSignature, DeckContext, tuple[str, ...]]:
  label = f"deck[{source_index}] in {source_path}"
  if not isinstance(deck, dict):
    raise ValueError(f"{label} must be an object")
  cards = deck.get("cards")
  if not isinstance(cards, list) or not cards:
    raise ValueError(f"{label} must contain a non-empty cards list")

  native_cards: list[tuple[str, int]] = []
  seen_card_ids: set[str] = set()
  for card_index, card in enumerate(cards):
    if not isinstance(card, dict):
      raise ValueError(f"{label} card[{card_index}] must be an object")
    card_id = card.get("card_id")
    quantity = card.get("quantity")
    if not isinstance(card_id, str) or not card_id:
      raise ValueError(f"{label} card[{card_index}] has invalid card_id")
    if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
      raise ValueError(f"{label} card[{card_index}] has invalid quantity {quantity!r}")
    if card_id in seen_card_ids:
      raise ValueError(f"{label} repeats card entry {card_id}")
    if card_id not in records_by_code:
      raise ValueError(f"{label} contains unsupported card {card_id}")
    seen_card_ids.add(card_id)
    native_cards.append((card_id, quantity))

  total_cards = sum(quantity for _, quantity in native_cards)
  if total_cards != EXPECTED_TOTAL_CARDS:
    raise ValueError(
      f"{label} has {total_cards} cards; expected a full engine deck with "
      f"{EXPECTED_MAIN_CARDS} main cards, one leader, one gate, and {IKZ_CARD_COUNT} {IKZ_CARD_CODE}"
    )

  native_deck: NativeDeck = tuple(native_cards)
  leader, gate = _validate_deck_shape(
    native_deck,
    records_by_code,
    deck_index=source_index,
  )
  allowed_types = MAIN_CARD_TYPES | {LEADER_CARD_TYPE, GATE_CARD_TYPE}
  main_total = 0
  off_element_cards: list[str] = []
  for card_id, quantity in native_deck:
    record = records_by_code[card_id]
    if card_id == IKZ_CARD_CODE:
      continue
    if record.card_type not in allowed_types:
      raise ValueError(f"{label} contains unsupported card type {record.card_type} ({card_id})")
    if record.card_type in MAIN_CARD_TYPES:
      main_total += quantity
      if record.element not in ("NORMAL", gate.element):
        off_element_cards.append(card_id)
      if quantity > MAX_MAIN_COPIES:
        raise ValueError(f"{label} has {quantity} copies of {card_id}; max {MAX_MAIN_COPIES}")
  if main_total != EXPECTED_MAIN_CARDS:
    raise ValueError(f"{label} has {main_total} main cards; expected {EXPECTED_MAIN_CARDS}")

  declared_leader = deck.get("leader_card_id")
  declared_gate = deck.get("gate_card_id")
  if not isinstance(declared_leader, str) or not declared_leader:
    raise ValueError(f"{label} must declare leader_card_id")
  if not isinstance(declared_gate, str) or not declared_gate:
    raise ValueError(f"{label} must declare gate_card_id")
  if declared_leader != leader.card_code:
    raise ValueError(
      f"{label} declares leader {declared_leader}, but its cards contain {leader.card_code}"
    )
  if declared_gate != gate.card_code:
    raise ValueError(f"{label} declares gate {declared_gate}, but its cards contain {gate.card_code}")
  declared_element = deck.get("element")
  if declared_element is not None and declared_element != leader.element:
    raise ValueError(
      f"{label} declares element {declared_element!r}, but leader/gate element is {leader.element}"
    )

  signature = tuple(sorted(native_deck))
  return (
    deepcopy(deck), native_deck, signature, (gate.card_code, leader.card_code),
    tuple(off_element_cards),
  )


def _expected_contexts(records: tuple[Any, ...]) -> tuple[DeckContext, ...]:
  gates = [record for record in records if record.card_type == GATE_CARD_TYPE]
  leaders = [record for record in records if record.card_type == LEADER_CARD_TYPE]
  contexts = tuple(
    sorted(
      (gate.card_code, leader.card_code)
      for gate in gates
      for leader in leaders
      if gate.element == leader.element
    )
  )
  if len(contexts) != EXPECTED_CONTEXT_COUNT:
    raise ValueError(
      f"Policy card metadata defines {len(contexts)} compatible gate/leader contexts; "
      f"expected {EXPECTED_CONTEXT_COUNT}"
    )
  return contexts


def _reference_indices(summary: object, deck_count: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
  if not isinstance(summary, dict):
    raise ValueError("Primary source must contain a summary object")
  if summary.get("reference_panel_size") != REFERENCE_PANEL_SIZE:
    raise ValueError(f"Primary source reference_panel_size must be {REFERENCE_PANEL_SIZE}")
  promotion = summary.get("promotion_reference_deck_indices")
  holdout = summary.get("holdout_reference_deck_indices")
  expected_promotion = list(range(0, REFERENCE_PANEL_SIZE, 2))
  expected_holdout = list(range(1, REFERENCE_PANEL_SIZE, 2))
  if promotion != expected_promotion or holdout != expected_holdout:
    raise ValueError("Primary source must preserve the original even/odd 18-deck reference panel")
  if deck_count < REFERENCE_PANEL_SIZE:
    raise ValueError(f"Primary source contains only {deck_count} decks; expected at least 18")
  return tuple(expected_promotion), tuple(expected_holdout)


def build_prebuilt_deck_pool(
  source_path: Path = DEFAULT_SOURCE,
  additional_pool_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
  """Build a deterministic pool, failing rather than manufacturing missing context decks."""
  source_path = source_path.expanduser().resolve()
  input_paths = (source_path, *(path.expanduser().resolve() for path in additional_pool_paths))
  loaded = [(*_load_pool(path), path) for path in input_paths]
  source_payload, _, _ = loaded[0]
  source_decks = source_payload["decks"]
  promotion_indices, holdout_indices = _reference_indices(source_payload.get("summary"), len(source_decks))

  records = _load_policy_card_records()
  records_by_code = {record.card_code: record for record in records}
  if len(records_by_code) != len(records):
    raise ValueError("Policy card metadata contains duplicate card codes")
  expected_contexts = _expected_contexts(records)

  normalized_inputs: list[
    list[tuple[dict[str, Any], NativeDeck, DeckSignature, DeckContext, tuple[str, ...]]]
  ] = []
  source_descriptors: list[dict[str, Any]] = []
  for input_position, (payload, raw, path) in enumerate(loaded):
    normalized = [
      _normalize_deck(
        deck,
        source_path=path,
        source_index=deck_index,
        records_by_code=records_by_code,
      )
      for deck_index, deck in enumerate(payload["decks"])
    ]
    normalized_inputs.append(normalized)
    source_descriptors.append(
      {
        "path": _portable(path),
        "sha256": _sha256(raw),
        "deck_count": len(normalized),
        "role": "primary_tournament_source" if input_position == 0 else "additional_pool",
      }
    )

  reference_rows = normalized_inputs[0][:REFERENCE_PANEL_SIZE]
  reference_signatures = {row[2] for row in reference_rows}
  if len(reference_signatures) != REFERENCE_PANEL_SIZE:
    raise ValueError("The original 18-deck reference panel contains duplicate card-list signatures")

  candidates: dict[DeckSignature, dict[str, Any]] = {}
  excluded_reference_rows = 0
  duplicate_training_rows = 0
  excluded_source_decks: list[dict[str, Any]] = []
  for input_position, normalized in enumerate(normalized_inputs):
    path = input_paths[input_position]
    for source_index, (deck_payload, _native, signature, context, off_element) in enumerate(normalized):
      if signature in reference_signatures:
        excluded_reference_rows += 1
        continue
      if off_element:
        if input_position != 0:
          raise ValueError(
            f"Additional deck[{source_index}] in {path} has off-element main cards: "
            + ", ".join(off_element)
          )
        excluded_source_decks.append({
          "source_deck_index": source_index,
          "reason": "off_element_main_cards",
          "card_ids": list(off_element),
        })
        continue
      provenance = {
        "source_path": _portable(path),
        "source_sha256": source_descriptors[input_position]["sha256"],
        "source_deck_index": source_index,
      }
      existing = candidates.get(signature)
      if existing is not None:
        existing["provenance"].append(provenance)
        duplicate_training_rows += 1
        continue
      candidates[signature] = {
        "payload": deck_payload,
        "context": context,
        "provenance": [provenance],
      }

  candidates_by_context: dict[DeckContext, list[tuple[DeckSignature, dict[str, Any]]]] = {
    context: [] for context in expected_contexts
  }
  for signature, candidate in candidates.items():
    context = candidate["context"]
    if context not in candidates_by_context:
      gate, leader = context
      raise ValueError(f"Deck declares unexpected compatible context {gate}/{leader}")
    candidates_by_context[context].append((signature, candidate))

  missing = [
    (context, len(rows))
    for context, rows in candidates_by_context.items()
    if len(rows) < MIN_DECKS_PER_CONTEXT
  ]
  if missing:
    details = ", ".join(
      f"{gate}/{leader}: found {count}, need {MIN_DECKS_PER_CONTEXT}"
      for (gate, leader), count in missing
    )
    raise ValueError(f"Insufficient distinct non-reference prebuilt decks by context: {details}")

  output_decks: list[dict[str, Any]] = []
  for source_index, (deck_payload, _native, _signature, _context, _off_element) in enumerate(reference_rows):
    deck_payload["prebuilt_provenance"] = [
      {
        "source_path": _portable(source_path),
        "source_sha256": source_descriptors[0]["sha256"],
        "source_deck_index": source_index,
      }
    ]
    output_decks.append(deck_payload)

  groups: list[dict[str, Any]] = []
  for gate_card_id, leader_card_id in expected_contexts:
    rows = sorted(candidates_by_context[(gate_card_id, leader_card_id)], key=lambda row: row[0])
    deck_indices: list[int] = []
    for _signature, candidate in rows:
      deck_payload = candidate["payload"]
      deck_payload["prebuilt_provenance"] = sorted(
        candidate["provenance"],
        key=lambda item: (item["source_path"], item["source_deck_index"]),
      )
      deck_indices.append(len(output_decks))
      output_decks.append(deck_payload)
    groups.append(
      {
        "gate_card_id": gate_card_id,
        "leader_card_id": leader_card_id,
        "deck_count": len(deck_indices),
        "deck_indices": deck_indices,
      }
    )

  training_indices = list(range(REFERENCE_PANEL_SIZE, len(output_decks)))
  return {
    "schema_version": 1,
    "source": {
      "builder": "scripts/build_prebuilt_deck_pool.py",
      "inputs": source_descriptors,
    },
    "normalization_notes": [
      "Preserved the original 18-deck evaluation reference panel at indices 0 through 17.",
      "Training decks contain 50 element-legal main cards, one declared leader, one declared gate, and IKZ-001 x10.",
      "The immutable evaluation panel retains historical lists, including off-element cards; no matching signature enters training.",
      "Excluded off-element primary-source candidates with an audit trail; additional training inputs must be element-legal.",
      "Excluded every card-list signature present in the reference panel, including signatures reintroduced by additional inputs.",
      "Deduplicated training decks by complete card/count signature without generating, swapping, or otherwise manufacturing deck lists.",
      "Ordered contexts by gate_card_id then leader_card_id and signatures lexicographically within each context.",
    ],
    "summary": {
      "reference_panel_size": REFERENCE_PANEL_SIZE,
      "promotion_reference_deck_indices": list(promotion_indices),
      "holdout_reference_deck_indices": list(holdout_indices),
      "prebuilt_training_deck_indices": training_indices,
      "prebuilt_training_deck_count": len(training_indices),
      "prebuilt_training_unique_signature_count": len(training_indices),
      "excluded_reference_signature_row_count": excluded_reference_rows,
      "deduplicated_training_row_count": duplicate_training_rows,
      "excluded_source_decks": excluded_source_decks,
      "reference_off_element_deck_indices": [
        index for index, row in enumerate(reference_rows) if row[4]
      ],
      "prebuilt_deck_groups": groups,
    },
    "decks": output_decks,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
  parser.add_argument(
    "--additional-pool",
    action="append",
    type=Path,
    default=[],
    help="Additional existing deck-pool JSON; repeat for multiple inputs",
  )
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  payload = build_prebuilt_deck_pool(args.source, tuple(args.additional_pool))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  summary = payload["summary"]
  print(
    f"wrote {summary['prebuilt_training_deck_count']} distinct prebuilt training decks "
    f"across {len(summary['prebuilt_deck_groups'])} contexts: {args.output}"
  )


if __name__ == "__main__":
  main()
