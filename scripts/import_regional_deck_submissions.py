#!/usr/bin/env python3
"""Convert regional 52-card submissions into the engine-ready training corpus."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / ".codex" / "docs" / "azuki_garden_arena_2026-08-15_deck_submissions.json"
DEFAULT_OUTPUT = REPO_ROOT / ".codex" / "docs" / "azuki_garden_arena_2026-08-15_decks.json"
DEFAULT_METADATA = REPO_ROOT / "python" / "config" / "policy_card_metadata_v1.json"
EVENT_NAME = "Azuki Garden Arena"
EVENT_DATE = "2026-08-15"
IKZ_CARD_CODE = "IKZ-001"
MAIN_CARD_TYPES = frozenset(("ENTITY", "SPELL", "WEAPON"))
PANEL_QUOTAS = (("FIRE", 6), ("LIGHTNING", 4), ("EARTH", 4), ("WATER", 4))


def _canonical_card_code(card_code: str) -> str:
  family, separator, number = card_code.partition("-")
  if separator and family in {"STT03", "STT04"} and number.isdigit():
    return f"{family}-{int(number):03d}"
  return card_code


def _load_metadata(path: Path) -> dict[str, dict[str, Any]]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  raw_records = payload.get("records") if isinstance(payload, dict) else None
  if not isinstance(raw_records, list) or not raw_records:
    raise ValueError(f"Card metadata must contain a non-empty records list: {path}")

  records: dict[str, dict[str, Any]] = {}
  for index, record in enumerate(raw_records):
    if not isinstance(record, dict):
      raise ValueError(f"Card metadata record {index} must be an object")
    card_code = record.get("card_code")
    card_type = record.get("card_type")
    element = record.get("element")
    if not all(isinstance(value, str) and value for value in (card_code, card_type, element)):
      raise ValueError(f"Card metadata record {index} has invalid identity fields")
    if card_code in records:
      raise ValueError(f"Card metadata contains duplicate card code {card_code}")
    records[card_code] = record
  return records


def _normalize_submission(
  submission: object,
  *,
  source_index: int,
  records: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
  if not isinstance(submission, list) or not submission:
    raise ValueError(f"Submission {source_index + 1} must be a non-empty list")

  cards: list[tuple[str, int]] = []
  seen_codes: set[str] = set()
  for card_index, row in enumerate(submission):
    if not isinstance(row, dict):
      raise ValueError(f"Submission {source_index + 1} card {card_index + 1} must be an object")
    raw_code = row.get("card_id")
    quantity = row.get("card_count")
    if not isinstance(raw_code, str) or not raw_code:
      raise ValueError(f"Submission {source_index + 1} card {card_index + 1} has invalid card_id")
    if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
      raise ValueError(
        f"Submission {source_index + 1} card {card_index + 1} has invalid card_count"
      )
    card_code = _canonical_card_code(raw_code)
    if card_code in seen_codes:
      raise ValueError(f"Submission {source_index + 1} repeats card entry {card_code}")
    seen_codes.add(card_code)
    cards.append((card_code, quantity))

  total = sum(quantity for _, quantity in cards)
  if total != 52:
    raise ValueError(f"Submission {source_index + 1} has {total} cards; expected 52")
  if IKZ_CARD_CODE in seen_codes:
    raise ValueError(f"Submission {source_index + 1} already contains {IKZ_CARD_CODE}")

  unsupported = sorted(seen_codes - records.keys())
  if unsupported:
    return None, unsupported

  type_totals: Counter[str] = Counter()
  leaders: list[str] = []
  gates: list[str] = []
  for card_code, quantity in cards:
    record = records[card_code]
    card_type = str(record["card_type"]).upper()
    type_totals[card_type] += quantity
    if card_type == "LEADER":
      leaders.extend([card_code] * quantity)
    elif card_type == "GATE":
      gates.extend([card_code] * quantity)
    elif card_type in MAIN_CARD_TYPES and quantity > 4:
      raise ValueError(f"Submission {source_index + 1} has {quantity} copies of {card_code}")

  main_total = sum(type_totals[card_type] for card_type in MAIN_CARD_TYPES)
  if len(leaders) != 1 or len(gates) != 1 or main_total != 50:
    raise ValueError(
      f"Submission {source_index + 1} must contain one leader, one gate, and 50 main cards"
    )
  if sum(type_totals.values()) != 52:
    raise ValueError(f"Submission {source_index + 1} contains unsupported card types")

  leader = leaders[0]
  gate = gates[0]
  leader_element = str(records[leader]["element"]).upper()
  gate_element = str(records[gate]["element"]).upper()
  if leader_element != gate_element:
    raise ValueError(
      f"Submission {source_index + 1} leader element {leader_element} "
      f"does not match gate element {gate_element}"
    )

  signature = tuple(sorted(cards))
  return {
    "source_index": source_index,
    "cards": cards,
    "leader": leader,
    "gate": gate,
    "element": leader_element,
    "signature": signature,
  }, []


def _select_reference_panel(
  supported: list[dict[str, Any]],
  signature_frequency: Counter[tuple[tuple[str, int], ...]],
) -> list[dict[str, Any]]:
  representatives: dict[tuple[tuple[str, int], ...], dict[str, Any]] = {}
  for deck in supported:
    representatives.setdefault(deck["signature"], deck)

  selected_by_element: dict[str, list[dict[str, Any]]] = {}
  for element, quota in PANEL_QUOTAS:
    candidates = [deck for deck in representatives.values() if deck["element"] == element]
    selected: list[dict[str, Any]] = []
    seen_leaders: set[str] = set()
    seen_gates: set[str] = set()
    while len(selected) < quota:
      if not candidates:
        raise ValueError(f"Not enough unique {element} decks for reference quota {quota}")
      candidates.sort(
        key=lambda deck: (
          -int(deck["leader"] not in seen_leaders) - int(deck["gate"] not in seen_gates),
          -signature_frequency[deck["signature"]],
          deck["source_index"],
        )
      )
      chosen = candidates.pop(0)
      selected.append(chosen)
      seen_leaders.add(chosen["leader"])
      seen_gates.add(chosen["gate"])
    selected_by_element[element] = selected

  panel: list[dict[str, Any]] = []
  for element, _ in PANEL_QUOTAS:
    selected = selected_by_element[element]
    for promotion, holdout in zip(selected[0::2], selected[1::2], strict=True):
      panel.extend((promotion, holdout))
  return panel


def _deck_payload(deck: dict[str, Any], *, pool_index: int) -> dict[str, Any]:
  signature_json = json.dumps(deck["signature"], separators=(",", ":"))
  source_number = int(deck["source_index"]) + 1
  payload = {
    "deck_name": f"Garden Arena 2026-08-15 submission {source_number:03d}",
    "deck_slug": f"garden-arena-2026-08-15-submission-{source_number:03d}",
    "source_submission_number": source_number,
    "content_sha256": hashlib.sha256(signature_json.encode()).hexdigest(),
    "element": deck["element"],
    "leader_card_id": deck["leader"],
    "gate_card_id": deck["gate"],
    "cards": [
      *({"card_id": card_code, "quantity": quantity} for card_code, quantity in deck["cards"]),
      {"card_id": IKZ_CARD_CODE, "quantity": 10},
    ],
  }
  if pool_index < 18:
    payload["reference_role"] = "promotion" if pool_index % 2 == 0 else "holdout"
  return payload


def build_corpus(source_path: Path, metadata_path: Path) -> dict[str, Any]:
  source_bytes = source_path.read_bytes()
  submissions = json.loads(source_bytes)
  if not isinstance(submissions, list) or not submissions:
    raise ValueError(f"Submission source must be a non-empty JSON list: {source_path}")
  records = _load_metadata(metadata_path)

  supported: list[dict[str, Any]] = []
  excluded: list[dict[str, Any]] = []
  unsupported_counts: Counter[str] = Counter()
  for source_index, submission in enumerate(submissions):
    deck, unsupported = _normalize_submission(
      submission,
      source_index=source_index,
      records=records,
    )
    if deck is not None:
      supported.append(deck)
      continue
    excluded.append(
      {
        "source_submission_number": source_index + 1,
        "unsupported_card_ids": unsupported,
      }
    )
    unsupported_counts.update(unsupported)

  signature_frequency = Counter(deck["signature"] for deck in supported)
  panel = _select_reference_panel(supported, signature_frequency)
  panel_source_indices = {int(deck["source_index"]) for deck in panel}
  ordered = panel + [
    deck for deck in supported if int(deck["source_index"]) not in panel_source_indices
  ]
  element_distribution = Counter(str(deck["element"]) for deck in supported)

  return {
    "schema_version": 1,
    "source": {
      "event": EVENT_NAME,
      "event_date": EVENT_DATE,
      "file_name": source_path.name,
      "sha256": hashlib.sha256(source_bytes).hexdigest(),
      "submission_count": len(submissions),
    },
    "normalization_notes": [
      "Canonicalized STT03 and STT04 numeric card codes to three digits.",
      "Excluded submissions containing card codes absent from policy_card_metadata_v1.json.",
      "Preserved duplicate supported submissions so pool sampling reflects the submitted field.",
      "Added IKZ-001 x10 to every supported 52-card submission for the engine-required 62-card representation.",
      "Placed an 18-deck reference panel first so existing even promotion and odd holdout indices remain stable.",
      "Selected the panel deterministically for element, leader, and gate coverage, then exact-list submission frequency.",
    ],
    "summary": {
      "input_submission_count": len(submissions),
      "engine_ready_submission_count": len(ordered),
      "excluded_submission_count": len(excluded),
      "unique_engine_ready_signature_count": len(signature_frequency),
      "duplicate_engine_ready_submission_count": len(ordered) - len(signature_frequency),
      "engine_ready_element_distribution": dict(sorted(element_distribution.items())),
      "unsupported_card_submission_counts": dict(sorted(unsupported_counts.items())),
      "reference_panel_size": 18,
      "promotion_reference_deck_indices": list(range(0, 18, 2)),
      "holdout_reference_deck_indices": list(range(1, 18, 2)),
    },
    "excluded_submissions": excluded,
    "decks": [_deck_payload(deck, pool_index=index) for index, deck in enumerate(ordered)],
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
  parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()

  payload = build_corpus(args.source, args.metadata)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  summary = payload["summary"]
  print(
    f"wrote {summary['engine_ready_submission_count']} engine-ready decks; "
    f"excluded {summary['excluded_submission_count']} submissions: {args.output}"
  )


if __name__ == "__main__":
  main()
