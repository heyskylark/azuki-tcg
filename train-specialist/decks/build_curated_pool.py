#!/usr/bin/env python3
"""Validate curated_decks.json and build the per-element specialist training pool.

Output schema (consumed by python/src/prebuilt_deck_pool.load_specialist_deck_groups):
- decks: held-out evaluation decks first, then training decks grouped by (gate, leader) context.
- summary.holdout_reference_deck_indices: held-out decks (never trained on).
- summary.prebuilt_deck_groups: only non-empty contexts, sorted by (gate_card_id, leader_card_id).
- summary.prebuilt_training_deck_indices: concatenation of group deck_indices in group order.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
  sys.path.insert(0, str(PYTHON_SRC))

from deck_building import (  # noqa: E402
  GATE_CARD_TYPE,
  IKZ_CARD_CODE,
  IKZ_CARD_COUNT,
  LEADER_CARD_TYPE,
  MAX_MAIN_COPIES,
  _load_policy_card_records,
)

DECKS_DIR = Path(__file__).resolve().parent
DEFAULT_CURATED = DECKS_DIR / "curated_decks.json"
DEFAULT_OUTPUT = DECKS_DIR / "curated_deck_pool.json"
OFFICIAL_CARDS = REPO_ROOT / "train-specialist" / "sources" / "official_cards_api_2026-09-30.json"
# Cards being added to the engine in the same change set; allowed even if metadata predates them.
PENDING_ENGINE_CARDS = frozenset({"AZK01-013", "AZK01-076", "AZK01-079", "AZK01-083", "AZK01-099"})
OFFICIAL_ELEMENTS = {"Fire": "FIRE", "Water": "WATER", "Earth": "EARTH", "Lightning": "LIGHTNING", "Neutral": "NORMAL"}
ELEMENT_ORDER = ("FIRE", "WATER", "EARTH", "LIGHTNING")
TIER_SAMPLING_WEIGHT = {"A": 2, "B": 1}
EXPECTED_MAIN_CARDS = 50


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path) -> str:
  resolved = path.resolve()
  try:
    return str(resolved.relative_to(REPO_ROOT))
  except ValueError:
    return str(resolved)


def _card_catalog() -> dict[str, tuple[str, str]]:
  """Map card code -> (category, engine element) from engine metadata plus pending official cards."""
  catalog = {record.card_code: (record.card_type, record.element) for record in _load_policy_card_records()}
  official = {card["id"]: card for card in json.loads(OFFICIAL_CARDS.read_text(encoding="utf-8"))}
  for code in PENDING_ENGINE_CARDS:
    if code in catalog:
      continue
    card = official[code]
    catalog[code] = (card["category"].upper(), OFFICIAL_ELEMENTS[card["element"]])
  missing_official = sorted(code for code in catalog if code != IKZ_CARD_CODE and code not in official)
  if missing_official:
    raise ValueError(f"Engine cards absent from the official card list: {missing_official}")
  return catalog


def _validate(deck: dict[str, Any], catalog: dict[str, tuple[str, str]]) -> tuple[tuple[str, int], ...]:
  label = deck["id"]
  counts: Counter[str] = Counter()
  for card in deck["cards"]:
    counts[card["card_id"]] += int(card["quantity"])
  unknown = sorted(code for code in counts if code not in catalog)
  if unknown:
    raise ValueError(f"{label}: cards unknown to engine and not pending: {unknown}")
  leaders = [code for code in counts if catalog[code][0] == LEADER_CARD_TYPE]
  gates = [code for code in counts if catalog[code][0] == GATE_CARD_TYPE]
  if leaders != [deck["leader_card_id"]] or gates != [deck["gate_card_id"]]:
    raise ValueError(f"{label}: leader/gate declarations do not match contents")
  if counts[leaders[0]] != 1 or counts[gates[0]] != 1:
    raise ValueError(f"{label}: leader and gate must appear once")
  gate_element = catalog[gates[0]][1]
  if catalog[leaders[0]][1] != gate_element or gate_element != deck["element"]:
    raise ValueError(f"{label}: leader/gate/declared element mismatch")
  if counts[IKZ_CARD_CODE] != IKZ_CARD_COUNT:
    raise ValueError(f"{label}: expected {IKZ_CARD_CODE} x{IKZ_CARD_COUNT}")
  main_total = 0
  for code, quantity in counts.items():
    if code in (leaders[0], gates[0], IKZ_CARD_CODE):
      continue
    if quantity > MAX_MAIN_COPIES:
      raise ValueError(f"{label}: {code} x{quantity} exceeds {MAX_MAIN_COPIES} copies")
    if catalog[code][1] not in ("NORMAL", gate_element):
      raise ValueError(f"{label}: off-element main card {code}")
    main_total += quantity
  if main_total != EXPECTED_MAIN_CARDS:
    raise ValueError(f"{label}: {main_total} main cards; expected {EXPECTED_MAIN_CARDS}")
  if deck["tier"] not in TIER_SAMPLING_WEIGHT:
    raise ValueError(f"{label}: untiered deck")
  if not deck["source_urls"] or not deck["placements"]:
    raise ValueError(f"{label}: missing source URL or placement evidence")
  return tuple(sorted(counts.items()))


def build(curated_path: Path) -> dict[str, Any]:
  curated = json.loads(curated_path.read_text(encoding="utf-8"))
  catalog = _card_catalog()
  entries = curated["decks"]
  signatures: dict[tuple[tuple[str, int], ...], str] = {}
  for entry in entries:
    signature = _validate(entry, catalog)
    if signature in signatures:
      raise ValueError(f"{entry['id']} duplicates {signatures[signature]}")
    signatures[signature] = entry["id"]

  element_rank = {element: rank for rank, element in enumerate(ELEMENT_ORDER)}
  heldout = sorted(
    (entry for entry in entries if entry["heldout_eval_only"]),
    key=lambda entry: element_rank[entry["element"]],
  )
  contexts: dict[tuple[str, str], list[dict[str, Any]]] = {}
  for entry in entries:
    if not entry["heldout_eval_only"]:
      contexts.setdefault((entry["gate_card_id"], entry["leader_card_id"]), []).append(entry)
  missing = [element for element in ELEMENT_ORDER if not any(e["element"] == element for g in contexts.values() for e in g)]
  if missing:
    raise ValueError(f"Elements without training decks: {missing}")

  pool_decks: list[dict[str, Any]] = []

  def _append(entry: dict[str, Any]) -> int:
    pool_decks.append({
      "deck_name": entry["deck_name"],
      "deck_slug": entry["id"],
      "element": entry["element"],
      "leader_card_id": entry["leader_card_id"],
      "gate_card_id": entry["gate_card_id"],
      "tier": entry["tier"],
      "sampling_weight": TIER_SAMPLING_WEIGHT[entry["tier"]],
      "heldout_eval_only": entry["heldout_eval_only"],
      "event": entry["event"],
      "date": entry["date"],
      "source_urls": entry["source_urls"],
      "cards": [{"card_id": card["card_id"], "quantity": card["quantity"]} for card in entry["cards"]],
    })
    return len(pool_decks) - 1

  holdout_indices = [_append(entry) for entry in heldout]
  groups = []
  training_indices: list[int] = []
  for gate, leader in sorted(contexts):
    members = contexts[(gate, leader)]
    indices = [_append(entry) for entry in members]
    training_indices.extend(indices)
    groups.append({
      "gate_card_id": gate,
      "leader_card_id": leader,
      "element": members[0]["element"],
      "deck_count": len(indices),
      "deck_indices": indices,
    })

  element_groups = []
  for element in ELEMENT_ORDER:
    train = [i for i in training_indices if pool_decks[i]["element"] == element]
    held = [i for i in holdout_indices if pool_decks[i]["element"] == element]
    element_groups.append({
      "element": element,
      "training_deck_indices": train,
      "heldout_deck_indices": held,
      "contexts": [
        {"gate_card_id": g["gate_card_id"], "leader_card_id": g["leader_card_id"], "deck_count": g["deck_count"]}
        for g in groups if g["element"] == element
      ],
      "training_tier_counts": dict(Counter(pool_decks[i]["tier"] for i in train)),
      "heldout_tier_counts": dict(Counter(pool_decks[i]["tier"] for i in held)),
    })

  return {
    "schema_version": 1,
    "source": {
      "builder": _portable(Path(__file__)),
      "inputs": [
        {"path": _portable(curated_path), "sha256": _sha256(curated_path), "deck_count": len(entries), "role": "curated_tournament_decks"},
        {"path": _portable(OFFICIAL_CARDS), "sha256": _sha256(OFFICIAL_CARDS), "role": "official_card_legality"},
      ],
    },
    "normalization_notes": [
      "Every deck is a curated, tiered tournament list (see curated_decks.json for evidence and URLs).",
      "Decks contain 1 leader, 1 gate, 50 element-legal main cards, and IKZ-001 x10.",
      "Held-out evaluation decks come first and are listed in summary.holdout_reference_deck_indices; they are never training decks.",
      "Training decks are grouped by (gate, leader) context; only contexts with curated decks are present.",
      "Rushfire Gate (AZK01-122) is intentionally legal; pending engine cards "
      + ", ".join(sorted(PENDING_ENGINE_CARDS)) + " are allowed.",
      "sampling_weight is informational (tier A=2, B=1).",
    ],
    "summary": {
      "reference_panel_size": len(holdout_indices),
      "holdout_reference_deck_indices": holdout_indices,
      "prebuilt_training_deck_indices": training_indices,
      "prebuilt_training_deck_count": len(training_indices),
      "prebuilt_deck_groups": groups,
      "element_groups": element_groups,
    },
    "decks": pool_decks,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()
  payload = build(args.curated.resolve())
  args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  summary = payload["summary"]
  print(
    f"wrote {args.output}: {len(payload['decks'])} decks, "
    f"{summary['prebuilt_training_deck_count']} training, {len(summary['holdout_reference_deck_indices'])} held out"
  )
  for group in summary["element_groups"]:
    print(
      f"  {group['element']}: train={len(group['training_deck_indices'])} {group['training_tier_counts']} "
      f"heldout={len(group['heldout_deck_indices'])} contexts={len(group['contexts'])}"
    )


if __name__ == "__main__":
  main()
