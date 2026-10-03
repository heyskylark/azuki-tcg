from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

from deck_building import (
  GATE_CARD_TYPE,
  LEADER_CARD_TYPE,
  IKZ_CARD_CODE,
  MAIN_CARD_TYPES,
  _load_policy_card_records,
  _validate_deck_shape,
)
from training_deck_pool import NativeDeck, load_training_deck_pool, resolve_training_deck_pool_path

REFERENCE_PANEL_SIZE = 18
MIN_DECKS_PER_CONTEXT = 2
EXPECTED_CONTEXT_COUNT = 16

PrebuiltDeckGroups = tuple[tuple[int, ...], ...]


def _deck_signature(deck: NativeDeck) -> tuple[tuple[str, int], ...]:
  counts: Counter[str] = Counter()
  for card_id, quantity in deck:
    counts[card_id] += int(quantity)
  return tuple(sorted(counts.items()))


def _expected_contexts() -> tuple[tuple[str, str], ...]:
  records = _load_policy_card_records()
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


def _integer_indices(value: object, *, label: str, deck_count: int) -> tuple[int, ...]:
  if not isinstance(value, list) or not value:
    raise ValueError(f"Prebuilt deck pool summary must contain a non-empty '{label}' list")
  indices: list[int] = []
  for position, index in enumerate(value):
    if not isinstance(index, int) or isinstance(index, bool):
      raise ValueError(f"{label}[{position}] must be an integer")
    if not 0 <= index < deck_count:
      raise ValueError(f"{label}[{position}] index {index} is outside the {deck_count}-deck pool")
    indices.append(index)
  if len(indices) != len(set(indices)):
    raise ValueError(f"{label} must not contain duplicate indices")
  return tuple(indices)


@lru_cache(maxsize=None)
def load_prebuilt_deck_groups(path: str | Path) -> PrebuiltDeckGroups:
  """Load and verify the production prebuilt curriculum groups from a deck-pool file."""
  resolved_path = resolve_training_deck_pool_path(path)
  payload = json.loads(resolved_path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError(f"Prebuilt deck pool file must contain a JSON object: {resolved_path}")
  if payload.get("schema_version") != 1:
    raise ValueError(f"Prebuilt deck pool must use schema_version 1: {resolved_path}")
  raw_decks = payload.get("decks")
  if not isinstance(raw_decks, list):
    raise ValueError(f"Prebuilt deck pool must contain a decks list: {resolved_path}")

  deck_pool = load_training_deck_pool(resolved_path)
  if len(deck_pool) < REFERENCE_PANEL_SIZE:
    raise ValueError(
      f"Prebuilt deck pool has {len(deck_pool)} decks; expected the first "
      f"{REFERENCE_PANEL_SIZE} reference decks"
    )

  summary = payload.get("summary")
  if not isinstance(summary, dict):
    raise ValueError(f"Prebuilt deck pool must contain a summary object: {resolved_path}")
  if summary.get("reference_panel_size") != REFERENCE_PANEL_SIZE:
    raise ValueError(f"Prebuilt deck pool reference_panel_size must be {REFERENCE_PANEL_SIZE}")
  promotion = _integer_indices(
    summary.get("promotion_reference_deck_indices"),
    label="promotion_reference_deck_indices",
    deck_count=len(deck_pool),
  )
  holdout = _integer_indices(
    summary.get("holdout_reference_deck_indices"),
    label="holdout_reference_deck_indices",
    deck_count=len(deck_pool),
  )
  if promotion != tuple(range(0, REFERENCE_PANEL_SIZE, 2)):
    raise ValueError("Promotion reference indices must preserve the original even indices 0..16")
  if holdout != tuple(range(1, REFERENCE_PANEL_SIZE, 2)):
    raise ValueError("Holdout reference indices must preserve the original odd indices 1..17")

  training_indices = _integer_indices(
    summary.get("prebuilt_training_deck_indices"),
    label="prebuilt_training_deck_indices",
    deck_count=len(deck_pool),
  )
  expected_training_indices = tuple(range(REFERENCE_PANEL_SIZE, len(deck_pool)))
  if training_indices != expected_training_indices:
    raise ValueError(
      "prebuilt_training_deck_indices must list every post-reference deck exactly once in pool order"
    )

  records = _load_policy_card_records()
  records_by_code = {record.card_code: record for record in records}
  actual_contexts: dict[int, tuple[str, str]] = {}
  for deck_index, deck in enumerate(deck_pool):
    if any(card_id not in records_by_code for card_id, _ in deck):
      raise ValueError(f"Prebuilt deck {deck_index} contains unsupported cards")
    if sum(quantity for _, quantity in deck) != 62:
      raise ValueError(f"Prebuilt deck {deck_index} must contain exactly 62 cards")
    leader, gate = _validate_deck_shape(deck, records_by_code, deck_index=deck_index)
    raw_deck = raw_decks[deck_index]
    if (
      raw_deck.get("gate_card_id") != gate.card_code
      or raw_deck.get("leader_card_id") != leader.card_code
    ):
      raise ValueError(f"Prebuilt deck {deck_index} has false gate/leader declarations")
    for card_id, _ in deck:
      record = records_by_code[card_id]
      if record.card_type in MAIN_CARD_TYPES:
        if deck_index >= REFERENCE_PANEL_SIZE and record.element not in ("NORMAL", gate.element):
          raise ValueError(f"Prebuilt deck {deck_index} contains off-element main card {card_id}")
      elif card_id != IKZ_CARD_CODE and record.card_type not in (GATE_CARD_TYPE, LEADER_CARD_TYPE):
        raise ValueError(f"Prebuilt deck {deck_index} contains unsupported card type {record.card_type}")
    actual_contexts[deck_index] = (gate.card_code, leader.card_code)

  reference_signatures = {_deck_signature(deck) for deck in deck_pool[:REFERENCE_PANEL_SIZE]}
  training_signatures: set[tuple[tuple[str, int], ...]] = set()
  for deck_index in training_indices:
    signature = _deck_signature(deck_pool[deck_index])
    if signature in reference_signatures:
      raise ValueError(f"Prebuilt training deck {deck_index} overlaps the reference panel")
    if signature in training_signatures:
      raise ValueError(f"Prebuilt training deck {deck_index} duplicates another training deck")
    training_signatures.add(signature)

  raw_groups = summary.get("prebuilt_deck_groups")
  if not isinstance(raw_groups, list):
    raise ValueError("Prebuilt deck pool summary must contain a prebuilt_deck_groups list")
  expected_contexts = _expected_contexts()
  if len(raw_groups) != len(expected_contexts):
    raise ValueError(
      f"prebuilt_deck_groups has {len(raw_groups)} contexts; expected {len(expected_contexts)}"
    )

  groups: list[tuple[int, ...]] = []
  declared_contexts: list[tuple[str, str]] = []
  grouped_indices: list[int] = []
  for group_index, raw_group in enumerate(raw_groups):
    if not isinstance(raw_group, dict):
      raise ValueError(f"prebuilt_deck_groups[{group_index}] must be an object")
    gate_card_id = raw_group.get("gate_card_id")
    leader_card_id = raw_group.get("leader_card_id")
    if not isinstance(gate_card_id, str) or not isinstance(leader_card_id, str):
      raise ValueError(f"prebuilt_deck_groups[{group_index}] has invalid gate/leader ids")
    context = (gate_card_id, leader_card_id)
    declared_contexts.append(context)
    indices = _integer_indices(
      raw_group.get("deck_indices"),
      label=f"prebuilt_deck_groups[{group_index}].deck_indices",
      deck_count=len(deck_pool),
    )
    declared_count = raw_group.get("deck_count")
    if declared_count != len(indices):
      raise ValueError(
        f"Prebuilt context {gate_card_id}/{leader_card_id} declares deck_count "
        f"{declared_count!r}; actual count is {len(indices)}"
      )
    if len(indices) < MIN_DECKS_PER_CONTEXT:
      raise ValueError(
        f"Prebuilt context {gate_card_id}/{leader_card_id} has {len(indices)} decks; "
        f"requires at least {MIN_DECKS_PER_CONTEXT}"
      )
    for deck_index in indices:
      if deck_index < REFERENCE_PANEL_SIZE:
        raise ValueError(
          f"Prebuilt context {gate_card_id}/{leader_card_id} includes reserved reference index {deck_index}"
        )
      if actual_contexts[deck_index] != context:
        actual_gate, actual_leader = actual_contexts[deck_index]
        raise ValueError(
          f"Prebuilt deck {deck_index} is {actual_gate}/{actual_leader}, not "
          f"declared context {gate_card_id}/{leader_card_id}"
        )
    groups.append(indices)
    grouped_indices.extend(indices)

  if tuple(declared_contexts) != expected_contexts:
    raise ValueError("prebuilt_deck_groups must contain all compatible contexts sorted by gate then leader")
  if tuple(grouped_indices) != training_indices:
    if set(grouped_indices) != set(training_indices):
      raise ValueError(
        "prebuilt_deck_groups must partition prebuilt_training_deck_indices without omissions or extras"
      )
    raise ValueError(
      "prebuilt_deck_groups deck indices must preserve the same deterministic order as "
      "prebuilt_training_deck_indices"
    )
  return tuple(groups)
