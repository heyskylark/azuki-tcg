from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_DECK_POOL_PATH = REPO_ROOT / ".codex" / "docs" / "azuki_tcg_decks_final.json"
EXPECTED_DECK_SIZE = 62

NativeDeckCard = tuple[str, int]
NativeDeck = tuple[NativeDeckCard, ...]
NativeDeckPool = tuple[NativeDeck, ...]
DeckLabelPool = tuple[str, ...]

_STARTER_DECKS: tuple[NativeDeck, ...] = (
  (
    ("STT01-001", 1),
    ("STT01-002", 1),
    ("STT01-003", 4),
    ("STT01-004", 4),
    ("STT01-005", 4),
    ("STT01-006", 2),
    ("STT01-007", 4),
    ("STT01-008", 4),
    ("STT01-009", 4),
    ("STT01-010", 2),
    ("STT01-011", 2),
    ("STT01-012", 4),
    ("STT01-013", 4),
    ("STT01-014", 4),
    ("STT01-015", 2),
    ("STT01-016", 2),
    ("STT01-017", 4),
    ("IKZ-001", 10),
  ),
  (
    ("STT02-001", 1),
    ("STT02-002", 1),
    ("STT02-003", 4),
    ("STT02-004", 4),
    ("STT02-005", 4),
    ("STT02-006", 4),
    ("STT02-007", 4),
    ("STT02-008", 4),
    ("STT02-009", 4),
    ("STT02-010", 2),
    ("STT02-011", 4),
    ("STT02-012", 4),
    ("STT02-013", 2),
    ("STT02-014", 2),
    ("STT02-015", 4),
    ("STT02-016", 2),
    ("STT02-017", 2),
    ("IKZ-001", 10),
  ),
)

_STARTER_DECK_LABELS: DeckLabelPool = (
  "starter_raizan",
  "starter_shao",
)


def resolve_training_deck_pool_path(path: str | Path | None = None) -> Path:
  if path is None:
    return DEFAULT_TRAINING_DECK_POOL_PATH

  resolved = Path(path).expanduser()
  if not resolved.is_absolute():
    resolved = REPO_ROOT / resolved
  return resolved.resolve()


@lru_cache(maxsize=None)
def load_training_deck_pool(path: str | Path | None = None) -> NativeDeckPool:
  resolved_path = resolve_training_deck_pool_path(path)
  payload = json.loads(resolved_path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"Training deck pool file must contain a JSON object: {resolved_path}")

  decks = payload.get("decks")
  if not isinstance(decks, list) or not decks:
    raise ValueError(f"Training deck pool file must contain a non-empty 'decks' list: {resolved_path}")

  native_decks = [_normalize_starter_deck(deck) for deck in _STARTER_DECKS]
  for index, deck in enumerate(decks):
    native_decks.append(_normalize_payload_deck(deck, index=index, source_path=resolved_path))
  return tuple(native_decks)


@lru_cache(maxsize=None)
def load_training_deck_labels(path: str | Path | None = None) -> DeckLabelPool:
  resolved_path = resolve_training_deck_pool_path(path)
  payload = json.loads(resolved_path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"Training deck pool file must contain a JSON object: {resolved_path}")

  decks = payload.get("decks")
  if not isinstance(decks, list) or not decks:
    raise ValueError(f"Training deck pool file must contain a non-empty 'decks' list: {resolved_path}")

  labels = list(_STARTER_DECK_LABELS)
  for index, deck in enumerate(decks):
    if not isinstance(deck, dict):
      raise ValueError(f"Deck entry {index} in {resolved_path} must be a JSON object")
    deck_slug = deck.get("deck_slug")
    deck_name = deck.get("deck_name")
    if isinstance(deck_slug, str) and deck_slug:
      labels.append(deck_slug)
    elif isinstance(deck_name, str) and deck_name:
      normalized = deck_name.strip().lower().replace(" ", "_")
      labels.append(normalized)
    else:
      labels.append(f"deck_{index + len(_STARTER_DECK_LABELS):02d}")
  return tuple(labels)


def _normalize_payload_deck(deck: object, *, index: int, source_path: Path) -> NativeDeck:
  if not isinstance(deck, dict):
    raise ValueError(f"Deck entry {index} in {source_path} must be a JSON object")

  deck_name = deck.get("deck_name")
  deck_label = deck_name if isinstance(deck_name, str) and deck_name else f"deck[{index}]"
  cards = deck.get("cards")
  if not isinstance(cards, list) or not cards:
    raise ValueError(f"{deck_label} in {source_path} must contain a non-empty 'cards' list")

  native_cards: list[NativeDeckCard] = []
  total_cards = 0
  for card_index, card in enumerate(cards):
    if not isinstance(card, dict):
      raise ValueError(f"{deck_label} card[{card_index}] in {source_path} must be an object")
    card_id = card.get("card_id")
    quantity = card.get("quantity")
    if not isinstance(card_id, str) or not card_id:
      raise ValueError(f"{deck_label} card[{card_index}] in {source_path} is missing a valid card_id")
    if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
      raise ValueError(f"{deck_label} card[{card_index}] in {source_path} has invalid quantity {quantity!r}")
    native_cards.append((card_id, quantity))
    total_cards += quantity

  if total_cards != EXPECTED_DECK_SIZE:
    raise ValueError(
      f"{deck_label} in {source_path} has {total_cards} cards; expected {EXPECTED_DECK_SIZE}"
    )
  return tuple(native_cards)


def _normalize_starter_deck(deck: NativeDeck) -> NativeDeck:
  total_cards = sum(quantity for _, quantity in deck)
  if total_cards != EXPECTED_DECK_SIZE:
    raise ValueError(f"Starter deck has {total_cards} cards; expected {EXPECTED_DECK_SIZE}")
  return deck
