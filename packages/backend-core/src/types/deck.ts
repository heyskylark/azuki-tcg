import type { CardElement, CardType } from "@core/types/cards";

export interface DeckSummary {
  id: string;
  name: string;
  isSystemDeck: boolean;
  cardCount: number;
}

export interface DeckCardDetail {
  cardId: string;
  cardCode: string;
  cardDefId: number;
  name: string;
  imageKey: string;
  cardType: CardType;
  element: CardElement;
  attack: number | null;
  health: number | null;
  ikzCost: number | null;
  quantity: number;
}

export interface DeckWithCards {
  id: string;
  name: string;
  isSystemDeck: boolean;
  cards: DeckCardDetail[];
}

export interface DeckBuilderCard {
  id: string;
  cardCode: string;
  name: string;
  imageKey: string;
  cardType: CardType;
  element: CardElement;
  attack: number | null;
  health: number | null;
  ikzCost: number | null;
}

export interface EditableDeck {
  id: string;
  name: string;
  isSystemDeck: boolean;
  cards: DeckBuilderSelectionCard[];
}

export interface DeckBuilderSelectionCard {
  cardId: string;
  quantity: number;
}

export interface UpsertDeckInput {
  name: string;
  cardCounts: Record<string, number>;
}
