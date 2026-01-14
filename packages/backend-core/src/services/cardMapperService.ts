import { and, eq, inArray, isNull } from "drizzle-orm";
import db, { type IDatabase, type ITransaction } from "@core/database";
import { Cards, Decks, DeckCardJunctions } from "@core/drizzle/schemas";
import { CardRarity, CardType, RarityOrdering } from "@core/types/cards";

type Database = IDatabase | ITransaction;

/**
 * CardDefId values matching include/generated/card_defs.h
 * These must stay in sync with the C engine card definitions.
 */
export enum CardDefId {
  IKZ_001 = 0,
  IKZ_002 = 1,
  STT01_001 = 2,
  STT01_002 = 3,
  STT01_003 = 4,
  STT01_004 = 5,
  STT01_005 = 6,
  STT01_006 = 7,
  STT01_007 = 8,
  STT01_008 = 9,
  STT01_009 = 10,
  STT01_010 = 11,
  STT01_011 = 12,
  STT01_012 = 13,
  STT01_013 = 14,
  STT01_014 = 15,
  STT01_015 = 16,
  STT01_016 = 17,
  STT01_017 = 18,
  STT02_001 = 19,
  STT02_002 = 20,
  STT02_003 = 21,
  STT02_004 = 22,
  STT02_005 = 23,
  STT02_006 = 24,
  STT02_007 = 25,
  STT02_008 = 26,
  STT02_009 = 27,
  STT02_010 = 28,
  STT02_011 = 29,
  STT02_012 = 30,
  STT02_013 = 31,
  STT02_014 = 32,
  STT02_015 = 33,
  STT02_016 = 34,
  STT02_017 = 35,
}

/**
 * Mapping from database cardCode to C engine CardDefId.
 * Generated from include/generated/card_defs.h CardDefId enum.
 */
const CARD_CODE_TO_DEF_ID: Record<string, CardDefId> = {
  "IKZ-001": CardDefId.IKZ_001,
  "IKZ-002": CardDefId.IKZ_002,
  "STT01-001": CardDefId.STT01_001,
  "STT01-002": CardDefId.STT01_002,
  "STT01-003": CardDefId.STT01_003,
  "STT01-004": CardDefId.STT01_004,
  "STT01-005": CardDefId.STT01_005,
  "STT01-006": CardDefId.STT01_006,
  "STT01-007": CardDefId.STT01_007,
  "STT01-008": CardDefId.STT01_008,
  "STT01-009": CardDefId.STT01_009,
  "STT01-010": CardDefId.STT01_010,
  "STT01-011": CardDefId.STT01_011,
  "STT01-012": CardDefId.STT01_012,
  "STT01-013": CardDefId.STT01_013,
  "STT01-014": CardDefId.STT01_014,
  "STT01-015": CardDefId.STT01_015,
  "STT01-016": CardDefId.STT01_016,
  "STT01-017": CardDefId.STT01_017,
  "STT02-001": CardDefId.STT02_001,
  "STT02-002": CardDefId.STT02_002,
  "STT02-003": CardDefId.STT02_003,
  "STT02-004": CardDefId.STT02_004,
  "STT02-005": CardDefId.STT02_005,
  "STT02-006": CardDefId.STT02_006,
  "STT02-007": CardDefId.STT02_007,
  "STT02-008": CardDefId.STT02_008,
  "STT02-009": CardDefId.STT02_009,
  "STT02-010": CardDefId.STT02_010,
  "STT02-011": CardDefId.STT02_011,
  "STT02-012": CardDefId.STT02_012,
  "STT02-013": CardDefId.STT02_013,
  "STT02-014": CardDefId.STT02_014,
  "STT02-015": CardDefId.STT02_015,
  "STT02-016": CardDefId.STT02_016,
  "STT02-017": CardDefId.STT02_017,
};

/**
 * Reverse mapping from CardDefId to database cardCode.
 */
const DEF_ID_TO_CARD_CODE: Record<CardDefId, string> = Object.fromEntries(
  Object.entries(CARD_CODE_TO_DEF_ID).map(([code, id]) => [id, code])
) as Record<CardDefId, string>;

/**
 * Convert a database cardCode (e.g., "STT01-001") to C engine CardDefId.
 * Returns null if the cardCode is not recognized.
 */
export function cardCodeToDefId(cardCode: string): CardDefId | null {
  return CARD_CODE_TO_DEF_ID[cardCode] ?? null;
}

/**
 * Convert a C engine CardDefId to database cardCode.
 * Returns null if the defId is not recognized.
 */
export function defIdToCardCode(defId: CardDefId): string | null {
  return DEF_ID_TO_CARD_CODE[defId] ?? null;
}

/**
 * Deck structure with CardDefIds ready for the C engine.
 */
export interface DeckAsDefIds {
  leader: CardDefId;
  gate: CardDefId;
  mainDeck: CardDefId[]; // Flattened with quantities (e.g., 4 copies = 4 entries)
  ikzPile: CardDefId[]; // Flattened with quantities
}

interface CardWithType {
  id: string;
  cardCode: string;
  cardType: CardType;
  rarity: CardRarity;
  quantity: number;
}

/**
 * Select the least rare card for each cardCode.
 * Excludes cards with specialRarity set (non-null).
 */
function selectLeastRareCards(
  cards: Array<{ id: string; cardCode: string; rarity: CardRarity }>
): Map<string, string> {
  const cardsByCode = new Map<string, Array<{ id: string; cardCode: string; rarity: CardRarity }>>();
  for (const card of cards) {
    const existing = cardsByCode.get(card.cardCode) ?? [];
    existing.push(card);
    cardsByCode.set(card.cardCode, existing);
  }

  const cardCodeToId = new Map<string, string>();
  for (const [cardCode, variants] of cardsByCode) {
    variants.sort((a, b) => RarityOrdering[a.rarity] - RarityOrdering[b.rarity]);
    const leastRare = variants[0];
    if (leastRare) {
      cardCodeToId.set(cardCode, leastRare.id);
    }
  }

  return cardCodeToId;
}

/**
 * Load a deck from the database and convert all cards to CardDefIds.
 * Separates cards into leader, gate, main deck, and IKZ pile based on CardType.
 *
 * @throws Error if deck not found or if any card code cannot be mapped to CardDefId
 */
export async function loadDeckAsDefIds(
  deckId: string,
  database: Database = db
): Promise<DeckAsDefIds> {
  // Fetch deck with all card junctions
  const deckExists = await database
    .select({ id: Decks.id })
    .from(Decks)
    .where(eq(Decks.id, deckId))
    .limit(1)
    .then((rows) => rows[0]);

  if (!deckExists) {
    throw new Error(`Deck not found: ${deckId}`);
  }

  // Get all cards in the deck with their types
  const deckCards = await database
    .select({
      cardId: DeckCardJunctions.cardId,
      quantity: DeckCardJunctions.quantity,
      cardCode: Cards.cardCode,
      cardType: Cards.cardType,
      rarity: Cards.rarity,
    })
    .from(DeckCardJunctions)
    .innerJoin(Cards, eq(DeckCardJunctions.cardId, Cards.id))
    .where(eq(DeckCardJunctions.deckId, deckId));

  if (deckCards.length === 0) {
    throw new Error(`Deck has no cards: ${deckId}`);
  }

  let leader: CardDefId | null = null;
  let gate: CardDefId | null = null;
  const mainDeck: CardDefId[] = [];
  const ikzPile: CardDefId[] = [];

  for (const card of deckCards) {
    const defId = cardCodeToDefId(card.cardCode);
    if (defId === null) {
      throw new Error(`Unknown cardCode: ${card.cardCode} - not found in CardDefId mapping`);
    }

    switch (card.cardType) {
      case CardType.LEADER:
        if (leader !== null) {
          throw new Error(`Deck has multiple leaders: ${deckId}`);
        }
        leader = defId;
        break;

      case CardType.GATE:
        if (gate !== null) {
          throw new Error(`Deck has multiple gates: ${deckId}`);
        }
        gate = defId;
        break;

      case CardType.IKZ:
      case CardType.EXTRA_IKZ:
        // Flatten by quantity
        for (let i = 0; i < card.quantity; i++) {
          ikzPile.push(defId);
        }
        break;

      case CardType.ENTITY:
      case CardType.WEAPON:
      case CardType.SPELL:
        // Flatten by quantity
        for (let i = 0; i < card.quantity; i++) {
          mainDeck.push(defId);
        }
        break;

      default:
        throw new Error(`Unknown card type: ${card.cardType}`);
    }
  }

  if (leader === null) {
    throw new Error(`Deck missing leader: ${deckId}`);
  }

  if (gate === null) {
    throw new Error(`Deck missing gate: ${deckId}`);
  }

  return {
    leader,
    gate,
    mainDeck,
    ikzPile,
  };
}

/**
 * Flatten a DeckAsDefIds structure into a single array suitable for engine initialization.
 * Format: [leader, gate, ...mainDeck, ...ikzPile]
 */
export function flattenDeckForEngine(deck: DeckAsDefIds): number[] {
  return [deck.leader, deck.gate, ...deck.mainDeck, ...deck.ikzPile];
}

/**
 * Entry format for native module deck initialization.
 */
export interface DeckCardEntry {
  cardId: number;
  count: number;
}

/**
 * Convert a DeckAsDefIds structure into DeckCardEntry[] for native module.
 * Groups duplicate CardDefIds and sets their counts.
 * Includes leader, gate, main deck, and IKZ pile cards.
 */
export function deckAsDefIdsToDeckEntries(deck: DeckAsDefIds): DeckCardEntry[] {
  const entries: DeckCardEntry[] = [];
  const cardCounts = new Map<number, number>();

  // Add leader (always count 1)
  entries.push({ cardId: deck.leader, count: 1 });

  // Add gate (always count 1)
  entries.push({ cardId: deck.gate, count: 1 });

  // Count main deck cards (already flattened, so we need to aggregate)
  for (const cardId of deck.mainDeck) {
    cardCounts.set(cardId, (cardCounts.get(cardId) ?? 0) + 1);
  }

  // Count IKZ pile cards
  for (const cardId of deck.ikzPile) {
    cardCounts.set(cardId, (cardCounts.get(cardId) ?? 0) + 1);
  }

  // Add aggregated counts
  for (const [cardId, count] of cardCounts) {
    entries.push({ cardId, count });
  }

  return entries;
}
