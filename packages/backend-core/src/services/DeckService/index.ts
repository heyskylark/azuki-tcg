import { and, eq, inArray, isNull, ne, sql } from "drizzle-orm";
import db, { type IDatabase, type ITransaction } from "@core/database";
import { Cards, Decks, DeckCardJunctions } from "@core/drizzle/schemas";
import { DeckStatus } from "@core/types";
import { CardRarity, RarityOrdering } from "@core/types/cards";
import type { DeckSummary } from "@core/types/deck";
import { starterDecks } from "@core/services/DeckService/constants";

type Database = IDatabase | ITransaction;

interface CardRecord {
  id: string;
  cardCode: string;
  rarity: CardRarity;
}

/**
 * Select the least rare card for each cardCode.
 * Excludes cards with specialRarity set (non-null).
 * Uses RarityOrdering to determine the least rare variant.
 */
function selectLeastRareCards(
  cards: CardRecord[]
): Map<string, string> {
  // Group cards by cardCode
  const cardsByCode = new Map<string, CardRecord[]>();
  for (const card of cards) {
    const existing = cardsByCode.get(card.cardCode) ?? [];
    existing.push(card);
    cardsByCode.set(card.cardCode, existing);
  }

  // For each cardCode, select the card with the lowest rarity ordering
  const cardCodeToId = new Map<string, string>();
  for (const [cardCode, variants] of cardsByCode) {
    // Sort by rarity ordering (ascending - least rare first)
    variants.sort(
      (a, b) => RarityOrdering[a.rarity] - RarityOrdering[b.rarity]
    );
    // Pick the least rare variant
    cardCodeToId.set(cardCode, variants[0]!.id);
  }

  return cardCodeToId;
}

export async function addStarterDecks(
  userId: string,
  database: Database = db
): Promise<void> {
  // Collect all unique card codes needed
  const allCardCodes = new Set<string>();
  for (const deck of starterDecks) {
    for (const card of deck.cards) {
      allCardCodes.add(card.cardCode);
    }
  }

  // Look up all cards by cardCode, excluding those with specialRarity set
  const cards = await database
    .select({
      id: Cards.id,
      cardCode: Cards.cardCode,
      rarity: Cards.rarity,
    })
    .from(Cards)
    .where(
      and(
        inArray(Cards.cardCode, Array.from(allCardCodes)),
        isNull(Cards.specialRarity)
      )
    );

  // Select the least rare card for each cardCode
  const cardCodeToId = selectLeastRareCards(cards);

  // Verify all required cards exist
  for (const cardCode of allCardCodes) {
    if (!cardCodeToId.has(cardCode)) {
      throw new Error(
        `Card with code "${cardCode}" not found in database (without specialRarity). Ensure cards are seeded before creating users.`
      );
    }
  }

  // Create each starter deck
  for (const deckConfig of starterDecks) {
    // Create the deck
    const [deck] = await database
      .insert(Decks)
      .values({
        name: deckConfig.name,
        userId,
        status: DeckStatus.COMPLETE,
        isSystemDeck: true,
      })
      .returning({ id: Decks.id });

    // Prepare deck-card junction entries
    const junctionEntries = deckConfig.cards.map((cardInfo) => ({
      deckId: deck!.id,
      cardId: cardCodeToId.get(cardInfo.cardCode)!,
      quantity: cardInfo.quantity,
    }));

    // Insert all card entries for this deck
    await database.insert(DeckCardJunctions).values(junctionEntries);
  }
}

export async function getUserDecks(
  userId: string,
  database: Database = db
): Promise<DeckSummary[]> {
  const result = await database
    .select({
      id: Decks.id,
      name: Decks.name,
      isSystemDeck: Decks.isSystemDeck,
      cardCount: sql<number>`COALESCE(SUM(${DeckCardJunctions.quantity}), 0)`.as(
        "card_count"
      ),
    })
    .from(Decks)
    .leftJoin(DeckCardJunctions, eq(Decks.id, DeckCardJunctions.deckId))
    .where(and(eq(Decks.userId, userId), ne(Decks.status, DeckStatus.DELETED)))
    .groupBy(Decks.id, Decks.name, Decks.isSystemDeck)
    .orderBy(Decks.createdAt);

  return result.map((row) => ({
    id: row.id,
    name: row.name,
    isSystemDeck: row.isSystemDeck,
    cardCount: Number(row.cardCount),
  }));
}

export interface DeckCardDetail {
  cardCode: string;
  name: string;
  imageKey: string;
  cardType: string;
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

/**
 * Get a deck with all its cards and card details.
 * Returns null if deck not found or deleted.
 */
export async function getDeckWithCards(
  deckId: string,
  database: Database = db
): Promise<DeckWithCards | null> {
  // First get the deck
  const deck = await database
    .select({
      id: Decks.id,
      name: Decks.name,
      isSystemDeck: Decks.isSystemDeck,
    })
    .from(Decks)
    .where(and(eq(Decks.id, deckId), ne(Decks.status, DeckStatus.DELETED)))
    .limit(1)
    .then((results) => results[0]);

  if (!deck) {
    return null;
  }

  // Get all cards in the deck with their details
  const cards = await database
    .select({
      cardCode: Cards.cardCode,
      name: Cards.name,
      imageKey: Cards.imageKey,
      cardType: Cards.cardType,
      attack: Cards.attack,
      health: Cards.health,
      ikzCost: Cards.ikzCost,
      quantity: DeckCardJunctions.quantity,
    })
    .from(DeckCardJunctions)
    .innerJoin(Cards, eq(DeckCardJunctions.cardId, Cards.id))
    .where(eq(DeckCardJunctions.deckId, deckId));

  return {
    id: deck.id,
    name: deck.name,
    isSystemDeck: deck.isSystemDeck,
    cards,
  };
}
