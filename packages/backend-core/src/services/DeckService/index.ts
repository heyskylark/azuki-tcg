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
