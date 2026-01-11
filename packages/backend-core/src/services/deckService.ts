import { inArray } from "drizzle-orm";
import db, { type IDatabase, type ITransaction } from "@/database";
import { Cards, Decks, DeckCardJunctions } from "@/drizzle/schemas";
import { DeckStatus } from "@/types";

type Database = IDatabase | ITransaction;

interface StarterCardInfo {
  cardCode: string;
  quantity: number;
}

// Shao Starter deck configuration (from world.c shaoDeckCardInfo)
const shaoStarterCards: StarterCardInfo[] = [
  { cardCode: "STT02_001", quantity: 1 },
  { cardCode: "STT02_002", quantity: 1 },
  { cardCode: "STT02_003", quantity: 4 },
  { cardCode: "STT02_004", quantity: 4 },
  { cardCode: "STT02_005", quantity: 4 },
  { cardCode: "STT02_006", quantity: 4 },
  { cardCode: "STT02_007", quantity: 4 },
  { cardCode: "STT02_008", quantity: 4 },
  { cardCode: "STT02_009", quantity: 4 },
  { cardCode: "STT02_010", quantity: 2 },
  { cardCode: "STT02_011", quantity: 4 },
  { cardCode: "STT02_012", quantity: 4 },
  { cardCode: "STT02_013", quantity: 2 },
  { cardCode: "STT02_014", quantity: 2 },
  { cardCode: "STT02_015", quantity: 4 },
  { cardCode: "STT02_016", quantity: 2 },
  { cardCode: "STT02_017", quantity: 2 },
  { cardCode: "IKZ_001", quantity: 10 },
];

// Raizan Starter deck configuration (from world.c raizenDeckCardInfo)
const raizanStarterCards: StarterCardInfo[] = [
  { cardCode: "STT01_001", quantity: 1 },
  { cardCode: "STT01_002", quantity: 1 },
  { cardCode: "STT01_003", quantity: 4 },
  { cardCode: "STT01_004", quantity: 4 },
  { cardCode: "STT01_005", quantity: 4 },
  { cardCode: "STT01_006", quantity: 2 },
  { cardCode: "STT01_007", quantity: 4 },
  { cardCode: "STT01_008", quantity: 4 },
  { cardCode: "STT01_009", quantity: 4 },
  { cardCode: "STT01_010", quantity: 2 },
  { cardCode: "STT01_011", quantity: 2 },
  { cardCode: "STT01_012", quantity: 4 },
  { cardCode: "STT01_013", quantity: 4 },
  { cardCode: "STT01_014", quantity: 4 },
  { cardCode: "STT01_015", quantity: 2 },
  { cardCode: "STT01_016", quantity: 2 },
  { cardCode: "STT01_017", quantity: 4 },
  { cardCode: "IKZ_001", quantity: 10 },
];

interface StarterDeckConfig {
  name: string;
  cards: StarterCardInfo[];
}

const starterDecks: StarterDeckConfig[] = [
  { name: "Shao Starter", cards: shaoStarterCards },
  { name: "Raizan Starter", cards: raizanStarterCards },
];

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

  // Look up all cards by cardCode in a single query
  const cards = await database
    .select({ id: Cards.id, cardCode: Cards.cardCode })
    .from(Cards)
    .where(inArray(Cards.cardCode, Array.from(allCardCodes)));

  // Build a map from cardCode to cardId
  const cardCodeToId = new Map<string, string>();
  for (const card of cards) {
    // Only store the first match (in case of multiple rarities)
    if (!cardCodeToId.has(card.cardCode)) {
      cardCodeToId.set(card.cardCode, card.id);
    }
  }

  // Verify all required cards exist
  for (const cardCode of allCardCodes) {
    if (!cardCodeToId.has(cardCode)) {
      throw new Error(
        `Card with code "${cardCode}" not found in database. Ensure cards are seeded before creating users.`
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
