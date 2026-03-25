import { and, eq, inArray, isNull, ne, sql } from "drizzle-orm";
import db, { type IDatabase, type ITransaction } from "@core/database";
import { Cards, Decks, DeckCardJunctions } from "@core/drizzle/schemas";
import { DeckNotFoundError, ForbiddenError, ValidationError } from "@core/errors";
import { DeckStatus } from "@core/types";
import { CardElement, CardRarity, CardType, RarityOrdering } from "@core/types/cards";
import type {
  DeckBuilderCard,
  DeckCardDetail,
  DeckSummary,
  DeckWithCards,
  EditableDeck,
  UpsertDeckInput,
} from "@core/types/deck";
import { starterDecks } from "@core/services/DeckService/constants";
import { cardCodeToDefId } from "@core/services/cardMapperService";

type Database = IDatabase | ITransaction;

const AUTO_IKZ_CARD_CODE = "IKZ-001";
const AUTO_IKZ_CARD_QUANTITY = 10;
const MAIN_DECK_CARD_COUNT = 50;

interface CardRecord {
  cardCode: string;
  rarity: CardRarity;
}

interface StarterCardRecord extends CardRecord {
  id: string;
}

interface DeckCardRow {
  cardId: string;
  cardCode: string;
  name: string;
  imageKey: string;
  cardType: CardType;
  element: CardElement;
  attack: number | null;
  health: number | null;
  ikzCost: number | null;
  quantity: number;
}

interface DeckBaseRecord {
  id: string;
  name: string;
  isSystemDeck: boolean;
  status: DeckStatus;
}

interface SelectedDeckCard {
  id: string;
  name: string;
  cardType: CardType;
  element: CardElement;
}

interface ResolvedDeckInput {
  gateCardId: string;
  leaderCardId: string;
  mainDeckEntries: Array<{ cardId: string; quantity: number }>;
  ikzCardId: string;
}

function selectLeastRareCards<T extends CardRecord>(cards: T[]): Map<string, T> {
  const selectedCards = new Map<string, T>();

  for (const card of cards) {
    const existingCard = selectedCards.get(card.cardCode);

    if (existingCard == null || RarityOrdering[card.rarity] < RarityOrdering[existingCard.rarity]) {
      selectedCards.set(card.cardCode, card);
    }
  }

  return selectedCards;
}

function normalizeDeckName(name: string): string {
  const trimmedName = name.trim();

  if (trimmedName.length === 0) {
    throw new ValidationError("Deck name is required");
  }

  return trimmedName;
}

function isAllowedDeckElement(cardElement: CardElement, gateElement: CardElement): boolean {
  return cardElement === CardElement.NORMAL || cardElement === gateElement;
}

function mapDeckCardRowsToDetails(cardRows: DeckCardRow[]): DeckCardDetail[] {
  return cardRows.map((row) => {
    const cardDefId = cardCodeToDefId(row.cardCode);

    if (cardDefId === null) {
      throw new Error(`Unknown cardCode: ${row.cardCode} - not found in CardDefId mapping`);
    }

    return {
      cardId: row.cardId,
      cardCode: row.cardCode,
      cardDefId,
      name: row.name,
      imageKey: row.imageKey,
      cardType: row.cardType,
      element: row.element,
      attack: row.attack,
      health: row.health,
      ikzCost: row.ikzCost,
      quantity: row.quantity,
    };
  });
}

async function getDeckBase(
  deckId: string,
  database: Database,
  userId?: string
): Promise<DeckBaseRecord | null> {
  const conditions = [eq(Decks.id, deckId), ne(Decks.status, DeckStatus.DELETED)];

  if (userId != null) {
    conditions.push(eq(Decks.userId, userId));
  }

  const results = await database
    .select({
      id: Decks.id,
      name: Decks.name,
      isSystemDeck: Decks.isSystemDeck,
      status: Decks.status,
    })
    .from(Decks)
    .where(and(...conditions))
    .limit(1);

  return results[0] ?? null;
}

async function getDeckCardRows(deckId: string, database: Database): Promise<DeckCardRow[]> {
  return database
    .select({
      cardId: Cards.id,
      cardCode: Cards.cardCode,
      name: Cards.name,
      imageKey: Cards.imageKey,
      cardType: Cards.cardType,
      element: Cards.element,
      attack: Cards.attack,
      health: Cards.health,
      ikzCost: Cards.ikzCost,
      quantity: DeckCardJunctions.quantity,
    })
    .from(DeckCardJunctions)
    .innerJoin(Cards, eq(DeckCardJunctions.cardId, Cards.id))
    .where(eq(DeckCardJunctions.deckId, deckId));
}

async function getAutoIkzCardId(database: Database): Promise<string> {
  const ikzCards = await database
    .select({
      id: Cards.id,
      cardCode: Cards.cardCode,
      rarity: Cards.rarity,
    })
    .from(Cards)
    .where(and(eq(Cards.cardCode, AUTO_IKZ_CARD_CODE), isNull(Cards.specialRarity)));

  const ikzCard = selectLeastRareCards(ikzCards).get(AUTO_IKZ_CARD_CODE);

  if (ikzCard == null) {
    throw new Error(`Card with code "${AUTO_IKZ_CARD_CODE}" not found in database`);
  }

  return ikzCard.id;
}

async function resolveDeckInput(
  input: UpsertDeckInput,
  database: Database
): Promise<ResolvedDeckInput> {
  const cardEntries = Object.entries(input.cardCounts);

  if (cardEntries.length === 0) {
    throw new ValidationError("Deck must include a gate, a leader, and 50 main deck cards");
  }

  for (const [cardId, quantity] of cardEntries) {
    if (!cardId || !Number.isInteger(quantity) || quantity <= 0) {
      throw new ValidationError("Deck contains an invalid card quantity");
    }
  }

  const selectedCards = await database
    .select({
      id: Cards.id,
      name: Cards.name,
      cardType: Cards.cardType,
      element: Cards.element,
    })
    .from(Cards)
    .where(
      and(
        inArray(
          Cards.id,
          cardEntries.map(([cardId]) => cardId)
        ),
        isNull(Cards.specialRarity)
      )
    );

  if (selectedCards.length !== cardEntries.length) {
    throw new ValidationError("Deck contains cards that cannot be selected");
  }

  let gateCard: SelectedDeckCard | null = null;
  let leaderCard: SelectedDeckCard | null = null;
  const mainDeckEntries: Array<{ cardId: string; quantity: number }> = [];
  let mainDeckCardCount = 0;

  for (const card of selectedCards) {
    const quantity = input.cardCounts[card.id];

    if (quantity == null) {
      throw new ValidationError("Deck contains an invalid card selection");
    }

    switch (card.cardType) {
      case CardType.GATE:
        if (quantity !== 1 || gateCard != null) {
          throw new ValidationError("Deck must include exactly one gate");
        }
        gateCard = card;
        break;

      case CardType.LEADER:
        if (quantity !== 1 || leaderCard != null) {
          throw new ValidationError("Deck must include exactly one leader");
        }
        leaderCard = card;
        break;

      case CardType.ENTITY:
      case CardType.SPELL:
      case CardType.WEAPON:
        mainDeckEntries.push({ cardId: card.id, quantity });
        mainDeckCardCount += quantity;
        break;

      case CardType.IKZ:
      case CardType.EXTRA_IKZ:
        throw new ValidationError("IKZ cards are added automatically");

      default:
        throw new ValidationError(`Unsupported card type in deck: ${card.name}`);
    }
  }

  if (gateCard == null) {
    throw new ValidationError("Deck must include exactly one gate");
  }

  if (leaderCard == null) {
    throw new ValidationError("Deck must include exactly one leader");
  }

  if (mainDeckCardCount !== MAIN_DECK_CARD_COUNT) {
    throw new ValidationError(`Deck must include exactly ${MAIN_DECK_CARD_COUNT} main deck cards`);
  }

  if (!isAllowedDeckElement(leaderCard.element, gateCard.element)) {
    throw new ValidationError("Leader must be NORMAL or match the selected gate element");
  }

  for (const entry of mainDeckEntries) {
    const card = selectedCards.find((selectedCard) => selectedCard.id === entry.cardId);

    if (card == null) {
      throw new ValidationError("Deck contains an invalid card selection");
    }

    if (!isAllowedDeckElement(card.element, gateCard.element)) {
      throw new ValidationError(
        "Main deck cards must be NORMAL or match the selected gate element"
      );
    }
  }

  const ikzCardId = await getAutoIkzCardId(database);

  return {
    gateCardId: gateCard.id,
    leaderCardId: leaderCard.id,
    mainDeckEntries,
    ikzCardId,
  };
}

function buildDeckJunctionEntries(
  deckId: string,
  resolvedDeckInput: ResolvedDeckInput
): Array<{ deckId: string; cardId: string; quantity: number }> {
  return [
    {
      deckId,
      cardId: resolvedDeckInput.gateCardId,
      quantity: 1,
    },
    {
      deckId,
      cardId: resolvedDeckInput.leaderCardId,
      quantity: 1,
    },
    ...resolvedDeckInput.mainDeckEntries.map((entry) => ({
      deckId,
      cardId: entry.cardId,
      quantity: entry.quantity,
    })),
    {
      deckId,
      cardId: resolvedDeckInput.ikzCardId,
      quantity: AUTO_IKZ_CARD_QUANTITY,
    },
  ];
}

async function requireOwnedDeck(
  deckId: string,
  userId: string,
  database: Database
): Promise<DeckBaseRecord> {
  const deck = await getDeckBase(deckId, database, userId);

  if (deck == null) {
    throw new DeckNotFoundError();
  }

  return deck;
}

/**
 * Create starter decks for a newly created user.
 */
export async function addStarterDecks(userId: string, database: Database = db): Promise<void> {
  const allCardCodes = new Set<string>();

  for (const deck of starterDecks) {
    for (const card of deck.cards) {
      allCardCodes.add(card.cardCode);
    }
  }

  const cards = await database
    .select({
      id: Cards.id,
      cardCode: Cards.cardCode,
      rarity: Cards.rarity,
    })
    .from(Cards)
    .where(and(inArray(Cards.cardCode, Array.from(allCardCodes)), isNull(Cards.specialRarity)));

  const cardCodeToCard = selectLeastRareCards<StarterCardRecord>(cards);

  for (const cardCode of allCardCodes) {
    if (!cardCodeToCard.has(cardCode)) {
      throw new Error(
        `Card with code "${cardCode}" not found in database (without specialRarity). Ensure cards are seeded before creating users.`
      );
    }
  }

  for (const deckConfig of starterDecks) {
    const insertedDecks = await database
      .insert(Decks)
      .values({
        name: deckConfig.name,
        userId,
        status: DeckStatus.COMPLETE,
        isSystemDeck: true,
      })
      .returning({ id: Decks.id });

    const deck = insertedDecks[0];
    if (deck == null) {
      throw new Error(`Failed to create starter deck "${deckConfig.name}"`);
    }

    const junctionEntries: Array<{ deckId: string; cardId: string; quantity: number }> = [];

    for (const cardInfo of deckConfig.cards) {
      const card = cardCodeToCard.get(cardInfo.cardCode);

      if (card == null) {
        throw new Error(`Card with code "${cardInfo.cardCode}" not found in database`);
      }

      junctionEntries.push({
        deckId: deck.id,
        cardId: card.id,
        quantity: cardInfo.quantity,
      });
    }

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
      cardCount: sql<number>`COALESCE(SUM(${DeckCardJunctions.quantity}), 0)`.as("card_count"),
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

/**
 * Returns all cards that can be used by the deck builder.
 * This excludes special-rarity variants and auto-added IKZ cards.
 */
export async function getDeckBuilderCards(database: Database = db): Promise<DeckBuilderCard[]> {
  const cards = await database
    .select({
      id: Cards.id,
      cardCode: Cards.cardCode,
      name: Cards.name,
      imageKey: Cards.imageKey,
      cardType: Cards.cardType,
      element: Cards.element,
      attack: Cards.attack,
      health: Cards.health,
      ikzCost: Cards.ikzCost,
      rarity: Cards.rarity,
    })
    .from(Cards)
    .where(
      and(
        isNull(Cards.specialRarity),
        ne(Cards.cardType, CardType.IKZ),
        ne(Cards.cardType, CardType.EXTRA_IKZ)
      )
    );

  const selectedCards = Array.from(selectLeastRareCards(cards).values());

  selectedCards.sort((leftCard, rightCard) => {
    const nameCompare = leftCard.name.localeCompare(rightCard.name);

    if (nameCompare !== 0) {
      return nameCompare;
    }

    return leftCard.cardCode.localeCompare(rightCard.cardCode);
  });

  return selectedCards.map((card) => ({
    id: card.id,
    cardCode: card.cardCode,
    name: card.name,
    imageKey: card.imageKey,
    cardType: card.cardType,
    element: card.element,
    attack: card.attack,
    health: card.health,
    ikzCost: card.ikzCost,
  }));
}

/**
 * Returns a deck and its card counts for the authenticated owner.
 * Used by the edit flow to seed the deck builder form.
 */
export async function getEditableDeckForUser(
  deckId: string,
  userId: string,
  database: Database = db
): Promise<EditableDeck | null> {
  const deck = await getDeckBase(deckId, database, userId);

  if (deck == null) {
    return null;
  }

  const deckCards = await database
    .select({
      cardId: DeckCardJunctions.cardId,
      quantity: DeckCardJunctions.quantity,
    })
    .from(DeckCardJunctions)
    .where(eq(DeckCardJunctions.deckId, deckId));

  return {
    id: deck.id,
    name: deck.name,
    isSystemDeck: deck.isSystemDeck,
    cards: deckCards,
  };
}

/**
 * Get a deck with all its cards and card details.
 * Returns null if deck not found or deleted.
 */
export async function getDeckWithCards(
  deckId: string,
  database: Database = db
): Promise<DeckWithCards | null> {
  const deck = await getDeckBase(deckId, database);

  if (deck == null) {
    return null;
  }

  const cardRows = await getDeckCardRows(deckId, database);

  return {
    id: deck.id,
    name: deck.name,
    isSystemDeck: deck.isSystemDeck,
    cards: mapDeckCardRowsToDetails(cardRows),
  };
}

export async function copyDeck(
  deckId: string,
  userId: string,
  newName: string,
  database: Database = db
): Promise<{ id: string }> {
  const sourceDeck = await requireOwnedDeck(deckId, userId, database);
  const normalizedName = normalizeDeckName(newName);

  return database.transaction(async (tx) => {
    const insertedDecks = await tx
      .insert(Decks)
      .values({
        name: normalizedName,
        userId,
        status: sourceDeck.status,
        isSystemDeck: false,
      })
      .returning({ id: Decks.id });

    const copiedDeck = insertedDecks[0];
    if (copiedDeck == null) {
      throw new Error("Failed to copy deck");
    }

    const sourceCards = await tx
      .select({
        cardId: DeckCardJunctions.cardId,
        quantity: DeckCardJunctions.quantity,
      })
      .from(DeckCardJunctions)
      .where(eq(DeckCardJunctions.deckId, deckId));

    if (sourceCards.length > 0) {
      await tx.insert(DeckCardJunctions).values(
        sourceCards.map((card) => ({
          deckId: copiedDeck.id,
          cardId: card.cardId,
          quantity: card.quantity,
        }))
      );
    }

    return { id: copiedDeck.id };
  });
}

export async function softDeleteDeck(
  deckId: string,
  userId: string,
  database: Database = db
): Promise<{ id: string; status: DeckStatus.DELETED }> {
  const deck = await requireOwnedDeck(deckId, userId, database);

  if (deck.isSystemDeck) {
    throw new ForbiddenError("System decks cannot be deleted");
  }

  const updatedDecks = await database
    .update(Decks)
    .set({ status: DeckStatus.DELETED })
    .where(eq(Decks.id, deckId))
    .returning({
      id: Decks.id,
      status: Decks.status,
    });

  const updatedDeck = updatedDecks[0];
  if (updatedDeck == null || updatedDeck.status !== DeckStatus.DELETED) {
    throw new Error("Failed to delete deck");
  }

  return {
    id: updatedDeck.id,
    status: DeckStatus.DELETED,
  };
}

export async function createDeck(
  userId: string,
  input: UpsertDeckInput,
  database: Database = db
): Promise<{ id: string }> {
  const normalizedName = normalizeDeckName(input.name);

  return database.transaction(async (tx) => {
    const resolvedDeckInput = await resolveDeckInput(
      {
        name: normalizedName,
        cardCounts: input.cardCounts,
      },
      tx
    );

    const insertedDecks = await tx
      .insert(Decks)
      .values({
        name: normalizedName,
        userId,
        status: DeckStatus.COMPLETE,
        isSystemDeck: false,
      })
      .returning({ id: Decks.id });

    const deck = insertedDecks[0];
    if (deck == null) {
      throw new Error("Failed to create deck");
    }

    await tx.insert(DeckCardJunctions).values(buildDeckJunctionEntries(deck.id, resolvedDeckInput));

    return { id: deck.id };
  });
}

export async function updateDeck(
  deckId: string,
  userId: string,
  input: UpsertDeckInput,
  database: Database = db
): Promise<{ id: string }> {
  const deck = await requireOwnedDeck(deckId, userId, database);

  if (deck.isSystemDeck) {
    throw new ForbiddenError("System decks cannot be edited");
  }

  const normalizedName = normalizeDeckName(input.name);

  return database.transaction(async (tx) => {
    const resolvedDeckInput = await resolveDeckInput(
      {
        name: normalizedName,
        cardCounts: input.cardCounts,
      },
      tx
    );

    await tx
      .update(Decks)
      .set({
        name: normalizedName,
        status: DeckStatus.COMPLETE,
      })
      .where(eq(Decks.id, deckId));

    await tx.delete(DeckCardJunctions).where(eq(DeckCardJunctions.deckId, deckId));

    await tx.insert(DeckCardJunctions).values(buildDeckJunctionEntries(deckId, resolvedDeckInput));

    return { id: deckId };
  });
}
