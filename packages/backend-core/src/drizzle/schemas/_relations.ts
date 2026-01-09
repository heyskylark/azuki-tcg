import { relations } from "drizzle-orm";
import { users } from "@/drizzle/schemas/users";
import { cards } from "@/drizzle/schemas/cards";
import { decks } from "@/drizzle/schemas/decks";
import { deckCardJunctions } from "@/drizzle/schemas/deck_card_junctions";
import { rooms } from "@/drizzle/schemas/rooms";
import { matchResults } from "@/drizzle/schemas/match_results";
import { gameLogs } from "@/drizzle/schemas/game_logs";

// User relations
export const usersRelations = relations(users, ({ many }) => ({
  decks: many(decks),
  roomsAsPlayer0: many(rooms, { relationName: "player0" }),
  roomsAsPlayer1: many(rooms, { relationName: "player1" }),
  matchResultsAsPlayer0: many(matchResults, { relationName: "player0" }),
  matchResultsAsPlayer1: many(matchResults, { relationName: "player1" }),
  matchResultsAsWinner: many(matchResults, { relationName: "winner" }),
}));

// Card relations
export const cardsRelations = relations(cards, ({ many }) => ({
  decksAsLeader: many(decks, { relationName: "leader" }),
  decksAsGate: many(decks, { relationName: "gate" }),
  deckCardJunctions: many(deckCardJunctions),
}));

// Deck relations
export const decksRelations = relations(decks, ({ one, many }) => ({
  user: one(users, {
    fields: [decks.userId],
    references: [users.id],
  }),
  leaderCard: one(cards, {
    fields: [decks.leaderCardId],
    references: [cards.id],
    relationName: "leader",
  }),
  gateCard: one(cards, {
    fields: [decks.gateCardId],
    references: [cards.id],
    relationName: "gate",
  }),
  deckCardJunctions: many(deckCardJunctions),
  roomsAsPlayer0Deck: many(rooms, { relationName: "player0Deck" }),
  roomsAsPlayer1Deck: many(rooms, { relationName: "player1Deck" }),
}));

// DeckCardJunction relations
export const deckCardJunctionsRelations = relations(
  deckCardJunctions,
  ({ one }) => ({
    deck: one(decks, {
      fields: [deckCardJunctions.deckId],
      references: [decks.id],
    }),
    card: one(cards, {
      fields: [deckCardJunctions.cardId],
      references: [cards.id],
    }),
  })
);

// Room relations
export const roomsRelations = relations(rooms, ({ one, many }) => ({
  player0: one(users, {
    fields: [rooms.player0Id],
    references: [users.id],
    relationName: "player0",
  }),
  player1: one(users, {
    fields: [rooms.player1Id],
    references: [users.id],
    relationName: "player1",
  }),
  player0Deck: one(decks, {
    fields: [rooms.player0DeckId],
    references: [decks.id],
    relationName: "player0Deck",
  }),
  player1Deck: one(decks, {
    fields: [rooms.player1DeckId],
    references: [decks.id],
    relationName: "player1Deck",
  }),
  matchResult: one(matchResults),
  gameLogs: many(gameLogs),
}));

// MatchResult relations
export const matchResultsRelations = relations(matchResults, ({ one }) => ({
  room: one(rooms, {
    fields: [matchResults.roomId],
    references: [rooms.id],
  }),
  player0: one(users, {
    fields: [matchResults.player0Id],
    references: [users.id],
    relationName: "player0",
  }),
  player1: one(users, {
    fields: [matchResults.player1Id],
    references: [users.id],
    relationName: "player1",
  }),
  winner: one(users, {
    fields: [matchResults.winnerId],
    references: [users.id],
    relationName: "winner",
  }),
}));

// GameLog relations
export const gameLogsRelations = relations(gameLogs, ({ one }) => ({
  room: one(rooms, {
    fields: [gameLogs.roomId],
    references: [rooms.id],
  }),
}));
