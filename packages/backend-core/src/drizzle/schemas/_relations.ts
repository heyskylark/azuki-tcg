import { relations } from "drizzle-orm";
import { Emails, Users } from "@core/drizzle/schemas/users";
import { Cards } from "@core/drizzle/schemas/cards";
import { Decks } from "@core/drizzle/schemas/decks";
import { DeckCardJunctions } from "@core/drizzle/schemas/deck_card_junctions";
import { Rooms } from "@core/drizzle/schemas/rooms";
import { MatchResults } from "@core/drizzle/schemas/match_results";
import { GameLogs } from "@core/drizzle/schemas/game_logs";
import { JwtTokens } from "@core/drizzle/schemas/jwt_tokens";

// User relations
export const UsersRelations = relations(Users, ({ many }) => ({
  emails: many(Emails),
  decks: many(Decks),
  jwtTokens: many(JwtTokens),
  roomsAsPlayer0: many(Rooms, { relationName: "player0" }),
  roomsAsPlayer1: many(Rooms, { relationName: "player1" }),
  matchResultsAsPlayer0: many(MatchResults, { relationName: "player0" }),
  matchResultsAsPlayer1: many(MatchResults, { relationName: "player1" }),
  matchResultsAsWinner: many(MatchResults, { relationName: "winner" }),
}));

// Card relations
export const cardsRelations = relations(Cards, ({ many }) => ({
  deckCardJunctions: many(DeckCardJunctions),
}));

// Deck relations
export const decksRelations = relations(Decks, ({ one, many }) => ({
  user: one(Users, {
    fields: [Decks.userId],
    references: [Users.id],
  }),
  deckCardJunctions: many(DeckCardJunctions),
  roomsAsPlayer0Deck: many(Rooms, { relationName: "player0Deck" }),
  roomsAsPlayer1Deck: many(Rooms, { relationName: "player1Deck" }),
}));

// DeckCardJunction relations
export const deckCardJunctionsRelations = relations(DeckCardJunctions, ({ one }) => ({
  deck: one(Decks, {
    fields: [DeckCardJunctions.deckId],
    references: [Decks.id],
  }),
  card: one(Cards, {
    fields: [DeckCardJunctions.cardId],
    references: [Cards.id],
  }),
}));

// Room relations
export const roomsRelations = relations(Rooms, ({ one, many }) => ({
  player0: one(Users, {
    fields: [Rooms.player0Id],
    references: [Users.id],
    relationName: "player0",
  }),
  player1: one(Users, {
    fields: [Rooms.player1Id],
    references: [Users.id],
    relationName: "player1",
  }),
  player0Deck: one(Decks, {
    fields: [Rooms.player0DeckId],
    references: [Decks.id],
    relationName: "player0Deck",
  }),
  player1Deck: one(Decks, {
    fields: [Rooms.player1DeckId],
    references: [Decks.id],
    relationName: "player1Deck",
  }),
  matchResult: one(MatchResults),
  gameLogs: many(GameLogs),
}));

// MatchResult relations
export const matchResultsRelations = relations(MatchResults, ({ one }) => ({
  room: one(Rooms, {
    fields: [MatchResults.roomId],
    references: [Rooms.id],
  }),
  player0: one(Users, {
    fields: [MatchResults.player0Id],
    references: [Users.id],
    relationName: "player0",
  }),
  player1: one(Users, {
    fields: [MatchResults.player1Id],
    references: [Users.id],
    relationName: "player1",
  }),
  winner: one(Users, {
    fields: [MatchResults.winnerId],
    references: [Users.id],
    relationName: "winner",
  }),
}));

// GameLog relations
export const gameLogsRelations = relations(GameLogs, ({ one }) => ({
  room: one(Rooms, {
    fields: [GameLogs.roomId],
    references: [Rooms.id],
  }),
}));

// JwtToken relations
export const jwtTokensRelations = relations(JwtTokens, ({ one }) => ({
  user: one(Users, {
    fields: [JwtTokens.userId],
    references: [Users.id],
  }),
}));
