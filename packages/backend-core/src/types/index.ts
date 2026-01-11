// User status
export enum UserStatus {
  ACTIVE = "ACTIVE",
  DELETED = "DELETED",
  BANNED = "BANNED",
}

// User type
export enum UserType {
  HUMAN = "HUMAN",
  AI = "AI",
}

// Deck status
export enum DeckStatus {
  COMPLETE = "COMPLETE",
  IN_PROGRESS = "IN_PROGRESS",
  DELETED = "DELETED",
}

// Room status
export enum RoomStatus {
  WAITING_FOR_PLAYERS = "WAITING_FOR_PLAYERS",
  DECK_SELECTION = "DECK_SELECTION",
  READY_CHECK = "READY_CHECK",
  STARTING = "STARTING",
  IN_MATCH = "IN_MATCH",
  COMPLETED = "COMPLETED",
  ABORTED = "ABORTED",
}

// Room type
export enum RoomType {
  PRIVATE = "PRIVATE",
  MATCH_MAKING = "MATCH_MAKING",
}

// Match result type
export enum WinType {
  WIN = "WIN",
  DRAW = "DRAW",
  ABANDON = "ABANDON",
  FORFEIT = "FORFEIT",
  TIMEOUT = "TIMEOUT",
}

// Re-export card types
export * from "@/types/cards";

// Re-export WebSocket types
export * from "@/types/ws";

// Re-export auth types
export * from "@/types/auth";
