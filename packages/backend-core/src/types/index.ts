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

// Card rarity (matches C engine)
export enum CardRarity {
  L = "L",
  G = "G",
  C = "C",
  UC = "UC",
  R = "R",
  SR = "SR",
  IKZ = "IKZ",
}

// Card element (matches C engine)
export enum CardElement {
  NORMAL = "NORMAL",
  LIGHTNING = "LIGHTNING",
  WATER = "WATER",
  EARTH = "EARTH",
  FIRE = "FIRE",
}

// Card type (matches C engine)
export enum CardType {
  LEADER = "LEADER",
  GATE = "GATE",
  ENTITY = "ENTITY",
  WEAPON = "WEAPON",
  SPELL = "SPELL",
  IKZ = "IKZ",
  EXTRA_IKZ = "EXTRA_IKZ",
}

// WebSocket message types - Client to Server
export interface ClientMessage {
  type:
    | "SELECT_DECK"
    | "READY"
    | "GAME_ACTION"
    | "FORFEIT"
    | "PING";
}

export interface SelectDeckMessage extends ClientMessage {
  type: "SELECT_DECK";
  deckId: string;
}

export interface ReadyMessage extends ClientMessage {
  type: "READY";
  ready: boolean;
}

export interface GameActionMessage extends ClientMessage {
  type: "GAME_ACTION";
  action: [number, number, number, number];
}

export interface ForfeitMessage extends ClientMessage {
  type: "FORFEIT";
}

export interface PingMessage extends ClientMessage {
  type: "PING";
}

// WebSocket message types - Server to Client
export interface ServerMessage {
  type:
    | "CONNECTION_ACK"
    | "ROOM_STATE"
    | "GAME_SNAPSHOT"
    | "GAME_LOG_BATCH"
    | "GAME_OVER"
    | "ERROR"
    | "PONG";
}

export interface ConnectionAckMessage extends ServerMessage {
  type: "CONNECTION_ACK";
  playerId: string;
  playerSlot: 0 | 1;
}

export interface ErrorMessage extends ServerMessage {
  type: "ERROR";
  code: string;
  message: string;
}

export interface PongMessage extends ServerMessage {
  type: "PONG";
}
