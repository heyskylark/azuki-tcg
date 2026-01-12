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

export interface PlayerInfo {
  id: string;
  username: string;
  deckSelected: boolean;
  ready: boolean;
  connected: boolean;
}

export interface RoomStateMessage extends ServerMessage {
  type: "ROOM_STATE";
  status: string;
  players: [PlayerInfo | null, PlayerInfo | null];
  deckSelectionDeadline: string | null;
  readyCountdownEnd: string | null;
}
