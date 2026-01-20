// WebSocket message types - Client to Server
export interface ClientMessage {
  type:
    | "SELECT_DECK"
    | "READY"
    | "GAME_ACTION"
    | "FORFEIT"
    | "PING"
    | "LEAVE_ROOM"
    | "CLOSE_ROOM"
    | "START_GAME";
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

export interface LeaveRoomMessage extends ClientMessage {
  type: "LEAVE_ROOM";
}

export interface CloseRoomMessage extends ClientMessage {
  type: "CLOSE_ROOM";
}

export interface StartGameMessage extends ClientMessage {
  type: "START_GAME";
}

// WebSocket message types - Server to Client
export interface ServerMessage {
  type:
    | "CONNECTION_ACK"
    | "ROOM_STATE"
    | "ROOM_CLOSED"
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

export interface RoomClosedMessage extends ServerMessage {
  type: "ROOM_CLOSED";
  reason: string;
}

export interface PlayerInfo {
  id: string;
  username: string;
  deckSelected: boolean;
  deckId: string | null;
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

export interface GameOverMessage extends ServerMessage {
  type: "GAME_OVER";
  winnerId: string | null;
  winnerSlot: 0 | 1 | null;
  winType: string;
  reason: string;
}

// Game snapshot types for connect/reconnect
export interface SnapshotStateContext {
  phase: string;
  abilitySubphase: string;
  activePlayer: 0 | 1;
  turnNumber: number;
}

export interface SnapshotLeader {
  cardId: string | null;
  cardDefId: number;
  zoneIndex: number;
  curHp: number;
  curAtk: number;
  tapped: boolean;
  cooldown: boolean;
}

export interface SnapshotGate {
  cardId: string | null;
  cardDefId: number;
  zoneIndex: number;
  tapped: boolean;
  cooldown: boolean;
}

export interface SnapshotCard {
  cardId: string | null;
  cardDefId: number;
  zoneIndex: number;
  curAtk: number | null;
  curHp: number | null;
  tapped: boolean;
  cooldown: boolean;
  isFrozen: boolean;
  isShocked: boolean;
  isEffectImmune: boolean;
}

export interface SnapshotIkz {
  cardId: string | null;
  cardDefId: number;
  tapped: boolean;
  cooldown: boolean;
}

export interface SnapshotPlayerBoard {
  leader: SnapshotLeader;
  gate: SnapshotGate;
  garden: (SnapshotCard | null)[];
  alley: (SnapshotCard | null)[];
  ikzArea: SnapshotIkz[];
  handCount: number;
  deckCount: number;
  discardCount: number;
  ikzPileCount: number;
  hasIkzToken: boolean;
}

export interface SnapshotHandCard {
  cardId: string | null;
  cardDefId: number;
  type: string;
  ikzCost: number;
}

export interface SnapshotCardMetadata {
  cardCode: string;
  name: string;
  imageKey: string;
  cardType: string;
  attack: number | null;
  health: number | null;
  ikzCost: number | null;
}

export interface SnapshotActionMask {
  primaryActionMask: boolean[];
  legalActionCount: number;
  legalPrimary: number[];
  legalSub1: number[];
  legalSub2: number[];
  legalSub3: number[];
}

export interface GameSnapshotMessage extends ServerMessage {
  type: "GAME_SNAPSHOT";
  stateContext: SnapshotStateContext;
  players: [SnapshotPlayerBoard, SnapshotPlayerBoard];
  yourHand: SnapshotHandCard[];
  cardMetadata: Record<string, SnapshotCardMetadata>;
  combatStack: unknown[];
  actionMask: SnapshotActionMask | null;
}

export interface GameLogBatchMessage extends ServerMessage {
  type: "GAME_LOG_BATCH";
  batchNumber: number;
  logs: unknown[];
  stateContext: SnapshotStateContext;
  actionMask?: SnapshotActionMask | null; // Included for the active player
}
