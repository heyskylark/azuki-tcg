import type { WebSocket } from "uWebSockets.js";
import type { RoomStatus } from "@tcg/backend-core/types";
import type { UserData } from "@/constants";

export interface PlayerConnection {
  ws: WebSocket<UserData> | null;
  userId: string;
  username: string;
  playerSlot: 0 | 1;
  connected: boolean;
  disconnectedAt: Date | null;
}

export interface RoomChannelState {
  roomId: string;
  status: RoomStatus;
  players: [PlayerConnection | null, PlayerConnection | null];
  player0DeckId: string | null;
  player1DeckId: string | null;
  player0Ready: boolean;
  player1Ready: boolean;
  deckSelectionDeadline: Date | null;
  readyCountdownStartedAt: Date | null;
}

export interface ConnectionInfo {
  userId: string;
  username: string;
  roomId: string;
  playerSlot: 0 | 1;
}
