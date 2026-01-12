import { env } from "@/env";

export const PORT = env.PORT;

export const TIMEOUT_SECONDS = 120;
export const MAX_PAYLOAD_LENGTH = 16 * 1024 * 1024; // 16MB
export const COMPRESSION = 0; // Disabled for now

export interface UserData {
  id: string;
  isAnonymous: boolean;
  roomId?: string;
  playerSlot?: 0 | 1;
}

// Timer constants
export const READY_COUNTDOWN_MS = 5000; // 5 seconds
export const DISCONNECT_GRACE_MS = 60000; // 60 seconds
export const DECK_SELECTION_TIMEOUT_MS = 120000; // 2 minutes
