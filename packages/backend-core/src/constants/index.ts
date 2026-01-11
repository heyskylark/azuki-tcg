// Shared constants between web and websocket apps

// WebSocket configuration
export const WS_RECONNECT_TIMEOUT_MS = 60_000; // 1 minute
export const WS_DECK_SELECTION_TIMEOUT_MS = 120_000; // 2 minutes

// Rate limiting
export const MAX_ACTIONS_PER_SECOND = 10;
export const MAX_WS_MESSAGES_PER_SECOND = 30;
export const MAX_ROOMS_PER_USER_PER_MINUTE = 5;

// Game configuration
export const DECK_SIZE = 40;
export const MAX_CARD_COPIES = 4;
export const GARDEN_SLOTS = 5;
export const ALLEY_SLOTS = 5;

// Re-export auth constants
export * from "@/constants/auth";
