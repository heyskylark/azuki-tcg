/**
 * Test fixtures for integration tests.
 * These provide pre-defined test data for users, decks, and game actions.
 */

export interface TestUser {
  username: string;
  email: string;
  password: string;
}

export interface TestDeck {
  name: string;
  leaderCardCode: string;
  gateCardCode: string;
  mainDeckCardCodes: string[];
}

// Test users with unique identifiers to avoid collisions
export const testUsers: Record<string, TestUser> = {
  player1: {
    username: "testPlayer1",
    email: "player1@test.com",
    password: "testpass123!",
  },
  player2: {
    username: "testPlayer2",
    email: "player2@test.com",
    password: "testpass123!",
  },
  player3: {
    username: "testPlayer3",
    email: "player3@test.com",
    password: "testpass123!",
  },
};

// Starter deck codes - these are seeded by migrations
export const starterDeckCodes = {
  STT01: "STT01", // Raizen starter
  STT02: "STT02", // Shao starter
};

// Helper to generate unique test user data
let userCounter = 0;
export function generateUniqueTestUser(): TestUser {
  userCounter++;
  const timestamp = Date.now();
  return {
    username: `testUser_${timestamp}_${userCounter}`,
    email: `test_${timestamp}_${userCounter}@test.com`,
    password: "testpass123!",
  };
}

// Test room password
export const testRoomPassword = "testRoomPass123!";

// Game action helpers
export const gameActions = {
  // NOOP action (do nothing)
  noop: [0, 0, 0, 0] as [number, number, number, number],

  // End turn action
  endTurn: [3, 0, 0, 0] as [number, number, number, number],

  // Helper to create a valid action from action mask
  fromMask(mask: boolean[][]): [number, number, number, number] {
    return mask.map((head) => {
      const validIndex = head.findIndex((valid) => valid);
      return validIndex >= 0 ? validIndex : 0;
    }) as [number, number, number, number];
  },
};

// WebSocket message types for testing
export const wsMessageTypes = {
  // Client to server
  SELECT_DECK: "SELECT_DECK",
  READY: "READY",
  GAME_ACTION: "GAME_ACTION",
  FORFEIT: "FORFEIT",
  PING: "PING",
  LEAVE_ROOM: "LEAVE_ROOM",
  CLOSE_ROOM: "CLOSE_ROOM",
  START_GAME: "START_GAME",

  // Server to client
  CONNECTION_ACK: "CONNECTION_ACK",
  ROOM_STATE: "ROOM_STATE",
  ROOM_CLOSED: "ROOM_CLOSED",
  GAME_SNAPSHOT: "GAME_SNAPSHOT",
  GAME_LOG_BATCH: "GAME_LOG_BATCH",
  GAME_OVER: "GAME_OVER",
  ERROR: "ERROR",
  PONG: "PONG",
};

// Room statuses matching backend-core types
export const roomStatuses = {
  WAITING_FOR_PLAYERS: "WAITING_FOR_PLAYERS",
  DECK_SELECTION: "DECK_SELECTION",
  READY_CHECK: "READY_CHECK",
  STARTING: "STARTING",
  IN_MATCH: "IN_MATCH",
  COMPLETED: "COMPLETED",
  ABORTED: "ABORTED",
  CLOSED: "CLOSED",
};
