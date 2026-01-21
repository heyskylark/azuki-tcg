import { TestClient } from "./TestClient";
import { generateUniqueTestUser, type TestUser } from "./fixtures";

/**
 * Helper to create and register a test user.
 * Returns the client with authenticated state.
 */
export async function createTestUser(
  client: TestClient,
  overrides?: Partial<TestUser>
): Promise<{ user: TestUser; response: unknown }> {
  const user = { ...generateUniqueTestUser(), ...overrides };
  const response = await client.register(user.username, user.email, user.password);
  return { user, response };
}

/**
 * Helper to create a room with an authenticated user.
 */
export async function createTestRoom(
  client: TestClient,
  password?: string
): Promise<{ roomId: string; response: unknown }> {
  const response = await client.createRoom(password);
  return { roomId: response.room.id, response };
}

/**
 * Helper to set up two players in a room ready for deck selection.
 * Returns both clients and room info.
 */
export async function setupTwoPlayerRoom(): Promise<{
  client1: TestClient;
  client2: TestClient;
  roomId: string;
  player1JoinToken: string;
  player2JoinToken: string;
}> {
  const client1 = new TestClient();
  const client2 = new TestClient();

  // Register both users
  await createTestUser(client1);
  await createTestUser(client2);

  // Player 1 creates room
  const { roomId } = await createTestRoom(client1);

  // Both players join the room
  const join1 = await client1.joinRoom(roomId);
  const join2 = await client2.joinRoom(roomId);

  return {
    client1,
    client2,
    roomId,
    player1JoinToken: join1.joinToken,
    player2JoinToken: join2.joinToken,
  };
}

/**
 * Helper to set up a game that's ready to start (both players selected decks and ready).
 */
export async function setupGameReadyToStart(): Promise<{
  client1: TestClient;
  client2: TestClient;
  roomId: string;
}> {
  const { client1, client2, roomId, player1JoinToken, player2JoinToken } =
    await setupTwoPlayerRoom();

  // Connect WebSockets
  await client1.connectWebSocket(player1JoinToken);
  await client2.connectWebSocket(player2JoinToken);

  // Wait for connection acks
  await client1.waitForMessage("CONNECTION_ACK");
  await client2.waitForMessage("CONNECTION_ACK");

  // Player 1 starts the game (transitions to DECK_SELECTION)
  await client1.sendMessage("START_GAME", {});

  // Wait for ROOM_STATE showing DECK_SELECTION
  await client1.waitForMessage("ROOM_STATE");
  await client2.waitForMessage("ROOM_STATE");

  // Get starter decks and select them
  const decks1 = await client1.getStarterDecks();
  const decks2 = await client2.getStarterDecks();

  // Both select decks
  if (decks1.length > 0) {
    await client1.sendMessage("SELECT_DECK", { deckId: decks1[0].id });
  }
  if (decks2.length > 0) {
    await client2.sendMessage("SELECT_DECK", { deckId: decks2[0].id });
  }

  // Wait for deck selection updates
  await client1.waitForMessage("ROOM_STATE");
  await client2.waitForMessage("ROOM_STATE");

  // Both ready up
  await client1.sendMessage("READY", { ready: true });
  await client2.sendMessage("READY", { ready: true });

  return { client1, client2, roomId };
}

/**
 * Extract a valid action from an action mask.
 * Returns the first valid action index for each head.
 */
export function extractValidActionFromMask(
  mask: boolean[][] | { primaryActionMask: boolean[]; legalPrimary: number[]; legalSub1: number[]; legalSub2: number[]; legalSub3: number[] }
): [number, number, number, number] {
  // Handle SnapshotActionMask format
  if ("primaryActionMask" in mask) {
    return [
      mask.legalPrimary[0] ?? 0,
      mask.legalSub1[0] ?? 0,
      mask.legalSub2[0] ?? 0,
      mask.legalSub3[0] ?? 0,
    ];
  }

  // Handle simple boolean array format
  return mask.map((head) => {
    const validIndex = head.findIndex((valid) => valid);
    return validIndex >= 0 ? validIndex : 0;
  }) as [number, number, number, number];
}

/**
 * Wait for a specified amount of time.
 */
export function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Retry a function until it succeeds or timeout.
 */
export async function retry<T>(
  fn: () => Promise<T>,
  options: { maxAttempts?: number; delayMs?: number; timeout?: number } = {}
): Promise<T> {
  const { maxAttempts = 5, delayMs = 100, timeout = 10000 } = options;
  const startTime = Date.now();
  let lastError: Error | undefined;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (Date.now() - startTime > timeout) {
      throw new Error(`Retry timeout after ${timeout}ms: ${lastError?.message}`);
    }

    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      await delay(delayMs);
    }
  }

  throw new Error(`Max retry attempts (${maxAttempts}) reached: ${lastError?.message}`);
}
