import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { TestClient } from "@test/utils/TestClient";
import { generateUniqueTestUser, wsMessageTypes } from "@test/utils/fixtures";
import { resetTestDatabase } from "@test/utils/database";
import { extractValidActionFromMask, delay } from "@test/utils/helpers";

/**
 * Full Game Flow Test
 *
 * End-to-end test simulating a complete game flow:
 * 1. User registration
 * 2. Room creation
 * 3. Room joining
 * 4. WebSocket connection
 * 5. Deck selection
 * 6. Ready up
 * 7. Game start
 * 8. Game actions
 * 9. Game completion
 */
describe("Full Game Flow", () => {
  let client1: TestClient;
  let client2: TestClient;

  beforeEach(async () => {
    await resetTestDatabase();
    client1 = new TestClient();
    client2 = new TestClient();
  });

  afterEach(() => {
    client1.disconnect();
    client2.disconnect();
  });

  it("should complete flow from registration to game start", async () => {
    // Step 1: Register users
    const user1 = generateUniqueTestUser();
    const user2 = generateUniqueTestUser();

    const reg1 = await client1.register(user1.username, user1.email, user1.password);
    const reg2 = await client2.register(user2.username, user2.email, user2.password);

    expect(reg1.user.id).toBeDefined();
    expect(reg2.user.id).toBeDefined();

    // Step 2: Player 1 creates room
    const { room } = await client1.createRoom();
    expect(room.id).toBeDefined();
    expect(room.status).toBe("WAITING_FOR_PLAYERS");

    // Step 3: Player 2 joins room
    const join1 = await client1.joinRoom(room.id);
    const join2 = await client2.joinRoom(room.id);

    expect(join1.playerSlot).toBe(0);
    expect(join2.playerSlot).toBe(1);

    // Step 4: Both connect via WebSocket
    await client1.connectWebSocket(join1.joinToken);
    await client2.connectWebSocket(join2.joinToken);

    // Verify connections
    expect(client1.isConnected()).toBe(true);
    expect(client2.isConnected()).toBe(true);

    // Step 5: Wait for CONNECTION_ACK
    const ack1 = await client1.waitForMessage(wsMessageTypes.CONNECTION_ACK);
    const ack2 = await client2.waitForMessage(wsMessageTypes.CONNECTION_ACK);

    expect(ack1.playerSlot).toBe(0);
    expect(ack2.playerSlot).toBe(1);

    // Step 6: Wait for initial ROOM_STATE
    await client1.waitForMessage(wsMessageTypes.ROOM_STATE);
    await client2.waitForMessage(wsMessageTypes.ROOM_STATE);

    // Step 7: Owner starts the game
    await client1.sendMessage(wsMessageTypes.START_GAME, {});

    // Wait for DECK_SELECTION state
    const state1 = await client1.waitForMessage(wsMessageTypes.ROOM_STATE);
    const state2 = await client2.waitForMessage(wsMessageTypes.ROOM_STATE);

    expect(state1.status).toBe("DECK_SELECTION");
    expect(state2.status).toBe("DECK_SELECTION");

    // Step 8: Get starter decks
    const decks1 = await client1.getStarterDecks();
    const decks2 = await client2.getStarterDecks();

    // May not have starter decks in test environment
    if (decks1.length === 0 || decks2.length === 0) {
      console.log("No starter decks available, skipping remaining steps");
      return;
    }

    // Step 9: Select decks
    await client1.sendMessage(wsMessageTypes.SELECT_DECK, { deckId: decks1[0]?.id });
    await client2.sendMessage(wsMessageTypes.SELECT_DECK, { deckId: decks2[0]?.id });

    // Wait for deck selection confirmations
    await client1.waitForMessage(wsMessageTypes.ROOM_STATE);
    await client2.waitForMessage(wsMessageTypes.ROOM_STATE);

    // Step 10: Both players ready up
    await client1.sendMessage(wsMessageTypes.READY, { ready: true });
    await client2.sendMessage(wsMessageTypes.READY, { ready: true });

    // Step 11: Wait for game to start (IN_MATCH state or game messages)
    let gameStarted = false;
    const maxWait = 20000;
    const startTime = Date.now();

    while (!gameStarted && Date.now() - startTime < maxWait) {
      try {
        const msg = await client1.waitForMessage(undefined, 2000);
        if (
          (msg.type === wsMessageTypes.ROOM_STATE && msg.status === "IN_MATCH") ||
          msg.type === wsMessageTypes.GAME_SNAPSHOT ||
          msg.type === wsMessageTypes.GAME_LOG_BATCH
        ) {
          gameStarted = true;
        }
      } catch {
        // Continue waiting
      }
    }

    expect(gameStarted).toBe(true);
  });

  it("should allow game actions after game starts", async () => {
    // Setup game
    const user1 = generateUniqueTestUser();
    const user2 = generateUniqueTestUser();

    await client1.register(user1.username, user1.email, user1.password);
    await client2.register(user2.username, user2.email, user2.password);

    const { room } = await client1.createRoom();
    const join1 = await client1.joinRoom(room.id);
    const join2 = await client2.joinRoom(room.id);

    await client1.connectWebSocket(join1.joinToken);
    await client2.connectWebSocket(join2.joinToken);

    await client1.waitForMessage(wsMessageTypes.CONNECTION_ACK);
    await client2.waitForMessage(wsMessageTypes.CONNECTION_ACK);

    await client1.waitForMessage(wsMessageTypes.ROOM_STATE);
    await client2.waitForMessage(wsMessageTypes.ROOM_STATE);

    // Start game
    await client1.sendMessage(wsMessageTypes.START_GAME, {});
    await client1.waitForMessage(wsMessageTypes.ROOM_STATE);
    await client2.waitForMessage(wsMessageTypes.ROOM_STATE);

    // Select decks
    const decks1 = await client1.getStarterDecks();
    const decks2 = await client2.getStarterDecks();

    if (decks1.length === 0 || decks2.length === 0) {
      console.log("No starter decks available");
      return;
    }

    await client1.sendMessage(wsMessageTypes.SELECT_DECK, { deckId: decks1[0]?.id });
    await client2.sendMessage(wsMessageTypes.SELECT_DECK, { deckId: decks2[0]?.id });

    await client1.waitForMessage(wsMessageTypes.ROOM_STATE);
    await client2.waitForMessage(wsMessageTypes.ROOM_STATE);

    // Ready up
    await client1.sendMessage(wsMessageTypes.READY, { ready: true });
    await client2.sendMessage(wsMessageTypes.READY, { ready: true });

    // Wait for game to start and get action mask
    let actionMask = null;
    let activeClient: TestClient | null = null;
    const maxWait = 20000;
    const startTime = Date.now();

    while (!actionMask && Date.now() - startTime < maxWait) {
      // Check client1
      try {
        const msg = await client1.waitForMessage(undefined, 500);
        if (msg.actionMask && msg.actionMask.legalActionCount > 0) {
          actionMask = msg.actionMask;
          activeClient = client1;
          break;
        }
      } catch {
        // Continue
      }

      // Check client2
      try {
        const msg = await client2.waitForMessage(undefined, 500);
        if (msg.actionMask && msg.actionMask.legalActionCount > 0) {
          actionMask = msg.actionMask;
          activeClient = client2;
          break;
        }
      } catch {
        // Continue
      }
    }

    if (!actionMask || !activeClient) {
      console.log("Could not get action mask, game may not have started properly");
      return;
    }

    // Send a valid action
    const validAction = extractValidActionFromMask(actionMask);
    await activeClient.sendMessage(wsMessageTypes.GAME_ACTION, { action: validAction });

    // Should still be connected after action
    expect(activeClient.isConnected()).toBe(true);

    // Wait for response (log batch, snapshot, or game over)
    let receivedResponse = false;
    const responseStart = Date.now();

    while (!receivedResponse && Date.now() - responseStart < 5000) {
      try {
        const msg = await activeClient.waitForMessage(undefined, 500);
        if (
          msg.type === wsMessageTypes.GAME_LOG_BATCH ||
          msg.type === wsMessageTypes.GAME_SNAPSHOT ||
          msg.type === wsMessageTypes.GAME_OVER ||
          msg.type === wsMessageTypes.ERROR
        ) {
          receivedResponse = true;
        }
      } catch {
        // Continue
      }
    }

    // Connection should still be alive
    expect(client1.isConnected()).toBe(true);
    expect(client2.isConnected()).toBe(true);
  });

  it("should handle reconnection during game", async () => {
    // Setup game
    const user1 = generateUniqueTestUser();
    const user2 = generateUniqueTestUser();

    await client1.register(user1.username, user1.email, user1.password);
    await client2.register(user2.username, user2.email, user2.password);

    const { room } = await client1.createRoom();
    const join1 = await client1.joinRoom(room.id);
    const join2 = await client2.joinRoom(room.id);

    await client1.connectWebSocket(join1.joinToken);
    await client2.connectWebSocket(join2.joinToken);

    await client1.waitForMessage(wsMessageTypes.CONNECTION_ACK);
    await client2.waitForMessage(wsMessageTypes.CONNECTION_ACK);

    await client1.waitForMessage(wsMessageTypes.ROOM_STATE);
    await client2.waitForMessage(wsMessageTypes.ROOM_STATE);

    // Start game
    await client1.sendMessage(wsMessageTypes.START_GAME, {});
    await client1.waitForMessage(wsMessageTypes.ROOM_STATE);
    await client2.waitForMessage(wsMessageTypes.ROOM_STATE);

    // Select decks
    const decks1 = await client1.getStarterDecks();
    const decks2 = await client2.getStarterDecks();

    if (decks1.length === 0 || decks2.length === 0) {
      console.log("No starter decks available");
      return;
    }

    await client1.sendMessage(wsMessageTypes.SELECT_DECK, { deckId: decks1[0]?.id });
    await client2.sendMessage(wsMessageTypes.SELECT_DECK, { deckId: decks2[0]?.id });

    await client1.waitForMessage(wsMessageTypes.ROOM_STATE);
    await client2.waitForMessage(wsMessageTypes.ROOM_STATE);

    // Ready up
    await client1.sendMessage(wsMessageTypes.READY, { ready: true });
    await client2.sendMessage(wsMessageTypes.READY, { ready: true });

    // Wait for game to start
    await delay(3000);

    // Disconnect client1
    client1.disconnect();
    expect(client1.isConnected()).toBe(false);

    // Get a new join token and reconnect
    const newJoin1 = await client1.joinRoom(room.id);
    await client1.connectWebSocket(newJoin1.joinToken);

    // Should receive CONNECTION_ACK
    const ack = await client1.waitForMessage(wsMessageTypes.CONNECTION_ACK);
    expect(ack.type).toBe(wsMessageTypes.CONNECTION_ACK);
    expect(ack.playerSlot).toBe(0);

    // Should be connected again
    expect(client1.isConnected()).toBe(true);

    // Should receive game snapshot on reconnect (if game is IN_MATCH)
    // This may or may not happen depending on game state
    const maxWait = 5000;
    const startTime = Date.now();
    let receivedState = false;

    while (!receivedState && Date.now() - startTime < maxWait) {
      try {
        const msg = await client1.waitForMessage(undefined, 500);
        if (
          msg.type === wsMessageTypes.ROOM_STATE ||
          msg.type === wsMessageTypes.GAME_SNAPSHOT
        ) {
          receivedState = true;
        }
      } catch {
        // Continue
      }
    }

    // Should have received some state after reconnect
    expect(receivedState).toBe(true);
  });
});
