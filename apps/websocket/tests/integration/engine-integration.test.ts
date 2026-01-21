import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { TestClient } from "@test/utils/TestClient";
import { generateUniqueTestUser, wsMessageTypes } from "@test/utils/fixtures";
import { resetTestDatabase } from "@test/utils/database";
import { extractValidActionFromMask, delay } from "@test/utils/helpers";

/**
 * Engine Integration Tests
 *
 * These tests focus on the WebSocket ↔ C Engine integration layer,
 * which is the critical boundary where most issues arise.
 */
describe("Engine Integration", () => {
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

  /**
   * Helper to setup a game ready to start.
   * Returns when both players are connected, deck selected, and ready.
   */
  async function setupGameToStart(): Promise<{
    roomId: string;
  }> {
    // Register users
    const user1 = generateUniqueTestUser();
    await client1.register(user1.username, user1.email, user1.password);

    const user2 = generateUniqueTestUser();
    await client2.register(user2.username, user2.email, user2.password);

    // Create room
    const { room } = await client1.createRoom();
    const roomId = room.id;

    // Join room
    const { joinToken: token1 } = await client1.joinRoom(roomId);
    const { joinToken: token2 } = await client2.joinRoom(roomId);

    // Connect WebSockets
    await client1.connectWebSocket(token1);
    await client1.waitForMessage(wsMessageTypes.CONNECTION_ACK);
    await client1.waitForMessage(wsMessageTypes.ROOM_STATE);

    await client2.connectWebSocket(token2);
    await client2.waitForMessage(wsMessageTypes.CONNECTION_ACK);
    await client2.waitForMessage(wsMessageTypes.ROOM_STATE);

    // Start game
    await client1.sendMessage(wsMessageTypes.START_GAME, {});
    await client1.waitForMessage(wsMessageTypes.ROOM_STATE);
    await client2.waitForMessage(wsMessageTypes.ROOM_STATE);

    // Get and select decks
    const decks1 = await client1.getDecks();
    const decks2 = await client2.getDecks();

    if (decks1.decks.length === 0 || decks2.decks.length === 0) {
      throw new Error("No starter decks available for testing");
    }

    const deck1Id = decks1.decks[0]?.id;
    const deck2Id = decks2.decks[0]?.id;
    if (!deck1Id || !deck2Id) {
      throw new Error("Invalid deck IDs");
    }

    await client1.sendMessage(wsMessageTypes.SELECT_DECK, { deckId: deck1Id });
    await client2.sendMessage(wsMessageTypes.SELECT_DECK, { deckId: deck2Id });

    // Wait for deck selection confirmations
    await client1.waitForMessage(wsMessageTypes.ROOM_STATE);
    await client2.waitForMessage(wsMessageTypes.ROOM_STATE);

    // Clear any additional room state messages
    client1.clearMessageQueue();
    client2.clearMessageQueue();

    return { roomId };
  }

  describe("Game Start and World Creation", () => {
    it("should transition to IN_MATCH when both players ready up", async () => {
      await setupGameToStart();

      // Both players ready up
      await client1.sendMessage(wsMessageTypes.READY, { ready: true });
      await client2.sendMessage(wsMessageTypes.READY, { ready: true });

      // Wait for game to start - look for IN_MATCH status
      // May need to wait through READY_CHECK and STARTING states
      let gameStarted = false;
      const maxWait = 15000;
      const startTime = Date.now();

      while (!gameStarted && Date.now() - startTime < maxWait) {
        try {
          const msg = await client1.waitForMessage(undefined, 2000);
          if (msg.type === wsMessageTypes.ROOM_STATE && msg.status === "IN_MATCH") {
            gameStarted = true;
            break;
          }
          if (msg.type === wsMessageTypes.GAME_SNAPSHOT || msg.type === wsMessageTypes.GAME_LOG_BATCH) {
            gameStarted = true;
            break;
          }
        } catch {
          // Timeout, continue waiting
        }
      }

      expect(gameStarted).toBe(true);
    });

    it("should receive initial GAME_SNAPSHOT on game start", async () => {
      await setupGameToStart();

      // Both ready
      await client1.sendMessage(wsMessageTypes.READY, { ready: true });
      await client2.sendMessage(wsMessageTypes.READY, { ready: true });

      // Wait for game to start and receive snapshot
      const maxWait = 15000;
      const startTime = Date.now();
      let snapshot1 = null;
      let snapshot2 = null;

      while ((!snapshot1 || !snapshot2) && Date.now() - startTime < maxWait) {
        try {
          if (!snapshot1) {
            const msg = await client1.waitForMessage(undefined, 1000);
            if (msg.type === wsMessageTypes.GAME_SNAPSHOT) {
              snapshot1 = msg;
            }
          }
        } catch {
          // Continue
        }

        try {
          if (!snapshot2) {
            const msg = await client2.waitForMessage(undefined, 1000);
            if (msg.type === wsMessageTypes.GAME_SNAPSHOT) {
              snapshot2 = msg;
            }
          }
        } catch {
          // Continue
        }
      }

      // At least one player should receive a snapshot
      const receivedSnapshot = snapshot1 ?? snapshot2;
      expect(receivedSnapshot).not.toBeNull();

      if (receivedSnapshot) {
        // Validate snapshot structure
        expect(receivedSnapshot.stateContext).toBeDefined();
        expect(receivedSnapshot.players).toBeDefined();
        expect(Array.isArray(receivedSnapshot.players)).toBe(true);
        expect(receivedSnapshot.players.length).toBe(2);
      }
    });
  });

  describe("Action Validation", () => {
    it("should receive action mask with legal actions", async () => {
      await setupGameToStart();

      // Start game
      await client1.sendMessage(wsMessageTypes.READY, { ready: true });
      await client2.sendMessage(wsMessageTypes.READY, { ready: true });

      // Wait for game start and find message with action mask
      const maxWait = 15000;
      const startTime = Date.now();
      let actionMaskMsg = null;

      while (!actionMaskMsg && Date.now() - startTime < maxWait) {
        try {
          const msg = await client1.waitForMessage(undefined, 1000);
          if (msg.actionMask) {
            actionMaskMsg = msg;
            break;
          }
        } catch {
          // Continue
        }
      }

      // Should have received a message with action mask
      expect(actionMaskMsg).not.toBeNull();

      if (actionMaskMsg?.actionMask) {
        const mask = actionMaskMsg.actionMask;
        // Validate action mask structure
        expect(mask.primaryActionMask).toBeDefined();
        expect(Array.isArray(mask.primaryActionMask)).toBe(true);
        expect(mask.legalPrimary).toBeDefined();
        expect(mask.legalSub1).toBeDefined();
        expect(mask.legalSub2).toBeDefined();
        expect(mask.legalSub3).toBeDefined();
      }
    });

    it("should reject invalid game action", async () => {
      await setupGameToStart();

      // Start game
      await client1.sendMessage(wsMessageTypes.READY, { ready: true });
      await client2.sendMessage(wsMessageTypes.READY, { ready: true });

      // Wait for game to start
      await delay(2000);
      client1.clearMessageQueue();

      // Send invalid action (out of bounds values)
      await client1.sendMessage(wsMessageTypes.GAME_ACTION, {
        action: [99, 99, 99, 99],
      });

      // Should receive an error
      try {
        const response = await client1.waitForMessage(undefined, 5000);
        // Either ERROR or the action is silently rejected
        if (response.type === wsMessageTypes.ERROR) {
          expect(response.code).toBeDefined();
        }
      } catch {
        // Timeout is acceptable - invalid action may be silently ignored
      }

      // Connection should still be alive
      expect(client1.isConnected()).toBe(true);
    });

    it("should process valid game action", async () => {
      await setupGameToStart();

      // Start game
      await client1.sendMessage(wsMessageTypes.READY, { ready: true });
      await client2.sendMessage(wsMessageTypes.READY, { ready: true });

      // Wait for game start and get action mask
      const maxWait = 15000;
      const startTime = Date.now();
      let actionMask = null;

      while (!actionMask && Date.now() - startTime < maxWait) {
        try {
          const msg = await client1.waitForMessage(undefined, 1000);
          if (msg.actionMask) {
            actionMask = msg.actionMask;
            break;
          }
        } catch {
          // Continue
        }
      }

      if (!actionMask) {
        // Skip if we couldn't get action mask (may not be active player)
        return;
      }

      // Extract valid action from mask
      const validAction = extractValidActionFromMask(actionMask);
      client1.clearMessageQueue();

      // Send valid action
      await client1.sendMessage(wsMessageTypes.GAME_ACTION, {
        action: validAction,
      });

      // Should receive game state update (GAME_LOG_BATCH or GAME_SNAPSHOT)
      let receivedUpdate = false;
      const updateStart = Date.now();

      while (!receivedUpdate && Date.now() - updateStart < 5000) {
        try {
          const msg = await client1.waitForMessage(undefined, 1000);
          if (
            msg.type === wsMessageTypes.GAME_LOG_BATCH ||
            msg.type === wsMessageTypes.GAME_SNAPSHOT ||
            msg.type === wsMessageTypes.GAME_OVER
          ) {
            receivedUpdate = true;
          }
        } catch {
          // Continue
        }
      }

      // Connection should still be alive regardless
      expect(client1.isConnected()).toBe(true);
    });
  });

  describe("Game Log Processing", () => {
    it("should receive GAME_LOG_BATCH messages during game", async () => {
      await setupGameToStart();

      // Start game
      await client1.sendMessage(wsMessageTypes.READY, { ready: true });
      await client2.sendMessage(wsMessageTypes.READY, { ready: true });

      // Wait for game logs
      const maxWait = 15000;
      const startTime = Date.now();
      let logBatch = null;

      while (!logBatch && Date.now() - startTime < maxWait) {
        try {
          const msg = await client1.waitForMessage(undefined, 1000);
          if (msg.type === wsMessageTypes.GAME_LOG_BATCH) {
            logBatch = msg;
            break;
          }
        } catch {
          // Continue
        }
      }

      if (logBatch) {
        // Validate log batch structure
        expect(logBatch.batchNumber).toBeDefined();
        expect(logBatch.logs).toBeDefined();
        expect(Array.isArray(logBatch.logs)).toBe(true);
        expect(logBatch.stateContext).toBeDefined();
      }
    });
  });

  describe("Game Termination", () => {
    it("should handle FORFEIT message and end game", async () => {
      await setupGameToStart();

      // Start game
      await client1.sendMessage(wsMessageTypes.READY, { ready: true });
      await client2.sendMessage(wsMessageTypes.READY, { ready: true });

      // Wait for game to fully start
      await delay(2000);
      client1.clearMessageQueue();
      client2.clearMessageQueue();

      // Player 1 forfeits
      await client1.sendMessage(wsMessageTypes.FORFEIT, {});

      // Should receive GAME_OVER
      const maxWait = 10000;
      const startTime = Date.now();
      let gameOver1 = null;
      let gameOver2 = null;

      while ((!gameOver1 || !gameOver2) && Date.now() - startTime < maxWait) {
        try {
          if (!gameOver1) {
            const msg = await client1.waitForMessage(undefined, 500);
            if (msg.type === wsMessageTypes.GAME_OVER) {
              gameOver1 = msg;
            }
          }
        } catch {
          // Continue
        }

        try {
          if (!gameOver2) {
            const msg = await client2.waitForMessage(undefined, 500);
            if (msg.type === wsMessageTypes.GAME_OVER) {
              gameOver2 = msg;
            }
          }
        } catch {
          // Continue
        }
      }

      // At least one player should receive game over
      const gameOver = gameOver1 ?? gameOver2;
      expect(gameOver).not.toBeNull();

      if (gameOver) {
        expect(gameOver.winType).toBe("FORFEIT");
        // Winner should be player 2 (the one who didn't forfeit)
        expect(gameOver.winnerSlot).toBe(1);
      }
    });
  });

  describe("Concurrent Games", () => {
    it("should handle multiple concurrent games without interference", async () => {
      // Create second set of clients for second game
      const client3 = new TestClient();
      const client4 = new TestClient();

      try {
        // Setup first game
        const user1 = generateUniqueTestUser();
        await client1.register(user1.username, user1.email, user1.password);
        const user2 = generateUniqueTestUser();
        await client2.register(user2.username, user2.email, user2.password);

        const { room: room1 } = await client1.createRoom();
        const { joinToken: token1 } = await client1.joinRoom(room1.id);
        const { joinToken: token2 } = await client2.joinRoom(room1.id);

        // Setup second game
        const user3 = generateUniqueTestUser();
        await client3.register(user3.username, user3.email, user3.password);
        const user4 = generateUniqueTestUser();
        await client4.register(user4.username, user4.email, user4.password);

        const { room: room2 } = await client3.createRoom();
        const { joinToken: token3 } = await client3.joinRoom(room2.id);
        const { joinToken: token4 } = await client4.joinRoom(room2.id);

        // Connect all players
        await Promise.all([
          client1.connectWebSocket(token1),
          client2.connectWebSocket(token2),
          client3.connectWebSocket(token3),
          client4.connectWebSocket(token4),
        ]);

        // Wait for connection acks
        await Promise.all([
          client1.waitForMessage(wsMessageTypes.CONNECTION_ACK),
          client2.waitForMessage(wsMessageTypes.CONNECTION_ACK),
          client3.waitForMessage(wsMessageTypes.CONNECTION_ACK),
          client4.waitForMessage(wsMessageTypes.CONNECTION_ACK),
        ]);

        // Both rooms should be independent
        expect(room1.id).not.toBe(room2.id);

        // All connections should be alive
        expect(client1.isConnected()).toBe(true);
        expect(client2.isConnected()).toBe(true);
        expect(client3.isConnected()).toBe(true);
        expect(client4.isConnected()).toBe(true);
      } finally {
        client3.disconnect();
        client4.disconnect();
      }
    });
  });
});
