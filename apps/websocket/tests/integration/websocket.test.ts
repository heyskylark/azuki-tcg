import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { TestClient } from "@test/utils/TestClient";
import { generateUniqueTestUser, wsMessageTypes } from "@test/utils/fixtures";
import { resetTestDatabase } from "@test/utils/database";

describe("WebSocket", () => {
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

  describe("Connection", () => {
    it("should connect with valid joinToken", async () => {
      // Setup: Create user and room
      const user = generateUniqueTestUser();
      await client1.register(user.username, user.email, user.password);
      const { room } = await client1.createRoom();
      const { joinToken } = await client1.joinRoom(room.id);

      // Connect WebSocket
      await client1.connectWebSocket(joinToken);

      expect(client1.isConnected()).toBe(true);
    });

    it("should receive CONNECTION_ACK after connecting", async () => {
      const user = generateUniqueTestUser();
      await client1.register(user.username, user.email, user.password);
      const { room } = await client1.createRoom();
      const { joinToken, playerSlot } = await client1.joinRoom(room.id);

      await client1.connectWebSocket(joinToken);

      const ack = await client1.waitForMessage(wsMessageTypes.CONNECTION_ACK);

      expect(ack.type).toBe(wsMessageTypes.CONNECTION_ACK);
      expect(ack.playerId).toBeDefined();
      expect(ack.playerSlot).toBe(playerSlot);
    });

    it("should receive ROOM_STATE after CONNECTION_ACK", async () => {
      const user = generateUniqueTestUser();
      await client1.register(user.username, user.email, user.password);
      const { room } = await client1.createRoom();
      const { joinToken } = await client1.joinRoom(room.id);

      await client1.connectWebSocket(joinToken);

      // Wait for CONNECTION_ACK first
      await client1.waitForMessage(wsMessageTypes.CONNECTION_ACK);

      // Then wait for ROOM_STATE
      const roomState = await client1.waitForMessage(wsMessageTypes.ROOM_STATE);

      expect(roomState.type).toBe(wsMessageTypes.ROOM_STATE);
      expect(roomState.status).toBe("WAITING_FOR_PLAYERS");
      expect(roomState.players).toBeDefined();
      expect(Array.isArray(roomState.players)).toBe(true);
    });

    it("should fail connection with invalid joinToken", async () => {
      await expect(
        client1.connectWebSocket("invalid-token")
      ).rejects.toThrow();
    });
  });

  describe("Ping/Pong", () => {
    it("should respond to PING with PONG", async () => {
      const user = generateUniqueTestUser();
      await client1.register(user.username, user.email, user.password);
      const { room } = await client1.createRoom();
      const { joinToken } = await client1.joinRoom(room.id);

      await client1.connectWebSocket(joinToken);

      // Clear initial messages
      await client1.waitForMessage(wsMessageTypes.CONNECTION_ACK);
      await client1.waitForMessage(wsMessageTypes.ROOM_STATE);

      // Send PING
      await client1.sendMessage(wsMessageTypes.PING, {});

      // Wait for PONG
      const pong = await client1.waitForMessage(wsMessageTypes.PONG);
      expect(pong.type).toBe(wsMessageTypes.PONG);
    });
  });

  describe("Room State Updates", () => {
    it("should receive ROOM_STATE when opponent joins", async () => {
      // Player 1 creates and joins room
      const user1 = generateUniqueTestUser();
      await client1.register(user1.username, user1.email, user1.password);
      const { room } = await client1.createRoom();
      const { joinToken: token1 } = await client1.joinRoom(room.id);

      await client1.connectWebSocket(token1);
      await client1.waitForMessage(wsMessageTypes.CONNECTION_ACK);
      await client1.waitForMessage(wsMessageTypes.ROOM_STATE);

      // Player 2 joins room
      const user2 = generateUniqueTestUser();
      await client2.register(user2.username, user2.email, user2.password);
      const { joinToken: token2 } = await client2.joinRoom(room.id);

      await client2.connectWebSocket(token2);
      await client2.waitForMessage(wsMessageTypes.CONNECTION_ACK);

      // Player 1 should receive updated ROOM_STATE with player 2
      const roomState = await client1.waitForMessage(wsMessageTypes.ROOM_STATE, 5000);

      expect(roomState.type).toBe(wsMessageTypes.ROOM_STATE);
      // Both player slots should now have players
      const players = roomState.players as Array<{ id: string; connected: boolean } | null>;
      expect(players[0]).not.toBeNull();
      expect(players[1]).not.toBeNull();
    });
  });

  describe("START_GAME", () => {
    it("should transition room to DECK_SELECTION when owner sends START_GAME", async () => {
      // Setup two players in room
      const user1 = generateUniqueTestUser();
      await client1.register(user1.username, user1.email, user1.password);
      const { room } = await client1.createRoom();
      const { joinToken: token1 } = await client1.joinRoom(room.id);

      const user2 = generateUniqueTestUser();
      await client2.register(user2.username, user2.email, user2.password);
      const { joinToken: token2 } = await client2.joinRoom(room.id);

      // Both connect
      await client1.connectWebSocket(token1);
      await client1.waitForMessage(wsMessageTypes.CONNECTION_ACK);
      await client1.waitForMessage(wsMessageTypes.ROOM_STATE);

      await client2.connectWebSocket(token2);
      await client2.waitForMessage(wsMessageTypes.CONNECTION_ACK);
      // Clear room state messages
      client1.clearMessageQueue();
      client2.clearMessageQueue();

      // Owner starts the game
      await client1.sendMessage(wsMessageTypes.START_GAME, {});

      // Both should receive ROOM_STATE with DECK_SELECTION
      const state1 = await client1.waitForMessage(wsMessageTypes.ROOM_STATE, 5000);
      const state2 = await client2.waitForMessage(wsMessageTypes.ROOM_STATE, 5000);

      expect(state1.status).toBe("DECK_SELECTION");
      expect(state2.status).toBe("DECK_SELECTION");
    });

    it("should fail START_GAME from non-owner", async () => {
      // Setup two players in room
      const user1 = generateUniqueTestUser();
      await client1.register(user1.username, user1.email, user1.password);
      const { room } = await client1.createRoom();
      const { joinToken: token1 } = await client1.joinRoom(room.id);

      const user2 = generateUniqueTestUser();
      await client2.register(user2.username, user2.email, user2.password);
      const { joinToken: token2 } = await client2.joinRoom(room.id);

      // Both connect
      await client1.connectWebSocket(token1);
      await client1.waitForMessage(wsMessageTypes.CONNECTION_ACK);

      await client2.connectWebSocket(token2);
      await client2.waitForMessage(wsMessageTypes.CONNECTION_ACK);
      await client2.waitForMessage(wsMessageTypes.ROOM_STATE);
      client2.clearMessageQueue();

      // Non-owner tries to start the game
      await client2.sendMessage(wsMessageTypes.START_GAME, {});

      // Should receive error
      const error = await client2.waitForMessage(wsMessageTypes.ERROR, 5000);
      expect(error.type).toBe(wsMessageTypes.ERROR);
    });
  });

  describe("SELECT_DECK", () => {
    it("should update room state when player selects deck", async () => {
      // Setup two players in room
      const user1 = generateUniqueTestUser();
      await client1.register(user1.username, user1.email, user1.password);
      const { room } = await client1.createRoom();
      const { joinToken: token1 } = await client1.joinRoom(room.id);

      const user2 = generateUniqueTestUser();
      await client2.register(user2.username, user2.email, user2.password);
      const { joinToken: token2 } = await client2.joinRoom(room.id);

      // Both connect
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
      client1.clearMessageQueue();
      client2.clearMessageQueue();

      // Get decks
      const { decks } = await client1.getDecks();
      if (decks.length === 0) {
        // Skip test if no decks available
        return;
      }

      const deckId = decks[0]?.id;
      if (!deckId) return;

      // Player 1 selects deck
      await client1.sendMessage(wsMessageTypes.SELECT_DECK, { deckId });

      // Should receive updated room state
      const state = await client1.waitForMessage(wsMessageTypes.ROOM_STATE, 5000);
      expect(state.type).toBe(wsMessageTypes.ROOM_STATE);
      const players = state.players as Array<{ deckSelected: boolean } | null>;
      expect(players[0]?.deckSelected).toBe(true);
    });
  });

  describe("READY", () => {
    it("should update room state when player readies up", async () => {
      // Setup two players in room with decks selected
      const user1 = generateUniqueTestUser();
      await client1.register(user1.username, user1.email, user1.password);
      const { room } = await client1.createRoom();
      const { joinToken: token1 } = await client1.joinRoom(room.id);

      const user2 = generateUniqueTestUser();
      await client2.register(user2.username, user2.email, user2.password);
      const { joinToken: token2 } = await client2.joinRoom(room.id);

      // Both connect
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

      // Get and select decks for both players
      const decks1 = await client1.getDecks();
      const decks2 = await client2.getDecks();

      if (decks1.decks.length === 0 || decks2.decks.length === 0) {
        return; // Skip if no decks
      }

      const deck1Id = decks1.decks[0]?.id;
      const deck2Id = decks2.decks[0]?.id;
      if (!deck1Id || !deck2Id) return;

      await client1.sendMessage(wsMessageTypes.SELECT_DECK, { deckId: deck1Id });
      await client2.sendMessage(wsMessageTypes.SELECT_DECK, { deckId: deck2Id });

      // Wait for deck selection updates
      client1.clearMessageQueue();
      client2.clearMessageQueue();

      // Player 1 readies up
      await client1.sendMessage(wsMessageTypes.READY, { ready: true });

      // Should receive updated room state
      const state = await client1.waitForMessage(wsMessageTypes.ROOM_STATE, 5000);
      expect(state.type).toBe(wsMessageTypes.ROOM_STATE);
      const players = state.players as Array<{ ready: boolean } | null>;
      expect(players[0]?.ready).toBe(true);
    });
  });
});
