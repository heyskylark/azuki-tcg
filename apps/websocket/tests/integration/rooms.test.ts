import { describe, it, expect, beforeEach } from "vitest";
import { TestClient } from "@test/utils/TestClient";
import { generateUniqueTestUser, testRoomPassword } from "@test/utils/fixtures";
import { resetTestDatabase } from "@test/utils/database";

describe("Rooms", () => {
  let client1: TestClient;
  let client2: TestClient;

  beforeEach(async () => {
    await resetTestDatabase();
    client1 = new TestClient();
    client2 = new TestClient();
  });

  describe("POST /api/rooms", () => {
    it("should create a public room as authenticated user", async () => {
      const user = generateUniqueTestUser();
      await client1.register(user.username, user.email, user.password);

      const response = await client1.createRoom();

      expect(response.room).toBeDefined();
      expect(response.room.id).toBeDefined();
      expect(response.room.status).toBe("WAITING_FOR_PLAYERS");
      expect(response.room.hasPassword).toBe(false);
      expect(response.room.player0Id).toBeDefined();
      expect(response.room.player1Id).toBeNull();
    });

    it("should create a private room with password", async () => {
      const user = generateUniqueTestUser();
      await client1.register(user.username, user.email, user.password);

      const response = await client1.createRoom(testRoomPassword);

      expect(response.room).toBeDefined();
      expect(response.room.hasPassword).toBe(true);
    });

    it("should fail to create room when not authenticated", async () => {
      await expect(client1.createRoom()).rejects.toThrow();
    });

    it("should fail to create room when user already in active room", async () => {
      const user = generateUniqueTestUser();
      await client1.register(user.username, user.email, user.password);

      // Create first room
      await client1.createRoom();

      // Try to create another room
      await expect(client1.createRoom()).rejects.toThrow();
    });
  });

  describe("POST /api/rooms/:room_id/join", () => {
    it("should join public room as second player", async () => {
      // Player 1 creates room
      const user1 = generateUniqueTestUser();
      await client1.register(user1.username, user1.email, user1.password);
      const { room } = await client1.createRoom();

      // Player 2 joins room
      const user2 = generateUniqueTestUser();
      await client2.register(user2.username, user2.email, user2.password);
      const joinResponse = await client2.joinRoom(room.id);

      expect(joinResponse.joinToken).toBeDefined();
      expect(joinResponse.playerSlot).toBe(1);
      expect(joinResponse.isNewJoin).toBe(true);
    });

    it("should return existing join token for room owner", async () => {
      const user = generateUniqueTestUser();
      await client1.register(user.username, user.email, user.password);
      const { room } = await client1.createRoom();

      // Owner joins their own room
      const joinResponse = await client1.joinRoom(room.id);

      expect(joinResponse.joinToken).toBeDefined();
      expect(joinResponse.playerSlot).toBe(0);
      expect(joinResponse.isNewJoin).toBe(false);
    });

    it("should fail to join private room with wrong password", async () => {
      // Player 1 creates private room
      const user1 = generateUniqueTestUser();
      await client1.register(user1.username, user1.email, user1.password);
      const { room } = await client1.createRoom(testRoomPassword);

      // Player 2 tries to join with wrong password
      const user2 = generateUniqueTestUser();
      await client2.register(user2.username, user2.email, user2.password);

      await expect(client2.joinRoom(room.id, "wrongpassword")).rejects.toThrow();
    });

    it("should join private room with correct password", async () => {
      // Player 1 creates private room
      const user1 = generateUniqueTestUser();
      await client1.register(user1.username, user1.email, user1.password);
      const { room } = await client1.createRoom(testRoomPassword);

      // Player 2 joins with correct password
      const user2 = generateUniqueTestUser();
      await client2.register(user2.username, user2.email, user2.password);
      const joinResponse = await client2.joinRoom(room.id, testRoomPassword);

      expect(joinResponse.joinToken).toBeDefined();
      expect(joinResponse.playerSlot).toBe(1);
    });

    it("should fail if user already in another active room", async () => {
      // Player 1 creates room 1
      const user1 = generateUniqueTestUser();
      await client1.register(user1.username, user1.email, user1.password);
      const { room: room1 } = await client1.createRoom();

      // Player 2 creates room 2
      const user2 = generateUniqueTestUser();
      await client2.register(user2.username, user2.email, user2.password);
      await client2.createRoom();

      // Player 2 tries to join room 1 while already in room 2
      await expect(client2.joinRoom(room1.id)).rejects.toThrow();
    });

    it("should fail to join non-existent room", async () => {
      const user = generateUniqueTestUser();
      await client1.register(user.username, user.email, user.password);

      await expect(
        client1.joinRoom("00000000-0000-0000-0000-000000000000")
      ).rejects.toThrow();
    });
  });
});
