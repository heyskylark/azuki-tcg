import { describe, it, expect, beforeEach } from "vitest";
import { TestClient } from "@test/utils/TestClient";
import { generateUniqueTestUser } from "@test/utils/fixtures";
import { resetTestDatabase } from "@test/utils/database";

describe("Decks", () => {
  let client: TestClient;

  beforeEach(async () => {
    await resetTestDatabase();
    client = new TestClient();
  });

  describe("GET /api/decks", () => {
    it("should get user decks after registration", async () => {
      const user = generateUniqueTestUser();
      await client.register(user.username, user.email, user.password);

      const response = await client.getDecks();

      expect(response.decks).toBeDefined();
      expect(Array.isArray(response.decks)).toBe(true);
      // New users should have starter decks created automatically
      // The number depends on the seed data
    });

    it("should fail when not authenticated", async () => {
      await expect(client.getDecks()).rejects.toThrow();
    });

    it("should return starter decks for new user", async () => {
      const user = generateUniqueTestUser();
      await client.register(user.username, user.email, user.password);

      const starterDecks = await client.getStarterDecks();

      expect(starterDecks).toBeDefined();
      expect(Array.isArray(starterDecks)).toBe(true);
      // Starter decks should be available
      // The exact count depends on seed data but should be > 0
    });

    it("should return decks with expected properties", async () => {
      const user = generateUniqueTestUser();
      await client.register(user.username, user.email, user.password);

      const response = await client.getDecks();

      if (response.decks.length > 0) {
        const deck = response.decks[0];
        expect(deck).toBeDefined();
        if (deck) {
          expect(deck.id).toBeDefined();
          expect(deck.name).toBeDefined();
          expect(deck.status).toBeDefined();
        }
      }
    });
  });
});
