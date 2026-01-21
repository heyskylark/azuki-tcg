import { describe, it, expect, beforeEach } from "vitest";
import { TestClient } from "@test/utils/TestClient";
import { generateUniqueTestUser } from "@test/utils/fixtures";
import { resetTestDatabase } from "@test/utils/database";

describe("Authentication", () => {
  let client: TestClient;

  beforeEach(async () => {
    await resetTestDatabase();
    client = new TestClient();
  });

  describe("POST /api/users/auth/sign-up", () => {
    it("should register a new user with valid credentials", async () => {
      const user = generateUniqueTestUser();

      const response = await client.register(user.username, user.email, user.password);

      expect(response.user).toBeDefined();
      expect(response.user.username).toBe(user.username);
      expect(response.user.email).toBe(user.email);
      expect(response.user.id).toBeDefined();
      expect(response.user.displayName).toBe(user.username);

      // Should have auth cookies set
      expect(client.getAccessToken()).toBeDefined();
      expect(client.getRefreshToken()).toBeDefined();
    });

    it("should fail with duplicate email", async () => {
      const user = generateUniqueTestUser();

      // First registration should succeed
      await client.register(user.username, user.email, user.password);

      // Second registration with same email should fail
      const client2 = new TestClient();
      const user2 = generateUniqueTestUser();
      user2.email = user.email; // Same email

      await expect(
        client2.register(user2.username, user2.email, user2.password)
      ).rejects.toThrow();
    });

    it("should fail with invalid email format", async () => {
      const user = generateUniqueTestUser();
      user.email = "invalid-email";

      await expect(
        client.register(user.username, user.email, user.password)
      ).rejects.toThrow();
    });

    it("should fail with short password", async () => {
      const user = generateUniqueTestUser();
      user.password = "short";

      await expect(
        client.register(user.username, user.email, user.password)
      ).rejects.toThrow();
    });

    it("should fail with invalid username", async () => {
      const user = generateUniqueTestUser();
      user.username = "ab"; // Too short

      await expect(
        client.register(user.username, user.email, user.password)
      ).rejects.toThrow();
    });
  });

  describe("POST /api/users/auth/login", () => {
    it("should login with correct credentials", async () => {
      const user = generateUniqueTestUser();

      // Register first
      await client.register(user.username, user.email, user.password);

      // Clear cookies and login
      client.clearCookies();
      const response = await client.login(user.email, user.password);

      expect(response.user).toBeDefined();
      expect(response.user.email).toBe(user.email);
      expect(client.getAccessToken()).toBeDefined();
      expect(client.getRefreshToken()).toBeDefined();
    });

    it("should fail with incorrect password", async () => {
      const user = generateUniqueTestUser();

      // Register first
      await client.register(user.username, user.email, user.password);

      // Try to login with wrong password
      client.clearCookies();
      await expect(client.login(user.email, "wrongpassword")).rejects.toThrow();
    });

    it("should fail with non-existent email", async () => {
      await expect(
        client.login("nonexistent@test.com", "somepassword")
      ).rejects.toThrow();
    });
  });

  describe("POST /api/users/auth/refresh", () => {
    it("should refresh tokens with valid refresh token", async () => {
      const user = generateUniqueTestUser();

      // Register and get initial tokens
      await client.register(user.username, user.email, user.password);
      const initialAccessToken = client.getAccessToken();

      // Refresh tokens
      const response = await client.refreshTokens();

      expect(response.user).toBeDefined();
      expect(response.user.email).toBe(user.email);

      // Should have new tokens
      const newAccessToken = client.getAccessToken();
      expect(newAccessToken).toBeDefined();
      // Note: Token might be the same if refreshed quickly, but should work
    });

    it("should fail without authentication", async () => {
      // No registration/login - no cookies
      await expect(client.refreshTokens()).rejects.toThrow();
    });
  });

  describe("POST /api/users/auth/logout", () => {
    it("should clear authentication", async () => {
      const user = generateUniqueTestUser();

      // Register
      await client.register(user.username, user.email, user.password);
      expect(client.getAccessToken()).toBeDefined();

      // Logout
      await client.logout();

      // Cookies should be cleared
      expect(client.getAccessToken()).toBeUndefined();
      expect(client.getRefreshToken()).toBeUndefined();
    });
  });
});
