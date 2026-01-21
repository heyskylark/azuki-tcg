/// <reference types="vitest" />
import { beforeAll, afterAll, beforeEach } from "vitest";
import { resetTestDatabase, closeTestDatabase } from "@test/utils/database";

// Test environment configuration
process.env.NODE_ENV = "test";

// Set test JWT secret if not already set
if (!process.env.JWT_SECRET) {
  process.env.JWT_SECRET = "test-jwt-secret-for-testing-purposes-only";
}

if (!process.env.JWT_ISSUER) {
  process.env.JWT_ISSUER = "azuki-tcg-test";
}

beforeAll(async () => {
  // Reset database before all tests run
  await resetTestDatabase();
});

beforeEach(async () => {
  // Clean up between tests if needed
  // Note: We don't reset between each test for speed
  // Individual test suites can call resetTestDatabase() in their beforeEach if needed
});

afterAll(async () => {
  await closeTestDatabase();
});
