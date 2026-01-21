import { sql } from "drizzle-orm";
import db from "@tcg/backend-core/database";

/**
 * Resets the test database by truncating all tables.
 * This is faster than dropping and recreating the schema.
 */
export async function resetTestDatabase(): Promise<void> {
  // Order matters due to foreign key constraints
  // Truncate in reverse dependency order
  const tables = [
    "match_results",
    "game_logs",
    "deck_cards",
    "decks",
    "jwt_tokens",
    "rooms",
    "users",
  ];

  for (const table of tables) {
    try {
      await db.execute(sql.raw(`TRUNCATE TABLE "${table}" CASCADE`));
    } catch (error) {
      // Table might not exist in some test scenarios
      console.warn(`Warning: Could not truncate table ${table}:`, error);
    }
  }
}

/**
 * Seeds the test database with required base data.
 * This includes cards and other static data needed for tests.
 */
export async function seedTestData(): Promise<void> {
  // Cards are seeded by migrations, so we don't need to seed them here
  // This function can be extended if we need additional test data
}

/**
 * Cleans up specific test data without full reset.
 * Useful for cleaning up after a specific test.
 */
export async function cleanupTestData(): Promise<void> {
  // Clean up in reverse dependency order
  await db.execute(sql.raw(`DELETE FROM "match_results"`));
  await db.execute(sql.raw(`DELETE FROM "game_logs"`));
  await db.execute(sql.raw(`DELETE FROM "deck_cards"`));
  await db.execute(sql.raw(`DELETE FROM "decks"`));
  await db.execute(sql.raw(`DELETE FROM "jwt_tokens"`));
  await db.execute(sql.raw(`DELETE FROM "rooms"`));
  await db.execute(sql.raw(`DELETE FROM "users"`));
}

/**
 * Closes the database connection pool.
 * Call this in afterAll to clean up.
 */
export async function closeTestDatabase(): Promise<void> {
  // The database connection is managed by drizzle-orm
  // We don't need to explicitly close it in most cases
  // as it will be cleaned up when the process exits
}
