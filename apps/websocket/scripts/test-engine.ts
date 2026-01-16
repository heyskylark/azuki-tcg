/**
 * Standalone test script for the native engine binding.
 * Tests world creation without going through the full WebSocket flow.
 *
 * Usage: yarn ws test:engine
 */

import { createRequire } from "module";
import { fileURLToPath } from "url";
import path from "path";

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Card definition IDs matching include/generated/card_defs.h
const CardDefId = {
  IKZ_001: 0,
  IKZ_002: 1,
  STT01_001: 2, // Raizen Leader
  STT01_002: 3, // Raizen Gate
  STT01_003: 4,
  STT01_004: 5,
  STT01_005: 6,
  STT01_006: 7,
  STT01_007: 8,
  STT01_008: 9,
  STT01_009: 10,
  STT01_010: 11,
  STT01_011: 12,
  STT01_012: 13,
  STT01_013: 14,
  STT01_014: 15,
  STT01_015: 16,
  STT01_016: 17,
  STT01_017: 18,
  STT02_001: 19, // Shao Leader
  STT02_002: 20, // Shao Gate
  STT02_003: 21,
  STT02_004: 22,
  STT02_005: 23,
  STT02_006: 24,
  STT02_007: 25,
  STT02_008: 26,
  STT02_009: 27,
  STT02_010: 28,
  STT02_011: 29,
  STT02_012: 30,
  STT02_013: 31,
  STT02_014: 32,
  STT02_015: 33,
  STT02_016: 34,
  STT02_017: 35,
} as const;

// DeckCardEntry format matching the native binding
interface DeckCardEntry {
  cardId: number;
  count: number;
}

// Raizen deck matching world.c raizenDeckCardInfo
const raizenDeck: DeckCardEntry[] = [
  { cardId: CardDefId.STT01_001, count: 1 }, // Leader
  { cardId: CardDefId.STT01_002, count: 1 }, // Gate
  { cardId: CardDefId.STT01_003, count: 4 },
  { cardId: CardDefId.STT01_004, count: 4 },
  { cardId: CardDefId.STT01_005, count: 4 },
  { cardId: CardDefId.STT01_006, count: 2 },
  { cardId: CardDefId.STT01_007, count: 4 },
  { cardId: CardDefId.STT01_008, count: 4 },
  { cardId: CardDefId.STT01_009, count: 4 },
  { cardId: CardDefId.STT01_010, count: 2 },
  { cardId: CardDefId.STT01_011, count: 2 },
  { cardId: CardDefId.STT01_012, count: 4 },
  { cardId: CardDefId.STT01_013, count: 4 },
  { cardId: CardDefId.STT01_014, count: 4 },
  { cardId: CardDefId.STT01_015, count: 2 },
  { cardId: CardDefId.STT01_016, count: 2 },
  { cardId: CardDefId.STT01_017, count: 4 },
  { cardId: CardDefId.IKZ_001, count: 10 }, // IKZ pile
];

// Shao deck matching world.c shaoDeckCardInfo
const shaoDeck: DeckCardEntry[] = [
  { cardId: CardDefId.STT02_001, count: 1 }, // Leader
  { cardId: CardDefId.STT02_002, count: 1 }, // Gate
  { cardId: CardDefId.STT02_003, count: 4 },
  { cardId: CardDefId.STT02_004, count: 4 },
  { cardId: CardDefId.STT02_005, count: 4 },
  { cardId: CardDefId.STT02_006, count: 4 },
  { cardId: CardDefId.STT02_007, count: 4 },
  { cardId: CardDefId.STT02_008, count: 4 },
  { cardId: CardDefId.STT02_009, count: 4 },
  { cardId: CardDefId.STT02_010, count: 2 },
  { cardId: CardDefId.STT02_011, count: 4 },
  { cardId: CardDefId.STT02_012, count: 4 },
  { cardId: CardDefId.STT02_013, count: 2 },
  { cardId: CardDefId.STT02_014, count: 2 },
  { cardId: CardDefId.STT02_015, count: 4 },
  { cardId: CardDefId.STT02_016, count: 2 },
  { cardId: CardDefId.STT02_017, count: 2 },
  { cardId: CardDefId.IKZ_001, count: 10 }, // IKZ pile
];

function calculateDeckStats(deck: DeckCardEntry[]): { total: number; entries: number } {
  const total = deck.reduce((sum, entry) => sum + entry.count, 0);
  return { total, entries: deck.length };
}

function loadBinding() {
  const releasePath = path.join(__dirname, "../native/build/Release/azuki_engine.node");
  const debugPath = path.join(__dirname, "../native/build/Debug/azuki_engine.node");

  try {
    console.log("Trying Release build:", releasePath);
    return require(releasePath);
  } catch {
    try {
      console.log("Trying Debug build:", debugPath);
      return require(debugPath);
    } catch (error) {
      console.error("Failed to load native binding from either path.");
      console.error("Make sure to run: yarn ws build:native");
      throw error;
    }
  }
}

function main() {
  console.log("=== Engine Test Script ===\n");

  // Load binding
  console.log("Loading native binding...");
  const binding = loadBinding();
  console.log("Native binding loaded successfully.\n");

  // Show deck stats
  const raizenStats = calculateDeckStats(raizenDeck);
  const shaoStats = calculateDeckStats(shaoDeck);
  console.log(`Raizen deck: ${raizenStats.entries} entries, ${raizenStats.total} total cards`);
  console.log(`Shao deck: ${shaoStats.entries} entries, ${shaoStats.total} total cards`);
  console.log();

  // Test 1: Create world with valid decks
  console.log("Test 1: Creating world with valid decks...");
  console.log("  Seed: 12345");
  console.log("  Player 0: Raizen deck");
  console.log("  Player 1: Shao deck");

  try {
    const result = binding.createWorldWithDecks(12345, raizenDeck, shaoDeck);
    console.log("  Result:", JSON.stringify(result, null, 2));

    if (result.success) {
      console.log("\n  World created successfully!");

      // Test getGameState
      console.log("\n  Getting game state...");
      const gameState = binding.getGameState(result.worldId);
      console.log("  Game state:", JSON.stringify(gameState, null, 2));

      // Test requiresAction
      const needsAction = binding.requiresAction(result.worldId);
      console.log("  Requires action:", needsAction);

      // Test getActivePlayer
      const activePlayer = binding.getActivePlayer(result.worldId);
      console.log("  Active player:", activePlayer);

      // Cleanup
      console.log("\n  Destroying world...");
      binding.destroyWorld(result.worldId);
      console.log("  World destroyed.");
    } else {
      console.log("\n  FAILED: World creation returned success=false");
      if (result.error) {
        console.log("  Error:", result.error);
      }
    }
  } catch (error) {
    console.error("\n  EXCEPTION during world creation:");
    console.error(" ", error);
  }

  // Test 2: Create world with minimal/invalid deck (to test error handling)
  console.log("\n\nTest 2: Creating world with empty decks (should fail)...");
  try {
    const result = binding.createWorldWithDecks(12345, [], []);
    console.log("  Result:", JSON.stringify(result, null, 2));
    if (!result.success) {
      console.log("  (Expected failure - empty decks)");
    }
  } catch (error) {
    console.error("  EXCEPTION:", error);
    console.log("  (This may crash Node.js due to exit() in C code)");
  }

  console.log("\n=== Tests Complete ===");
}

// Also export the test data for other scripts to use
export { raizenDeck, shaoDeck, CardDefId };

main();
