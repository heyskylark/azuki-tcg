/**
 * Game log database service.
 * Stores game logs for replay functionality.
 */

import db from "@tcg/backend-core/database";
import { GameLogs } from "@tcg/backend-core/drizzle/schemas/game_logs";
import type { GameLog } from "@/engine/types";
import logger from "@/logger";

/**
 * Store a batch of game logs to the database.
 * Logs are stored with batch and sequence numbers for ordered replay.
 */
export async function storeGameLogs(
  roomId: string,
  batchNumber: number,
  logs: GameLog[]
): Promise<void> {
  if (logs.length === 0) {
    return;
  }

  const entries = logs.map((log, index) => ({
    roomId,
    batchNumber,
    sequenceNumber: index,
    logType: log.type,
    player: extractPlayerFromLog(log),
    logData: log.data,
  }));

  try {
    await db.insert(GameLogs).values(entries);
    logger.debug("Stored game logs", {
      roomId,
      batchNumber,
      count: logs.length,
    });
  } catch (error) {
    logger.error("Failed to store game logs", {
      roomId,
      batchNumber,
      error,
    });
    throw error;
  }
}

/**
 * Extract the player index from a game log if applicable.
 * Returns null if the log doesn't have a specific player.
 */
function extractPlayerFromLog(log: GameLog): number | null {
  // Type-based extraction - each log type has known properties
  switch (log.type) {
    // Logs with a card reference containing player
    case "CARD_ZONE_MOVED":
    case "CARD_STAT_CHANGE":
    case "KEYWORDS_CHANGED":
    case "CARD_TAP_STATE_CHANGED":
    case "STATUS_EFFECT_APPLIED":
    case "STATUS_EFFECT_EXPIRED":
    case "ENTITY_DIED":
    case "EFFECT_QUEUED":
    case "CARD_EFFECT_ENABLED":
      return (log.data as { card: { player: number } }).card.player;

    // Logs with attacker reference
    case "COMBAT_DECLARED":
    case "COMBAT_DAMAGE":
      return (log.data as { attacker: { player: number } }).attacker.player;

    // Logs with defender reference
    case "DEFENDER_DECLARED":
      return (log.data as { defender: { player: number } }).defender.player;

    // Logs with direct player field
    case "DECK_SHUFFLED":
    case "TURN_STARTED":
    case "TURN_ENDED":
      return (log.data as { player: number }).player;

    // Game ended has winner, not a specific player action
    case "GAME_ENDED":
      return null;

    default:
      return null;
  }
}
