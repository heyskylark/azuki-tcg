/**
 * Log processor for game state logs.
 * Handles visibility redaction for each player based on web-service.md spec.
 */

import type {
  GameLog,
  GameLogTapChange,
  GameLogZoneMoved,
  StateContext,
  ZoneType,
} from "@/engine/types";

import type { SnapshotSelectionCard } from "@tcg/backend-core/types/ws";

// Extended state context for log batches that can include selection cards
export interface LogBatchStateContext extends StateContext {
  // Use abilitySubphase for client compatibility (same as abilityPhase)
  abilitySubphase?: string;
  // Selection cards for SELECTION_PICK and BOTTOM_DECK phases
  selectionCards?: SnapshotSelectionCard[];
}

export interface GameLogBatchMessage {
  type: "GAME_LOG_BATCH";
  batchNumber: number;
  logs: ProcessedGameLog[];
  stateContext: LogBatchStateContext;
  actionMask?: unknown; // Included for the active player
}

export interface ProcessedGameLog {
  type: string;
  data: unknown;
}

/**
 * Zones that are private (hidden from opponent).
 */
const PRIVATE_ZONES: Set<ZoneType> = new Set(["DECK", "HAND", "IKZ_PILE"]);

/**
 * Check if a zone transition involves private information.
 * Private transitions: Deck→Hand, IKZ_PILE→Hand, etc.
 */
function isPrivateZoneTransition(fromZone: ZoneType, toZone: ZoneType): boolean {
  // Transitions FROM private zones (except when going to public zones like board)
  if (PRIVATE_ZONES.has(fromZone)) {
    // Going to hand stays private
    if (toZone === "HAND") {
      return true;
    }
    // Going to selection (for viewing) stays private for opponent
    if (toZone === "SELECTION") {
      return true;
    }
  }

  // Hand to hand shuffles (mulligan)
  if (fromZone === "HAND" && toZone === "DECK") {
    return true;
  }

  return false;
}

/**
 * Redact a CARD_ZONE_MOVED log for the viewing player.
 * Per spec: opponent's private zone transitions show card_id: "HIDDEN", metadata: null
 */
function redactZoneMoved(
  log: GameLogZoneMoved,
  viewingPlayer: 0 | 1
): ProcessedGameLog {
  const isOwnCard = log.card.player === viewingPlayer;
  const isPrivate = isPrivateZoneTransition(log.fromZone, log.toZone);

  if (!isOwnCard && isPrivate) {
    // Redact opponent's private zone transitions
    return {
      type: "CARD_ZONE_MOVED",
      data: {
        card: {
          player: log.card.player,
          cardDefId: null, // Hidden
          zone: log.card.zone,
          zoneIndex: log.card.zoneIndex,
        },
        fromZone: log.fromZone,
        fromIndex: log.fromIndex,
        toZone: log.toZone,
        toIndex: log.toIndex,
        metadata: null, // Hidden
      },
    };
  }

  // Full visibility
  return {
    type: "CARD_ZONE_MOVED",
    data: log,
  };
}

/**
 * Process a CARD_TAP_STATE_CHANGED log.
 * Handles cards being tapped, untapped, or put into cooldown.
 * Tap state changes are always visible to both players.
 */
function processTapStateChanged(log: GameLogTapChange): ProcessedGameLog {
  return {
    type: "CARD_TAP_STATE_CHANGED",
    data: {
      card: {
        player: log.card.player,
        cardDefId: log.card.cardDefId,
        zone: log.card.zone,
        zoneIndex: log.card.zoneIndex,
      },
      newState: log.newState,
    },
  };
}

/**
 * Process a single game log for a specific player.
 * Applies visibility redaction based on log type and ownership.
 */
function processLogForPlayer(
  log: GameLog,
  viewingPlayer: 0 | 1
): ProcessedGameLog {
  switch (log.type) {
    case "CARD_ZONE_MOVED":
      return redactZoneMoved(log.data as GameLogZoneMoved, viewingPlayer);

    case "DECK_SHUFFLED":
      // Shuffle events are public but don't reveal card order
      return { type: log.type, data: log.data };

    case "CARD_TAP_STATE_CHANGED":
      return processTapStateChanged(log.data as GameLogTapChange);

    case "CARD_STAT_CHANGE":
    case "KEYWORDS_CHANGED":
    case "STATUS_EFFECT_APPLIED":
    case "STATUS_EFFECT_EXPIRED":
    case "COMBAT_DECLARED":
    case "DEFENDER_DECLARED":
    case "COMBAT_DAMAGE":
    case "ENTITY_DIED":
    case "EFFECT_QUEUED":
    case "CARD_EFFECT_ENABLED":
    case "TURN_STARTED":
    case "TURN_ENDED":
    case "GAME_ENDED":
      // All other log types are fully visible
      return { type: log.type, data: log.data };

    default:
      // Unknown log type - pass through
      return { type: log.type, data: log.data };
  }
}

/**
 * Process all game logs for a specific player.
 * Creates a log batch message with appropriate redaction.
 */
export function processLogsForPlayer(
  logs: GameLog[],
  viewingPlayer: 0 | 1,
  stateContext: StateContext,
  batchNumber: number = 0
): GameLogBatchMessage {
  const processedLogs = logs.map((log) =>
    processLogForPlayer(log, viewingPlayer)
  );

  // Create extended state context with abilitySubphase for client compatibility
  const extendedStateContext: LogBatchStateContext = {
    ...stateContext,
    abilitySubphase: stateContext.abilityPhase,
  };

  return {
    type: "GAME_LOG_BATCH",
    batchNumber,
    logs: processedLogs,
    stateContext: extendedStateContext,
  };
}

/**
 * Check if a card reference belongs to the viewing player.
 */
export function isOwnCard(
  cardPlayer: 0 | 1,
  viewingPlayer: 0 | 1
): boolean {
  return cardPlayer === viewingPlayer;
}
