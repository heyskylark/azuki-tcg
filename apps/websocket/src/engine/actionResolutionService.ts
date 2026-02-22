import { getRoomChannel } from "@/state/RoomRegistry";
import {
  incrementBatchNumber,
  getPlayerObservationBySlot,
  flushEngineDebugLogs,
} from "@/engine/WorldManager";
import { processLogsForPlayer } from "@/engine/logProcessor";
import { sendToPlayer } from "@/utils/broadcast";
import { storeGameLogs } from "@/engine/gameLogService";
import { handleGameOver } from "@/engine/gameOverHandler";
import logger from "@/logger";
import type { ActionResult } from "@/engine/types";

/**
 * Broadcast and persist an accepted action result.
 */
export async function resolveAcceptedActionResult(
  roomId: string,
  result: ActionResult
): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot resolve action result: channel not found", { roomId });
    flushEngineDebugLogs();
    return;
  }

  const batchNumber = incrementBatchNumber(roomId);

  storeGameLogs(roomId, batchNumber, result.logs).catch((error) => {
    logger.error("Failed to store game logs", { roomId, batchNumber, error });
  });

  for (const slot of [0, 1] as const) {
    const playerConnection = channel.players[slot];
    if (!playerConnection?.ws || !playerConnection.connected) {
      continue;
    }

    try {
      const logBatch = processLogsForPlayer(
        result.logs,
        slot,
        result.stateContext,
        batchNumber
      );

      if (!result.gameOver && result.stateContext.activePlayer === slot) {
        const observation = getPlayerObservationBySlot(roomId, slot);
        if (observation?.actionMask) {
          logBatch.actionMask = observation.actionMask;
        }

        const abilityPhase = result.stateContext.abilityPhase;
        if (
          observation &&
          (abilityPhase === "SELECTION_PICK" || abilityPhase === "BOTTOM_DECK")
        ) {
          logBatch.stateContext.selectionCards =
            observation.myObservationData.selection.flatMap((card) => {
              if (!card) {
                return [];
              }

              return [{
                cardId: card.cardCode,
                cardDefId: card.cardDefId,
                zoneIndex: card.zoneIndex,
                type: card.type,
                ikzCost: card.ikzCost,
                curAtk: card.curAtk,
                curHp: card.curHp,
              }];
            });
        }
      }

      sendToPlayer(channel, slot, logBatch);
    } catch (error) {
      logger.error("Error processing/sending log batch", {
        roomId,
        slot,
        error: String(error),
        stack: error instanceof Error ? error.stack : undefined,
      });
    }
  }

  // Flush C engine debug logs after all engine operations including observation building.
  flushEngineDebugLogs();

  if (result.gameOver) {
    await handleGameOver(roomId, result);
  }
}
