/**
 * Handler for GAME_ACTION WebSocket messages.
 * Routes player actions to the game engine and broadcasts results.
 */

import type { WebSocket } from "uWebSockets.js";
import type { UserData } from "@/constants";
import { RoomStatus } from "@tcg/backend-core/types";
import { getRoomChannel } from "@/state/RoomRegistry";
import {
  getWorldByRoomId,
  submitPlayerAction,
  incrementBatchNumber,
  getPlayerObservationBySlot,
  flushEngineDebugLogs,
  type SubmitActionResult,
} from "@/engine/WorldManager";
import { processLogsForPlayer } from "@/engine/logProcessor";
import { sendToPlayer } from "@/utils/broadcast";
import { storeGameLogs } from "@/engine/gameLogService";
import { handleGameOver } from "@/engine/gameOverHandler";
import logger from "@/logger";
import type { ConnectionInfo } from "@/state/types";
import type { ActionTuple, ActionResult } from "@/engine/types";

export interface GameActionMessage {
  type: "GAME_ACTION";
  action: [number, number, number, number];
}

function sendError(
  ws: WebSocket<UserData>,
  code: string,
  message: string
): void {
  ws.send(
    JSON.stringify({
      type: "ERROR",
      code,
      message,
    })
  );
}

function isActionResult(
  result: SubmitActionResult
): result is ActionResult {
  return "success" in result && !("code" in result);
}

/**
 * Handle a GAME_ACTION message from a player.
 */
export async function handleGameAction(
  ws: WebSocket<UserData>,
  message: GameActionMessage,
  connectionInfo: ConnectionInfo
): Promise<void> {
  const { roomId, userId, playerSlot } = connectionInfo;

  // Validate room is in IN_MATCH state
  const channel = getRoomChannel(roomId);
  if (!channel || channel.status !== RoomStatus.IN_MATCH) {
    sendError(ws, "INVALID_STATE", "Game not in progress");
    return;
  }

  // Get active world
  const world = getWorldByRoomId(roomId);
  if (!world) {
    sendError(ws, "NO_WORLD", "Game world not found");
    return;
  }

  // Validate action format
  const action = message.action;
  if (!Array.isArray(action) || action.length !== 4) {
    sendError(ws, "INVALID_ACTION", "Action must be array of 4 integers");
    return;
  }

  // Submit action to engine
  logger.info("Submitting player action", {
    roomId,
    userId,
    playerSlot,
    action,
  });
  const result = submitPlayerAction(roomId, userId, action as ActionTuple);

  logger.info("Action result", {
    roomId,
    playerSlot,
    action,
    isActionResult: isActionResult(result),
    result: isActionResult(result) ? {
      success: result.success,
      gameOver: result.gameOver,
      phase: result.stateContext?.phase,
      activePlayer: result.stateContext?.activePlayer,
    } : result,
  });

  // Handle error responses
  if (!isActionResult(result)) {
    switch (result.code) {
      case "NOT_FOUND":
        sendError(ws, "NO_WORLD", result.error);
        return;
      case "NOT_YOUR_TURN":
        sendError(ws, "NOT_YOUR_TURN", result.error);
        return;
      case "NOT_AWAITING_ACTION":
        sendError(ws, "NOT_AWAITING_ACTION", result.error);
        return;
      case "INVALID_ACTION":
        sendError(ws, "INVALID_ACTION", result.error);
        return;
    }
  }

  // Action was accepted
  logger.debug("Action submitted successfully", {
    roomId,
    playerSlot,
    action,
    gameOver: result.gameOver,
  });

  // Increment batch number for log storage
  const batchNumber = incrementBatchNumber(roomId);

  // Store logs to database (non-blocking)
  storeGameLogs(roomId, batchNumber, result.logs).catch((error) => {
    logger.error("Failed to store game logs", { roomId, batchNumber, error });
  });

  // Process and send logs to each player with appropriate redaction
  for (const slot of [0, 1] as const) {
    const playerConnection = channel.players[slot];

    if (playerConnection?.ws && playerConnection.connected) {
      // Process logs with visibility redaction for this player
      const logBatch = processLogsForPlayer(
        result.logs,
        slot,
        result.stateContext,
        batchNumber
      );

      // Add action mask if it's this player's turn
      if (
        !result.gameOver &&
        result.stateContext.activePlayer === slot
      ) {
        const observation = getPlayerObservationBySlot(roomId, slot);
        if (observation?.actionMask) {
          logger.info("Action handler - action mask for next player", {
            roomId,
            slot,
            phase: result.stateContext.phase,
            legalActionCount: observation.actionMask.legalActionCount,
            legalPrimary: observation.actionMask.legalPrimary?.slice(0, Math.min(10, observation.actionMask.legalActionCount)),
            primaryActionMaskTrue: observation.actionMask.primaryActionMask
              ?.map((v, i) => v ? i : -1)
              .filter(i => i >= 0),
          });
          logBatch.actionMask = observation.actionMask;
        }
      }

      sendToPlayer(channel, slot, logBatch);
    }
  }

  // Flush C engine debug logs to Winston (after all engine operations including observation building)
  flushEngineDebugLogs();

  // Handle game over
  if (result.gameOver) {
    await handleGameOver(roomId, result);
  }
}
