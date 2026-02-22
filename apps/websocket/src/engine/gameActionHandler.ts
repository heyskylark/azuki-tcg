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
  type SubmitActionResult,
} from "@/engine/WorldManager";
import { resolveAcceptedActionResult } from "@/engine/actionResolutionService";
import { maybeRunAiTurns } from "@/engine/aiOpponentService";
import { transitionToAborted } from "@/handlers/stateTransitionHandler";
import logger from "@/logger";
import type { ConnectionInfo } from "@/state/types";
import type { ActionTuple } from "@/engine/types";

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
): result is Extract<SubmitActionResult, { success: boolean }> {
  return "success" in result && !("code" in result);
}

function parseActionTuple(action: unknown): ActionTuple | null {
  if (!Array.isArray(action) || action.length !== 4) {
    return null;
  }

  const [a0, a1, a2, a3] = action;
  if (
    !Number.isInteger(a0) ||
    !Number.isInteger(a1) ||
    !Number.isInteger(a2) ||
    !Number.isInteger(a3)
  ) {
    return null;
  }

  return [a0, a1, a2, a3];
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
  const action = parseActionTuple(message.action);
  if (!action) {
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
  const result = submitPlayerAction(roomId, userId, action);

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
  logger.info("Action submitted successfully", {
    roomId,
    playerSlot,
    action,
    gameOver: result.gameOver,
    phase: result.stateContext.phase,
    abilityPhase: result.stateContext.abilityPhase,
    logsCount: result.logs.length,
  });
  await resolveAcceptedActionResult(roomId, result);
  if (result.gameOver) {
    return;
  }

  try {
    await maybeRunAiTurns(roomId);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    logger.error("AI turn processing failed after player action", {
      roomId,
      error: message,
    });
    await transitionToAborted(roomId, `AI inference failed: ${message}`);
  }
}
