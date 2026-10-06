/**
 * Handler for game over logic.
 * Stores match results, broadcasts game over, and cleans up resources.
 */

import db from "@tcg/backend-core/database";
import { MatchResults } from "@tcg/backend-core/drizzle/schemas/match_results";
import { RoomStatus, WinType } from "@tcg/backend-core/types";
import type { GameOverMessage } from "@tcg/backend-core/types/ws";
import { findRoomById, updateRoomStatus } from "@tcg/backend-core/services/roomService";
import { finalizeHumanEvaluationMatch } from "@tcg/backend-core/services/humanEvaluationService";
import { getRoomChannel, removeRoomChannel, updateRoomChannelStatus } from "@/state/RoomRegistry";
import {
  getWorldByRoomId,
  destroyGameWorld,
  getPlayerUserId,
  getGameState,
} from "@/engine/WorldManager";
import { clearAiOpponentForRoom } from "@/engine/aiOpponentService";
import { broadcastToRoom } from "@/utils/broadcast";
import logger from "@/logger";
import type { StateContext } from "@/engine/types";

export interface GameOverResult {
  gameOver: boolean;
  winner: number | null;
  stateContext: StateContext;
  logs: Array<{ type: string; data: unknown }>;
}

/**
 * Map the terminal engine log to the persisted result category.
 */
function getResultWinType(result: GameOverResult): WinType {
  if (result.winner === null) {
    return WinType.DRAW;
  }

  for (const log of result.logs) {
    if (
      log.type === "GAME_ENDED" &&
      typeof log.data === "object" &&
      log.data !== null &&
      "reason" in log.data &&
      log.data.reason === "CONCEDE"
    ) {
      return WinType.FORFEIT;
    }
  }
  return WinType.WIN;
}

/**
 * Handle game over for a room.
 * Stores match result, broadcasts game over message, and cleans up.
 */
export async function handleGameOver(roomId: string, result: GameOverResult): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.error("Cannot handle game over: room channel not found", { roomId });
    return;
  }

  const world = getWorldByRoomId(roomId);
  if (!world) {
    logger.error("Cannot handle game over: world not found", { roomId });
    return;
  }

  const room = await findRoomById(roomId);
  if (!room) {
    logger.error("Cannot handle game over: room data not found", { roomId });
    return;
  }

  // Determine winner info
  const winnerSlot = result.winner === 0 || result.winner === 1 ? result.winner : null;
  const winnerId = winnerSlot !== null ? getPlayerUserId(roomId, winnerSlot) : null;
  const publicWinnerId =
    winnerSlot !== null && channel.evaluation !== null && channel.players[winnerSlot]?.isAi
      ? `evaluation-opponent-${winnerSlot}`
      : winnerId;

  // Calculate game duration
  const durationSeconds = Math.floor((Date.now() - world.createdAt.getTime()) / 1000);
  const winType = getResultWinType(result);

  if (channel.evaluation) {
    await finalizeHumanEvaluationMatch({
      roomId,
      winnerId,
      winType,
      totalTurns: result.stateContext.turnNumber,
      durationSeconds,
    });
  } else {
    try {
      await db.insert(MatchResults).values({
        roomId,
        player0Id: world.player0UserId,
        player1Id: world.player1UserId,
        aiModelId: room.aiModelId,
        winnerId,
        winType,
        totalTurns: result.stateContext.turnNumber,
        durationSeconds,
      });
    } catch (error) {
      logger.error("Failed to store match result", { roomId, error });
    }
  }
  logger.info("Stored match result", {
    roomId,
    winnerId,
    totalTurns: result.stateContext.turnNumber,
    durationSeconds,
  });

  // Broadcast GAME_OVER to all players
  const gameOverMessage: GameOverMessage = {
    type: "GAME_OVER",
    winnerId: publicWinnerId,
    winnerSlot,
    winType,
    reason: getGameOverReason(result),
  };
  broadcastToRoom(channel, gameOverMessage);

  if (!channel.evaluation) {
    await updateRoomStatus(roomId, RoomStatus.COMPLETED);
  }
  updateRoomChannelStatus(roomId, { status: RoomStatus.COMPLETED });
  logger.info("Updated room status to COMPLETED", { roomId });

  await clearAiOpponentForRoom(roomId);

  // Clean up game world
  destroyGameWorld(roomId);

  // Remove room channel after a delay to allow clients to receive final messages
  setTimeout(() => {
    removeRoomChannel(roomId);
    logger.debug("Removed room channel", { roomId });
  }, 5000);
}

/**
 * Get a human-readable reason for game over.
 */
function getGameOverReason(result: GameOverResult): string {
  if (result.winner === null) {
    return "Draw";
  }
  return `Player ${result.winner} wins`;
}

/**
 * Handle a player forfeiting the game.
 */
export async function handleForfeit(roomId: string, forfeitingPlayerSlot: 0 | 1): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.error("Cannot handle forfeit: room channel not found", { roomId });
    return;
  }

  const world = getWorldByRoomId(roomId);
  if (!world) {
    logger.error("Cannot handle forfeit: world not found", { roomId });
    return;
  }

  const room = await findRoomById(roomId);
  if (!room) {
    logger.error("Cannot handle forfeit: room data not found", { roomId });
    return;
  }

  // Winner is the opponent
  const winnerSlot = forfeitingPlayerSlot === 0 ? 1 : 0;
  const winnerId = getPlayerUserId(roomId, winnerSlot);
  const forfeiterId = getPlayerUserId(roomId, forfeitingPlayerSlot);
  const publicWinnerId =
    channel.evaluation !== null && channel.players[winnerSlot]?.isAi
      ? `evaluation-opponent-${winnerSlot}`
      : winnerId;

  // Calculate game duration
  const durationSeconds = Math.floor((Date.now() - world.createdAt.getTime()) / 1000);
  const totalTurns = getGameState(roomId)?.turnNumber ?? 0;

  if (channel.evaluation) {
    await finalizeHumanEvaluationMatch({
      roomId,
      winnerId,
      winType: WinType.FORFEIT,
      totalTurns,
      durationSeconds,
    });
  } else {
    try {
      await db.insert(MatchResults).values({
        roomId,
        player0Id: world.player0UserId,
        player1Id: world.player1UserId,
        aiModelId: room.aiModelId,
        winnerId,
        winType: WinType.FORFEIT,
        totalTurns,
        durationSeconds,
      });
    } catch (error) {
      logger.error("Failed to store forfeit match result", { roomId, error });
    }
  }
  logger.info("Stored forfeit match result", {
    roomId,
    winnerId,
    forfeiterId,
    durationSeconds,
  });

  // Broadcast GAME_OVER to all players
  const gameOverMessage: GameOverMessage = {
    type: "GAME_OVER",
    winnerId: publicWinnerId,
    winnerSlot,
    winType: WinType.FORFEIT,
    reason: `Player ${forfeitingPlayerSlot} forfeited`,
  };
  broadcastToRoom(channel, gameOverMessage);

  if (!channel.evaluation) {
    await updateRoomStatus(roomId, RoomStatus.COMPLETED);
  }
  updateRoomChannelStatus(roomId, { status: RoomStatus.COMPLETED });
  logger.info("Updated room status to COMPLETED after forfeit", { roomId });

  await clearAiOpponentForRoom(roomId);

  // Clean up game world
  destroyGameWorld(roomId);

  // Remove room channel after a delay
  setTimeout(() => {
    removeRoomChannel(roomId);
    logger.debug("Removed room channel after forfeit", { roomId });
  }, 5000);
}
