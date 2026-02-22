/**
 * Handler for game over logic.
 * Stores match results, broadcasts game over, and cleans up resources.
 */

import db from "@tcg/backend-core/database";
import { MatchResults } from "@tcg/backend-core/drizzle/schemas/match_results";
import { RoomStatus, WinType } from "@tcg/backend-core/types";
import type { GameOverMessage } from "@tcg/backend-core/types/ws";
import { updateRoomStatus } from "@tcg/backend-core/services/roomService";
import { getRoomChannel, removeRoomChannel, updateRoomChannelStatus } from "@/state/RoomRegistry";
import { getWorldByRoomId, destroyGameWorld, getPlayerUserId } from "@/engine/WorldManager";
import { clearAiOpponentForRoom } from "@/engine/aiOpponentService";
import { broadcastToRoom } from "@/utils/broadcast";
import logger from "@/logger";
import type { GameEndReason, StateContext } from "@/engine/types";

export interface GameOverResult {
  gameOver: boolean;
  winner: number | null;
  stateContext: StateContext;
}

/**
 * Map C engine GameEndReason to WinType.
 */
function mapGameEndReasonToWinType(reason: GameEndReason): WinType {
  switch (reason) {
    case "LEADER_DEFEATED":
      return WinType.WIN;
    case "DECK_OUT":
      return WinType.WIN;
    case "CONCEDE":
      return WinType.FORFEIT;
    default:
      return WinType.WIN;
  }
}

/**
 * Handle game over for a room.
 * Stores match result, broadcasts game over message, and cleans up.
 */
export async function handleGameOver(
  roomId: string,
  result: GameOverResult
): Promise<void> {
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

  // Determine winner info
  const winnerSlot = result.winner as 0 | 1 | null;
  const winnerId = winnerSlot !== null ? getPlayerUserId(roomId, winnerSlot) : null;

  // Calculate game duration
  const durationSeconds = Math.floor(
    (Date.now() - world.createdAt.getTime()) / 1000
  );

  // Store match result
  try {
    await db.insert(MatchResults).values({
      roomId,
      player0Id: world.player0UserId,
      player1Id: world.player1UserId,
      winnerId,
      winType: WinType.WIN, // Default to WIN, could be FORFEIT for concede
      totalTurns: result.stateContext.turnNumber,
      durationSeconds,
    });
    logger.info("Stored match result", {
      roomId,
      winnerId,
      totalTurns: result.stateContext.turnNumber,
      durationSeconds,
    });
  } catch (error) {
    logger.error("Failed to store match result", { roomId, error });
  }

  // Broadcast GAME_OVER to all players
  const gameOverMessage: GameOverMessage = {
    type: "GAME_OVER",
    winnerId,
    winnerSlot,
    winType: WinType.WIN,
    reason: getGameOverReason(result),
  };
  broadcastToRoom(channel, gameOverMessage);

  // Update room status to COMPLETED
  try {
    await updateRoomStatus(roomId, RoomStatus.COMPLETED);
    updateRoomChannelStatus(roomId, { status: RoomStatus.COMPLETED });
    logger.info("Updated room status to COMPLETED", { roomId });
  } catch (error) {
    logger.error("Failed to update room status", { roomId, error });
  }

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
export async function handleForfeit(
  roomId: string,
  forfeitingPlayerSlot: 0 | 1
): Promise<void> {
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

  // Winner is the opponent
  const winnerSlot = forfeitingPlayerSlot === 0 ? 1 : 0;
  const winnerId = getPlayerUserId(roomId, winnerSlot);
  const forfeiterId = getPlayerUserId(roomId, forfeitingPlayerSlot);

  // Calculate game duration
  const durationSeconds = Math.floor(
    (Date.now() - world.createdAt.getTime()) / 1000
  );

  // Store match result
  try {
    await db.insert(MatchResults).values({
      roomId,
      player0Id: world.player0UserId,
      player1Id: world.player1UserId,
      winnerId,
      winType: WinType.FORFEIT,
      totalTurns: 0, // We don't track turn number for forfeits
      durationSeconds,
    });
    logger.info("Stored forfeit match result", {
      roomId,
      winnerId,
      forfeiterId,
      durationSeconds,
    });
  } catch (error) {
    logger.error("Failed to store forfeit match result", { roomId, error });
  }

  // Broadcast GAME_OVER to all players
  const gameOverMessage: GameOverMessage = {
    type: "GAME_OVER",
    winnerId,
    winnerSlot,
    winType: WinType.FORFEIT,
    reason: `Player ${forfeitingPlayerSlot} forfeited`,
  };
  broadcastToRoom(channel, gameOverMessage);

  // Update room status to COMPLETED
  try {
    await updateRoomStatus(roomId, RoomStatus.COMPLETED);
    updateRoomChannelStatus(roomId, { status: RoomStatus.COMPLETED });
    logger.info("Updated room status to COMPLETED after forfeit", { roomId });
  } catch (error) {
    logger.error("Failed to update room status after forfeit", { roomId, error });
  }

  await clearAiOpponentForRoom(roomId);

  // Clean up game world
  destroyGameWorld(roomId);

  // Remove room channel after a delay
  setTimeout(() => {
    removeRoomChannel(roomId);
    logger.debug("Removed room channel after forfeit", { roomId });
  }, 5000);
}
