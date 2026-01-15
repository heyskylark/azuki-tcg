/**
 * WorldManager handles multi-game routing between room IDs and engine world IDs.
 * Manages the lifecycle of game worlds and provides access control.
 */

import { getNativeBinding, type ActionResult, type ActionTuple, type DeckCardEntry, type ObservationData, type StateContext, type GameLog } from "@/engine/EngineBinding";
import logger from "@/logger";

export interface ActiveWorld {
  worldId: string;
  roomId: string;
  player0UserId: string;
  player1UserId: string;
  createdAt: Date;
  batchNumber: number;
}

// In-memory map of roomId -> ActiveWorld
const activeWorlds = new Map<string, ActiveWorld>();

// Reverse lookup: worldId -> roomId
const worldIdToRoomId = new Map<string, string>();

/**
 * Create a new game world for a room.
 */
export function createGameWorld(
  roomId: string,
  seed: number,
  player0UserId: string,
  player0Deck: DeckCardEntry[],
  player1UserId: string,
  player1Deck: DeckCardEntry[]
): ActiveWorld {
  // Check if a world already exists for this room
  if (activeWorlds.has(roomId)) {
    throw new Error(`World already exists for room ${roomId}`);
  }

  const binding = getNativeBinding();
  const result = binding.createWorldWithDecks(seed, player0Deck, player1Deck);

  if (!result.success) {
    const errorMsg = result.error ?? "Unknown error";
    logger.error("Engine failed to create world", { roomId, error: errorMsg });
    throw new Error(`Failed to create world for room ${roomId}: ${errorMsg}`);
  }

  const world: ActiveWorld = {
    worldId: result.worldId,
    roomId,
    player0UserId,
    player1UserId,
    createdAt: new Date(),
    batchNumber: 0,
  };

  activeWorlds.set(roomId, world);
  worldIdToRoomId.set(result.worldId, roomId);

  logger.info("Created game world", { roomId, worldId: result.worldId });

  return world;
}

/**
 * Destroy a game world for a room.
 */
export function destroyGameWorld(roomId: string): void {
  const world = activeWorlds.get(roomId);
  if (!world) {
    logger.warn("Attempted to destroy non-existent world", { roomId });
    return;
  }

  const binding = getNativeBinding();
  binding.destroyWorld(world.worldId);

  worldIdToRoomId.delete(world.worldId);
  activeWorlds.delete(roomId);

  logger.info("Destroyed game world", { roomId, worldId: world.worldId });
}

/**
 * Get the active world for a room.
 */
export function getWorldByRoomId(roomId: string): ActiveWorld | null {
  return activeWorlds.get(roomId) ?? null;
}

/**
 * Get the room ID for a world ID.
 */
export function getRoomIdByWorldId(worldId: string): string | null {
  return worldIdToRoomId.get(worldId) ?? null;
}

/**
 * Convert a user ID to player slot (0 or 1).
 * Returns -1 if user is not a player in the world.
 */
export function getUserPlayerSlot(roomId: string, userId: string): 0 | 1 | -1 {
  const world = activeWorlds.get(roomId);
  if (!world) {
    return -1;
  }
  if (world.player0UserId === userId) {
    return 0;
  }
  if (world.player1UserId === userId) {
    return 1;
  }
  return -1;
}

/**
 * Get the user ID for a player slot.
 */
export function getPlayerUserId(roomId: string, playerSlot: 0 | 1): string | null {
  const world = activeWorlds.get(roomId);
  if (!world) {
    return null;
  }
  return playerSlot === 0 ? world.player0UserId : world.player1UserId;
}

export type SubmitActionResult =
  | ActionResult
  | { error: string; code: "NOT_FOUND" | "NOT_YOUR_TURN" | "NOT_AWAITING_ACTION" | "INVALID_ACTION" };

/**
 * Submit a player action to the game world.
 * Validates that it's the player's turn and the game is expecting input.
 */
export function submitPlayerAction(
  roomId: string,
  userId: string,
  action: ActionTuple
): SubmitActionResult {
  const world = activeWorlds.get(roomId);
  if (!world) {
    return { error: "World not found", code: "NOT_FOUND" };
  }

  const playerSlot = getUserPlayerSlot(roomId, userId);
  if (playerSlot === -1) {
    return { error: "User is not a player in this game", code: "NOT_FOUND" };
  }

  const binding = getNativeBinding();

  // Check if it's this player's turn
  const activePlayer = binding.getActivePlayer(world.worldId);
  if (activePlayer !== playerSlot) {
    return { error: "Not your turn", code: "NOT_YOUR_TURN" };
  }

  // Check if the game is expecting input
  if (!binding.requiresAction(world.worldId)) {
    return { error: "Game is not expecting player input", code: "NOT_AWAITING_ACTION" };
  }

  // Submit the action
  const result = binding.submitAction(world.worldId, playerSlot, action);

  if (!result.success && result.invalid) {
    return { error: result.error ?? "Invalid action", code: "INVALID_ACTION" };
  }

  return result;
}

/**
 * Get the observation for a player.
 */
export function getPlayerObservation(roomId: string, userId: string): ObservationData | null {
  const world = activeWorlds.get(roomId);
  if (!world) {
    return null;
  }

  const playerSlot = getUserPlayerSlot(roomId, userId);
  if (playerSlot === -1) {
    return null;
  }

  const binding = getNativeBinding();
  return binding.getObservation(world.worldId, playerSlot);
}

/**
 * Get the observation for a player by slot.
 */
export function getPlayerObservationBySlot(roomId: string, playerSlot: 0 | 1): ObservationData | null {
  const world = activeWorlds.get(roomId);
  if (!world) {
    return null;
  }

  const binding = getNativeBinding();
  return binding.getObservation(world.worldId, playerSlot);
}

/**
 * Get the game state for a room.
 */
export function getGameState(roomId: string): StateContext | null {
  const world = activeWorlds.get(roomId);
  if (!world) {
    return null;
  }

  const binding = getNativeBinding();
  return binding.getGameState(world.worldId);
}

/**
 * Get the game logs for a room.
 */
export function getGameLogs(roomId: string): GameLog[] {
  const world = activeWorlds.get(roomId);
  if (!world) {
    return [];
  }

  const binding = getNativeBinding();
  return binding.getGameLogs(world.worldId);
}

/**
 * Check if the game requires action from a player.
 */
export function requiresAction(roomId: string): boolean {
  const world = activeWorlds.get(roomId);
  if (!world) {
    return false;
  }

  const binding = getNativeBinding();
  return binding.requiresAction(world.worldId);
}

/**
 * Check if the game is over.
 */
export function isGameOver(roomId: string): boolean {
  const world = activeWorlds.get(roomId);
  if (!world) {
    return true;
  }

  const binding = getNativeBinding();
  return binding.isGameOver(world.worldId);
}

/**
 * Get the active player for a room.
 * Returns -1 if game is over or world not found.
 */
export function getActivePlayer(roomId: string): 0 | 1 | -1 {
  const world = activeWorlds.get(roomId);
  if (!world) {
    return -1;
  }

  const binding = getNativeBinding();
  const player = binding.getActivePlayer(world.worldId);
  return player as 0 | 1 | -1;
}

/**
 * Increment and return the batch number for log storage.
 */
export function incrementBatchNumber(roomId: string): number {
  const world = activeWorlds.get(roomId);
  if (!world) {
    return 0;
  }
  world.batchNumber++;
  return world.batchNumber;
}

/**
 * Get all active world IDs (for debugging/monitoring).
 */
export function getActiveWorldCount(): number {
  return activeWorlds.size;
}

/**
 * Destroy all worlds (for cleanup on server shutdown).
 */
export function destroyAllWorlds(): void {
  const roomIds = Array.from(activeWorlds.keys());
  for (const roomId of roomIds) {
    destroyGameWorld(roomId);
  }
  logger.info("Destroyed all game worlds", { count: roomIds.length });
}
