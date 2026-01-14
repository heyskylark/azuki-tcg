import { RoomStatus } from "@tcg/backend-core/types";
import { updateRoomStatus, findRoomById } from "@tcg/backend-core/services/roomService";
import {
  loadDeckAsDefIds,
  deckAsDefIdsToDeckEntries,
} from "@tcg/backend-core/services/cardMapperService";
import { getRoomChannel, updateRoomChannelStatus, removeRoomChannel } from "@/state/RoomRegistry";
import {
  startDeckSelectionTimeout,
  startReadyCountdown,
  cancelReadyCountdown,
  cancelDeckSelectionTimeout,
  clearAllTimersForRoom,
} from "@/state/TimerManager";
import { broadcastRoomState, broadcastToRoom, sendToPlayer } from "@/utils/broadcast";
import { DECK_SELECTION_TIMEOUT_MS } from "@/constants";
import { createGameWorld } from "@/engine/WorldManager";
import { generateSnapshot } from "@/engine/snapshotGenerator";
import logger from "@/logger";

export async function transitionToDeckSelection(roomId: string): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot transition to DECK_SELECTION: channel not found", { roomId });
    return;
  }

  const deadline = new Date(Date.now() + DECK_SELECTION_TIMEOUT_MS);

  await updateRoomStatus(roomId, RoomStatus.DECK_SELECTION, {
    deckSelectionDeadline: deadline,
  });

  updateRoomChannelStatus(roomId, {
    status: RoomStatus.DECK_SELECTION,
    deckSelectionDeadline: deadline,
  });

  startDeckSelectionTimeout(roomId, deadline, async () => {
    await transitionToAborted(roomId, "Deck selection timeout");
  });

  broadcastRoomState(channel);

  logger.info("Room transitioned to DECK_SELECTION", { roomId, deadline });
}

export async function transitionToReadyCheck(roomId: string): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot transition to READY_CHECK: channel not found", { roomId });
    return;
  }

  await updateRoomStatus(roomId, RoomStatus.READY_CHECK);

  updateRoomChannelStatus(roomId, {
    status: RoomStatus.READY_CHECK,
    readyCountdownStartedAt: new Date(),
  });

  startReadyCountdown(roomId, async () => {
    await transitionToStarting(roomId);
  });

  broadcastRoomState(channel);

  logger.info("Room transitioned to READY_CHECK", { roomId });
}

export async function transitionToStarting(roomId: string): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot transition to STARTING: channel not found", { roomId });
    return;
  }

  cancelDeckSelectionTimeout(roomId);

  const rngSeed = Math.floor(Math.random() * 2147483647);

  await updateRoomStatus(roomId, RoomStatus.STARTING, {
    rngSeed,
  });

  updateRoomChannelStatus(roomId, {
    status: RoomStatus.STARTING,
    readyCountdownStartedAt: null,
  });

  broadcastRoomState(channel);

  logger.info("Room transitioned to STARTING", { roomId, rngSeed });

  // Immediately transition to IN_MATCH
  await transitionToInMatch(roomId, rngSeed);
}

export async function transitionToInMatch(roomId: string, rngSeed: number): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot transition to IN_MATCH: channel not found", { roomId });
    return;
  }

  // Get room data with player and deck IDs
  const room = await findRoomById(roomId);
  if (!room) {
    logger.error("Cannot transition to IN_MATCH: room not found", { roomId });
    await transitionToAborted(roomId, "Room data not found");
    return;
  }

  if (!room.player0Id || !room.player1Id) {
    logger.error("Cannot transition to IN_MATCH: missing players", { roomId });
    await transitionToAborted(roomId, "Missing players");
    return;
  }

  if (!room.player0DeckId || !room.player1DeckId) {
    logger.error("Cannot transition to IN_MATCH: missing decks", { roomId });
    await transitionToAborted(roomId, "Missing decks");
    return;
  }

  try {
    // Load decks as CardDefIds for the engine
    const player0Deck = await loadDeckAsDefIds(room.player0DeckId);
    const player1Deck = await loadDeckAsDefIds(room.player1DeckId);

    // Convert decks to DeckCardEntry format for engine initialization
    const player0DeckEntries = deckAsDefIdsToDeckEntries(player0Deck);
    const player1DeckEntries = deckAsDefIdsToDeckEntries(player1Deck);

    // Create the game world
    const world = createGameWorld(
      roomId,
      rngSeed,
      room.player0Id,
      player0DeckEntries,
      room.player1Id,
      player1DeckEntries
    );

    // Update room status to IN_MATCH
    await updateRoomStatus(roomId, RoomStatus.IN_MATCH);

    updateRoomChannelStatus(roomId, {
      status: RoomStatus.IN_MATCH,
    });

    logger.info("Room transitioned to IN_MATCH", {
      roomId,
      worldId: world.worldId,
    });

    // Send initial snapshots to each player
    for (const slot of [0, 1] as const) {
      const playerConnection = channel.players[slot];
      if (playerConnection?.ws && playerConnection.connected) {
        const snapshot = generateSnapshot(roomId, slot);
        if (snapshot) {
          sendToPlayer(channel, slot, snapshot);
        }
      }
    }
  } catch (error) {
    logger.error("Failed to create game world", { roomId, error });
    await transitionToAborted(roomId, "Failed to initialize game");
  }
}

export async function transitionToClosed(roomId: string, reason: string): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot transition to CLOSED: channel not found", { roomId });
    return;
  }

  clearAllTimersForRoom(roomId);

  await updateRoomStatus(roomId, RoomStatus.CLOSED);

  updateRoomChannelStatus(roomId, {
    status: RoomStatus.CLOSED,
  });

  // Send ROOM_CLOSED message to all connected players
  broadcastToRoom(channel, {
    type: "ROOM_CLOSED",
    reason,
  });

  removeRoomChannel(roomId);

  logger.info("Room transitioned to CLOSED", { roomId, reason });
}

export async function transitionToAborted(roomId: string, reason: string): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot transition to ABORTED: channel not found", { roomId });
    return;
  }

  clearAllTimersForRoom(roomId);

  await updateRoomStatus(roomId, RoomStatus.ABORTED);

  updateRoomChannelStatus(roomId, {
    status: RoomStatus.ABORTED,
  });

  broadcastToRoom(channel, {
    type: "ERROR",
    code: "ROOM_ABORTED",
    message: reason,
  });

  broadcastRoomState(channel);

  removeRoomChannel(roomId);

  logger.info("Room transitioned to ABORTED", { roomId, reason });
}

export async function revertToDeckSelection(roomId: string): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot revert to DECK_SELECTION: channel not found", { roomId });
    return;
  }

  cancelReadyCountdown(roomId);

  await updateRoomStatus(roomId, RoomStatus.DECK_SELECTION);

  updateRoomChannelStatus(roomId, {
    status: RoomStatus.DECK_SELECTION,
    readyCountdownStartedAt: null,
  });

  broadcastRoomState(channel);

  logger.info("Room reverted to DECK_SELECTION", { roomId });
}

export async function checkAndTransitionToReadyCheck(roomId: string): Promise<boolean> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    return false;
  }

  if (channel.status !== RoomStatus.DECK_SELECTION) {
    return false;
  }

  if (!channel.player0Ready || !channel.player1Ready) {
    return false;
  }

  await transitionToReadyCheck(roomId);
  return true;
}
