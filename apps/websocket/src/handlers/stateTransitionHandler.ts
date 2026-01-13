import { RoomStatus } from "@tcg/backend-core/types";
import { updateRoomStatus } from "@tcg/backend-core/services/roomService";
import { getRoomChannel, updateRoomChannelStatus, removeRoomChannel } from "@/state/RoomRegistry";
import {
  startDeckSelectionTimeout,
  startReadyCountdown,
  cancelReadyCountdown,
  cancelDeckSelectionTimeout,
  clearAllTimersForRoom,
} from "@/state/TimerManager";
import { broadcastRoomState, broadcastToRoom } from "@/utils/broadcast";
import { DECK_SELECTION_TIMEOUT_MS } from "@/constants";
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
