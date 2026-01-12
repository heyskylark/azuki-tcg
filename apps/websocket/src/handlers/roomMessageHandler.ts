import type { WebSocket } from "uWebSockets.js";
import { RoomStatus } from "@tcg/backend-core/types";
import {
  updatePlayerDeck,
  updatePlayerReady,
  verifyDeckOwnership,
} from "@tcg/backend-core/services/roomService";
import type { SelectDeckMessage, ReadyMessage } from "@tcg/backend-core/types/ws";

import type { UserData } from "@/constants";
import type { ConnectionInfo } from "@/state/types";
import { getRoomChannel, updateRoomChannelStatus } from "@/state/RoomRegistry";
import { cancelReadyCountdown } from "@/state/TimerManager";
import { sendJson } from "@/utils";
import { broadcastRoomState } from "@/utils/broadcast";
import {
  checkAndTransitionToReadyCheck,
  revertToDeckSelection,
} from "@/handlers/stateTransitionHandler";
import logger from "@/logger";

export async function handleSelectDeck(
  ws: WebSocket<UserData>,
  message: SelectDeckMessage,
  connectionInfo: ConnectionInfo
): Promise<void> {
  const { roomId, playerSlot, userId } = connectionInfo;
  const { deckId } = message;

  const channel = getRoomChannel(roomId);
  if (!channel) {
    sendJson(ws, { type: "ERROR", code: "ROOM_NOT_FOUND", message: "Room channel not found" });
    return;
  }

  if (channel.status !== RoomStatus.DECK_SELECTION) {
    sendJson(ws, {
      type: "ERROR",
      code: "INVALID_STATE",
      message: "Deck selection is not allowed in current room state",
    });
    return;
  }

  const isReady = playerSlot === 0 ? channel.player0Ready : channel.player1Ready;
  if (isReady) {
    sendJson(ws, {
      type: "ERROR",
      code: "ALREADY_READY",
      message: "Cannot change deck while ready. Unready first.",
    });
    return;
  }

  const ownsDecks = await verifyDeckOwnership(userId, deckId);
  if (!ownsDecks) {
    sendJson(ws, {
      type: "ERROR",
      code: "DECK_NOT_OWNED",
      message: "You do not own this deck",
    });
    return;
  }

  await updatePlayerDeck(roomId, playerSlot, deckId);

  if (playerSlot === 0) {
    updateRoomChannelStatus(roomId, { player0DeckId: deckId });
  } else {
    updateRoomChannelStatus(roomId, { player1DeckId: deckId });
  }

  logger.info("Player selected deck", { roomId, playerSlot, deckId });

  broadcastRoomState(channel);
}

export async function handleReady(
  ws: WebSocket<UserData>,
  message: ReadyMessage,
  connectionInfo: ConnectionInfo
): Promise<void> {
  const { roomId, playerSlot } = connectionInfo;
  const { ready } = message;

  const channel = getRoomChannel(roomId);
  if (!channel) {
    sendJson(ws, { type: "ERROR", code: "ROOM_NOT_FOUND", message: "Room channel not found" });
    return;
  }

  if (channel.status !== RoomStatus.DECK_SELECTION && channel.status !== RoomStatus.READY_CHECK) {
    sendJson(ws, {
      type: "ERROR",
      code: "INVALID_STATE",
      message: "Ready is not allowed in current room state",
    });
    return;
  }

  if (ready) {
    const deckId = playerSlot === 0 ? channel.player0DeckId : channel.player1DeckId;
    if (!deckId) {
      sendJson(ws, {
        type: "ERROR",
        code: "DECK_NOT_SELECTED",
        message: "You must select a deck before readying up",
      });
      return;
    }

    await updatePlayerReady(roomId, playerSlot, true);

    if (playerSlot === 0) {
      updateRoomChannelStatus(roomId, { player0Ready: true });
    } else {
      updateRoomChannelStatus(roomId, { player1Ready: true });
    }

    logger.info("Player ready", { roomId, playerSlot });

    const bothReady = channel.player0Ready && channel.player1Ready;
    if (bothReady && channel.status === RoomStatus.DECK_SELECTION) {
      await checkAndTransitionToReadyCheck(roomId);
    } else {
      broadcastRoomState(channel);
    }
  } else {
    const wasInReadyCheck = channel.status === RoomStatus.READY_CHECK;

    await updatePlayerReady(roomId, playerSlot, false);

    if (playerSlot === 0) {
      updateRoomChannelStatus(roomId, { player0Ready: false });
    } else {
      updateRoomChannelStatus(roomId, { player1Ready: false });
    }

    logger.info("Player unready", { roomId, playerSlot });

    if (wasInReadyCheck) {
      await revertToDeckSelection(roomId);
    } else {
      broadcastRoomState(channel);
    }
  }
}
