import type { WebSocket } from "uWebSockets.js";
import { RoomStatus } from "@tcg/backend-core/types";
import {
  updatePlayerDeck,
  updatePlayerReady,
  verifyDeckOwnership,
  removePlayer1FromRoom,
} from "@tcg/backend-core/services/roomService";
import type { SelectDeckMessage, ReadyMessage } from "@tcg/backend-core/types/ws";

import type { UserData } from "@/constants";
import type { ConnectionInfo } from "@/state/types";
import { getRoomChannel, updateRoomChannelStatus } from "@/state/RoomRegistry";
import { cancelReadyCountdown } from "@/state/TimerManager";
import { sendJson } from "@/utils";
import { broadcastRoomState, getConnectedWebSockets } from "@/utils/broadcast";
import {
  checkAndTransitionToReadyCheck,
  revertToDeckSelection,
  transitionToDeckSelection,
  transitionToAborted,
  transitionToClosed,
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

export async function handleLeaveRoom(
  ws: WebSocket<UserData>,
  connectionInfo: ConnectionInfo
): Promise<void> {
  const { roomId, playerSlot } = connectionInfo;

  const channel = getRoomChannel(roomId);
  if (!channel) {
    sendJson(ws, { type: "ERROR", code: "ROOM_NOT_FOUND", message: "Room channel not found" });
    return;
  }

  // If player 1 leaves during WAITING_FOR_PLAYERS, just remove them from the room
  if (channel.status === RoomStatus.WAITING_FOR_PLAYERS && playerSlot === 1) {
    logger.info("Player 1 leaving room during WAITING_FOR_PLAYERS", { roomId });

    // Remove player 1 from database
    await removePlayer1FromRoom(roomId);

    // Remove from channel
    channel.players[1] = null;

    // Broadcast updated state to remaining player
    broadcastRoomState(channel);

    // Close the leaving player's WebSocket
    ws.close();

    logger.info("Player 1 left room", { roomId });
    return;
  }

  // For all other cases (owner leaving, or leaving during active phases), abort the room
  logger.info("Player leaving room, aborting", { roomId, playerSlot });
  await transitionToAborted(roomId, `Player ${playerSlot} left the room`);

  // Close all WebSockets
  const sockets = getConnectedWebSockets(channel);
  for (const socket of sockets) {
    socket.close();
  }
}

export async function handleCloseRoom(
  ws: WebSocket<UserData>,
  connectionInfo: ConnectionInfo
): Promise<void> {
  const { roomId, playerSlot } = connectionInfo;

  // Only owner (slot 0) can close the room
  if (playerSlot !== 0) {
    sendJson(ws, {
      type: "ERROR",
      code: "NOT_OWNER",
      message: "Only the room owner can close the room",
    });
    return;
  }

  const channel = getRoomChannel(roomId);
  if (!channel) {
    sendJson(ws, { type: "ERROR", code: "ROOM_NOT_FOUND", message: "Room channel not found" });
    return;
  }

  logger.info("Owner closing room", { roomId });

  // Transition to CLOSED - this broadcasts ROOM_CLOSED to all players
  await transitionToClosed(roomId, "Room closed by owner");

  // Close all WebSockets
  const sockets = getConnectedWebSockets(channel);
  for (const socket of sockets) {
    socket.close();
  }
}

export async function handleStartGame(
  ws: WebSocket<UserData>,
  connectionInfo: ConnectionInfo
): Promise<void> {
  const { roomId, playerSlot } = connectionInfo;

  // Only owner (slot 0) can start the game
  if (playerSlot !== 0) {
    sendJson(ws, {
      type: "ERROR",
      code: "NOT_OWNER",
      message: "Only the room owner can start the game",
    });
    return;
  }

  const channel = getRoomChannel(roomId);
  if (!channel) {
    sendJson(ws, { type: "ERROR", code: "ROOM_NOT_FOUND", message: "Room channel not found" });
    return;
  }

  // Only valid in WAITING_FOR_PLAYERS status
  if (channel.status !== RoomStatus.WAITING_FOR_PLAYERS) {
    sendJson(ws, {
      type: "ERROR",
      code: "INVALID_STATE",
      message: "Game can only be started in WAITING_FOR_PLAYERS state",
    });
    return;
  }

  // Both players must be connected
  const player0 = channel.players[0];
  const player1 = channel.players[1];

  if (!player0?.connected || !player1?.connected) {
    sendJson(ws, {
      type: "ERROR",
      code: "PLAYERS_NOT_READY",
      message: "Both players must be connected to start the game",
    });
    return;
  }

  logger.info("Owner starting game", { roomId });

  // Transition to DECK_SELECTION
  await transitionToDeckSelection(roomId);
}
