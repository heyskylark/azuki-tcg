import type { WebSocket } from "uWebSockets.js";
import type { RoomStateMessage, PlayerInfo, ServerMessage } from "@tcg/backend-core/types/ws";
import type { UserData } from "@/constants";
import type { RoomChannelState } from "@/state/types";
import { sendJson } from "@/utils";
import { getReadyCountdownEnd } from "@/state/TimerManager";

export function buildPlayerInfo(channel: RoomChannelState, playerSlot: 0 | 1): PlayerInfo | null {
  const player = channel.players[playerSlot];
  if (!player) {
    return null;
  }

  const deckId = playerSlot === 0 ? channel.player0DeckId : channel.player1DeckId;
  const ready = playerSlot === 0 ? channel.player0Ready : channel.player1Ready;

  return {
    id: player.userId,
    username: player.username,
    isAi: player.isAi,
    deckSelected: deckId !== null,
    deckId,
    ready,
    connected: player.connected,
  };
}

export function buildRoomStateMessage(channel: RoomChannelState): RoomStateMessage {
  const readyCountdownEnd = getReadyCountdownEnd(channel.roomId);

  return {
    type: "ROOM_STATE",
    status: channel.status,
    players: [
      buildPlayerInfo(channel, 0),
      buildPlayerInfo(channel, 1),
    ],
    deckSelectionDeadline: channel.deckSelectionDeadline?.toISOString() ?? null,
    readyCountdownEnd: readyCountdownEnd?.toISOString() ?? null,
  };
}

export function broadcastToRoom<T extends ServerMessage>(
  channel: RoomChannelState,
  message: T
): void {
  for (const player of channel.players) {
    if (player?.ws && player.connected) {
      sendJson(player.ws, message);
    }
  }
}

export function sendToPlayer<T extends ServerMessage>(
  channel: RoomChannelState,
  playerSlot: 0 | 1,
  message: T
): boolean {
  const player = channel.players[playerSlot];
  if (!player?.ws || !player.connected) {
    return false;
  }
  return sendJson(player.ws, message);
}

export function broadcastRoomState(channel: RoomChannelState): void {
  const message = buildRoomStateMessage(channel);
  broadcastToRoom(channel, message);
}

export function getConnectedWebSockets(channel: RoomChannelState): WebSocket<UserData>[] {
  const sockets: WebSocket<UserData>[] = [];
  for (const player of channel.players) {
    if (player?.ws && player.connected) {
      sockets.push(player.ws);
    }
  }
  return sockets;
}
