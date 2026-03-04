import type { RoomData } from "@tcg/backend-core/services/roomService";
import type { RoomChannelState, PlayerConnection } from "@/state/types";

const roomChannels = new Map<string, RoomChannelState>();

export function createRoomChannel(
  roomId: string,
  roomData: RoomData,
  player0Username: string | null,
  player1Username: string | null,
  player0IsAi = false,
  player1IsAi = false,
  player0ModelKey: string | null = null,
  player1ModelKey: string | null = null
): RoomChannelState {
  const player0: PlayerConnection | null = roomData.player0Id
    ? {
        ws: null,
        userId: roomData.player0Id,
        username: player0Username ?? "Unknown",
        isAi: player0IsAi,
        modelKey: player0ModelKey,
        playerSlot: 0,
        connected: player0IsAi,
        disconnectedAt: null,
      }
    : null;

  const player1: PlayerConnection | null = roomData.player1Id
    ? {
        ws: null,
        userId: roomData.player1Id,
        username: player1Username ?? "Unknown",
        isAi: player1IsAi,
        modelKey: player1ModelKey,
        playerSlot: 1,
        connected: player1IsAi,
        disconnectedAt: null,
      }
    : null;

  const channel: RoomChannelState = {
    roomId,
    status: roomData.status,
    players: [player0, player1],
    player0DeckId: roomData.player0DeckId,
    player1DeckId: roomData.player1DeckId,
    player0Ready: roomData.player0Ready,
    player1Ready: roomData.player1Ready,
    deckSelectionDeadline: roomData.deckSelectionDeadline,
    readyCountdownStartedAt: null,
  };

  roomChannels.set(roomId, channel);
  return channel;
}

export function getRoomChannel(roomId: string): RoomChannelState | null {
  return roomChannels.get(roomId) ?? null;
}

export function getOrCreateRoomChannel(
  roomId: string,
  roomData: RoomData,
  player0Username: string | null,
  player1Username: string | null,
  player0IsAi = false,
  player1IsAi = false,
  player0ModelKey: string | null = null,
  player1ModelKey: string | null = null
): RoomChannelState {
  const existing = roomChannels.get(roomId);
  if (existing) {
    return existing;
  }

  return createRoomChannel(
    roomId,
    roomData,
    player0Username,
    player1Username,
    player0IsAi,
    player1IsAi,
    player0ModelKey,
    player1ModelKey
  );
}

export function removeRoomChannel(roomId: string): boolean {
  return roomChannels.delete(roomId);
}

export function getAllRoomChannels(): Map<string, RoomChannelState> {
  return roomChannels;
}

export function updateRoomChannelStatus(
  roomId: string,
  updates: Partial<RoomChannelState>
): RoomChannelState | null {
  const channel = roomChannels.get(roomId);
  if (!channel) {
    return null;
  }

  Object.assign(channel, updates);
  return channel;
}
