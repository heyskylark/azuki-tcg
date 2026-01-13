"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useWebSocket } from "@/contexts/WebSocketContext";
import { authenticatedFetch } from "@/lib/api/authenticatedFetch";
import type {
  RoomStateMessage,
  ConnectionAckMessage,
  ErrorMessage,
} from "@tcg/backend-core/types/ws";

type ConnectionState = "idle" | "joining" | "connecting" | "connected" | "error" | "inactive";

const INACTIVE_ROOM_STATUSES = ["COMPLETED", "CLOSED", "ABORTED"];

interface UseRoomConnectionOptions {
  roomId: string;
  userId: string;
  isInRoom: boolean;
  hasPassword: boolean;
  roomStatus: string;
}

interface UseRoomConnectionReturn {
  connectionState: ConnectionState;
  error: string | null;
  playerSlot: 0 | 1 | null;
  roomState: RoomStateMessage | null;
  needsPassword: boolean;
  join: (password?: string) => Promise<void>;
  clearError: () => void;
}

interface JoinResponse {
  joinToken: string;
  playerSlot: 0 | 1;
  isNewJoin: boolean;
}

export function useRoomConnection({
  roomId,
  userId,
  isInRoom,
  hasPassword,
  roomStatus,
}: UseRoomConnectionOptions): UseRoomConnectionReturn {
  const { status, connect, lastMessage } = useWebSocket();

  const isRoomInactive = INACTIVE_ROOM_STATUSES.includes(roomStatus);
  const [connectionState, setConnectionState] = useState<ConnectionState>(
    isRoomInactive ? "inactive" : "idle"
  );
  const [error, setError] = useState<string | null>(null);
  const [playerSlot, setPlayerSlot] = useState<0 | 1 | null>(null);
  const [roomState, setRoomState] = useState<RoomStateMessage | null>(null);
  const [needsPassword, setNeedsPassword] = useState(false);

  const hasAttemptedJoin = useRef(false);

  const join = useCallback(
    async (password?: string) => {
      setConnectionState("joining");
      setError(null);

      try {
        const body: { password?: string } = {};
        if (password) {
          body.password = password;
        }

        const response = await authenticatedFetch(`/api/rooms/${roomId}/join`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });

        if (!response.ok) {
          const data = await response.json();
          const errorMessage = data.message || "Failed to join room";

          // Handle specific error codes
          if (response.status === 403 && data.code === "ROOM_FULL") {
            setError("Room is full");
            setConnectionState("error");
            return;
          }

          if (response.status === 403 && data.code === "INVALID_ROOM_PASSWORD") {
            setError("Invalid password");
            setNeedsPassword(true);
            setConnectionState("idle");
            return;
          }

          throw new Error(errorMessage);
        }

        const data: JoinResponse = await response.json();
        setPlayerSlot(data.playerSlot);
        setNeedsPassword(false);
        setConnectionState("connecting");

        // Connect to WebSocket with the join token
        connect(data.joinToken);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to join room");
        setConnectionState("error");
      }
    },
    [roomId, connect]
  );

  // Auto-join on mount based on conditions
  useEffect(() => {
    if (hasAttemptedJoin.current) {
      return;
    }

    // Don't auto-join if we're already joining/connecting/connected
    // This prevents the effect from resetting state after a manual password join
    if (connectionState !== "idle") {
      return;
    }

    // Don't join inactive rooms
    if (isRoomInactive) {
      return;
    }

    if (isInRoom) {
      // User is already in the room, auto-join without password
      hasAttemptedJoin.current = true;
      join();
    } else if (!hasPassword) {
      // Room doesn't require password, auto-join
      hasAttemptedJoin.current = true;
      join();
    } else {
      // Room requires password and user is not in room
      setNeedsPassword(true);
    }
  }, [isInRoom, hasPassword, join, connectionState, isRoomInactive]);

  // Update connection state based on WebSocket status
  useEffect(() => {
    if (connectionState === "connecting") {
      if (status === "connected") {
        setConnectionState("connected");
      } else if (status === "error") {
        setError("WebSocket connection failed");
        setConnectionState("error");
      }
    }

    // Handle disconnection while connected
    if (connectionState === "connected" && status === "disconnected") {
      setConnectionState("error");
      setError("Connection lost");
    }
  }, [status, connectionState]);

  // Process incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage) {
      return;
    }

    const message = lastMessage as { type: string; [key: string]: unknown };

    switch (message.type) {
      case "CONNECTION_ACK": {
        const ack = message as unknown as ConnectionAckMessage;
        setPlayerSlot(ack.playerSlot);
        break;
      }

      case "ROOM_STATE": {
        const state = message as unknown as RoomStateMessage;
        setRoomState(state);
        break;
      }

      case "ERROR": {
        const err = message as unknown as ErrorMessage;
        setError(err.message);
        break;
      }
    }
  }, [lastMessage]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    connectionState,
    error,
    playerSlot,
    roomState,
    needsPassword,
    join,
    clearError,
  };
}
