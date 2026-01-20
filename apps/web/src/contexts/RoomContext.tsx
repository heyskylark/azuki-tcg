"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  type ReactNode,
} from "react";
import { useAuth } from "@/contexts/AuthContext";
import { authenticatedFetch } from "@/lib/api/authenticatedFetch";
import type {
  RoomStateMessage,
  ConnectionAckMessage,
  ErrorMessage,
  RoomClosedMessage,
  GameSnapshotMessage,
  GameLogBatchMessage,
} from "@tcg/backend-core/types/ws";

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

interface ActiveRoom {
  id: string;
  playerSlot: 0 | 1;
}

interface WebSocketMessage {
  type: string;
  [key: string]: unknown;
}

type GameMessage = GameSnapshotMessage | GameLogBatchMessage;

interface RoomContextValue {
  activeRoom: ActiveRoom | null;
  roomState: RoomStateMessage | null;
  connectionStatus: ConnectionStatus;
  error: string | null;
  lastMessage: WebSocketMessage | null;
  onGameMessage: (callback: (message: GameMessage) => void) => () => void;
  join: (roomId: string, password?: string) => Promise<boolean>;
  leave: () => void;
  close: () => void;
  startGame: () => void;
  send: (message: WebSocketMessage) => void;
  clearError: () => void;
  clearActiveRoom: () => void;
}

const RoomContext = createContext<RoomContextValue | null>(null);

interface RoomProviderProps {
  children: ReactNode;
}

interface JoinResponse {
  joinToken: string;
  playerSlot: 0 | 1;
  isNewJoin: boolean;
}

interface ActiveRoomData {
  id: string;
  status: string;
  player0Id: string | null;
  player1Id: string | null;
}

interface ActiveRoomResponse {
  room: ActiveRoomData | null;
}

const INACTIVE_ROOM_STATUSES = ["COMPLETED", "CLOSED", "ABORTED"];

export function RoomProvider({ children }: RoomProviderProps) {
  const { user, isLoading: authLoading } = useAuth();

  const [activeRoom, setActiveRoom] = useState<ActiveRoom | null>(null);
  const [roomState, setRoomState] = useState<RoomStateMessage | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("disconnected");
  const [error, setError] = useState<string | null>(null);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const hasCheckedActiveRoom = useRef(false);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 3;
  const gameMessageCallbacks = useRef<Set<(message: GameMessage) => void>>(new Set());

  // Check for active room on mount (after auth is loaded)
  useEffect(() => {
    if (authLoading || !user || hasCheckedActiveRoom.current) {
      return;
    }

    hasCheckedActiveRoom.current = true;

    const checkActiveRoom = async () => {
      try {
        const response = await authenticatedFetch("/api/users/rooms/active");
        if (response.ok) {
          const data: ActiveRoomResponse = await response.json();
          const room = data.room;
          if (room && !INACTIVE_ROOM_STATUSES.includes(room.status)) {
            // User has an active room, determine their slot and auto-reconnect
            const playerSlot = room.player0Id === user.id ? 0 : 1;
            await joinInternal(room.id, playerSlot);
          }
        }
      } catch {
        // No active room or error checking - that's fine
      }
    };

    checkActiveRoom();
  }, [authLoading, user]);

  // Reset state when user logs out
  useEffect(() => {
    if (!user && !authLoading) {
      disconnect();
      setActiveRoom(null);
      setRoomState(null);
      hasCheckedActiveRoom.current = false;
    }
  }, [user, authLoading]);

  const connectWebSocket = useCallback((joinToken: string, roomId: string, playerSlot: 0 | 1) => {
    if (socketRef.current) {
      socketRef.current.close();
    }

    setConnectionStatus("connecting");

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:3001";
    const ws = new WebSocket(`${wsUrl}?token=${joinToken}`);

    ws.onopen = () => {
      // Only update state if this socket is still the current one
      if (socketRef.current === ws) {
        setConnectionStatus("connected");
        setActiveRoom({ id: roomId, playerSlot });
        reconnectAttempts.current = 0;
      }
    };

    ws.onmessage = (event) => {
      // Only process messages if this socket is still the current one
      if (socketRef.current !== ws) {
        return;
      }
      try {
        const message = JSON.parse(event.data) as WebSocketMessage;
        setLastMessage(message);
        handleMessage(message);
      } catch {
        console.error("Failed to parse WebSocket message");
      }
    };

    ws.onerror = () => {
      // Only update state if this socket is still the current one
      if (socketRef.current === ws) {
        setConnectionStatus("error");
      }
    };

    ws.onclose = () => {
      // Only update state if this socket is still the current one
      // This prevents a closing old socket from clearing a newly created socket
      if (socketRef.current === ws) {
        setConnectionStatus("disconnected");
        socketRef.current = null;
      }
    };

    socketRef.current = ws;
  }, []);

  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case "CONNECTION_ACK": {
        const ack = message as unknown as ConnectionAckMessage;
        setActiveRoom((prev) => prev ? { ...prev, playerSlot: ack.playerSlot } : null);
        break;
      }

      case "ROOM_STATE": {
        const state = message as unknown as RoomStateMessage;
        setRoomState(state);

        // If room became inactive, clear active room
        if (INACTIVE_ROOM_STATUSES.includes(state.status)) {
          setActiveRoom(null);
          setRoomState(null);
        }
        break;
      }

      case "ROOM_CLOSED": {
        const closed = message as unknown as RoomClosedMessage;
        setError(`Room closed: ${closed.reason}`);
        setActiveRoom(null);
        setRoomState(null);
        break;
      }

      case "ERROR": {
        const err = message as unknown as ErrorMessage;
        setError(err.message);
        break;
      }

      case "GAME_SNAPSHOT":
      case "GAME_LOG_BATCH": {
        const gameMessage = message as unknown as GameMessage;
        for (const callback of gameMessageCallbacks.current) {
          callback(gameMessage);
        }
        break;
      }
    }
  }, []);

  const onGameMessage = useCallback((callback: (message: GameMessage) => void) => {
    gameMessageCallbacks.current.add(callback);
    return () => {
      gameMessageCallbacks.current.delete(callback);
    };
  }, []);

  const joinInternal = useCallback(async (roomId: string, expectedSlot?: 0 | 1, password?: string): Promise<boolean> => {
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
        setError(data.message || "Failed to join room");
        return false;
      }

      const data: JoinResponse = await response.json();
      connectWebSocket(data.joinToken, roomId, expectedSlot ?? data.playerSlot);
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to join room");
      return false;
    }
  }, [connectWebSocket]);

  const join = useCallback(async (roomId: string, password?: string): Promise<boolean> => {
    return joinInternal(roomId, undefined, password);
  }, [joinInternal]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
    setConnectionStatus("disconnected");
  }, []);

  const send = useCallback((message: WebSocketMessage) => {
    if (!socketRef.current) {
      console.error("[RoomContext] Cannot send message: WebSocket not initialized", {
        messageType: message.type,
      });
      return;
    }
    if (connectionStatus !== "connected") {
      console.error("[RoomContext] Cannot send message: not connected", {
        messageType: message.type,
        connectionStatus,
      });
      return;
    }
    socketRef.current.send(JSON.stringify(message));
  }, [connectionStatus]);

  const leave = useCallback(() => {
    send({ type: "LEAVE_ROOM" });
    // Server will close the connection, but we can clear state preemptively
    setActiveRoom(null);
    setRoomState(null);
  }, [send]);

  const close = useCallback(() => {
    send({ type: "CLOSE_ROOM" });
    // Server will close the connection, but we can clear state preemptively
    setActiveRoom(null);
    setRoomState(null);
  }, [send]);

  const startGame = useCallback(() => {
    send({ type: "START_GAME" });
  }, [send]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const clearActiveRoom = useCallback(() => {
    disconnect();
    setActiveRoom(null);
    setRoomState(null);
  }, [disconnect]);

  return (
    <RoomContext.Provider
      value={{
        activeRoom,
        roomState,
        connectionStatus,
        error,
        lastMessage,
        onGameMessage,
        join,
        leave,
        close,
        startGame,
        send,
        clearError,
        clearActiveRoom,
      }}
    >
      {children}
    </RoomContext.Provider>
  );
}

export function useRoom(): RoomContextValue {
  const context = useContext(RoomContext);
  if (!context) {
    throw new Error("useRoom must be used within a RoomProvider");
  }
  return context;
}
