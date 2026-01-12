"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

interface WebSocketMessage {
  type: string;
  [key: string]: unknown;
}

interface WebSocketContextValue {
  status: ConnectionStatus;
  connect: (joinToken: string) => void;
  disconnect: () => void;
  send: (message: WebSocketMessage) => void;
  lastMessage: WebSocketMessage | null;
}

const WebSocketContext = createContext<WebSocketContextValue | null>(null);

interface WebSocketProviderProps {
  children: ReactNode;
}

export function WebSocketProvider({ children }: WebSocketProviderProps) {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  const connect = useCallback(
    (joinToken: string) => {
      if (socket) {
        socket.close();
      }

      setStatus("connecting");

      const wsUrl =
        process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:3001";
      const ws = new WebSocket(`${wsUrl}?token=${joinToken}`);

      ws.onopen = () => {
        setStatus("connected");
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;
          setLastMessage(message);
        } catch {
          console.error("Failed to parse WebSocket message");
        }
      };

      ws.onerror = () => {
        setStatus("error");
      };

      ws.onclose = () => {
        setStatus("disconnected");
        setSocket(null);
      };

      setSocket(ws);
    },
    [socket]
  );

  const disconnect = useCallback(() => {
    if (socket) {
      socket.close();
      setSocket(null);
      setStatus("disconnected");
    }
  }, [socket]);

  const send = useCallback(
    (message: WebSocketMessage) => {
      if (socket && status === "connected") {
        socket.send(JSON.stringify(message));
      }
    },
    [socket, status]
  );

  useEffect(() => {
    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, [socket]);

  return (
    <WebSocketContext.Provider
      value={{ status, connect, disconnect, send, lastMessage }}
    >
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket(): WebSocketContextValue {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error("useWebSocket must be used within a WebSocketProvider");
  }
  return context;
}
