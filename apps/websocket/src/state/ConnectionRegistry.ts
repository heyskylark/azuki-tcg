import type { WebSocket } from "uWebSockets.js";
import type { UserData } from "@/constants";
import type { ConnectionInfo } from "@/state/types";

const connections = new Map<WebSocket<UserData>, ConnectionInfo>();
const userConnections = new Map<string, WebSocket<UserData>>();

export function registerConnection(
  ws: WebSocket<UserData>,
  info: ConnectionInfo
): void {
  connections.set(ws, info);
  userConnections.set(info.userId, ws);
}

export function unregisterConnection(
  ws: WebSocket<UserData>
): ConnectionInfo | null {
  const info = connections.get(ws);
  if (!info) {
    return null;
  }

  connections.delete(ws);

  const currentWs = userConnections.get(info.userId);
  if (currentWs === ws) {
    userConnections.delete(info.userId);
  }

  return info;
}

export function getConnectionInfo(
  ws: WebSocket<UserData>
): ConnectionInfo | null {
  return connections.get(ws) ?? null;
}

export function getConnectionByUserId(
  userId: string
): WebSocket<UserData> | null {
  return userConnections.get(userId) ?? null;
}

export function getAllConnections(): Map<WebSocket<UserData>, ConnectionInfo> {
  return connections;
}
