import type { WebSocket } from "uWebSockets.js";

import type { UserData } from "@/constants";

/**
 * Safely send a JSON message to a WebSocket client
 */
export function sendJson<T>(ws: WebSocket<UserData>, data: T): boolean {
  try {
    const message = JSON.stringify(data);
    return ws.send(message, false) !== 0;
  } catch {
    return false;
  }
}

/**
 * Parse a binary or string message to JSON
 */
export function parseMessage<T>(
  message: ArrayBuffer,
  _isBinary: boolean
): T | null {
  try {
    const text = Buffer.from(message).toString("utf-8");
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}
