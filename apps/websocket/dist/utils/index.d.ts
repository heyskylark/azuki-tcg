import type { WebSocket } from "uWebSockets.js";
import type { UserData } from "../constants";
/**
 * Safely send a JSON message to a WebSocket client
 */
export declare function sendJson<T>(ws: WebSocket<UserData>, data: T): boolean;
/**
 * Parse a binary or string message to JSON
 */
export declare function parseMessage<T>(message: ArrayBuffer, _isBinary: boolean): T | null;
//# sourceMappingURL=index.d.ts.map