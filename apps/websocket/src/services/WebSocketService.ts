import type {
  HttpRequest,
  HttpResponse,
  us_socket_context_t,
  WebSocket,
} from "uWebSockets.js";

import type { UserData } from "@/constants";
import logger from "@/logger";
import { parseMessage, sendJson } from "@/utils";

export class WebSocketService {
  private static instance: WebSocketService;

  private constructor() {}

  public static getInstance(): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService();
    }
    return WebSocketService.instance;
  }

  public handleWebSocketUpgrade(
    res: HttpResponse,
    req: HttpRequest,
    context: us_socket_context_t
  ): void {
    const url = req.getUrl();
    const secWebSocketKey = req.getHeader("sec-websocket-key");
    const secWebSocketProtocol = req.getHeader("sec-websocket-protocol");
    const secWebSocketExtensions = req.getHeader("sec-websocket-extensions");

    // TODO: Validate join token from query params
    // const query = req.getQuery();

    res.upgrade<UserData>(
      {
        id: "", // Will be set after authentication
        isAnonymous: true,
      },
      secWebSocketKey,
      secWebSocketProtocol,
      secWebSocketExtensions,
      context
    );

    logger.debug("WebSocket upgrade requested", { url });
  }

  public handleWebSocketOpen(ws: WebSocket<UserData>): void {
    const userData = ws.getUserData();
    logger.info("WebSocket connection opened", {
      userId: userData.isAnonymous ? null : userData.id,
    });

    // Send connection acknowledgment
    sendJson(ws, {
      type: "CONNECTION_ACK",
      message: "Connected to Azuki TCG WebSocket server",
    });
  }

  public handleMessage(
    ws: WebSocket<UserData>,
    message: ArrayBuffer,
    isBinary: boolean
  ): void {
    const userData = ws.getUserData();
    const parsed = parseMessage<{ type: string }>(message, isBinary);

    if (!parsed) {
      logger.warn("Failed to parse WebSocket message", {
        userId: userData.isAnonymous ? null : userData.id,
      });
      sendJson(ws, { type: "ERROR", code: "PARSE_ERROR", message: "Invalid message format" });
      return;
    }

    logger.debug("Received message", {
      type: parsed.type,
      userId: userData.isAnonymous ? null : userData.id,
    });

    // Handle message types
    switch (parsed.type) {
      case "PING":
        sendJson(ws, { type: "PONG" });
        break;
      default:
        logger.warn("Unknown message type", { type: parsed.type });
        sendJson(ws, {
          type: "ERROR",
          code: "UNKNOWN_MESSAGE_TYPE",
          message: `Unknown message type: ${parsed.type}`,
        });
    }
  }

  public handleDroppedMessage(
    ws: WebSocket<UserData>,
    _message: ArrayBuffer,
    _isBinary: boolean
  ): void {
    const userData = ws.getUserData();
    logger.warn("WebSocket message dropped due to backpressure", {
      userId: userData.isAnonymous ? null : userData.id,
    });
  }

  public async handleCloseWebSocket(
    ws: WebSocket<UserData>,
    code: number,
    _message: ArrayBuffer
  ): Promise<void> {
    const userData = ws.getUserData();
    logger.info("WebSocket connection closed", {
      userId: userData.isAnonymous ? null : userData.id,
      code,
    });

    // TODO: Handle cleanup (remove from room, notify others, etc.)
  }
}
