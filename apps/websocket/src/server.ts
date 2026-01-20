import {
  App as UWebsocketApp,
  type HttpRequest,
  type HttpResponse,
  type WebSocketBehavior,
} from "uWebSockets.js";

import {
  COMPRESSION,
  MAX_PAYLOAD_LENGTH,
  PORT,
  TIMEOUT_SECONDS,
  type UserData,
} from "@/constants";
import logger from "@/logger";
import { WebSocketService } from "@/services/WebSocketService";
import { setEngineDebugLogging } from "@/engine/WorldManager";

const uWebSocketApp = UWebsocketApp();

const webSocketBehavior: WebSocketBehavior<UserData> = {
  /* Options */
  compression: COMPRESSION,
  maxPayloadLength: MAX_PAYLOAD_LENGTH,
  idleTimeout: TIMEOUT_SECONDS,
  sendPingsAutomatically: false,

  /* Handlers */
  upgrade: (res, req, context) =>
    WebSocketService.getInstance().handleWebSocketUpgrade(res, req, context),
  open: (ws) => WebSocketService.getInstance().handleWebSocketOpen(ws),
  message: (ws, message, isBinary) =>
    WebSocketService.getInstance().handleMessage(ws, message, isBinary),
  drain: (ws) => {
    const userData = ws.getUserData();
    logger.warn("WebSocket backpressure detected", {
      bufferedAmount: ws.getBufferedAmount(),
      userId: userData.isAnonymous ? null : userData.id,
    });
  },
  dropped: (ws, message, isBinary) =>
    WebSocketService.getInstance().handleDroppedMessage(ws, message, isBinary),
  close: async (ws, code, message) =>
    await WebSocketService.getInstance().handleCloseWebSocket(ws, code, message),
};

async function main(): Promise<void> {
  logger.info("Initializing WebSocket server...");

  // Enable C engine debug logging (always on for debugging action mask issues)
  setEngineDebugLogging(true);
  logger.info("Engine debug logging enabled");

  uWebSocketApp.ws<UserData>("/*", webSocketBehavior);

  uWebSocketApp.get("/health", (res: HttpResponse, _req: HttpRequest) => {
    res.cork(() => {
      res.writeStatus("200 OK").end("OK");
    });
  });

  uWebSocketApp.listen(PORT, (listenSocket) => {
    if (listenSocket) {
      logger.info("WebSocket server started", {
        port: PORT,
        healthCheck: `http://localhost:${PORT}/health`,
      });
    } else {
      logger.error("Failed to start WebSocket server", { port: PORT });
      process.exit(1);
    }
  });
}

main().catch((error: unknown) => {
  const logError = error instanceof Error ? error : undefined;
  const logMeta = error instanceof Error ? {} : { error: String(error) };
  logger.error("Fatal startup error", logError, logMeta);
  process.exit(1);
});
