import { App as UWebsocketApp, } from "uWebSockets.js";
import { COMPRESSION, MAX_PAYLOAD_LENGTH, PORT, TIMEOUT_SECONDS, } from "./constants";
import logger from "./logger";
import { WebSocketService } from "./services";
const uWebSocketApp = UWebsocketApp();
const webSocketBehavior = {
    /* Options */
    compression: COMPRESSION,
    maxPayloadLength: MAX_PAYLOAD_LENGTH,
    idleTimeout: TIMEOUT_SECONDS,
    sendPingsAutomatically: false,
    /* Handlers */
    upgrade: (res, req, context) => WebSocketService.getInstance().handleWebSocketUpgrade(res, req, context),
    open: (ws) => WebSocketService.getInstance().handleWebSocketOpen(ws),
    message: (ws, message, isBinary) => WebSocketService.getInstance().handleMessage(ws, message, isBinary),
    drain: (ws) => {
        const userData = ws.getUserData();
        logger.warn("WebSocket backpressure detected", {
            bufferedAmount: ws.getBufferedAmount(),
            userId: userData.isAnonymous ? null : userData.id,
        });
    },
    dropped: (ws, message, isBinary) => WebSocketService.getInstance().handleDroppedMessage(ws, message, isBinary),
    close: async (ws, code, message) => await WebSocketService.getInstance().handleCloseWebSocket(ws, code, message),
};
async function main() {
    logger.info("Initializing WebSocket server...");
    uWebSocketApp.ws("/*", webSocketBehavior);
    uWebSocketApp.get("/health", (res, _req) => {
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
        }
        else {
            logger.error("Failed to start WebSocket server", { port: PORT });
            process.exit(1);
        }
    });
}
main().catch((error) => {
    const logError = error instanceof Error ? error : undefined;
    const logMeta = error instanceof Error ? {} : { error: String(error) };
    logger.error("Fatal startup error", logError, logMeta);
    process.exit(1);
});
//# sourceMappingURL=server.js.map