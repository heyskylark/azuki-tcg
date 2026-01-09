/**
 * Safely send a JSON message to a WebSocket client
 */
export function sendJson(ws, data) {
    try {
        const message = JSON.stringify(data);
        return ws.send(message, false) !== 0;
    }
    catch {
        return false;
    }
}
/**
 * Parse a binary or string message to JSON
 */
export function parseMessage(message, _isBinary) {
    try {
        const text = Buffer.from(message).toString("utf-8");
        return JSON.parse(text);
    }
    catch {
        return null;
    }
}
//# sourceMappingURL=index.js.map