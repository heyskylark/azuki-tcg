export class WebSocketError extends Error {
    code;
    constructor(message, code) {
        super(message);
        this.name = "WebSocketError";
        this.code = code;
    }
}
export class AuthenticationError extends WebSocketError {
    constructor(message) {
        super(message, "AUTHENTICATION_ERROR");
        this.name = "AuthenticationError";
    }
}
export class InvalidActionError extends WebSocketError {
    constructor(message) {
        super(message, "INVALID_ACTION");
        this.name = "InvalidActionError";
    }
}
export class RoomNotFoundError extends WebSocketError {
    constructor(roomId) {
        super(`Room not found: ${roomId}`, "ROOM_NOT_FOUND");
        this.name = "RoomNotFoundError";
    }
}
//# sourceMappingURL=index.js.map