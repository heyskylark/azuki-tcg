export class WebSocketError extends Error {
  public readonly code: string;

  constructor(message: string, code: string) {
    super(message);
    this.name = "WebSocketError";
    this.code = code;
  }
}

export class AuthenticationError extends WebSocketError {
  constructor(message: string) {
    super(message, "AUTHENTICATION_ERROR");
    this.name = "AuthenticationError";
  }
}

export class InvalidActionError extends WebSocketError {
  constructor(message: string) {
    super(message, "INVALID_ACTION");
    this.name = "InvalidActionError";
  }
}

export class RoomNotFoundError extends WebSocketError {
  constructor(roomId: string) {
    super(`Room not found: ${roomId}`, "ROOM_NOT_FOUND");
    this.name = "RoomNotFoundError";
  }
}
