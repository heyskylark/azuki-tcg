/**
 * Base API error class with HTTP status code
 */
export class ApiError extends Error {
  public readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// 400 Bad Request
export class ValidationError extends ApiError {
  constructor(message: string) {
    super(message, 400);
    this.name = "ValidationError";
  }
}

// 401 Unauthorized
export class InvalidCredentialsError extends ApiError {
  constructor() {
    super("Invalid email or password", 401);
    this.name = "InvalidCredentialsError";
  }
}

export class UnauthorizedError extends ApiError {
  constructor(message = "Unauthorized") {
    super(message, 401);
    this.name = "UnauthorizedError";
  }
}

export class TokenExpiredError extends ApiError {
  constructor() {
    super("Token has expired", 401);
    this.name = "TokenExpiredError";
  }
}

export class TokenRevokedError extends ApiError {
  constructor() {
    super("Token has been revoked", 401);
    this.name = "TokenRevokedError";
  }
}

export class InvalidTokenError extends ApiError {
  constructor() {
    super("Invalid token", 401);
    this.name = "InvalidTokenError";
  }
}

// 403 Forbidden
export class ForbiddenError extends ApiError {
  constructor(message = "Forbidden") {
    super(message, 403);
    this.name = "ForbiddenError";
  }
}

export class UserBannedError extends ApiError {
  constructor() {
    super("User account is banned", 403);
    this.name = "UserBannedError";
  }
}

export class UserDeletedError extends ApiError {
  constructor() {
    super("User account has been deleted", 403);
    this.name = "UserDeletedError";
  }
}

// 404 Not Found
export class UserNotFoundError extends ApiError {
  constructor() {
    super("User not found", 404);
    this.name = "UserNotFoundError";
  }
}

// 409 Conflict
export class EmailAlreadyExistsError extends ApiError {
  constructor() {
    super("Email already registered", 409);
    this.name = "EmailAlreadyExistsError";
  }
}

export class UsernameAlreadyExistsError extends ApiError {
  constructor() {
    super("Username already taken", 409);
    this.name = "UsernameAlreadyExistsError";
  }
}

// Deck errors

export class DeckNotFoundError extends ApiError {
  constructor() {
    super("Deck not found", 404);
    this.name = "DeckNotFoundError";
  }
}

// Room errors

export class RoomNotFoundError extends ApiError {
  constructor() {
    super("Room not found", 404);
    this.name = "RoomNotFoundError";
  }
}

export class NotRoomOwnerError extends ApiError {
  constructor() {
    super("Only the room owner can perform this action", 403);
    this.name = "NotRoomOwnerError";
  }
}

export class RoomFullError extends ApiError {
  constructor() {
    super("Room is full", 409);
    this.name = "RoomFullError";
  }
}

export class InvalidRoomStatusError extends ApiError {
  constructor(message = "Room is not in the correct status for this action") {
    super(message, 409);
    this.name = "InvalidRoomStatusError";
  }
}

export class InvalidRoomPasswordError extends ApiError {
  constructor() {
    super("Invalid room password", 401);
    this.name = "InvalidRoomPasswordError";
  }
}

export class UserAlreadyInRoomError extends ApiError {
  constructor() {
    super("User is already in an active room", 400);
    this.name = "UserAlreadyInRoomError";
  }
}

export class RoomClosedError extends ApiError {
  constructor() {
    super("Room is closed", 400);
    this.name = "RoomClosedError";
  }
}

// Update errors
export class UpdateFailedError extends ApiError {
  constructor(message = "Failed to update resource") {
    super(message, 500);
    this.name = "UpdateFailedError";
  }
}
