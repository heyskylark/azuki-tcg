export declare class WebSocketError extends Error {
    readonly code: string;
    constructor(message: string, code: string);
}
export declare class AuthenticationError extends WebSocketError {
    constructor(message: string);
}
export declare class InvalidActionError extends WebSocketError {
    constructor(message: string);
}
export declare class RoomNotFoundError extends WebSocketError {
    constructor(roomId: string);
}
//# sourceMappingURL=index.d.ts.map