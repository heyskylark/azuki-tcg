import type { HttpRequest, HttpResponse, us_socket_context_t, WebSocket } from "uWebSockets.js";
import type { UserData } from "../constants";
export declare class WebSocketService {
    private static instance;
    private constructor();
    static getInstance(): WebSocketService;
    handleWebSocketUpgrade(res: HttpResponse, req: HttpRequest, context: us_socket_context_t): void;
    handleWebSocketOpen(ws: WebSocket<UserData>): void;
    handleMessage(ws: WebSocket<UserData>, message: ArrayBuffer, isBinary: boolean): void;
    handleDroppedMessage(ws: WebSocket<UserData>, _message: ArrayBuffer, _isBinary: boolean): void;
    handleCloseWebSocket(ws: WebSocket<UserData>, code: number, _message: ArrayBuffer): Promise<void>;
}
//# sourceMappingURL=WebSocketService.d.ts.map