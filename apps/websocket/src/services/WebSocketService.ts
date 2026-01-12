import type {
  HttpRequest,
  HttpResponse,
  us_socket_context_t,
  WebSocket,
} from "uWebSockets.js";

import { RoomStatus } from "@tcg/backend-core/types";
import { verifyJoinToken } from "@tcg/backend-core/services/authService";
import { findRoomById } from "@tcg/backend-core/services/roomService";
import { findUserById } from "@tcg/backend-core/services/userService";
import type { AuthConfig } from "@tcg/backend-core/types/auth";
import type { ConnectionAckMessage, SelectDeckMessage, ReadyMessage } from "@tcg/backend-core/types/ws";

import type { UserData } from "@/constants";
import { env } from "@/env";
import logger from "@/logger";
import { parseMessage, sendJson } from "@/utils";
import {
  registerConnection,
  unregisterConnection,
  getConnectionInfo,
} from "@/state/ConnectionRegistry";
import {
  getOrCreateRoomChannel,
  getRoomChannel,
  updateRoomChannelStatus,
} from "@/state/RoomRegistry";
import {
  startDisconnectGrace,
  cancelDisconnectGrace,
} from "@/state/TimerManager";
import { broadcastRoomState } from "@/utils/broadcast";
import { transitionToDeckSelection, transitionToAborted } from "@/handlers/stateTransitionHandler";
import { handleSelectDeck, handleReady } from "@/handlers/roomMessageHandler";

const INACTIVE_ROOM_STATUSES = [
  RoomStatus.COMPLETED,
  RoomStatus.ABORTED,
  RoomStatus.CLOSED,
];

const authConfig: AuthConfig = {
  jwtSecret: env.JWT_SECRET,
  jwtIssuer: env.JWT_ISSUER,
  saltRounds: 12,
};

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
    const query = req.getQuery();
    const secWebSocketKey = req.getHeader("sec-websocket-key");
    const secWebSocketProtocol = req.getHeader("sec-websocket-protocol");
    const secWebSocketExtensions = req.getHeader("sec-websocket-extensions");

    const token = new URLSearchParams(query).get("token");

    if (!token) {
      logger.warn("WebSocket upgrade rejected: missing token");
      res.writeStatus("401 Unauthorized").end("Missing token");
      return;
    }

    res.onAborted(() => {
      logger.debug("WebSocket upgrade aborted");
    });

    verifyJoinToken(token, authConfig)
      .then(async (payload) => {
        const { roomId, playerSlot, sub: userId } = payload;

        const room = await findRoomById(roomId);
        if (!room) {
          logger.warn("WebSocket upgrade rejected: room not found", { roomId });
          res.cork(() => {
            res.writeStatus("404 Not Found").end("Room not found");
          });
          return;
        }

        if (INACTIVE_ROOM_STATUSES.includes(room.status)) {
          logger.warn("WebSocket upgrade rejected: room not active", { roomId, status: room.status });
          res.cork(() => {
            res.writeStatus("410 Gone").end("Room is no longer active");
          });
          return;
        }

        const expectedPlayerId = playerSlot === 0 ? room.player0Id : room.player1Id;
        if (expectedPlayerId !== userId) {
          logger.warn("WebSocket upgrade rejected: player slot mismatch", {
            roomId,
            playerSlot,
            expectedPlayerId,
            userId,
          });
          res.cork(() => {
            res.writeStatus("403 Forbidden").end("Player slot mismatch");
          });
          return;
        }

        const user = await findUserById(userId);
        if (!user) {
          logger.warn("WebSocket upgrade rejected: user not found", { userId });
          res.cork(() => {
            res.writeStatus("404 Not Found").end("User not found");
          });
          return;
        }

        res.cork(() => {
          res.upgrade<UserData>(
            {
              id: userId,
              isAnonymous: false,
              roomId,
              playerSlot,
            },
            secWebSocketKey,
            secWebSocketProtocol,
            secWebSocketExtensions,
            context
          );
        });

        logger.debug("WebSocket upgrade completed", { url, userId, roomId, playerSlot });
      })
      .catch((error) => {
        logger.warn("WebSocket upgrade rejected: token validation failed", { error: String(error) });
        res.cork(() => {
          res.writeStatus("401 Unauthorized").end("Invalid or expired token");
        });
      });
  }

  public async handleWebSocketOpen(ws: WebSocket<UserData>): Promise<void> {
    const userData = ws.getUserData();
    const { id: userId, roomId, playerSlot } = userData;

    if (!roomId || playerSlot === undefined) {
      logger.warn("WebSocket opened without room info", { userId });
      sendJson(ws, { type: "ERROR", code: "INVALID_STATE", message: "Missing room info" });
      ws.close();
      return;
    }

    logger.info("WebSocket connection opened", { userId, roomId, playerSlot });

    const room = await findRoomById(roomId);
    if (!room) {
      sendJson(ws, { type: "ERROR", code: "ROOM_NOT_FOUND", message: "Room not found" });
      ws.close();
      return;
    }

    const user = await findUserById(userId);
    if (!user) {
      sendJson(ws, { type: "ERROR", code: "USER_NOT_FOUND", message: "User not found" });
      ws.close();
      return;
    }

    const player0User = room.player0Id ? await findUserById(room.player0Id) : null;
    const player1User = room.player1Id ? await findUserById(room.player1Id) : null;

    const channel = getOrCreateRoomChannel(
      roomId,
      room,
      player0User?.username ?? null,
      player1User?.username ?? null
    );

    const player = channel.players[playerSlot];
    if (player) {
      player.ws = ws;
      player.connected = true;
      player.disconnectedAt = null;
      player.username = user.username;
    }

    registerConnection(ws, {
      userId,
      username: user.username,
      roomId,
      playerSlot,
    });

    cancelDisconnectGrace(roomId, playerSlot);

    const ackMessage: ConnectionAckMessage = {
      type: "CONNECTION_ACK",
      playerId: userId,
      playerSlot,
    };
    sendJson(ws, ackMessage);

    const shouldTransitionToDeckSelection =
      room.status === RoomStatus.WAITING_FOR_PLAYERS &&
      playerSlot === 1 &&
      channel.players[0]?.connected;

    if (shouldTransitionToDeckSelection) {
      await transitionToDeckSelection(roomId);
    } else {
      broadcastRoomState(channel);
    }
  }

  public async handleMessage(
    ws: WebSocket<UserData>,
    message: ArrayBuffer,
    isBinary: boolean
  ): Promise<void> {
    const userData = ws.getUserData();
    const connectionInfo = getConnectionInfo(ws);

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

    switch (parsed.type) {
      case "PING":
        sendJson(ws, { type: "PONG" });
        break;

      case "SELECT_DECK":
        if (!connectionInfo) {
          sendJson(ws, { type: "ERROR", code: "NOT_AUTHENTICATED", message: "Not authenticated" });
          return;
        }
        await handleSelectDeck(ws, parsed as SelectDeckMessage, connectionInfo);
        break;

      case "READY":
        if (!connectionInfo) {
          sendJson(ws, { type: "ERROR", code: "NOT_AUTHENTICATED", message: "Not authenticated" });
          return;
        }
        await handleReady(ws, parsed as ReadyMessage, connectionInfo);
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
    const connectionInfo = unregisterConnection(ws);
    const userData = ws.getUserData();

    logger.info("WebSocket connection closed", {
      userId: userData.isAnonymous ? null : userData.id,
      code,
    });

    if (!connectionInfo) {
      return;
    }

    const { roomId, playerSlot } = connectionInfo;
    const channel = getRoomChannel(roomId);

    if (!channel) {
      return;
    }

    const player = channel.players[playerSlot];
    if (player) {
      player.ws = null;
      player.connected = false;
      player.disconnectedAt = new Date();
    }

    const otherSlot = playerSlot === 0 ? 1 : 0;
    const otherPlayer = channel.players[otherSlot];
    const bothDisconnected = !otherPlayer?.connected;

    if (bothDisconnected) {
      logger.info("Both players disconnected, aborting room", { roomId });
      await transitionToAborted(roomId, "Both players disconnected");
      return;
    }

    broadcastRoomState(channel);

    startDisconnectGrace(roomId, playerSlot, async () => {
      logger.info("Disconnect grace period expired", { roomId, playerSlot });
      await transitionToAborted(roomId, `Player ${playerSlot} disconnect timeout`);
    });
  }
}
