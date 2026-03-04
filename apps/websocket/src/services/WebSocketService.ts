import type {
  HttpRequest,
  HttpResponse,
  us_socket_context_t,
  WebSocket,
} from "uWebSockets.js";

import { RoomStatus, UserType } from "@tcg/backend-core/types";
import { verifyJoinToken } from "@tcg/backend-core/services/authService";
import { findRoomById } from "@tcg/backend-core/services/roomService";
import { findUserById } from "@tcg/backend-core/services/userService";
import type { AuthConfig } from "@tcg/backend-core/types/auth";
import type {
  ConnectionAckMessage,
  SelectDeckMessage,
  ReadyMessage,
  GameActionMessage as WsGameActionMessage,
  ForfeitMessage,
} from "@tcg/backend-core/types/ws";

import type { UserData } from "@/constants";
import { handleGameAction, type GameActionMessage } from "@/engine/gameActionHandler";
import { handleForfeit } from "@/engine/gameOverHandler";
import { generateSnapshot } from "@/engine/snapshotGenerator";
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
import {
  handleSelectDeck,
  handleReady,
  handleLeaveRoom,
  handleCloseRoom,
  handleStartGame,
} from "@/handlers/roomMessageHandler";

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

    let isAborted = false;
    res.onAborted(() => {
      isAborted = true;
      logger.debug("WebSocket upgrade aborted");
    });

    verifyJoinToken(token, authConfig)
      .then(async (payload) => {
        const { roomId, playerSlot, sub: userId } = payload;

        const room = await findRoomById(roomId);
        if (isAborted) return;
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
        if (isAborted) return;
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
        if (isAborted) return;
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
      player1User?.username ?? null,
      player0User?.type === UserType.AI,
      player1User?.type === UserType.AI,
      player0User?.modelKey ?? null,
      player1User?.modelKey ?? null
    );

    const existingPlayer = channel.players[playerSlot];
    if (existingPlayer) {
      existingPlayer.ws = ws;
      existingPlayer.connected = true;
      existingPlayer.disconnectedAt = null;
      existingPlayer.username = user.username;
    } else {
      // Player slot was null (e.g., player1 joined after channel was created)
      // Create a new player connection entry
      channel.players[playerSlot] = {
        ws,
        userId,
        username: user.username,
        isAi: false,
        modelKey: null,
        playerSlot,
        connected: true,
        disconnectedAt: null,
      };
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

    // Broadcast room state to all connected players
    // Note: Transition to DECK_SELECTION is now triggered by owner via START_GAME message
    broadcastRoomState(channel);

    // If room is IN_MATCH, send game snapshot for reconnection
    if (room.status === RoomStatus.IN_MATCH) {
      const snapshot = await generateSnapshot(roomId, playerSlot);
      if (snapshot) {
        sendJson(ws, snapshot);
        logger.debug("Sent game snapshot on reconnection", { roomId, playerSlot });
      }
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

      case "GAME_ACTION":
        if (!connectionInfo) {
          sendJson(ws, { type: "ERROR", code: "NOT_AUTHENTICATED", message: "Not authenticated" });
          return;
        }
        await handleGameAction(ws, parsed as GameActionMessage, connectionInfo);
        break;

      case "FORFEIT":
        if (!connectionInfo) {
          sendJson(ws, { type: "ERROR", code: "NOT_AUTHENTICATED", message: "Not authenticated" });
          return;
        }
        await handleForfeit(connectionInfo.roomId, connectionInfo.playerSlot);
        break;

      case "LEAVE_ROOM":
        if (!connectionInfo) {
          sendJson(ws, { type: "ERROR", code: "NOT_AUTHENTICATED", message: "Not authenticated" });
          return;
        }
        await handleLeaveRoom(ws, connectionInfo);
        break;

      case "CLOSE_ROOM":
        if (!connectionInfo) {
          sendJson(ws, { type: "ERROR", code: "NOT_AUTHENTICATED", message: "Not authenticated" });
          return;
        }
        await handleCloseRoom(ws, connectionInfo);
        break;

      case "START_GAME":
        if (!connectionInfo) {
          sendJson(ws, { type: "ERROR", code: "NOT_AUTHENTICATED", message: "Not authenticated" });
          return;
        }
        await handleStartGame(ws, connectionInfo);
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
    // Only consider "both disconnected" if the other player slot was actually filled
    const bothDisconnected = otherPlayer !== null && !otherPlayer.connected;

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
