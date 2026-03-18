import type { WebSocket } from "uWebSockets.js";
import { z } from "zod";
import { RoomStatus } from "@tcg/backend-core/types";
import type { UserData } from "@/constants";
import { env } from "@/env";
import { resolveAcceptedActionResult } from "@/engine/actionResolutionService";
import { debugGrantPlayerIkz } from "@/engine/WorldManager";
import logger from "@/logger";
import { getRoomChannel } from "@/state/RoomRegistry";
import type { ConnectionInfo } from "@/state/types";

const debugIkzMessageSchema = z
  .object({
    type: z.literal("DEBUG_IKZ"),
    count: z.number().int().min(0),
  })
  .strict();

function sendError(ws: WebSocket<UserData>, code: string, message: string): void {
  ws.send(
    JSON.stringify({
      type: "ERROR",
      code,
      message,
    })
  );
}

function isDebugIkzAcceptedResult(
  result: ReturnType<typeof debugGrantPlayerIkz>
): result is Extract<ReturnType<typeof debugGrantPlayerIkz>, { success: boolean }> {
  return "success" in result && !("code" in result);
}

export async function handleDebugIkz(
  ws: WebSocket<UserData>,
  message: unknown,
  connectionInfo: ConnectionInfo
): Promise<void> {
  if (!env.AZK_DEBUG_ACTIONS_ENABLED) {
    sendError(ws, "DEBUG_ACTIONS_DISABLED", "Debug IKZ grant is not enabled");
    return;
  }

  const parsedMessage = debugIkzMessageSchema.safeParse(message);
  if (!parsedMessage.success) {
    sendError(ws, "INVALID_DEBUG_IKZ", "Debug IKZ requires a non-negative integer count");
    return;
  }

  const { roomId, userId, playerSlot } = connectionInfo;
  const channel = getRoomChannel(roomId);
  if (!channel || channel.status !== RoomStatus.IN_MATCH) {
    sendError(ws, "INVALID_STATE", "Game not in progress");
    return;
  }

  const { count } = parsedMessage.data;

  logger.info("Processing debug IKZ grant request", {
    roomId,
    userId,
    playerSlot,
    count,
  });

  const result = debugGrantPlayerIkz(roomId, userId, count);
  if (!isDebugIkzAcceptedResult(result)) {
    const code = result.code === "NOT_FOUND" ? "NO_WORLD" : "DEBUG_IKZ_FAILED";
    sendError(ws, code, result.error);
    return;
  }

  await resolveAcceptedActionResult(roomId, result);
}
