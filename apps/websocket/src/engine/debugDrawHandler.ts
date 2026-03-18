import type { WebSocket } from "uWebSockets.js";
import { z } from "zod";
import { RoomStatus } from "@tcg/backend-core/types";
import { cardCodeToDefId } from "@tcg/backend-core/services/cardMapperService";
import type { UserData } from "@/constants";
import { env } from "@/env";
import { resolveAcceptedActionResult } from "@/engine/actionResolutionService";
import { debugDrawPlayerCard } from "@/engine/WorldManager";
import logger from "@/logger";
import { getRoomChannel } from "@/state/RoomRegistry";
import type { ConnectionInfo } from "@/state/types";

const debugDrawMessageSchema = z
  .object({
    type: z.literal("DEBUG_DRAW"),
    cardCode: z.string().trim().min(1),
  })
  .strict();

function sendError(
  ws: WebSocket<UserData>,
  code: string,
  message: string
): void {
  ws.send(
    JSON.stringify({
      type: "ERROR",
      code,
      message,
    })
  );
}

function isDebugDrawAcceptedResult(
  result: ReturnType<typeof debugDrawPlayerCard>
): result is Extract<ReturnType<typeof debugDrawPlayerCard>, { success: boolean }> {
  return "success" in result && !("code" in result);
}

export async function handleDebugDraw(
  ws: WebSocket<UserData>,
  message: unknown,
  connectionInfo: ConnectionInfo
): Promise<void> {
  if (!env.AZK_DEBUG_ACTIONS_ENABLED) {
    sendError(ws, "DEBUG_ACTIONS_DISABLED", "Debug draw is not enabled");
    return;
  }

  const parsedMessage = debugDrawMessageSchema.safeParse(message);
  if (!parsedMessage.success) {
    sendError(ws, "INVALID_DEBUG_DRAW", "Debug draw requires a cardCode string");
    return;
  }

  const { roomId, userId, playerSlot } = connectionInfo;
  const channel = getRoomChannel(roomId);
  if (!channel || channel.status !== RoomStatus.IN_MATCH) {
    sendError(ws, "INVALID_STATE", "Game not in progress");
    return;
  }

  const normalizedCardCode = parsedMessage.data.cardCode.toUpperCase();
  const cardDefId = cardCodeToDefId(normalizedCardCode);
  if (cardDefId === null) {
    sendError(ws, "INVALID_CARD_CODE", `Unknown card code: ${normalizedCardCode}`);
    return;
  }

  logger.info("Processing debug draw request", {
    roomId,
    userId,
    playerSlot,
    cardCode: normalizedCardCode,
    cardDefId,
  });

  const result = debugDrawPlayerCard(roomId, userId, cardDefId);
  if (!isDebugDrawAcceptedResult(result)) {
    const code = result.code === "NOT_FOUND" ? "NO_WORLD" : "DEBUG_DRAW_FAILED";
    sendError(ws, code, result.error);
    return;
  }

  await resolveAcceptedActionResult(roomId, result);
}
