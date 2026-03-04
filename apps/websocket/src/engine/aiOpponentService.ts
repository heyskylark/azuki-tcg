import { RoomStatus } from "@tcg/backend-core/types";
import { getRoomChannel } from "@/state/RoomRegistry";
import {
  getActivePlayer,
  getPlayerTrainingObservationPackedBySlot,
  getWorldByRoomId,
  requiresAction,
  submitPlayerAction,
  type SubmitActionResult,
} from "@/engine/WorldManager";
import { resolveAcceptedActionResult } from "@/engine/actionResolutionService";
import { endInferenceSession, inferAction } from "@/services/inferenceClient";
import logger from "@/logger";

interface AiOpponentConfig {
  roomId: string;
  playerSlot: 0 | 1;
  userId: string;
  modelKey: string;
  sessionKey: string;
  running: boolean;
  resetSession: boolean;
}

interface RegisterAiOpponentParams {
  roomId: string;
  playerSlot: 0 | 1;
  userId: string;
  modelKey: string;
}

const aiOpponents = new Map<string, AiOpponentConfig>();
const MAX_AI_ACTIONS_PER_RUN = 256;

function isAcceptedActionResult(
  result: SubmitActionResult
): result is Extract<SubmitActionResult, { success: boolean }> {
  return "success" in result && !("code" in result);
}

function createSessionKey(roomId: string, playerSlot: 0 | 1): string {
  return `${roomId}:${playerSlot}`;
}

export function registerAiOpponent(params: RegisterAiOpponentParams): void {
  const existing = aiOpponents.get(params.roomId);
  if (existing && existing.sessionKey !== createSessionKey(params.roomId, params.playerSlot)) {
    void endInferenceSession(existing.sessionKey);
  }

  aiOpponents.set(params.roomId, {
    roomId: params.roomId,
    playerSlot: params.playerSlot,
    userId: params.userId,
    modelKey: params.modelKey,
    sessionKey: createSessionKey(params.roomId, params.playerSlot),
    running: false,
    resetSession: true,
  });
}

export function hasAiOpponent(roomId: string): boolean {
  return aiOpponents.has(roomId);
}

export async function clearAiOpponentForRoom(roomId: string): Promise<void> {
  const existing = aiOpponents.get(roomId);
  if (!existing) {
    return;
  }

  aiOpponents.delete(roomId);
  await endInferenceSession(existing.sessionKey);
}

async function runAiTurns(config: AiOpponentConfig): Promise<void> {
  let actionCount = 0;

  while (true) {
    const currentConfig = aiOpponents.get(config.roomId);
    if (!currentConfig || currentConfig.sessionKey !== config.sessionKey) {
      return;
    }

    const channel = getRoomChannel(config.roomId);
    if (!channel || channel.status !== RoomStatus.IN_MATCH) {
      return;
    }

    const world = getWorldByRoomId(config.roomId);
    if (!world) {
      return;
    }

    if (!requiresAction(config.roomId)) {
      return;
    }

    const activePlayer = getActivePlayer(config.roomId);
    if (activePlayer !== config.playerSlot) {
      return;
    }

    if (actionCount >= MAX_AI_ACTIONS_PER_RUN) {
      throw new Error(
        `AI action loop exceeded ${MAX_AI_ACTIONS_PER_RUN} actions in a single run`
      );
    }

    const packedObservation = getPlayerTrainingObservationPackedBySlot(
      config.roomId,
      config.playerSlot
    );
    if (!packedObservation) {
      throw new Error("Failed to build packed training observation for AI turn");
    }

    const action = await inferAction({
      modelKey: config.modelKey,
      sessionKey: config.sessionKey,
      observationPacked: packedObservation,
      resetSession: config.resetSession,
    });
    config.resetSession = false;

    logger.info("Submitting AI action", {
      roomId: config.roomId,
      playerSlot: config.playerSlot,
      action,
    });

    const submitResult = submitPlayerAction(config.roomId, config.userId, action);
    if (!isAcceptedActionResult(submitResult)) {
      throw new Error(
        `AI action rejected (${submitResult.code}): ${submitResult.error}`
      );
    }

    await resolveAcceptedActionResult(config.roomId, submitResult);
    actionCount += 1;

    if (submitResult.gameOver) {
      return;
    }
  }
}

export async function maybeRunAiTurns(roomId: string): Promise<void> {
  const config = aiOpponents.get(roomId);
  if (!config) {
    return;
  }

  if (config.running) {
    return;
  }

  config.running = true;
  try {
    await runAiTurns(config);
  } finally {
    const current = aiOpponents.get(roomId);
    if (current && current.sessionKey === config.sessionKey) {
      current.running = false;
    }
  }
}
