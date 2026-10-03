import { RoomStatus, UserType } from "@tcg/backend-core/types";
import {
  updateRoomStatus,
  findRoomById,
  updatePlayerReady,
} from "@tcg/backend-core/services/roomService";
import {
  loadDeckAsDefIds,
  deckAsDefIdsToDeckEntries,
} from "@tcg/backend-core/services/cardMapperService";
import { findUserById } from "@tcg/backend-core/services/userService";
import {
  abortHumanEvaluationRoom,
  getHumanEvaluationMatchByRoomId,
  materializeHumanEvaluationDeck,
} from "@tcg/backend-core/services/humanEvaluationService";
import type { HumanEvaluationRuntimeMatch } from "@tcg/backend-core/types/humanEvaluations";
import { getRoomChannel, updateRoomChannelStatus, removeRoomChannel } from "@/state/RoomRegistry";
import {
  startDeckSelectionTimeout,
  startReadyCountdown,
  cancelReadyCountdown,
  cancelDeckSelectionTimeout,
  clearAllTimersForRoom,
} from "@/state/TimerManager";
import { broadcastRoomState, broadcastToRoom, sendToPlayer } from "@/utils/broadcast";
import { DECK_SELECTION_TIMEOUT_MS } from "@/constants";
import { createGameWorld, destroyGameWorld, getWorldByRoomId } from "@/engine/WorldManager";
import {
  clearAiOpponentForRoom,
  maybeRunAiTurns,
  registerAiOpponent,
} from "@/engine/aiOpponentService";
import { generateSnapshot } from "@/engine/snapshotGenerator";
import { endInferenceSession, generateDeck, hasInferenceSession } from "@/services/inferenceClient";
import logger from "@/logger";
import type { RoomChannelState } from "@/state/types";

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
function battleInitializationAbortReason(
  channel: RoomChannelState,
  ordinaryReason: string
): string {
  return channel.evaluation ? "Evaluation battle could not be initialized" : ordinaryReason;
}

async function prepareEvaluationDeck(
  evaluationMatch: HumanEvaluationRuntimeMatch,
  channel: RoomChannelState
): Promise<void> {
  const artifact = await generateDeck({
    modelKey: evaluationMatch.modelKey,
    sessionKey: evaluationMatch.sessionKey,
    draftSeed: evaluationMatch.draftSeed,
    aiSlot: evaluationMatch.aiSlot,
    gateCardCode: evaluationMatch.gateCardCode,
    leaderCardCode: evaluationMatch.leaderCardCode,
    premadeDeckSlug: evaluationMatch.premadeDeckSlug,
  });
  const materialized = await materializeHumanEvaluationDeck(evaluationMatch.matchId, artifact);
  if (evaluationMatch.aiSlot === 0) {
    channel.player0DeckId = materialized.deckId;
  } else {
    channel.player1DeckId = materialized.deckId;
  }
}

export async function transitionToDeckSelection(roomId: string): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot transition to DECK_SELECTION: channel not found", { roomId });
    return;
  }

  const deadline = new Date(Date.now() + DECK_SELECTION_TIMEOUT_MS);

  await updateRoomStatus(roomId, RoomStatus.DECK_SELECTION, {
    deckSelectionDeadline: deadline,
  });

  let player0Ready = channel.player0Ready;
  let player1Ready = channel.player1Ready;
  const evaluationMatch = await getHumanEvaluationMatchByRoomId(roomId);
  if (channel.evaluation && !evaluationMatch) {
    await transitionToAborted(roomId, "Evaluation assignment not found");
    return;
  }
  if (evaluationMatch) {
    try {
      await prepareEvaluationDeck(evaluationMatch, channel);
    } catch (error) {
      logger.error("Failed to prepare evaluation opponent deck", { roomId, error });
      await endInferenceSession(evaluationMatch.sessionKey);
      await transitionToAborted(roomId, "Evaluation opponent could not be prepared");
      return;
    }
  }

  for (const slot of [0, 1] as const) {
    const player = channel.players[slot];
    if (!player?.isAi) {
      continue;
    }

    const deckId = slot === 0 ? channel.player0DeckId : channel.player1DeckId;
    if (!deckId) {
      logger.error("Cannot transition AI player to DECK_SELECTION: missing deck", {
        roomId,
        playerSlot: slot,
        userId: player.userId,
      });
      await transitionToAborted(
        roomId,
        channel.evaluation
          ? "Evaluation opponent could not be prepared"
          : `AI player ${slot} is missing a deck`
      );
      return;
    }

    await updatePlayerReady(roomId, slot, true);
    if (slot === 0) {
      player0Ready = true;
    } else {
      player1Ready = true;
    }
  }

  updateRoomChannelStatus(roomId, {
    status: RoomStatus.DECK_SELECTION,
    deckSelectionDeadline: deadline,
    player0Ready,
    player1Ready,
  });

  startDeckSelectionTimeout(roomId, deadline, async () => {
    await transitionToAborted(roomId, "Deck selection timeout");
  });

  broadcastRoomState(channel);
  if (evaluationMatch && player0Ready && player1Ready) {
    await transitionToReadyCheck(roomId);
  }

  logger.info("Room transitioned to DECK_SELECTION", { roomId, deadline });
}

export async function transitionToReadyCheck(roomId: string): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot transition to READY_CHECK: channel not found", { roomId });
    return;
  }

  await updateRoomStatus(roomId, RoomStatus.READY_CHECK);

  updateRoomChannelStatus(roomId, {
    status: RoomStatus.READY_CHECK,
    readyCountdownStartedAt: new Date(),
  });

  startReadyCountdown(roomId, async () => {
    await transitionToStarting(roomId);
  });

  broadcastRoomState(channel);

  logger.info("Room transitioned to READY_CHECK", { roomId });
}

export async function transitionToStarting(roomId: string): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot transition to STARTING: channel not found", { roomId });
    return;
  }

  cancelDeckSelectionTimeout(roomId);

  const evaluationMatch = channel.evaluation ? await getHumanEvaluationMatchByRoomId(roomId) : null;
  if (channel.evaluation && !evaluationMatch) {
    await transitionToAborted(roomId, "Evaluation assignment not found");
    return;
  }
  if (evaluationMatch) {
    try {
      const sessionActive = await hasInferenceSession(evaluationMatch.sessionKey);
      if (!sessionActive) {
        await prepareEvaluationDeck(evaluationMatch, channel);
      }
    } catch (error) {
      logger.error("Failed to restore evaluation inference session", { roomId, error });
      await endInferenceSession(evaluationMatch.sessionKey);
      await transitionToAborted(roomId, "Evaluation opponent could not be prepared");
      return;
    }
  }
  const rngSeed = evaluationMatch?.battleSeed ?? Math.floor(Math.random() * 2147483647);

  await updateRoomStatus(roomId, RoomStatus.STARTING, {
    rngSeed,
  });

  updateRoomChannelStatus(roomId, {
    status: RoomStatus.STARTING,
    readyCountdownStartedAt: null,
  });

  broadcastRoomState(channel);

  logger.info("Room transitioned to STARTING", { roomId, rngSeed });

  // Immediately transition to IN_MATCH
  await transitionToInMatch(roomId, rngSeed);
}

export async function transitionToInMatch(roomId: string, rngSeed: number): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot transition to IN_MATCH: channel not found", { roomId });
    return;
  }

  // Get room data with player and deck IDs
  const room = await findRoomById(roomId);
  if (!room) {
    logger.error("Cannot transition to IN_MATCH: room not found", { roomId });
    await transitionToAborted(
      roomId,
      battleInitializationAbortReason(channel, "Room data not found")
    );
    return;
  }

  if (!room.player0Id || !room.player1Id) {
    logger.error("Cannot transition to IN_MATCH: missing players", { roomId });
    await transitionToAborted(roomId, battleInitializationAbortReason(channel, "Missing players"));
    return;
  }

  if (!room.player0DeckId || !room.player1DeckId) {
    logger.error("Cannot transition to IN_MATCH: missing decks", { roomId });
    await transitionToAborted(roomId, battleInitializationAbortReason(channel, "Missing decks"));
    return;
  }

  try {
    const evaluationMatch = channel.evaluation
      ? await getHumanEvaluationMatchByRoomId(roomId)
      : null;
    if (channel.evaluation && !evaluationMatch) {
      throw new Error("Evaluation assignment not found");
    }

    const player0User = await findUserById(room.player0Id);
    const player1User = await findUserById(room.player1Id);
    if (!player0User || !player1User) {
      logger.error("Cannot transition to IN_MATCH: missing user records", {
        roomId,
        player0Found: !!player0User,
        player1Found: !!player1User,
      });
      await transitionToAborted(
        roomId,
        battleInitializationAbortReason(channel, "Missing user records")
      );
      return;
    }

    const aiPlayers: Array<{
      slot: 0 | 1;
      userId: string;
      modelKey: string | null;
    }> = [];
    if (player0User.type === UserType.AI) {
      aiPlayers.push({ slot: 0, userId: player0User.id, modelKey: player0User.modelKey });
    }
    if (player1User.type === UserType.AI) {
      aiPlayers.push({ slot: 1, userId: player1User.id, modelKey: player1User.modelKey });
    }

    if (aiPlayers.length > 1) {
      logger.error("Cannot transition to IN_MATCH: multiple AI players are not supported", {
        roomId,
      });
      await transitionToAborted(
        roomId,
        battleInitializationAbortReason(channel, "Only one AI player is supported per match")
      );
      return;
    }

    const detectedAiPlayer = aiPlayers[0] ?? null;
    const aiPlayer = evaluationMatch
      ? {
          slot: evaluationMatch.aiSlot,
          userId: evaluationMatch.aiSlot === 0 ? room.player0Id : room.player1Id,
          modelKey: evaluationMatch.modelKey,
        }
      : detectedAiPlayer;
    if (aiPlayer && !aiPlayer.modelKey) {
      logger.error("Cannot transition to IN_MATCH: AI player missing model key", {
        roomId,
        aiSlot: aiPlayer.slot,
        aiUserId: aiPlayer.userId,
      });
      await transitionToAborted(
        roomId,
        battleInitializationAbortReason(channel, "AI player is missing a model key")
      );
      return;
    }

    if (!evaluationMatch) {
      await clearAiOpponentForRoom(roomId);
    }

    // Load decks as CardDefIds for the engine
    const player0Deck = await loadDeckAsDefIds(room.player0DeckId);
    const player1Deck = await loadDeckAsDefIds(room.player1DeckId);

    // Convert decks to DeckCardEntry format for engine initialization
    const player0DeckEntries = deckAsDefIdsToDeckEntries(player0Deck);
    const player1DeckEntries = deckAsDefIdsToDeckEntries(player1Deck);

    logger.info("Loaded player0 and player 1 decks and deck entries");

    // Create the game world
    const world = createGameWorld(
      roomId,
      rngSeed,
      room.player0Id,
      player0DeckEntries,
      room.player1Id,
      player1DeckEntries,
      evaluationMatch?.startingPlayer
    );

    logger.info("Created game world", { roomId, worldId: world.worldId });

    // Update room status to IN_MATCH
    await updateRoomStatus(roomId, RoomStatus.IN_MATCH);

    updateRoomChannelStatus(roomId, {
      status: RoomStatus.IN_MATCH,
    });

    broadcastRoomState(channel);

    logger.info("Room transitioned to IN_MATCH", {
      roomId,
      worldId: world.worldId,
    });

    if (aiPlayer && aiPlayer.modelKey) {
      if (evaluationMatch) {
        registerAiOpponent({
          roomId,
          playerSlot: aiPlayer.slot,
          userId: aiPlayer.userId,
          modelKey: aiPlayer.modelKey,
          sessionKey: evaluationMatch.sessionKey,
          resetSession: false,
          requireSession: true,
        });
      } else {
        registerAiOpponent({
          roomId,
          playerSlot: aiPlayer.slot,
          userId: aiPlayer.userId,
          modelKey: aiPlayer.modelKey,
        });
      }
    }

    // Send initial snapshots to each player
    for (const slot of [0, 1] as const) {
      const playerConnection = channel.players[slot];
      if (playerConnection?.ws && playerConnection.connected) {
        const snapshot = await generateSnapshot(roomId, slot);
        if (snapshot) {
          sendToPlayer(channel, slot, snapshot);
        }
      }
    }

    if (aiPlayer) {
      try {
        await maybeRunAiTurns(roomId);
      } catch (error) {
        const errorMsg = getErrorMessage(error);
        logger.error("AI turn failed during IN_MATCH transition", {
          roomId,
          error: errorMsg,
        });
        await transitionToAborted(
          roomId,
          channel.evaluation
            ? "Evaluation opponent failed during battle"
            : `AI inference failed: ${errorMsg}`
        );
      }
    }
  } catch (error) {
    const errorMsg = getErrorMessage(error);
    logger.error("Failed to create game world", { roomId, error: errorMsg });
    await transitionToAborted(
      roomId,
      channel.evaluation
        ? "Evaluation battle could not be initialized"
        : `Failed to initialize game: ${errorMsg}`
    );
  }
}

export async function transitionToClosed(roomId: string, reason: string): Promise<void> {
  clearAllTimersForRoom(roomId);
  await clearAiOpponentForRoom(roomId);

  if (getWorldByRoomId(roomId)) {
    destroyGameWorld(roomId);
  }

  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot transition to CLOSED: channel not found", { roomId });
    return;
  }

  await updateRoomStatus(roomId, RoomStatus.CLOSED);

  updateRoomChannelStatus(roomId, {
    status: RoomStatus.CLOSED,
  });

  // Send ROOM_CLOSED message to all connected players
  broadcastToRoom(channel, {
    type: "ROOM_CLOSED",
    reason,
  });

  removeRoomChannel(roomId);

  logger.info("Room transitioned to CLOSED", { roomId, reason });
}

export async function transitionToAborted(roomId: string, reason: string): Promise<void> {
  clearAllTimersForRoom(roomId);
  await clearAiOpponentForRoom(roomId);

  if (getWorldByRoomId(roomId)) {
    destroyGameWorld(roomId);
  }

  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot transition to ABORTED: channel not found", { roomId });
    return;
  }
  if (channel.evaluation) {
    await abortHumanEvaluationRoom(roomId);
  }

  await updateRoomStatus(roomId, RoomStatus.ABORTED);

  updateRoomChannelStatus(roomId, {
    status: RoomStatus.ABORTED,
  });

  broadcastToRoom(channel, {
    type: "ERROR",
    code: "ROOM_ABORTED",
    message: reason,
  });

  broadcastRoomState(channel);

  removeRoomChannel(roomId);

  logger.info("Room transitioned to ABORTED", { roomId, reason });
}

export async function revertToDeckSelection(roomId: string): Promise<void> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    logger.warn("Cannot revert to DECK_SELECTION: channel not found", { roomId });
    return;
  }

  cancelReadyCountdown(roomId);

  await updateRoomStatus(roomId, RoomStatus.DECK_SELECTION);

  updateRoomChannelStatus(roomId, {
    status: RoomStatus.DECK_SELECTION,
    readyCountdownStartedAt: null,
  });

  broadcastRoomState(channel);

  logger.info("Room reverted to DECK_SELECTION", { roomId });
}

export async function checkAndTransitionToReadyCheck(roomId: string): Promise<boolean> {
  const channel = getRoomChannel(roomId);
  if (!channel) {
    return false;
  }

  if (channel.status !== RoomStatus.DECK_SELECTION) {
    return false;
  }

  if (!channel.player0Ready || !channel.player1Ready) {
    return false;
  }

  await transitionToReadyCheck(roomId);
  return true;
}
