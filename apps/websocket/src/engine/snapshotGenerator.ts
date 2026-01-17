/**
 * Snapshot generator for game state.
 * Creates GAME_SNAPSHOT messages for connect/reconnect scenarios.
 */

import type {
  GameSnapshotMessage,
  SnapshotStateContext,
  SnapshotPlayerBoard,
  SnapshotHandCard,
  SnapshotLeader,
  SnapshotGate,
  SnapshotCard,
  SnapshotIkz,
  SnapshotActionMask,
  SnapshotCardMetadata,
} from "@tcg/backend-core/types/ws";
import { getCardMetadataByCardCodes } from "@tcg/backend-core/services/cardMetadataService";
import {
  getWorldByRoomId,
  getPlayerObservationBySlot,
  getGameState,
  getActivePlayer,
} from "@/engine/WorldManager";
import type {
  ObservationData,
  StateContext,
  LeaderObservation,
  GateObservation,
  CardObservation,
  IKZObservation,
  ActionMask,
} from "@/engine/types";
import logger from "@/logger";

/**
 * Generate a GAME_SNAPSHOT message for a player.
 */
export async function generateSnapshot(
  roomId: string,
  playerSlot: 0 | 1
): Promise<GameSnapshotMessage | null> {
  const world = getWorldByRoomId(roomId);
  if (!world) {
    logger.error("Cannot generate snapshot: world not found", { roomId });
    return null;
  }

  // Get game state
  const gameState = getGameState(roomId);
  if (!gameState) {
    logger.error("Cannot generate snapshot: game state not found", { roomId });
    return null;
  }

  // Get observations for both players
  const myObservation = getPlayerObservationBySlot(roomId, playerSlot);
  if (!myObservation) {
    logger.error("Cannot generate snapshot: observation not found", {
      roomId,
      playerSlot,
    });
    return null;
  }

  const opponentSlot = playerSlot === 0 ? 1 : 0;
  const opponentObservation = getPlayerObservationBySlot(roomId, opponentSlot);
  if (!opponentObservation) {
    logger.error("Cannot generate snapshot: opponent observation not found", {
      roomId,
      opponentSlot,
    });
    return null;
  }

  // Build state context
  const stateContext: SnapshotStateContext = {
    phase: gameState.phase,
    abilitySubphase: "", // Not currently tracked
    activePlayer: gameState.activePlayer,
    turnNumber: gameState.turnNumber,
  };

  // Build player boards
  const myBoard = buildPlayerBoard(myObservation, true);
  const opponentBoard = buildPlayerBoard(opponentObservation, false);

  // Order boards by player slot (0 first, then 1)
  const players: [SnapshotPlayerBoard, SnapshotPlayerBoard] =
    playerSlot === 0 ? [myBoard, opponentBoard] : [opponentBoard, myBoard];

  // Build hand cards (only for the requesting player)
  const yourHand = buildHandCards(myObservation);

  // Get action mask if it's this player's turn
  const activePlayer = getActivePlayer(roomId);
  const actionMask =
    activePlayer === playerSlot ? buildActionMask(myObservation.actionMask) : null;

  const cardCodes = collectSnapshotCardCodes(players, yourHand);
  let cardMetadata: Record<string, SnapshotCardMetadata> = {};
  try {
    const cardMetadataMap = await getCardMetadataByCardCodes([...cardCodes]);
    cardMetadata = Object.fromEntries(cardMetadataMap.entries());
  } catch (error) {
    logger.error("Cannot generate snapshot: failed to load card metadata", {
      roomId,
      playerSlot,
      error: String(error),
    });
  }

  return {
    type: "GAME_SNAPSHOT",
    stateContext,
    players,
    yourHand,
    cardMetadata,
    combatStack: [], // TODO: populate combat stack when combat system is ready
    actionMask,
  };
}

/**
 * Build a player board from observation data.
 */
function buildPlayerBoard(
  observation: ObservationData,
  isOwnBoard: boolean
): SnapshotPlayerBoard {
  const myObs = observation.myObservationData;
  const oppObs = observation.opponentObservationData;

  // Use the appropriate observation based on whether this is the player's own board
  if (isOwnBoard) {
    return {
      leader: buildLeader(myObs.leader),
      gate: buildGate(myObs.gate),
      garden: myObs.garden.map((card) =>
        card ? buildCard(card) : null
      ),
      alley: myObs.alley.map((card) => (card ? buildCard(card) : null)),
      ikzArea: myObs.ikzArea.map((ikz) => buildIkz(ikz)),
      handCount: myObs.hand.length,
      deckCount: myObs.deckCount,
      discardCount: myObs.discard.length,
      ikzPileCount: myObs.ikzPileCount,
      hasIkzToken: myObs.hasIkzToken,
    };
  } else {
    return {
      leader: buildLeader(oppObs.leader),
      gate: buildGate(oppObs.gate),
      garden: oppObs.garden.map((card) =>
        card ? buildCard(card) : null
      ),
      alley: oppObs.alley.map((card) => (card ? buildCard(card) : null)),
      ikzArea: oppObs.ikzArea.map((ikz) => buildIkz(ikz)),
      handCount: oppObs.handCount,
      deckCount: oppObs.deckCount,
      discardCount: oppObs.discard.length,
      ikzPileCount: oppObs.ikzPileCount,
      hasIkzToken: oppObs.hasIkzToken,
    };
  }
}

/**
 * Build a leader snapshot from observation.
 */
function buildLeader(leader: LeaderObservation): SnapshotLeader {
  return {
    cardId: leader.cardCode,
    cardDefId: leader.cardDefId,
    zoneIndex: 0,
    curHp: leader.curHp,
    curAtk: leader.curAtk,
    tapped: leader.tapped,
    cooldown: leader.cooldown,
  };
}

/**
 * Build a gate snapshot from observation.
 */
function buildGate(gate: GateObservation): SnapshotGate {
  return {
    cardId: gate.cardCode,
    cardDefId: gate.cardDefId,
    zoneIndex: 0,
    tapped: gate.tapped,
    cooldown: gate.cooldown,
  };
}

/**
 * Build a card snapshot from observation.
 */
function buildCard(card: CardObservation): SnapshotCard {
  return {
    cardId: card.cardCode,
    cardDefId: card.cardDefId,
    zoneIndex: card.zoneIndex,
    curAtk: card.curAtk,
    curHp: card.curHp,
    tapped: card.tapped,
    cooldown: card.cooldown,
    isFrozen: card.isFrozen,
    isShocked: card.isShocked,
    isEffectImmune: card.isEffectImmune,
  };
}

/**
 * Build an IKZ snapshot from observation.
 */
function buildIkz(ikz: IKZObservation): SnapshotIkz {
  return {
    cardId: ikz.cardCode,
    cardDefId: ikz.cardDefId,
    tapped: ikz.tapped,
    cooldown: ikz.cooldown,
  };
}

/**
 * Build hand cards from observation.
 */
function buildHandCards(observation: ObservationData): SnapshotHandCard[] {
  return observation.myObservationData.hand.map((card) => ({
    cardId: card.cardCode,
    cardDefId: card.cardDefId,
    type: card.type,
    ikzCost: card.ikzCost,
  }));
}

/**
 * Build action mask from observation.
 */
function buildActionMask(mask: ActionMask): SnapshotActionMask {
  return {
    primaryActionMask: mask.primaryActionMask,
    legalActionCount: mask.legalActionCount,
    legalPrimary: mask.legalPrimary,
    legalSub1: mask.legalSub1,
    legalSub2: mask.legalSub2,
    legalSub3: mask.legalSub3,
  };
}

function collectSnapshotCardCodes(
  players: [SnapshotPlayerBoard, SnapshotPlayerBoard],
  hand: SnapshotHandCard[]
): Set<string> {
  const cardCodes = new Set<string>();

  for (const board of players) {
    addCardCode(board.leader.cardId, cardCodes);
    addCardCode(board.gate.cardId, cardCodes);
    for (const card of board.garden) {
      addCardCode(card?.cardId ?? null, cardCodes);
    }
    for (const card of board.alley) {
      addCardCode(card?.cardId ?? null, cardCodes);
    }
    for (const ikz of board.ikzArea) {
      addCardCode(ikz.cardId, cardCodes);
    }
  }

  for (const card of hand) {
    addCardCode(card.cardId, cardCodes);
  }

  return cardCodes;
}

function addCardCode(cardCode: string | null, cardCodes: Set<string>): void {
  if (cardCode) {
    cardCodes.add(cardCode);
  }
}
