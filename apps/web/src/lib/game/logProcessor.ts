/**
 * Log processor for applying game logs to client state.
 * Transforms GAME_LOG_BATCH messages into GameState updates.
 */

import type { GameState, CardMapping, ResolvedCard, ResolvedHandCard, ResolvedIkz } from "@/types/game";
import { buildImageUrl } from "@/types/game";
import type {
  ProcessedGameLog,
  CardZoneMovedData,
  CardStatChangeData,
  CardTapChangeData,
  KeywordsChangedData,
  StatusAppliedData,
  StatusExpiredData,
  ZoneType,
  ZoneMovedMetadata,
  LogCardRef,
} from "@/types/gameLogs";

// ============================================
// Main entry point
// ============================================

/**
 * Apply a batch of game logs to the current state.
 * Returns a new state with all log effects applied.
 */
export function applyLogBatch(
  state: GameState,
  logs: ProcessedGameLog[],
  playerSlot: 0 | 1,
  cardMappings: Map<string, CardMapping>,
  cardDefIdMap: Map<number, CardMapping>
): GameState {
  let newState = { ...state };

  for (const log of logs) {
    newState = applyLog(newState, log, playerSlot, cardMappings, cardDefIdMap);
  }

  return newState;
}

/**
 * Apply a single game log to the state.
 */
function applyLog(
  state: GameState,
  log: ProcessedGameLog,
  playerSlot: 0 | 1,
  cardMappings: Map<string, CardMapping>,
  cardDefIdMap: Map<number, CardMapping>
): GameState {
  switch (log.type) {
    case "ZONE_MOVED":
      return applyZoneMoved(state, log.data, playerSlot, cardMappings, cardDefIdMap);

    case "STAT_CHANGE":
      return applyStatChange(state, log.data, playerSlot);

    case "KEYWORDS_CHANGED":
      return applyKeywordsChange(state, log.data, playerSlot);

    case "TAP_CHANGED":
      return applyTapStateChange(state, log.data, playerSlot);

    case "STATUS_EFFECT_APPLIED":
      return applyStatusEffect(state, log.data, playerSlot, true);

    case "STATUS_EFFECT_EXPIRED":
      return applyStatusEffect(state, log.data as StatusAppliedData, playerSlot, false);

    // These logs don't require state updates (handled by stateContext)
    case "TURN_STARTED":
    case "TURN_ENDED":
    case "DECK_SHUFFLED":
    case "COMBAT_DECLARED":
    case "DEFENDER_DECLARED":
    case "COMBAT_DAMAGE":
    case "ENTITY_DIED":
    case "EFFECT_QUEUED":
    case "CARD_EFFECT_ENABLED":
    case "GAME_ENDED":
      return state;

    default: {
      console.error(`Unknown log type: ${(log as ProcessedGameLog).type}`);
      return state;
    }
  }
}

// ============================================
// Zone movement handling
// ============================================

function applyZoneMoved(
  state: GameState,
  data: CardZoneMovedData,
  playerSlot: 0 | 1,
  cardMappings: Map<string, CardMapping>,
  cardDefIdMap: Map<number, CardMapping>
): GameState {
  console.log('Applying zone move:', data);
  const isMyCard = data.card.player === playerSlot;

  // Create mutable copies
  let newState = deepCopyState(state);

  // Remove from source zone
  newState = removeFromZone(newState, data.fromZone, data.fromIndex, isMyCard, data.card.player, playerSlot);

  // Add to destination zone
  newState = addToZone(
    newState,
    data.toZone,
    data.toIndex,
    isMyCard,
    data.card,
    data.metadata,
    cardMappings,
    cardDefIdMap
  );

  return newState;
}

function removeFromZone(
  state: GameState,
  zone: ZoneType,
  index: number,
  isMyCard: boolean,
  cardPlayer: 0 | 1,
  viewerSlot: 0 | 1
): GameState {
  switch (zone) {
    case "HAND":
      if (isMyCard) {
        // Remove from my hand array
        const newHand = state.myHand.filter((_, i) => i !== index);
        if (newHand.length === state.myHand.length) {
          console.error(`Warning: Attempted to remove card from my hand at invalid index ${index}`);
        }

        state.myHand = newHand;
      } else {
        // Decrement opponent hand count
        state.opponentBoard = {
          ...state.opponentBoard,
          handCount: Math.max(0, state.opponentBoard.handCount - 1),
        };
      }
      break;

    case "DECK":
      if (isMyCard) {
        state.myBoard = {
          ...state.myBoard,
          deckCount: Math.max(0, state.myBoard.deckCount - 1),
        };
      } else {
        state.opponentBoard = {
          ...state.opponentBoard,
          deckCount: Math.max(0, state.opponentBoard.deckCount - 1),
        };
      }
      break;

    case "GARDEN":
      if (isMyCard) {
        const newGarden = state.myBoard.garden.map((card, i) => {
          if (i === index) {
            if (card === null) {
              console.error(`Warning: Attempted to remove card from my garden at invalid index ${index}`);
            }

            return null;
          } else {
            return card
          }
      });
        state.myBoard = {
          ...state.myBoard,
          garden: newGarden,
        };
      } else {
        const newGarden = state.opponentBoard.garden.map((card, i) => {
          if (i === index) {
            if (card === null) {
              console.error(`Warning: Attempted to remove card from opponent garden at invalid index ${index}`);
            }

            return null;
          } else {
            return card
          }
        });

        state.opponentBoard = {
          ...state.opponentBoard,
          garden: newGarden
        };
      }
      break;

    case "ALLEY":
      if (isMyCard) {
        const newAlley = state.myBoard.alley.map((card, i) => {
          if (i === index) {
            if (card === null) {
              console.error(`Warning: Attempted to remove card from my alley at invalid index ${index}`);
            }

            return null;
          } else {
            return card
          }
        });

        state.myBoard = {
          ...state.myBoard,
          alley: newAlley,
        };
      } else {
        const newAlley = state.opponentBoard.alley.map((card, i) => {
          if (i === index) {
            if (card === null) {
              console.error(`Warning: Attempted to remove card from opponent alley at invalid index ${index}`);
            }

            return null;
          } else {
            return card
          }
        });

        state.opponentBoard = {
          ...state.opponentBoard,
          alley: newAlley
        };
      }
      break;

    case "IKZ_PILE":
      if (isMyCard) {
        state.myBoard = {
          ...state.myBoard,
          ikzPileCount: Math.max(0, state.myBoard.ikzPileCount - 1),
        };
      } else {
        state.opponentBoard = {
          ...state.opponentBoard,
          ikzPileCount: Math.max(0, state.opponentBoard.ikzPileCount - 1),
        };
      }
      break;

    case "IKZ_AREA":
      if (isMyCard) {
        const newIkzArea = state.myBoard.ikzArea.filter((_, i) => i !== index);
        if (newIkzArea.length === state.myBoard.ikzArea.length) {
          console.error(`Warning: Attempted to remove IKZ from my IKZ area at invalid index ${index}`);
        }

        state.myBoard = {
          ...state.myBoard,
          ikzArea: newIkzArea,
        };
      } else {
        const newIkzArea = state.opponentBoard.ikzArea.filter((_, i) => i !== index);
        if (newIkzArea.length === state.opponentBoard.ikzArea.length) {
          console.error(`Warning: Attempted to remove IKZ from opponent IKZ area at invalid index ${index}`);
        }

        state.opponentBoard = {
          ...state.opponentBoard,
          ikzArea: newIkzArea,
        };
      }
      break;

    case "DISCARD":
      if (isMyCard) {
        state.myBoard = {
          ...state.myBoard,
          discardCount: Math.max(0, state.myBoard.discardCount - 1),
        };
      } else {
        state.opponentBoard = {
          ...state.opponentBoard,
          discardCount: Math.max(0, state.opponentBoard.discardCount - 1),
        };
      }
      break;

    case "EQUIPPED":
      // No-op: equipped weapons aren't tracked separately in client state
      // They're attached to entities and tracked via metadata
      break;

    // LEADER, GATE, SELECTION - no removal needed for these zones
    default:
      break;
  }

  return state;
}

function addToZone(
  state: GameState,
  zone: ZoneType,
  index: number,
  isMyCard: boolean,
  cardRef: LogCardRef,
  metadata: ZoneMovedMetadata | null,
  cardMappings: Map<string, CardMapping>,
  cardDefIdMap: Map<number, CardMapping>
): GameState {
  switch (zone) {
    case "HAND":
      const cardRefCardDefId = cardRef.cardDefId;
      if (cardRefCardDefId === null) {
        console.error("Warning: Cannot add card to hand with null cardDefId");
        break;
      }

      if (isMyCard) {
        // Add to my hand
        const handCard = resolveHandCard(cardRefCardDefId, cardDefIdMap, cardMappings);
        if (handCard == null) {
          console.error(`Warning: Could not resolve hand card with def ID ${cardRefCardDefId}`);
          return state;
        }

        // Insert at the specified index, or append if index is at/beyond the end
        if (index >= 0 && index < state.myHand.length) {
          console.debug(`Inserting card into my hand at index ${index}`);

          // Insert at specific position
          const newHand = [...state.myHand];
          newHand.splice(index, 0, handCard);
          state.myHand = newHand;
        } else {
          console.debug(`Appending card to end of my hand`);
          
          // Append to end (index is at or beyond current length)
          state.myHand = [...state.myHand, handCard];
        }
      } else {
        // Increment opponent hand count
        state.opponentBoard = {
          ...state.opponentBoard,
          handCount: state.opponentBoard.handCount + 1,
        };
      }
      break;

    case "DECK":
      if (isMyCard) {
        state.myBoard = {
          ...state.myBoard,
          deckCount: state.myBoard.deckCount + 1,
        };
      } else {
        state.opponentBoard = {
          ...state.opponentBoard,
          deckCount: state.opponentBoard.deckCount + 1,
        };
      }
      break;

    case "GARDEN":
      if (index >= 0 && index < 5 && cardRef.cardDefId !== null) {
        const resolvedCard = resolveCard(cardRef, metadata, cardDefIdMap, cardMappings);
        if (resolvedCard) {
          if (isMyCard) {
            const garden = [...state.myBoard.garden];
            garden[index] = resolvedCard;
            state.myBoard = { ...state.myBoard, garden };
          } else {
            const garden = [...state.opponentBoard.garden];
            garden[index] = resolvedCard;
            state.opponentBoard = { ...state.opponentBoard, garden };
          }
        }
      }
      break;

    case "ALLEY":
      if (index >= 0 && index < 5 && cardRef.cardDefId !== null) {
        const resolvedCard = resolveCard(cardRef, metadata, cardDefIdMap, cardMappings);
        if (resolvedCard) {
          if (isMyCard) {
            const alley = [...state.myBoard.alley];
            alley[index] = resolvedCard;
            state.myBoard = { ...state.myBoard, alley };
          } else {
            const alley = [...state.opponentBoard.alley];
            alley[index] = resolvedCard;
            state.opponentBoard = { ...state.opponentBoard, alley };
          }
        }
      }
      break;

    case "IKZ_AREA":
      if (cardRef.cardDefId !== null) {
        const ikz = resolveIkz(cardRef.cardDefId, metadata, cardDefIdMap, cardMappings);
        if (ikz) {
          if (isMyCard) {
            state.myBoard = {
              ...state.myBoard,
              ikzArea: [...state.myBoard.ikzArea, ikz],
            };
          } else {
            state.opponentBoard = {
              ...state.opponentBoard,
              ikzArea: [...state.opponentBoard.ikzArea, ikz],
            };
          }
        }
      }
      break;

    case "IKZ_PILE":
      if (isMyCard) {
        state.myBoard = {
          ...state.myBoard,
          ikzPileCount: state.myBoard.ikzPileCount + 1,
        };
      } else {
        state.opponentBoard = {
          ...state.opponentBoard,
          ikzPileCount: state.opponentBoard.ikzPileCount + 1,
        };
      }
      break;

    case "DISCARD":
      if (isMyCard) {
        state.myBoard = {
          ...state.myBoard,
          discardCount: state.myBoard.discardCount + 1,
        };
      } else {
        state.opponentBoard = {
          ...state.opponentBoard,
          discardCount: state.opponentBoard.discardCount + 1,
        };
      }
      break;

    case "EQUIPPED":
      // No-op: equipped weapons aren't tracked separately in client state
      // The entity's attached_weapons metadata will reflect this
      break;

    // LEADER, GATE, SELECTION - handled specially or not needed
    default:
      break;
  }

  return state;
}

// ============================================
// Stat change handling
// ============================================

function applyStatChange(
  state: GameState,
  data: CardStatChangeData,
  playerSlot: 0 | 1
): GameState {
  const isMyCard = data.card.player === playerSlot;
  const board = isMyCard ? state.myBoard : state.opponentBoard;

  // Find and update the card in its zone
  switch (data.card.zone) {
    case "LEADER":
      const newLeader = {
        ...board.leader,
        curAtk: data.newAtk,
        curHp: data.newHp,
      };
      if (isMyCard) {
        return { ...state, myBoard: { ...state.myBoard, leader: newLeader } };
      } else {
        return { ...state, opponentBoard: { ...state.opponentBoard, leader: newLeader } };
      }

    case "GARDEN":
      const gardenCard = board.garden[data.card.zoneIndex];
      if (gardenCard) {
        const newCard = { ...gardenCard, curAtk: data.newAtk, curHp: data.newHp };
        const garden = [...board.garden];
        garden[data.card.zoneIndex] = newCard;
        if (isMyCard) {
          return { ...state, myBoard: { ...state.myBoard, garden } };
        } else {
          return { ...state, opponentBoard: { ...state.opponentBoard, garden } };
        }
      }
      break;

    case "ALLEY":
      const alleyCard = board.alley[data.card.zoneIndex];
      if (alleyCard) {
        const newCard = { ...alleyCard, curAtk: data.newAtk, curHp: data.newHp };
        const alley = [...board.alley];
        alley[data.card.zoneIndex] = newCard;
        if (isMyCard) {
          return { ...state, myBoard: { ...state.myBoard, alley } };
        } else {
          return { ...state, opponentBoard: { ...state.opponentBoard, alley } };
        }
      }
      break;
  }

  return state;
}

// ============================================
// Keyword change handling
// ============================================

function applyKeywordsChange(
  state: GameState,
  data: KeywordsChangedData,
  playerSlot: 0 | 1
): GameState {
  const isMyCard = data.card.player === playerSlot;
  const board = isMyCard ? state.myBoard : state.opponentBoard;

  const keywordUpdate = {
    hasCharge: data.hasCharge,
    hasDefender: data.hasDefender,
    hasInfiltrate: data.hasInfiltrate,
  };

  switch (data.card.zone) {
    case "LEADER": {
      const newLeader = { ...board.leader, ...keywordUpdate };
      if (isMyCard) {
        return { ...state, myBoard: { ...state.myBoard, leader: newLeader } };
      }
      return { ...state, opponentBoard: { ...state.opponentBoard, leader: newLeader } };
    }

    case "GARDEN": {
      const gardenCard = board.garden[data.card.zoneIndex];
      if (gardenCard) {
        const newCard = { ...gardenCard, ...keywordUpdate };
        const garden = [...board.garden];
        garden[data.card.zoneIndex] = newCard;
        if (isMyCard) {
          return { ...state, myBoard: { ...state.myBoard, garden } };
        }
        return { ...state, opponentBoard: { ...state.opponentBoard, garden } };
      }
      break;
    }

    case "ALLEY": {
      const alleyCard = board.alley[data.card.zoneIndex];
      if (alleyCard) {
        const newCard = { ...alleyCard, ...keywordUpdate };
        const alley = [...board.alley];
        alley[data.card.zoneIndex] = newCard;
        if (isMyCard) {
          return { ...state, myBoard: { ...state.myBoard, alley } };
        }
        return { ...state, opponentBoard: { ...state.opponentBoard, alley } };
      }
      break;
    }
  }

  return state;
}

// ============================================
// Tap state handling
// ============================================

function applyTapStateChange(
  state: GameState,
  data: CardTapChangeData,
  playerSlot: 0 | 1
): GameState {
  console.log("Applying tap state change:", data);

  const isMyCard = data.card.player === playerSlot;
  const board = isMyCard ? state.myBoard : state.opponentBoard;

  const tapped = data.newState === "TAPPED";
  const cooldown = data.newState === "COOLDOWN";

  switch (data.card.zone) {
    case "LEADER":
      const newLeader = { ...board.leader, tapped, cooldown };
      if (isMyCard) {
        return { ...state, myBoard: { ...state.myBoard, leader: newLeader } };
      } else {
        return { ...state, opponentBoard: { ...state.opponentBoard, leader: newLeader } };
      }

    case "GATE":
      const newGate = { ...board.gate, tapped, cooldown };
      if (isMyCard) {
        return { ...state, myBoard: { ...state.myBoard, gate: newGate } };
      } else {
        return { ...state, opponentBoard: { ...state.opponentBoard, gate: newGate } };
      }

    case "GARDEN": {
      const gardenCard = board.garden[data.card.zoneIndex];
      if (gardenCard == null) {
        console.error(`Warning: Could not find card in ${isMyCard ? "my" : "opponent"} garden at index ${data.card.zoneIndex} to update tap state`);
        break;
      }

      const newCard = { ...gardenCard, tapped, cooldown };
      const garden = [...board.garden];
      garden[data.card.zoneIndex] = newCard;
      if (isMyCard) {
        return { ...state, myBoard: { ...state.myBoard, garden } };
      } else {
        return { ...state, opponentBoard: { ...state.opponentBoard, garden } };
      }
    }

    case "ALLEY": {
      const alleyCard = board.alley[data.card.zoneIndex];
      if (alleyCard == null) {
        console.error(`Warning: Could not find card in ${isMyCard ? "my" : "opponent"} alley at index ${data.card.zoneIndex} to update tap state`);
        break;
      }

      const newCard = { ...alleyCard, tapped, cooldown };
      const alley = [...board.alley];
      alley[data.card.zoneIndex] = newCard;
      if (isMyCard) {
        return { ...state, myBoard: { ...state.myBoard, alley } };
      } else {
        return { ...state, opponentBoard: { ...state.opponentBoard, alley } };
      }
    }

    case "IKZ_AREA": {
      const ikzCard = board.ikzArea[data.card.zoneIndex];
      if (ikzCard == null) {
        console.error(`Warning: Could not find IKZ in ${isMyCard ? "my" : "opponent"} IKZ area at index ${data.card.zoneIndex} to update tap state`);
        break;
      }

      const newIkz = { ...ikzCard, tapped, cooldown };
      const ikzArea = [...board.ikzArea];
      ikzArea[data.card.zoneIndex] = newIkz;
      if (isMyCard) {
        return { ...state, myBoard: { ...state.myBoard, ikzArea } };
      } else {
        return { ...state, opponentBoard: { ...state.opponentBoard, ikzArea } };
      }
    }
  }

  return state;
}

// ============================================
// Status effect handling
// ============================================

function applyStatusEffect(
  state: GameState,
  data: StatusAppliedData | StatusExpiredData,
  playerSlot: 0 | 1,
  isApplied: boolean
): GameState {
  const isMyCard = data.card.player === playerSlot;
  const board = isMyCard ? state.myBoard : state.opponentBoard;

  const effectUpdate = {
    isFrozen: data.effect === "FROZEN" ? isApplied : undefined,
    isShocked: data.effect === "SHOCKED" ? isApplied : undefined,
    isEffectImmune: data.effect === "EFFECT_IMMUNE" ? isApplied : undefined,
  };

  // Remove undefined values
  const cleanUpdate = Object.fromEntries(
    Object.entries(effectUpdate).filter(([_, v]) => v !== undefined)
  );

  switch (data.card.zone) {
    case "GARDEN":
      const gardenCard = board.garden[data.card.zoneIndex];
      if (gardenCard) {
        const newCard = { ...gardenCard, ...cleanUpdate };
        const garden = [...board.garden];
        garden[data.card.zoneIndex] = newCard as ResolvedCard;
        if (isMyCard) {
          return { ...state, myBoard: { ...state.myBoard, garden } };
        } else {
          return { ...state, opponentBoard: { ...state.opponentBoard, garden } };
        }
      }
      break;

    case "ALLEY":
      const alleyCard = board.alley[data.card.zoneIndex];
      if (alleyCard) {
        const newCard = { ...alleyCard, ...cleanUpdate };
        const alley = [...board.alley];
        alley[data.card.zoneIndex] = newCard as ResolvedCard;
        if (isMyCard) {
          return { ...state, myBoard: { ...state.myBoard, alley } };
        } else {
          return { ...state, opponentBoard: { ...state.opponentBoard, alley } };
        }
      }
      break;
  }

  return state;
}

// ============================================
// Card resolution helpers
// ============================================

function resolveCard(
  cardRef: LogCardRef,
  metadata: ZoneMovedMetadata | null,
  cardDefIdMap: Map<number, CardMapping>,
  cardMappings: Map<string, CardMapping>
): ResolvedCard | null {
  if (cardRef.cardDefId === null) {
    return null;
  }

  const mapping = cardDefIdMap.get(cardRef.cardDefId);
  if (!mapping) {
    return null;
  }

  return {
    cardCode: mapping.cardCode,
    cardDefId: cardRef.cardDefId,
    imageUrl: buildImageUrl(mapping.imageKey),
    name: mapping.name,
    curAtk: metadata?.curAtk ?? mapping.attack,
    curHp: metadata?.curHp ?? mapping.health,
    tapped: metadata?.tapped ?? false,
    cooldown: metadata?.cooldown ?? false,
    isFrozen: metadata?.isFrozen ?? false,
    isShocked: false, // Not in metadata
    isEffectImmune: metadata?.isEffectImmune ?? false,
    hasCharge: metadata?.hasCharge ?? false,
    hasDefender: metadata?.hasDefender ?? false,
    hasInfiltrate: metadata?.hasInfiltrate ?? false,
    zoneIndex: cardRef.zoneIndex,
  };
}

function resolveHandCard(
  cardDefId: number,
  cardDefIdMap: Map<number, CardMapping>,
  cardMappings: Map<string, CardMapping>
): ResolvedHandCard | null {
  const mapping = cardDefIdMap.get(cardDefId);
  if (!mapping) {
    return null;
  }

  return {
    cardCode: mapping.cardCode,
    cardDefId,
    imageUrl: buildImageUrl(mapping.imageKey),
    name: mapping.name,
    type: mapping.cardType,
    ikzCost: mapping.ikzCost ?? 0,
  };
}

function resolveIkz(
  cardDefId: number,
  metadata: ZoneMovedMetadata | null,
  cardDefIdMap: Map<number, CardMapping>,
  cardMappings: Map<string, CardMapping>
): ResolvedIkz | null {
  const mapping = cardDefIdMap.get(cardDefId);
  if (!mapping) {
    return null;
  }

  return {
    cardCode: mapping.cardCode,
    cardDefId,
    imageUrl: buildImageUrl(mapping.imageKey),
    name: mapping.name,
    tapped: metadata?.tapped ?? false,
    cooldown: metadata?.cooldown ?? false,
  };
}

// ============================================
// Deep copy helper
// ============================================

function deepCopyState(state: GameState): GameState {
  return {
    ...state,
    myBoard: {
      ...state.myBoard,
      garden: [...state.myBoard.garden],
      alley: [...state.myBoard.alley],
      ikzArea: [...state.myBoard.ikzArea],
    },
    opponentBoard: {
      ...state.opponentBoard,
      garden: [...state.opponentBoard.garden],
      alley: [...state.opponentBoard.alley],
      ikzArea: [...state.opponentBoard.ikzArea],
    },
    myHand: [...state.myHand],
  };
}
