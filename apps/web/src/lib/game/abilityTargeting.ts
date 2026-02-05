import type { GameState, ResolvedPlayerBoard } from "@/types/game";

export enum AbilityTargetType {
  NONE = 0,
  SELF = 1,
  FRIENDLY_HAND = 2,
  FRIENDLY_IKZ = 3,
  FRIENDLY_GARDEN_ENTITY = 4,
  FRIENDLY_ALLEY_ENTITY = 5,
  FRIENDLY_ENTITY_WITH_WEAPON = 6,
  FRIENDLY_LEADER = 7,
  ENEMY_GARDEN_ENTITY = 8,
  ENEMY_LEADER = 9,
  ENEMY_LEADER_OR_GARDEN_ENTITY = 10,
  ANY_LEADER_OR_GARDEN_ENTITY = 11,
  ANY_GARDEN_ENTITY = 12,
  FRIENDLY_SELECTION = 13,
  FRIENDLY_SELECTION_WEAPON = 14,
  FRIENDLY_HAND_WEAPON = 15,
  ANY_LEADER = 16,
}

export interface AbilityTargetMaps {
  hand: Map<number, number>;
  selfGarden: Map<number, number>;
  opponentGarden: Map<number, number>;
  selfLeader: number | null;
  opponentLeader: number | null;
}

function findGardenSlotIndex(
  board: ResolvedPlayerBoard,
  zoneIndex: number
): number | null {
  const slotIndex = board.garden.findIndex(
    (card) => card && card.zoneIndex === zoneIndex
  );
  if (slotIndex >= 0) return slotIndex;
  if (zoneIndex >= 0 && zoneIndex < board.garden.length) return zoneIndex;
  return null;
}

function addGardenTargetByZoneIndex(
  map: Map<number, number>,
  board: ResolvedPlayerBoard,
  zoneIndex: number,
  targetIndex: number
) {
  const slotIndex = findGardenSlotIndex(board, zoneIndex);
  if (slotIndex === null) return;
  map.set(slotIndex, targetIndex);
}

function addGardenTargetByOrderedIndex(
  map: Map<number, number>,
  board: ResolvedPlayerBoard,
  orderedIndex: number,
  targetIndex: number
) {
  if (orderedIndex < 0 || orderedIndex >= board.garden.length) return;
  map.set(orderedIndex, targetIndex);
}

export function isHandTargetType(targetType?: number): boolean {
  return (
    targetType === AbilityTargetType.FRIENDLY_HAND ||
    targetType === AbilityTargetType.FRIENDLY_HAND_WEAPON
  );
}

export function buildAbilityTargetMaps({
  targetType,
  targetIndices,
  gameState,
}: {
  targetType?: number;
  targetIndices: number[];
  gameState: GameState;
}): AbilityTargetMaps {
  const maps: AbilityTargetMaps = {
    hand: new Map(),
    selfGarden: new Map(),
    opponentGarden: new Map(),
    selfLeader: null,
    opponentLeader: null,
  };

  if (targetType === undefined || targetType === null) {
    return maps;
  }

  const myBoard = gameState.myBoard;
  const opponentBoard = gameState.opponentBoard;

  switch (targetType) {
    case AbilityTargetType.FRIENDLY_HAND:
    case AbilityTargetType.FRIENDLY_HAND_WEAPON:
      for (const idx of targetIndices) {
        maps.hand.set(idx, idx);
      }
      break;

    case AbilityTargetType.FRIENDLY_GARDEN_ENTITY:
      for (const idx of targetIndices) {
        addGardenTargetByZoneIndex(maps.selfGarden, myBoard, idx, idx);
      }
      break;

    case AbilityTargetType.ENEMY_GARDEN_ENTITY:
      for (const idx of targetIndices) {
        addGardenTargetByZoneIndex(maps.opponentGarden, opponentBoard, idx, idx);
      }
      break;

    case AbilityTargetType.ANY_GARDEN_ENTITY:
      for (const idx of targetIndices) {
        if (idx < 5) {
          addGardenTargetByZoneIndex(maps.selfGarden, myBoard, idx, idx);
        } else {
          addGardenTargetByZoneIndex(maps.opponentGarden, opponentBoard, idx - 5, idx);
        }
      }
      break;

    case AbilityTargetType.ENEMY_LEADER_OR_GARDEN_ENTITY:
      for (const idx of targetIndices) {
        if (idx === 5) {
          maps.opponentLeader = idx;
        } else {
          addGardenTargetByZoneIndex(maps.opponentGarden, opponentBoard, idx, idx);
        }
      }
      break;

    case AbilityTargetType.ANY_LEADER:
      for (const idx of targetIndices) {
        if (idx === 0) {
          maps.selfLeader = idx;
        } else if (idx === 1) {
          maps.opponentLeader = idx;
        }
      }
      break;

    case AbilityTargetType.FRIENDLY_LEADER:
      if (targetIndices.length > 0) {
        maps.selfLeader = targetIndices[0];
      }
      break;

    case AbilityTargetType.ENEMY_LEADER:
      if (targetIndices.length > 0) {
        maps.opponentLeader = targetIndices[0];
      }
      break;

    default:
      break;
  }

  return maps;
}
