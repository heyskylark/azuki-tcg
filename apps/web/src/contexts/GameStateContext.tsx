"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from "react";
import type {
  GameSnapshotMessage,
  GameLogBatchMessage,
} from "@tcg/backend-core/types/ws";
import type { GameState, CardMapping } from "@/types/game";
import type { ProcessedGameLog } from "@/types/gameLogs";
import { applyLogBatch } from "@/lib/game/logProcessor";

interface GameStateContextValue {
  gameState: GameState | null;
  isLoading: boolean;
  error: string | null;

  // Card mappings for resolving cardCode -> imageUrl
  cardMappings: Map<string, CardMapping>;
  setCardMappings: (mappings: Map<string, CardMapping>) => void;

  // Reverse lookup: cardDefId -> CardMapping (built from deck data or snapshot)
  cardDefIdMap: Map<number, CardMapping>;
  setCardDefIdMap: (map: Map<number, CardMapping>) => void;

  // For dev/testing: directly set mock state
  setMockState: (state: GameState) => void;

  // For production: process WebSocket messages
  processSnapshot: (
    snapshot: GameSnapshotMessage,
    playerSlot: 0 | 1,
    cardMappingsOverride?: Map<string, CardMapping>
  ) => void;
  processLogBatch: (batch: GameLogBatchMessage, playerSlot: 0 | 1) => void;

  // Clear state
  clearGameState: () => void;
}

const GameStateContext = createContext<GameStateContextValue | null>(null);

interface GameStateProviderProps {
  children: ReactNode;
  initialState?: GameState | null;
}

export function GameStateProvider({
  children,
  initialState = null,
}: GameStateProviderProps) {
  const [gameState, setGameState] = useState<GameState | null>(initialState);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cardMappings, setCardMappingsState] = useState<Map<string, CardMapping>>(
    new Map()
  );
  const [cardDefIdMap, setCardDefIdMap] = useState<Map<number, CardMapping>>(
    new Map()
  );

  const setCardMappings = useCallback((mappings: Map<string, CardMapping>) => {
    setCardMappingsState(mappings);
  }, []);

  const setCardDefIdMapCallback = useCallback((map: Map<number, CardMapping>) => {
    setCardDefIdMap(map);
  }, []);

  const setMockState = useCallback((state: GameState) => {
    setGameState(state);
    setIsLoading(false);
    setError(null);
  }, []);

  const processSnapshot = useCallback(
    (
      snapshot: GameSnapshotMessage,
      playerSlot: 0 | 1,
      cardMappingsOverride?: Map<string, CardMapping>
    ) => {
      try {
        setIsLoading(true);

        // Transform snapshot to GameState using card mappings
        const mappings = cardMappingsOverride ?? cardMappings;
        const transformed = transformSnapshot(snapshot, mappings, playerSlot);

        // Only build cardDefIdMap from snapshot if it's not already populated
        // (e.g., when pre-built from deck data during loading)
        setCardDefIdMap((currentMap) => {
          if (currentMap.size > 0) {
            // Map already populated from deck data, skip snapshot-based building
            return currentMap;
          }
          // Build from snapshot as fallback
          return buildCardDefIdMap(snapshot, mappings);
        });

        setGameState(transformed);
        setError(null);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to process snapshot"
        );
      } finally {
        setIsLoading(false);
      }
    },
    [cardMappings]
  );

  const processLogBatch = useCallback(
    (batch: GameLogBatchMessage, playerSlot: 0 | 1) => {
      setGameState((prevState) => {
        if (!prevState) {
          return null;
        }

        // Apply log entries to update game state
        const logs = batch.logs as ProcessedGameLog[];
        const updatedState = applyLogBatch(
          prevState,
          logs,
          playerSlot,
          cardMappings,
          cardDefIdMap
        );

        // Resolve selection cards if present in the batch
        const stateContext = batch.stateContext as {
          phase: string;
          abilitySubphase: string;
          activePlayer: 0 | 1;
          turnNumber: number;
          abilitySourceCardDefId?: number;
          abilityCostTargetType?: number;
          abilityEffectTargetType?: number;
          selectionCards?: Array<{
            cardId: string | null;
            cardDefId: number;
            zoneIndex: number;
            type: string;
            ikzCost: number;
            curAtk: number | null;
            curHp: number | null;
          }>;
        };

        const selectionCards = stateContext.selectionCards?.map((card) => {
          const cardCode = card.cardId ?? "unknown";
          const mapping = cardMappings.get(cardCode);
          return {
            cardCode,
            cardDefId: card.cardDefId,
            zoneIndex: card.zoneIndex,
            imageUrl: mapping?.imageUrl ?? "",
            name: mapping?.name ?? cardCode,
            type: card.type,
            ikzCost: card.ikzCost,
            curAtk: card.curAtk,
            curHp: card.curHp,
          };
        });

        // Also update state context and action mask from the batch
        return {
          ...updatedState,
          phase: stateContext.phase,
          abilitySubphase: stateContext.abilitySubphase,
          activePlayer: stateContext.activePlayer,
          turnNumber: stateContext.turnNumber,
          abilitySourceCardDefId: stateContext.abilitySourceCardDefId,
          abilityCostTargetType: stateContext.abilityCostTargetType,
          abilityEffectTargetType: stateContext.abilityEffectTargetType,
          selectionCards,
          // Use action mask from batch if present (for active player), otherwise clear it
          actionMask: batch.actionMask ?? null,
        };
      });
    },
    [cardMappings, cardDefIdMap]
  );

  const clearGameState = useCallback(() => {
    setGameState(null);
    setIsLoading(false);
    setError(null);
  }, []);

  return (
    <GameStateContext.Provider
      value={{
        gameState,
        isLoading,
        error,
        cardMappings,
        setCardMappings,
        cardDefIdMap,
        setCardDefIdMap: setCardDefIdMapCallback,
        setMockState,
        processSnapshot,
        processLogBatch,
        clearGameState,
      }}
    >
      {children}
    </GameStateContext.Provider>
  );
}

export function useGameState() {
  const context = useContext(GameStateContext);
  if (!context) {
    throw new Error("useGameState must be used within a GameStateProvider");
  }
  return context;
}

// ============================================
// Snapshot transformation
// ============================================

import type {
  SnapshotCard,
  SnapshotLeader,
  SnapshotGate,
  SnapshotIkz,
  SnapshotHandCard,
  SnapshotPlayerBoard,
} from "@tcg/backend-core/types/ws";
import type {
  ResolvedCard,
  ResolvedLeader,
  ResolvedGate,
  ResolvedIkz,
  ResolvedHandCard,
  ResolvedPlayerBoard,
  ResolvedSelectionCard,
} from "@/types/game";
import { buildImageUrl } from "@/types/game";
import type { SnapshotSelectionCard } from "@tcg/backend-core/types/ws";

function resolveCard(
  card: SnapshotCard,
  cardMappings: Map<string, CardMapping>
): ResolvedCard {
  const cardCode = card.cardId ?? "unknown"; // cardId IS the cardCode, null for hidden cards
  const mapping = cardMappings.get(cardCode);

  return {
    cardCode,
    cardDefId: card.cardDefId,
    imageUrl: mapping ? buildImageUrl(mapping.imageKey) : "",
    name: mapping?.name ?? cardCode,
    curAtk: card.curAtk,
    curHp: card.curHp,
    tapped: card.tapped,
    cooldown: card.cooldown,
    isFrozen: card.isFrozen,
    isShocked: card.isShocked,
    isEffectImmune: card.isEffectImmune,
    hasCharge: card.hasCharge,
    hasDefender: card.hasDefender,
    hasInfiltrate: card.hasInfiltrate,
    zoneIndex: card.zoneIndex,
  };
}

function resolveLeader(
  leader: SnapshotLeader,
  cardMappings: Map<string, CardMapping>
): ResolvedLeader {
  const cardCode = leader.cardId ?? "unknown";
  const mapping = cardMappings.get(cardCode);

  return {
    cardCode,
    cardDefId: leader.cardDefId,
    imageUrl: mapping ? buildImageUrl(mapping.imageKey) : "",
    name: mapping?.name ?? cardCode,
    curAtk: leader.curAtk,
    curHp: leader.curHp,
    tapped: leader.tapped,
    cooldown: leader.cooldown,
    hasCharge: leader.hasCharge,
    hasDefender: leader.hasDefender,
    hasInfiltrate: leader.hasInfiltrate,
  };
}

function resolveGate(
  gate: SnapshotGate,
  cardMappings: Map<string, CardMapping>
): ResolvedGate {
  const cardCode = gate.cardId ?? "unknown";
  const mapping = cardMappings.get(cardCode);

  return {
    cardCode,
    cardDefId: gate.cardDefId,
    imageUrl: mapping ? buildImageUrl(mapping.imageKey) : "",
    name: mapping?.name ?? cardCode,
    tapped: gate.tapped,
    cooldown: gate.cooldown,
  };
}

function resolveIkz(
  ikz: SnapshotIkz,
  cardMappings: Map<string, CardMapping>
): ResolvedIkz {
  const cardCode = ikz.cardId ?? "unknown";
  const mapping = cardMappings.get(cardCode);

  return {
    cardCode,
    cardDefId: ikz.cardDefId,
    imageUrl: mapping ? buildImageUrl(mapping.imageKey) : "",
    name: mapping?.name ?? cardCode,
    tapped: ikz.tapped,
    cooldown: ikz.cooldown,
  };
}

function resolveHandCard(
  card: SnapshotHandCard,
  cardMappings: Map<string, CardMapping>
): ResolvedHandCard {
  const cardCode = card.cardId ?? "unknown";
  const mapping = cardMappings.get(cardCode);

  return {
    cardCode,
    cardDefId: card.cardDefId,
    imageUrl: mapping ? buildImageUrl(mapping.imageKey) : "",
    name: mapping?.name ?? cardCode,
    type: card.type,
    ikzCost: card.ikzCost,
  };
}

function resolveSelectionCard(
  card: SnapshotSelectionCard,
  cardMappings: Map<string, CardMapping>
): ResolvedSelectionCard {
  const cardCode = card.cardId ?? "unknown";
  const mapping = cardMappings.get(cardCode);

  return {
    cardCode,
    cardDefId: card.cardDefId,
    zoneIndex: card.zoneIndex,
    imageUrl: mapping ? buildImageUrl(mapping.imageKey) : "",
    name: mapping?.name ?? cardCode,
    type: card.type,
    ikzCost: card.ikzCost,
    curAtk: card.curAtk,
    curHp: card.curHp,
  };
}

function resolvePlayerBoard(
  board: SnapshotPlayerBoard,
  cardMappings: Map<string, CardMapping>
): ResolvedPlayerBoard {
  return {
    leader: resolveLeader(board.leader, cardMappings),
    gate: resolveGate(board.gate, cardMappings),
    garden: board.garden.map((card) =>
      card ? resolveCard(card, cardMappings) : null
    ),
    alley: board.alley.map((card) =>
      card ? resolveCard(card, cardMappings) : null
    ),
    ikzArea: board.ikzArea.map((ikz) => resolveIkz(ikz, cardMappings)),
    handCount: board.handCount,
    deckCount: board.deckCount,
    discardCount: board.discardCount,
    ikzPileCount: board.ikzPileCount,
    hasIkzToken: board.hasIkzToken,
  };
}

function transformSnapshot(
  snapshot: GameSnapshotMessage,
  cardMappings: Map<string, CardMapping>,
  playerSlot: 0 | 1
): GameState {
  // players[0] is always player 0's board, players[1] is player 1's board
  // Use playerSlot to orient "my" and "opponent" boards for the viewer
  const myBoardIndex = playerSlot === 0 ? 0 : 1;
  const opponentBoardIndex = playerSlot === 0 ? 1 : 0;

  // Resolve selection cards if present
  const selectionCards = snapshot.stateContext.selectionCards?.map((card) =>
    resolveSelectionCard(card, cardMappings)
  );

  return {
    phase: snapshot.stateContext.phase,
    abilitySubphase: snapshot.stateContext.abilitySubphase,
    activePlayer: snapshot.stateContext.activePlayer,
    turnNumber: snapshot.stateContext.turnNumber,
    abilitySourceCardDefId: snapshot.stateContext.abilitySourceCardDefId,
    abilityCostTargetType: snapshot.stateContext.abilityCostTargetType,
    abilityEffectTargetType: snapshot.stateContext.abilityEffectTargetType,
    myBoard: resolvePlayerBoard(snapshot.players[myBoardIndex], cardMappings),
    opponentBoard: resolvePlayerBoard(snapshot.players[opponentBoardIndex], cardMappings),
    myHand: snapshot.yourHand.map((card) => resolveHandCard(card, cardMappings)),
    selectionCards,
    actionMask: snapshot.actionMask,
    combatStack: snapshot.combatStack,
  };
}

/**
 * Build a reverse lookup map from cardDefId to CardMapping.
 * Scans all visible cards in the snapshot to extract cardDefId -> cardCode mappings,
 * then resolves to the full CardMapping.
 */
function buildCardDefIdMap(
  snapshot: GameSnapshotMessage,
  cardMappings: Map<string, CardMapping>
): Map<number, CardMapping> {
  const defIdMap = new Map<number, CardMapping>();

  // Helper to add a card if it has both cardDefId and cardId
  const addCard = (cardDefId: number, cardId: string | null) => {
    if (cardId && !defIdMap.has(cardDefId)) {
      const mapping = cardMappings.get(cardId);
      if (mapping) {
        defIdMap.set(cardDefId, mapping);
      }
    }
  };

  // Scan both player boards
  for (const board of snapshot.players) {
    // Leader
    addCard(board.leader.cardDefId, board.leader.cardId);

    // Gate
    addCard(board.gate.cardDefId, board.gate.cardId);

    // Garden
    for (const card of board.garden) {
      if (card) {
        addCard(card.cardDefId, card.cardId);
      }
    }

    // Alley
    for (const card of board.alley) {
      if (card) {
        addCard(card.cardDefId, card.cardId);
      }
    }

    // IKZ Area
    for (const ikz of board.ikzArea) {
      addCard(ikz.cardDefId, ikz.cardId);
    }
  }

  // Scan hand cards
  for (const card of snapshot.yourHand) {
    addCard(card.cardDefId, card.cardId);
  }

  return defIdMap;
}
