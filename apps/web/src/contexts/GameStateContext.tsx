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

interface GameStateContextValue {
  gameState: GameState | null;
  isLoading: boolean;
  error: string | null;

  // Card mappings for resolving cardCode -> imageUrl
  cardMappings: Map<string, CardMapping>;
  setCardMappings: (mappings: Map<string, CardMapping>) => void;

  // For dev/testing: directly set mock state
  setMockState: (state: GameState) => void;

  // For production: process WebSocket messages
  processSnapshot: (
    snapshot: GameSnapshotMessage,
    playerSlot: 0 | 1,
    cardMappingsOverride?: Map<string, CardMapping>
  ) => void;
  processLogBatch: (batch: GameLogBatchMessage) => void;

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

  const setCardMappings = useCallback((mappings: Map<string, CardMapping>) => {
    setCardMappingsState(mappings);
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

  const processLogBatch = useCallback((_batch: GameLogBatchMessage) => {
    // TODO: Implement incremental state updates from game log
    // For now, we rely on full snapshots
  }, []);

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
} from "@/types/game";
import { buildImageUrl } from "@/types/game";

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

  return {
    phase: snapshot.stateContext.phase,
    abilitySubphase: snapshot.stateContext.abilitySubphase,
    activePlayer: snapshot.stateContext.activePlayer,
    turnNumber: snapshot.stateContext.turnNumber,
    myBoard: resolvePlayerBoard(snapshot.players[myBoardIndex], cardMappings),
    opponentBoard: resolvePlayerBoard(snapshot.players[opponentBoardIndex], cardMappings),
    myHand: snapshot.yourHand.map((card) => resolveHandCard(card, cardMappings)),
    actionMask: snapshot.actionMask,
    combatStack: snapshot.combatStack,
  };
}
