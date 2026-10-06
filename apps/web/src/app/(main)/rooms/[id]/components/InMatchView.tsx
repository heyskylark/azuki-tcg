"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { GameScene } from "@/components/game/GameScene";
import { GameConsoleCommands } from "@/components/game/GameConsoleCommands";
import { LoadingScreen } from "@/components/game/LoadingScreen";
import { DevDebugOverlay } from "@/components/game/DevDebugOverlay";
import { Button } from "@/components/ui/button";
import { useAssets } from "@/contexts/AssetContext";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import { authenticatedFetch } from "@/lib/api/authenticatedFetch";
import type { DeckCard, DeckWithCards } from "@/types/game";
import { buildCardDefIdMapFromDeckCards } from "@/types/game";
import type { RoomEvaluationMetadata } from "@tcg/backend-core/types/ws";

interface DeckApiResponse {
  deck: DeckWithCards;
}

interface InMatchViewProps {
  evaluation: RoomEvaluationMetadata | null;
  evaluationCardCatalog: DeckCard[] | null;
}

async function fetchDeckCards(deckId: string): Promise<DeckCard[]> {
  const response = await authenticatedFetch(`/api/decks/${deckId}`);

  if (!response.ok) {
    throw new Error("Failed to fetch deck data");
  }

  const payload: DeckApiResponse = await response.json();

  return payload.deck.cards;
}

function formatPhaseLabel(phase: string | undefined): string {
  if (!phase) {
    return "-";
  }

  return phase
    .toLowerCase()
    .split("_")
    .filter((part) => part.length > 0)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function InMatchView({ evaluation, evaluationCardCatalog }: InMatchViewProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { loadingState, preloadDeckCards } = useAssets();
  const { gameState, isLoading, setCardMappings, setCardDefIdMap } = useGameState();
  const { roomState, activeRoom, connectionStatus, send } = useRoom();

  const [isDeckLoading, setIsDeckLoading] = useState(true);
  const [deckLoadError, setDeckLoadError] = useState<string | null>(null);

  const isDevMode = searchParams.get("dev") === "true";
  const isInMatch = roomState?.status === "IN_MATCH";
  const player0DeckId = roomState?.players[0]?.deckId ?? null;
  const player1DeckId = roomState?.players[1]?.deckId ?? null;

  // In an evaluation room the opponent's deck id is redacted, so the only deck the
  // client may fetch is the reviewer's own — the one non-null deck id.
  const isEvaluationRoom = evaluation !== null;
  const evaluationHumanDeckId = player0DeckId ?? player1DeckId;

  // Fetch deck data and preload assets when entering match
  useEffect(() => {
    if (!isInMatch) {
      return;
    }

    const isEvaluationMatch = isEvaluationRoom;

    const hasRequiredDecks = isEvaluationMatch
      ? evaluationHumanDeckId !== null && evaluationCardCatalog !== null
      : player0DeckId !== null && player1DeckId !== null;

    if (!hasRequiredDecks) {
      console.error("Missing deck data in IN_MATCH state");
      setDeckLoadError("Missing deck information");
      setIsDeckLoading(false);
      return;
    }

    let isCancelled = false;

    async function loadGameAssets() {
      try {
        setDeckLoadError(null);
        setIsDeckLoading(true);

        // Evaluation matches never fetch the generated opponent deck. The reviewer's
        // own deck is loaded normally and the full playable catalog covers every card
        // the opponent can reveal later in the match, without leaking its contents.
        let allCards: DeckCard[];

        if (isEvaluationMatch) {
          if (evaluationHumanDeckId === null || evaluationCardCatalog === null) {
            throw new Error("Missing deck information");
          }

          allCards = [...(await fetchDeckCards(evaluationHumanDeckId)), ...evaluationCardCatalog];
        } else {
          if (player0DeckId === null || player1DeckId === null) {
            throw new Error("Missing deck information");
          }

          const [player0Cards, player1Cards] = await Promise.all([
            fetchDeckCards(player0DeckId),
            fetchDeckCards(player1DeckId),
          ]);

          allCards = [...player0Cards, ...player1Cards];
        }

        // Preload card textures (shows loading progress)
        const mappings = await preloadDeckCards(allCards);
        if (isCancelled) {
          return;
        }
        setCardMappings(mappings);

        // Build cardDefIdMap from complete deck data
        const defIdMap = buildCardDefIdMapFromDeckCards(allCards);
        setCardDefIdMap(defIdMap);

        setIsDeckLoading(false);
      } catch (err) {
        if (isCancelled) {
          return;
        }
        console.error("Failed to load game assets:", err);
        setDeckLoadError(err instanceof Error ? err.message : "Failed to load game assets");
        setIsDeckLoading(false);
      }
    }

    void loadGameAssets();

    return () => {
      isCancelled = true;
    };
  }, [
    isInMatch,
    isEvaluationRoom,
    evaluationCardCatalog,
    evaluationHumanDeckId,
    player0DeckId,
    player1DeckId,
    preloadDeckCards,
    setCardMappings,
    setCardDefIdMap,
  ]);

  const isBusy = isDeckLoading || loadingState.isLoading || isLoading || !gameState;
  const turnPlayerLabel = !gameState
    ? "-"
    : activeRoom?.playerSlot === undefined
      ? `Player ${gameState.activePlayer}`
      : gameState.activePlayer === activeRoom.playerSlot
        ? `Player ${gameState.activePlayer} (You)`
        : `Player ${gameState.activePlayer} (Opponent)`;
  const mainPhaseLabel = formatPhaseLabel(gameState?.phase);
  const subPhaseLabel = formatPhaseLabel(gameState?.abilitySubphase);
  const canForfeit = connectionStatus === "connected" && roomState?.status === "IN_MATCH";

  // Neutral label supplied by the room (e.g. "Opponent 3"): never a model name.
  const opponentSlot = activeRoom?.playerSlot === 0 ? 1 : 0;
  const opponentLabel = roomState?.players[opponentSlot]?.username ?? "Opponent";

  const handleForfeit = useCallback(() => {
    if (!canForfeit) {
      return;
    }
    const confirmed = window.confirm("Forfeit this match?");
    if (!confirmed) {
      return;
    }
    send({ type: "FORFEIT" });
  }, [canForfeit, send]);

  const handleReturnToDashboard = useCallback(() => {
    router.push(evaluation === null ? "/dashboard" : `/evaluations/${evaluation.sessionId}`);
  }, [evaluation, router]);

  return (
    <div className="fixed inset-0 z-50 bg-black">
      {deckLoadError ? (
        <div className="flex items-center justify-center h-full text-white">
          <div className="text-center">
            <p className="text-red-500 mb-4">Error: {deckLoadError}</p>
            <p className="text-gray-400">Please refresh the page to try again.</p>
          </div>
        </div>
      ) : isBusy ? (
        <LoadingScreen
          progress={loadingState.progress}
          message={
            isDeckLoading
              ? evaluation === null
                ? "Loading deck data..."
                : "Loading your deck and the card catalog..."
              : "Loading game assets..."
          }
        />
      ) : (
        <>
          <GameConsoleCommands />
          <GameScene />
          <div className="absolute top-4 right-4 z-50 pointer-events-auto flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleReturnToDashboard}>
              {evaluation === null ? "Dashboard" : "Session"}
            </Button>
            {canForfeit && (
              <Button variant="destructive" size="sm" onClick={handleForfeit}>
                Forfeit
              </Button>
            )}
          </div>
          <div className="absolute top-4 left-4 z-40 pointer-events-none rounded-md border border-gray-700 bg-black/70 px-3 py-2 text-xs text-white shadow-lg">
            {evaluation === null ? null : (
              <p className="mb-1 border-b border-gray-700 pb-1 font-medium">
                Blind evaluation · match {evaluation.ordinal}/{evaluation.totalMatches} ·{" "}
                {opponentLabel}
              </p>
            )}
            <p>
              <span className="text-gray-300">Turn:</span> {turnPlayerLabel}
            </p>
            <p>
              <span className="text-gray-300">Main Phase:</span> {mainPhaseLabel}
            </p>
            <p>
              <span className="text-gray-300">Sub Phase:</span> {subPhaseLabel}
            </p>
          </div>
          {isDevMode && <DevDebugOverlay />}
        </>
      )}
    </div>
  );
}
