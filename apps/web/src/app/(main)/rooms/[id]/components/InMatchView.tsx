"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { GameScene } from "@/components/game/GameScene";
import { LoadingScreen } from "@/components/game/LoadingScreen";
import { DevDebugOverlay } from "@/components/game/DevDebugOverlay";
import { Button } from "@/components/ui/button";
import { useAssets } from "@/contexts/AssetContext";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import { authenticatedFetch } from "@/lib/api/authenticatedFetch";
import type { DeckCard, DeckWithCards } from "@/types/game";
import { buildCardDefIdMapFromDeckCards } from "@/types/game";

interface DeckApiResponse {
  deck: DeckWithCards;
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

export function InMatchView() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { loadingState, preloadDeckCards } = useAssets();
  const { gameState, isLoading, setCardMappings, setCardDefIdMap } = useGameState();
  const { roomState, activeRoom, connectionStatus, send } = useRoom();

  const [isDeckLoading, setIsDeckLoading] = useState(true);
  const [deckLoadError, setDeckLoadError] = useState<string | null>(null);

  const isDevMode = searchParams.get("dev") === "true";

  // Fetch both decks and preload assets when entering match
  useEffect(() => {
    async function loadGameAssets() {
      // In IN_MATCH status, both deckIds are guaranteed to exist
      const player0DeckId = roomState?.players[0]?.deckId;
      const player1DeckId = roomState?.players[1]?.deckId;

      if (!player0DeckId || !player1DeckId) {
        console.error("Missing deck IDs in IN_MATCH state");
        setDeckLoadError("Missing deck information");
        setIsDeckLoading(false);
        return;
      }

      try {
        // Fetch both decks in parallel
        const [deck0Res, deck1Res] = await Promise.all([
          authenticatedFetch(`/api/decks/${player0DeckId}`),
          authenticatedFetch(`/api/decks/${player1DeckId}`),
        ]);

        if (!deck0Res.ok || !deck1Res.ok) {
          throw new Error("Failed to fetch deck data");
        }

        const deck0Data: DeckApiResponse = await deck0Res.json();
        const deck1Data: DeckApiResponse = await deck1Res.json();

        // Combine all cards from both decks
        const allCards: DeckCard[] = [
          ...deck0Data.deck.cards,
          ...deck1Data.deck.cards,
        ];

        // Preload card textures (shows loading progress)
        const mappings = await preloadDeckCards(allCards);
        setCardMappings(mappings);

        // Build cardDefIdMap from complete deck data
        const defIdMap = buildCardDefIdMapFromDeckCards(allCards);
        setCardDefIdMap(defIdMap);

        setIsDeckLoading(false);
      } catch (err) {
        console.error("Failed to load game assets:", err);
        setDeckLoadError(err instanceof Error ? err.message : "Failed to load game assets");
        setIsDeckLoading(false);
      }
    }

    loadGameAssets();
  }, [roomState, preloadDeckCards, setCardMappings, setCardDefIdMap]);

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
  const canForfeit =
    connectionStatus === "connected" &&
    roomState?.status === "IN_MATCH";

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
    router.push("/dashboard");
  }, [router]);

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
          message={isDeckLoading ? "Loading deck data..." : "Loading game assets..."}
        />
      ) : (
        <>
          <GameScene />
          <div className="absolute top-4 right-4 z-50 pointer-events-auto flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleReturnToDashboard}>
              Dashboard
            </Button>
            {canForfeit && (
              <Button
                variant="destructive"
                size="sm"
                onClick={handleForfeit}
              >
                Forfeit
              </Button>
            )}
          </div>
          <div className="absolute top-4 left-4 z-40 pointer-events-none rounded-md border border-gray-700 bg-black/70 px-3 py-2 text-xs text-white shadow-lg">
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
