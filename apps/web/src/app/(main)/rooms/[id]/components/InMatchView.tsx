"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { GameScene } from "@/components/game/GameScene";
import { LoadingScreen } from "@/components/game/LoadingScreen";
import { ActionPanel } from "@/components/game/ActionPanel";
import { DevDebugOverlay } from "@/components/game/DevDebugOverlay";
import { useAssets } from "@/contexts/AssetContext";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import { authenticatedFetch } from "@/lib/api/authenticatedFetch";
import type { DeckCard, DeckWithCards } from "@/types/game";
import { buildCardDefIdMapFromDeckCards } from "@/types/game";

interface DeckApiResponse {
  deck: DeckWithCards;
}

export function InMatchView() {
  const searchParams = useSearchParams();
  const { loadingState, preloadDeckCards } = useAssets();
  const { gameState, isLoading, setCardMappings, setCardDefIdMap } = useGameState();
  const { roomState } = useRoom();

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
          <ActionPanel />
          {isDevMode && <DevDebugOverlay />}
        </>
      )}
    </div>
  );
}
