"use client";

import { GameScene } from "@/components/game/GameScene";
import { LoadingScreen } from "@/components/game/LoadingScreen";
import { useAssets } from "@/contexts/AssetContext";
import { useGameState } from "@/contexts/GameStateContext";

export function InMatchView() {
  const { loadingState } = useAssets();
  const { gameState, isLoading } = useGameState();

  const isBusy = loadingState.isLoading || isLoading || !gameState;

  return (
    <div className="h-[70vh] min-h-[480px] w-full bg-black">
      {isBusy ? (
        <LoadingScreen
          progress={loadingState.progress}
          message="Loading game assets..."
        />
      ) : (
        <GameScene />
      )}
    </div>
  );
}
