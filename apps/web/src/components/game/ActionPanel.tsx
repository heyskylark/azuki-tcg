"use client";

import { useCallback } from "react";
import { Button } from "@/components/ui/button";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";

// Action type constants (from action space head 0)
const ACTION_NOOP = 0;
const ACTION_MULLIGAN = 23;

export function ActionPanel() {
  const { gameState } = useGameState();
  const { send } = useRoom();

  const actionMask = gameState?.actionMask;
  const legalPrimary = actionMask?.legalPrimary ?? [];

  const canNoop = legalPrimary.includes(ACTION_NOOP);
  const canMulligan = legalPrimary.includes(ACTION_MULLIGAN);
  const isMulliganPhase = gameState?.phase === "PREGAME_MULLIGAN"

  const handleAction = useCallback(
    (actionType: number) => {
      console.log("[ActionPanel] handleAction called with:", actionType);
      console.log("[ActionPanel] send function exists:", !!send);
      send({
        type: "GAME_ACTION",
        action: [actionType, 0, 0, 0],
      });
    },
    [send]
  );

  // Don't render if no actions available
  if (!canNoop && !canMulligan) {
    return null;
  }

  return (
    <div className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50 flex gap-4">
      {canMulligan && (
        <Button
          size="lg"
          variant="default"
          onClick={() => handleAction(ACTION_MULLIGAN)}
          className="min-w-32 text-lg font-semibold"
        >
          Mulligan
        </Button>
      )}
      {canNoop && (
        <Button
          size="lg"
          variant="secondary"
          onClick={() => handleAction(ACTION_NOOP)}
          className="min-w-32 text-lg font-semibold"
        >
          {isMulliganPhase ? "Keep Hand" : "Pass"}
        </Button>
      )}
    </div>
  );
}
