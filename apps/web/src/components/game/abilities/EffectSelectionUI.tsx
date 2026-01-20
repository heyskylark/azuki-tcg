"use client";

import { useCallback } from "react";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import {
  getValidEffectTargets,
  hasNoopAction,
  buildEffectTargetAction,
  buildNoopAction,
} from "@/lib/game/actionValidation";

/**
 * UI for selecting effect targets on the board.
 * Shown during the EFFECT_SELECTION ability phase.
 * The actual target highlighting happens in the 3D components (Card3D, Board).
 * This component shows instructions and an optional skip button.
 */
export function EffectSelectionUI() {
  const { gameState } = useGameState();
  const { send } = useRoom();

  const actionMask = gameState?.actionMask ?? null;
  const validTargets = getValidEffectTargets(actionMask);
  const canSkip = hasNoopAction(actionMask);

  const handleSkip = useCallback(() => {
    if (!canSkip) return;
    send({
      type: "GAME_ACTION",
      action: buildNoopAction(),
    });
  }, [canSkip, send]);

  // This function will be used by the 3D board components to handle target clicks
  // For now, we expose the send function through context or props

  return (
    <div className="absolute inset-0 flex flex-col items-center pointer-events-none">
      {/* Header banner */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-blue-600/90 text-white px-6 py-2 rounded-lg shadow-lg pointer-events-auto">
        <p className="text-lg font-semibold">
          Select a target for the ability
        </p>
        <p className="text-sm text-blue-200">
          Click a highlighted card on the board ({validTargets.length} valid targets)
        </p>
      </div>

      {/* Skip button (if available) */}
      {canSkip && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 pointer-events-auto">
          <button
            onClick={handleSkip}
            className="px-6 py-3 bg-slate-600 hover:bg-slate-500 text-white rounded-lg shadow-lg transition-colors"
          >
            Skip Target Selection
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * Hook to get the effect target click handler.
 * Used by 3D components to send the selection action.
 */
export function useEffectTargetHandler() {
  const { send } = useRoom();
  const { gameState } = useGameState();

  const actionMask = gameState?.actionMask ?? null;
  const validTargets = getValidEffectTargets(actionMask);

  const handleEffectTargetClick = useCallback(
    (targetIndex: number) => {
      if (!validTargets.includes(targetIndex)) return;
      send({
        type: "GAME_ACTION",
        action: buildEffectTargetAction(targetIndex),
      });
    },
    [validTargets, send]
  );

  return {
    validTargets,
    handleEffectTargetClick,
  };
}
