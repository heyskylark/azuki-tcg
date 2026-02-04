"use client";

import { useCallback } from "react";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import {
  getValidEffectTargets,
  buildEffectTargetAction,
} from "@/lib/game/actionValidation";
import { isHandTargetType } from "@/lib/game/abilityTargeting";

/**
 * UI for selecting effect targets on the board.
 * Shown during the EFFECT_SELECTION ability phase.
 * The actual target highlighting happens in the 3D components (Card3D, Board).
 * This component shows instructions; skip is rendered in 3D.
 */
export function EffectSelectionUI() {
  const { gameState } = useGameState();

  const actionMask = gameState?.actionMask ?? null;
  const validTargets = getValidEffectTargets(actionMask);
  const effectTargetType = gameState?.abilityEffectTargetType;
  const isHandTarget = isHandTargetType(effectTargetType);

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
          {isHandTarget
            ? `Click a highlighted card in your hand (${validTargets.length} valid targets)`
            : `Click a highlighted card on the board (${validTargets.length} valid targets)`}
        </p>
      </div>

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
