"use client";

import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import { ConfirmationModal } from "@/components/game/abilities/ConfirmationModal";
import { CostSelectionUI } from "@/components/game/abilities/CostSelectionUI";
import { EffectSelectionUI } from "@/components/game/abilities/EffectSelectionUI";
import { SelectionPickUI } from "@/components/game/abilities/SelectionPickUI";
import { BottomDeckUI } from "@/components/game/abilities/BottomDeckUI";

/**
 * HTML overlay that renders on top of the 3D canvas when in an ability phase.
 * Provides UI for confirming abilities, selecting targets, etc.
 */
export function AbilityOverlay() {
  const { gameState } = useGameState();
  const { activeRoom } = useRoom();

  const playerSlot = activeRoom?.playerSlot;
  const isMyTurn = gameState?.activePlayer === playerSlot;
  const abilityPhase = gameState?.abilitySubphase;

  // Don't render if not in an ability phase or not my turn
  if (!isMyTurn || !abilityPhase || abilityPhase === "NONE") {
    return null;
  }

  return (
    <div className="absolute inset-0 pointer-events-none z-50">
      {abilityPhase === "CONFIRMATION" && <ConfirmationModal />}
      {abilityPhase === "COST_SELECTION" && <CostSelectionUI />}
      {abilityPhase === "EFFECT_SELECTION" && <EffectSelectionUI />}
      {abilityPhase === "SELECTION_PICK" && <SelectionPickUI />}
      {abilityPhase === "BOTTOM_DECK" && <BottomDeckUI />}
    </div>
  );
}
