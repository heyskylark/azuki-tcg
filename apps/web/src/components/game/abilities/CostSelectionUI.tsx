"use client";

import { useCallback } from "react";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import {
  getValidCostTargets,
  buildCostTargetAction,
} from "@/lib/game/actionValidation";
import { isHandTargetType } from "@/lib/game/abilityTargeting";

/**
 * UI for selecting cost targets (e.g., discard a card from hand).
 * Shown during the COST_SELECTION ability phase.
 * Displays a list of hand cards that can be selected as cost.
 */
export function CostSelectionUI() {
  const { gameState } = useGameState();
  const { send } = useRoom();

  const actionMask = gameState?.actionMask ?? null;
  const validTargets = getValidCostTargets(actionMask);
  const myHand = gameState?.myHand ?? [];
  const costTargetType = gameState?.abilityCostTargetType;
  const isHandTarget = isHandTargetType(costTargetType);

  const handleSelectCost = useCallback(
    (handIndex: number) => {
      if (!validTargets.includes(handIndex)) return;
      send({
        type: "GAME_ACTION",
        action: buildCostTargetAction(handIndex),
      });
    },
    [validTargets, send]
  );

  return (
    <div className="absolute inset-0 flex flex-col items-center justify-end pb-32 pointer-events-auto">
      {/* Header banner */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-amber-600/90 text-white px-6 py-2 rounded-lg shadow-lg">
        <p className="text-lg font-semibold">Select a card to pay the cost</p>
      </div>

      {/* Hand card selection - overlay style */}
      <div className="bg-slate-800/90 border border-slate-600 rounded-lg p-4 max-w-3xl w-full mx-4">
        {isHandTarget ? (
          <>
            <p className="text-slate-300 text-sm mb-3">
              Click a highlighted card in your hand to select it as cost:
            </p>
            <div className="flex gap-2 flex-wrap justify-center">
              {myHand.map((card, index) => {
                const isValid = validTargets.includes(index);
                return (
                  <button
                    key={`cost-${index}-${card.cardCode}`}
                    onClick={() => handleSelectCost(index)}
                    disabled={!isValid}
                    className={`
                      relative p-2 rounded-md border-2 transition-all
                      ${
                        isValid
                          ? "border-amber-400 bg-amber-400/20 hover:bg-amber-400/40 cursor-pointer"
                          : "border-slate-600 bg-slate-700/50 opacity-50 cursor-not-allowed"
                      }
                    `}
                  >
                    <div className="w-16 h-20 bg-slate-700 rounded flex items-center justify-center overflow-hidden">
                      {card.imageUrl ? (
                        <img
                          src={card.imageUrl}
                          alt={card.name}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <span className="text-xs text-slate-400 text-center px-1">
                          {card.name}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-white mt-1 text-center truncate max-w-16">
                      {card.name}
                    </p>
                    {isValid && (
                      <div className="absolute -top-1 -right-1 w-4 h-4 bg-amber-400 rounded-full flex items-center justify-center">
                        <span className="text-xs text-black font-bold">!</span>
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </>
        ) : (
          <p className="text-slate-300 text-sm">
            Select a highlighted target on the board to pay the cost.
          </p>
        )}
      </div>
    </div>
  );
}
