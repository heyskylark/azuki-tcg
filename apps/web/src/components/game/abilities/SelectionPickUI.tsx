"use client";

import { useCallback } from "react";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import {
  getValidSelectionTargets,
  hasNoopAction,
  buildSelectionPickAction,
  buildNoopAction,
} from "@/lib/game/actionValidation";

/**
 * UI for picking cards from the selection zone.
 * Shown during the SELECTION_PICK ability phase.
 * Displays revealed cards and allows the player to pick one.
 */
export function SelectionPickUI() {
  const { gameState } = useGameState();
  const { send } = useRoom();

  const actionMask = gameState?.actionMask ?? null;
  const validTargets = getValidSelectionTargets(actionMask);
  const canSkip = hasNoopAction(actionMask);
  const selectionCards = gameState?.selectionCards ?? [];

  const handleSelectCard = useCallback(
    (selectionIndex: number) => {
      if (!validTargets.includes(selectionIndex)) return;
      send({
        type: "GAME_ACTION",
        action: buildSelectionPickAction(selectionIndex),
      });
    },
    [validTargets, send]
  );

  const handleSkip = useCallback(() => {
    if (!canSkip) return;
    send({
      type: "GAME_ACTION",
      action: buildNoopAction(),
    });
  }, [canSkip, send]);

  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-auto">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" />

      {/* Selection panel */}
      <div className="relative bg-slate-800 border border-slate-600 rounded-lg shadow-xl p-6 max-w-4xl w-full mx-4">
        <h2 className="text-xl font-bold text-white mb-2">Select a Card</h2>
        <p className="text-slate-300 mb-4">
          Choose a card from the selection ({validTargets.length} available)
        </p>

        {/* Card grid */}
        <div className="flex gap-3 flex-wrap justify-center mb-4">
          {selectionCards.length > 0 ? (
            selectionCards.map((card, index) => {
              const isValid = validTargets.includes(index);
              return (
                <button
                  key={`selection-${index}-${card.cardCode}`}
                  onClick={() => handleSelectCard(index)}
                  disabled={!isValid}
                  className={`
                    relative p-2 rounded-md border-2 transition-all
                    ${
                      isValid
                        ? "border-purple-400 bg-purple-400/20 hover:bg-purple-400/40 cursor-pointer"
                        : "border-slate-600 bg-slate-700/50 opacity-50 cursor-not-allowed"
                    }
                  `}
                >
                  <div className="w-20 h-28 bg-slate-700 rounded flex items-center justify-center overflow-hidden">
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
                  <p className="text-xs text-white mt-1 text-center truncate max-w-20">
                    {card.name}
                  </p>
                </button>
              );
            })
          ) : (
            <p className="text-slate-400 italic">
              Waiting for selection cards...
            </p>
          )}
        </div>

        {/* Skip button */}
        {canSkip && (
          <div className="flex justify-center">
            <button
              onClick={handleSkip}
              className="px-4 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded-md transition-colors"
            >
              Skip
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
