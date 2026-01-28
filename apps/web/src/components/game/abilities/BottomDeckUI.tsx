"use client";

import { useCallback } from "react";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import {
  getValidBottomDeckTargets,
  hasBottomDeckAllAction,
  buildBottomDeckCardAction,
  buildBottomDeckAllAction,
} from "@/lib/game/actionValidation";

/**
 * UI for ordering cards to the bottom of the deck.
 * Shown during the BOTTOM_DECK ability phase.
 * Allows player to select order of cards going to bottom of deck.
 */
export function BottomDeckUI() {
  const { gameState } = useGameState();
  const { send } = useRoom();

  const actionMask = gameState?.actionMask ?? null;
  const validTargets = getValidBottomDeckTargets(actionMask);
  const canBottomAll = hasBottomDeckAllAction(actionMask);
  const selectionCards = (gameState?.selectionCards ?? []).filter(
    (card) => card.cardCode !== "unknown" && card.cardDefId !== 0
  );

  const selectionCardsWithIndex = selectionCards.map((card, index) => {
    const zoneIndex = card.zoneIndex ?? null;
    const resolvedIndex =
      zoneIndex !== null && validTargets.includes(zoneIndex)
        ? zoneIndex
        : validTargets.length === selectionCards.length
          ? validTargets[index]
          : zoneIndex ?? index;

    return { card, selectionIndex: resolvedIndex };
  });

  const handleSelectCard = useCallback(
    (selectionIndex: number) => {
      if (!validTargets.includes(selectionIndex)) return;
      send({
        type: "GAME_ACTION",
        action: buildBottomDeckCardAction(selectionIndex),
      });
    },
    [validTargets, send]
  );

  const handleBottomAll = useCallback(() => {
    if (!canBottomAll) return;
    send({
      type: "GAME_ACTION",
      action: buildBottomDeckAllAction(),
    });
  }, [canBottomAll, send]);

  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-auto">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" />

      {/* Selection panel */}
      <div className="relative bg-slate-800 border border-slate-600 rounded-lg shadow-xl p-6 max-w-4xl w-full mx-4">
        <h2 className="text-xl font-bold text-white mb-2">
          Order Cards to Bottom of Deck
        </h2>
        <p className="text-slate-300 mb-4">
          Select cards in the order you want them at the bottom of your deck
          (first selected will be on bottom)
        </p>

        {/* Card grid */}
        <div className="flex gap-3 flex-wrap justify-center mb-4">
          {selectionCardsWithIndex.length > 0 ? (
            selectionCardsWithIndex.map(({ card, selectionIndex }) => {
              const isValid = validTargets.includes(selectionIndex);
              return (
                <button
                  key={`bottom-${selectionIndex}-${card.cardCode}`}
                  onClick={() => handleSelectCard(selectionIndex)}
                  disabled={!isValid}
                  className={`
                    relative p-2 rounded-md border-2 transition-all
                    ${
                      isValid
                        ? "border-orange-400 bg-orange-400/20 hover:bg-orange-400/40 cursor-pointer"
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

        {/* Bottom all button */}
        {canBottomAll && (
          <div className="flex justify-center">
            <button
              onClick={handleBottomAll}
              className="px-4 py-2 bg-orange-600 hover:bg-orange-500 text-white rounded-md transition-colors"
            >
              Bottom All (Random Order)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
