"use client";

import { useCallback } from "react";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import {
  getValidTopDeckTargets,
  getValidBottomDeckTargets,
  hasBottomDeckAllAction,
  buildTopDeckCardAction,
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
  const validTopTargets = getValidTopDeckTargets(actionMask);
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

  const handleTopDeckCard = useCallback(
    (selectionIndex: number) => {
      if (!validTopTargets.includes(selectionIndex)) return;
      send({
        type: "GAME_ACTION",
        action: buildTopDeckCardAction(selectionIndex),
      });
    },
    [validTopTargets, send]
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
          {validTopTargets.length > 0
            ? "Order Cards On Top Or Bottom Of Deck"
            : "Order Cards to Bottom of Deck"}
        </h2>
        <p className="text-slate-300 mb-4">
          {validTopTargets.length > 0
            ? "Select cards one by one and choose whether each goes to the top or bottom of your deck"
            : "Select cards in the order you want them at the bottom of your deck (first selected will be on bottom)"}
        </p>

        {/* Card grid */}
        <div className="flex gap-3 flex-wrap justify-center mb-4">
          {selectionCardsWithIndex.length > 0 ? (
            selectionCardsWithIndex.map(({ card, selectionIndex }) => {
              const isValid = validTargets.includes(selectionIndex);
              return (
                <div
                  key={`bottom-${selectionIndex}-${card.cardCode}`}
                  className="relative p-2 rounded-md border-2 border-slate-600 bg-slate-700/40"
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
                  <div className="mt-2 flex gap-2 justify-center">
                    {validTopTargets.includes(selectionIndex) && (
                      <button
                        onClick={() => handleTopDeckCard(selectionIndex)}
                        className="px-2 py-1 text-[10px] font-semibold rounded bg-sky-600 hover:bg-sky-500 text-white"
                      >
                        Top
                      </button>
                    )}
                    <button
                      onClick={() => handleSelectCard(selectionIndex)}
                      disabled={!isValid}
                      className={`
                        px-2 py-1 text-[10px] font-semibold rounded text-white
                        ${
                          isValid
                            ? "bg-orange-600 hover:bg-orange-500"
                            : "bg-slate-600 opacity-50 cursor-not-allowed"
                        }
                      `}
                    >
                      Bottom
                    </button>
                  </div>
                </div>
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
