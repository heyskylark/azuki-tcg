"use client";

import { useCallback, useState } from "react";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import {
  getSelectionActionInfo,
  getSelectionActionInfoByIndex,
  hasNoopAction,
  buildSelectionPickAction,
  buildSelectToAlleyAction,
  buildSelectToGardenAction,
  buildSelectToEquipAction,
  buildNoopAction,
  type SelectionActionInfo,
} from "@/lib/game/actionValidation";

/**
 * UI for picking cards from the selection zone.
 * Shown during the SELECTION_PICK ability phase.
 * Displays revealed cards and allows the player to pick one.
 *
 * Supports follow-up placement flows for cards that can:
 * 1. be added to hand
 * 2. be played into the Garden or Alley
 * 3. be equipped to a target entity
 */
export function SelectionPickUI() {
  const { gameState } = useGameState();
  const { send } = useRoom();

  const actionMask = gameState?.actionMask ?? null;
  const selectionActionInfo = getSelectionActionInfo(actionMask);
  const validTargets = selectionActionInfo.map((info) => info.selectionIndex);
  const canSkip = hasNoopAction(actionMask);
  const selectionCards = (gameState?.selectionCards ?? []).filter(
    (card) => card.cardCode !== "unknown" && card.cardDefId !== 0
  );

  const [selectedSelectionIndex, setSelectedSelectionIndex] = useState<number | null>(null);
  const [selectedActionInfo, setSelectedActionInfo] = useState<SelectionActionInfo | null>(null);

  const handleSelectCard = useCallback(
    (selectionIndex: number) => {
      if (!validTargets.includes(selectionIndex)) return;

      // Get action info for this selection
      const info = getSelectionActionInfoByIndex(actionMask, selectionIndex);
      if (!info) return;

      const needsFollowUp =
        info.canSelectToEquip ||
        info.canSelectToGarden ||
        info.canSelectToAlley;

      if (needsFollowUp) {
        setSelectedSelectionIndex(selectionIndex);
        setSelectedActionInfo(info);
        return;
      }

      if (info.canAddToHand) {
        send({
          type: "GAME_ACTION",
          action: buildSelectionPickAction(selectionIndex),
        });
      }
    },
    [validTargets, actionMask, send]
  );

  const resetSelectionFlow = useCallback(() => {
    setSelectedSelectionIndex(null);
    setSelectedActionInfo(null);
  }, []);

  const handleSelectEquipTarget = useCallback(
    (entitySlot: number) => {
      if (selectedSelectionIndex === null) return;
      if (!selectedActionInfo?.equipTargetSlots.includes(entitySlot)) return;

      send({
        type: "GAME_ACTION",
        action: buildSelectToEquipAction(selectedSelectionIndex, entitySlot),
      });

      resetSelectionFlow();
    },
    [selectedSelectionIndex, selectedActionInfo, send, resetSelectionFlow]
  );

  const handleAddToHand = useCallback(() => {
    if (selectedSelectionIndex === null || !selectedActionInfo?.canAddToHand) {
      return;
    }

    send({
      type: "GAME_ACTION",
      action: buildSelectionPickAction(selectedSelectionIndex),
    });

    resetSelectionFlow();
  }, [selectedSelectionIndex, selectedActionInfo, send, resetSelectionFlow]);

  const handleSelectGardenTarget = useCallback(
    (gardenSlot: number) => {
      if (selectedSelectionIndex === null) return;
      if (!selectedActionInfo?.gardenSlots.includes(gardenSlot)) return;

      send({
        type: "GAME_ACTION",
        action: buildSelectToGardenAction(selectedSelectionIndex, gardenSlot),
      });

      resetSelectionFlow();
    },
    [selectedSelectionIndex, selectedActionInfo, send, resetSelectionFlow]
  );

  const handleSelectAlleyTarget = useCallback(
    (alleySlot: number) => {
      if (selectedSelectionIndex === null) return;
      if (!selectedActionInfo?.alleySlots.includes(alleySlot)) return;

      send({
        type: "GAME_ACTION",
        action: buildSelectToAlleyAction(selectedSelectionIndex, alleySlot),
      });

      resetSelectionFlow();
    },
    [selectedSelectionIndex, selectedActionInfo, send, resetSelectionFlow]
  );

  const handleSkip = useCallback(() => {
    if (!canSkip) return;
    send({
      type: "GAME_ACTION",
      action: buildNoopAction(),
    });
  }, [canSkip, send]);

  if (selectedSelectionIndex !== null && selectedActionInfo) {
    const selectedCard = selectionCards.find(
      (card) => (card.zoneIndex ?? -1) === selectedSelectionIndex
    );
    const garden = gameState?.myBoard?.garden ?? [];
    const alley = gameState?.myBoard?.alley ?? [];
    const leader = gameState?.myBoard?.leader;
    const isEquipFlow =
      selectedActionInfo.canSelectToEquip &&
      !selectedActionInfo.canSelectToGarden &&
      !selectedActionInfo.canSelectToAlley;

    return (
      <div className="absolute inset-0 flex items-center justify-center pointer-events-auto">
        {/* Backdrop */}
        <div className="absolute inset-0 bg-black/50" />

        {/* Target selection panel */}
        <div className="relative bg-slate-800 border border-slate-600 rounded-lg shadow-xl p-6 max-w-4xl w-full mx-4">
          <h2 className="text-xl font-bold text-white mb-2">
            {isEquipFlow ? "Select Equip Target" : "Choose a Destination"}
          </h2>
          <p className="text-slate-300 mb-4">
            {isEquipFlow
              ? `Choose an entity to equip ${selectedCard?.name ?? "the weapon"} to`
              : `Choose what to do with ${selectedCard?.name ?? "this card"}`}
          </p>

          {!isEquipFlow && selectedActionInfo.canAddToHand && (
            <div className="mb-4 flex justify-center">
              <button
                onClick={handleAddToHand}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-md transition-colors"
              >
                Add To Hand
              </button>
            </div>
          )}

          {/* Garden entities */}
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-slate-400 mb-2">Garden</h3>
            <div className="flex gap-3 flex-wrap justify-center">
              {garden.map((entity, slot) => {
                const isValid = isEquipFlow
                  ? selectedActionInfo.equipTargetSlots.includes(slot)
                  : selectedActionInfo.gardenSlots.includes(slot);
                const isEmpty = entity === null;

                if (isEmpty && isEquipFlow) {
                  return (
                    <div
                      key={`garden-slot-${slot}`}
                      className="w-20 h-28 border-2 border-dashed border-slate-600 rounded-md flex items-center justify-center"
                    >
                      <span className="text-xs text-slate-500">Empty</span>
                    </div>
                  );
                }

                return (
                  <button
                    key={`garden-entity-${slot}-${entity?.cardCode}`}
                    onClick={() =>
                      isEquipFlow
                        ? handleSelectEquipTarget(slot)
                        : handleSelectGardenTarget(slot)
                    }
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
                    <div className="w-20 h-28 bg-slate-700 rounded flex items-center justify-center overflow-hidden">
                      {entity?.imageUrl ? (
                        <img
                          src={entity.imageUrl}
                          alt={entity.name}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <span className="text-xs text-slate-400 text-center px-1">
                          {entity?.name ?? "Empty"}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-white mt-1 text-center truncate max-w-20">
                      {entity?.name ?? `Slot ${slot + 1}`}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>

          {!isEquipFlow && selectedActionInfo.canSelectToAlley && (
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-slate-400 mb-2">Alley</h3>
              <div className="flex gap-3 flex-wrap justify-center">
                {alley.map((entity, slot) => {
                  const isValid = selectedActionInfo.alleySlots.includes(slot);
                  return (
                    <button
                      key={`alley-slot-${slot}-${entity?.cardCode ?? "empty"}`}
                      onClick={() => handleSelectAlleyTarget(slot)}
                      disabled={!isValid}
                      className={`
                        relative p-2 rounded-md border-2 transition-all
                        ${
                          isValid
                            ? "border-cyan-400 bg-cyan-400/20 hover:bg-cyan-400/40 cursor-pointer"
                            : "border-slate-600 bg-slate-700/50 opacity-50 cursor-not-allowed"
                        }
                      `}
                    >
                      <div className="w-20 h-28 bg-slate-700 rounded flex items-center justify-center overflow-hidden">
                        {entity?.imageUrl ? (
                          <img
                            src={entity.imageUrl}
                            alt={entity.name}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <span className="text-xs text-slate-400 text-center px-1">
                            Empty
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-white mt-1 text-center truncate max-w-20">
                        {entity?.name ?? `Slot ${slot + 1}`}
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Leader (slot 5) */}
          {isEquipFlow && selectedActionInfo.equipTargetSlots.includes(5) && leader && (
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-slate-400 mb-2">Leader</h3>
              <div className="flex justify-center">
                <button
                  onClick={() => handleSelectEquipTarget(5)}
                  className="relative p-2 rounded-md border-2 border-amber-400 bg-amber-400/20 hover:bg-amber-400/40 cursor-pointer transition-all"
                >
                  <div className="w-20 h-28 bg-slate-700 rounded flex items-center justify-center overflow-hidden">
                    {leader.imageUrl ? (
                      <img
                        src={leader.imageUrl}
                        alt={leader.name}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <span className="text-xs text-slate-400 text-center px-1">
                        {leader.name}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-white mt-1 text-center truncate max-w-20">
                    {leader.name}
                  </p>
                </button>
              </div>
            </div>
          )}

          {/* Cancel button */}
          <div className="flex justify-center">
            <button
              onClick={resetSelectionFlow}
              className="px-4 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded-md transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Default: weapon/card selection UI
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
              const selectionIndex = card.zoneIndex ?? index;
              const info = selectionActionInfo.find(
                (i) => i.selectionIndex === selectionIndex
              );
              const isValid = info !== undefined;

              const hasFollowUpOptions =
                info?.canSelectToEquip ||
                info?.canSelectToGarden ||
                info?.canSelectToAlley;

              return (
                <button
                  key={`selection-${selectionIndex}-${card.cardCode}`}
                  onClick={() => handleSelectCard(selectionIndex)}
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
                  {hasFollowUpOptions && (
                    <span className="absolute -top-1 -right-1 px-1.5 py-0.5 text-[10px] font-semibold bg-amber-500 text-black rounded">
                      PICK
                    </span>
                  )}
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
