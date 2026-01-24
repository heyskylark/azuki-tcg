"use client";

import { useCallback, useState } from "react";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import {
  getSelectionActionInfo,
  getSelectionActionInfoByIndex,
  hasNoopAction,
  buildSelectionPickAction,
  buildSelectToEquipAction,
  buildNoopAction,
  type SelectionActionInfo,
} from "@/lib/game/actionValidation";

/**
 * UI for picking cards from the selection zone.
 * Shown during the SELECTION_PICK ability phase.
 * Displays revealed cards and allows the player to pick one.
 *
 * Supports two-step selection for equip actions:
 * 1. Select a weapon from the selection cards
 * 2. Select a target entity (garden slot or leader) to equip
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

  // State for two-step equip selection
  const [selectedWeaponIndex, setSelectedWeaponIndex] = useState<number | null>(null);
  const [selectedWeaponInfo, setSelectedWeaponInfo] = useState<SelectionActionInfo | null>(null);

  const handleSelectCard = useCallback(
    (selectionIndex: number) => {
      if (!validTargets.includes(selectionIndex)) return;

      // Get action info for this selection
      const info = getSelectionActionInfoByIndex(actionMask, selectionIndex);
      if (!info) return;

      // If only equip action is available (no add to hand), enter two-step flow
      if (info.canSelectToEquip && !info.canAddToHand) {
        setSelectedWeaponIndex(selectionIndex);
        setSelectedWeaponInfo(info);
        return;
      }

      // Default: add to hand (action 18)
      if (info.canAddToHand) {
        send({
          type: "GAME_ACTION",
          action: buildSelectionPickAction(selectionIndex),
        });
      }
    },
    [validTargets, actionMask, send]
  );

  const handleSelectEquipTarget = useCallback(
    (entitySlot: number) => {
      if (selectedWeaponIndex === null) return;
      if (!selectedWeaponInfo?.equipTargetSlots.includes(entitySlot)) return;

      send({
        type: "GAME_ACTION",
        action: buildSelectToEquipAction(selectedWeaponIndex, entitySlot),
      });

      // Reset state
      setSelectedWeaponIndex(null);
      setSelectedWeaponInfo(null);
    },
    [selectedWeaponIndex, selectedWeaponInfo, send]
  );

  const handleCancelEquip = useCallback(() => {
    setSelectedWeaponIndex(null);
    setSelectedWeaponInfo(null);
  }, []);

  const handleSkip = useCallback(() => {
    if (!canSkip) return;
    send({
      type: "GAME_ACTION",
      action: buildNoopAction(),
    });
  }, [canSkip, send]);

  // If we're in the equip target selection step
  if (selectedWeaponIndex !== null && selectedWeaponInfo) {
    const selectedCard = selectionCards.find(
      (card) => (card.zoneIndex ?? -1) === selectedWeaponIndex
    );
    const garden = gameState?.myBoard?.garden ?? [];
    const leader = gameState?.myBoard?.leader;

    return (
      <div className="absolute inset-0 flex items-center justify-center pointer-events-auto">
        {/* Backdrop */}
        <div className="absolute inset-0 bg-black/50" />

        {/* Target selection panel */}
        <div className="relative bg-slate-800 border border-slate-600 rounded-lg shadow-xl p-6 max-w-4xl w-full mx-4">
          <h2 className="text-xl font-bold text-white mb-2">Select Equip Target</h2>
          <p className="text-slate-300 mb-4">
            Choose an entity to equip {selectedCard?.name ?? "the weapon"} to
          </p>

          {/* Garden entities */}
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-slate-400 mb-2">Garden</h3>
            <div className="flex gap-3 flex-wrap justify-center">
              {garden.map((entity, slot) => {
                const isValid = selectedWeaponInfo.equipTargetSlots.includes(slot);
                const isEmpty = entity === null;

                if (isEmpty) {
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
                    onClick={() => handleSelectEquipTarget(slot)}
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
                          {entity?.name ?? "Entity"}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-white mt-1 text-center truncate max-w-20">
                      {entity?.name ?? "Entity"}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Leader (slot 5) */}
          {selectedWeaponInfo.equipTargetSlots.includes(5) && leader && (
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
              onClick={handleCancelEquip}
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

              // Show badge for equip-only cards
              const isEquipOnly = info?.canSelectToEquip && !info?.canAddToHand;

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
                  {isEquipOnly && (
                    <span className="absolute -top-1 -right-1 px-1.5 py-0.5 text-[10px] font-semibold bg-amber-500 text-black rounded">
                      EQUIP
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
