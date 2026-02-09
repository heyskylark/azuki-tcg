"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import {
  hasConfirmAbilityAction,
  hasNoopAction,
  buildConfirmAbilityAction,
  buildNoopAction,
} from "@/lib/game/actionValidation";

/**
 * Modal dialog for confirming or declining an ability.
 * Shown during the CONFIRMATION ability phase.
 */
export function ConfirmationModal() {
  const { gameState } = useGameState();
  const { send } = useRoom();

  const actionMask = gameState?.actionMask ?? null;
  const pendingConfirmationCount = gameState?.pendingConfirmationCount ?? null;
  const canConfirm = hasConfirmAbilityAction(actionMask);
  const canDecline = hasNoopAction(actionMask);
  const progressTotalRef = useRef<number | null>(null);
  const [queueProgress, setQueueProgress] = useState<{
    current: number;
    total: number;
  } | null>(null);

  useEffect(() => {
    if (pendingConfirmationCount === null || pendingConfirmationCount <= 0) {
      progressTotalRef.current = null;
      setQueueProgress(null);
      return;
    }

    if (
      progressTotalRef.current === null ||
      pendingConfirmationCount > progressTotalRef.current
    ) {
      progressTotalRef.current = pendingConfirmationCount;
    }

    const total = progressTotalRef.current;
    if (total === null || total <= 1) {
      setQueueProgress(null);
      return;
    }

    const current = Math.max(
      1,
      Math.min(total, total - pendingConfirmationCount + 1)
    );
    setQueueProgress({ current, total });
  }, [pendingConfirmationCount]);

  const handleConfirm = useCallback(() => {
    if (!canConfirm) return;
    send({
      type: "GAME_ACTION",
      action: buildConfirmAbilityAction(),
    });
  }, [canConfirm, send]);

  const handleDecline = useCallback(() => {
    if (!canDecline) return;
    send({
      type: "GAME_ACTION",
      action: buildNoopAction(),
    });
  }, [canDecline, send]);

  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-auto">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/40" />

      {/* Modal */}
      <div className="relative bg-slate-800 border border-slate-600 rounded-lg shadow-xl p-6 max-w-sm w-full mx-4">
        <div className="mb-2 flex items-start justify-between gap-3">
          <h2 className="text-xl font-bold text-white">Activate Ability?</h2>
          {queueProgress && (
            <span className="rounded-md bg-slate-700 px-2 py-1 text-xs font-semibold text-slate-200">
              {queueProgress.current}/{queueProgress.total} actions
            </span>
          )}
        </div>
        <p className="text-slate-300 mb-6">
          This card has an ability that can be activated. Do you want to use it?
        </p>

        <div className="flex gap-3 justify-end">
          {canDecline && (
            <button
              onClick={handleDecline}
              className="px-4 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded-md transition-colors"
            >
              Decline
            </button>
          )}
          {canConfirm && (
            <button
              onClick={handleConfirm}
              className="px-4 py-2 bg-green-600 hover:bg-green-500 text-white rounded-md transition-colors"
            >
              Activate
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
