"use client";

import { useCallback, useEffect } from "react";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import { isSpellActionLegal } from "@/lib/game/actionValidation";
import {
  isSpellModeSelectionCurrent,
  useSpellModeStore,
} from "@/stores/spellModeStore";

const SPELL_MODE_DESCRIPTIONS: Record<string, Record<number, string>> = {
  "STT03-017": {
    0: "Add 1 IKZ from your IKZ pile to your IKZ area; the IKZ enters the field tapped. Then, heal up to 1 to your leader.",
    1: "Draw 1.",
  },
};

function getSpellModeDescription(cardCode: string, abilityIndex: number): string {
  return SPELL_MODE_DESCRIPTIONS[cardCode]?.[abilityIndex] ?? `Option ${abilityIndex + 1}`;
}

export function SpellModeChooser() {
  const { gameState } = useGameState();
  const { activeRoom, send } = useRoom();
  const pending = useSpellModeStore((state) => state.pending);
  const clear = useSpellModeStore((state) => state.clear);
  const playerSlot = activeRoom?.playerSlot ?? null;
  const isCurrent =
    pending !== null && isSpellModeSelectionCurrent(pending, gameState, playerSlot);

  useEffect(() => {
    if (pending && !isCurrent) {
      clear();
    }
  }, [clear, isCurrent, pending]);

  useEffect(() => {
    if (!pending) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        clear();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [clear, pending]);

  const handleSelect = useCallback(
    (actionIndex: number) => {
      const current = useSpellModeStore.getState().pending;
      if (
        !current ||
        !isSpellModeSelectionCurrent(current, gameState, playerSlot)
      ) {
        clear();
        return;
      }

      const action = current.actions[actionIndex];
      if (!action || !isSpellActionLegal(gameState?.actionMask ?? null, action)) {
        clear();
        return;
      }

      clear();
      send({ type: "GAME_ACTION", action });
    },
    [clear, gameState, playerSlot, send]
  );

  if (!pending || !isCurrent) return null;

  return (
    <div className="absolute inset-0 z-[60] flex items-center justify-center pointer-events-auto">
      <button
        type="button"
        aria-label="Cancel spell mode selection"
        className="absolute inset-0 cursor-default bg-black/50"
        onClick={clear}
      />
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="spell-mode-title"
        className="relative mx-4 w-full max-w-lg rounded-lg border border-slate-600 bg-slate-800 p-6 shadow-xl"
      >
        <h2 id="spell-mode-title" className="text-xl font-bold text-white">
          Choose an effect for {pending.cardName}
        </h2>
        <p className="mt-2 text-sm text-slate-300">
          Choose one effect. Your spell will not be played until you make a selection.
        </p>

        <div className="mt-5 grid gap-3">
          {pending.actions.map((action, actionIndex) => (
            <button
              key={action[2]}
              type="button"
              className="rounded-md border border-slate-500 bg-slate-700 px-4 py-3 text-left text-white transition-colors hover:border-green-400 hover:bg-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-green-400"
              onClick={() => handleSelect(actionIndex)}
            >
              <span className="block text-xs font-semibold uppercase tracking-wide text-green-300">
                Option {actionIndex + 1}
              </span>
              <span className="mt-1 block">
                {getSpellModeDescription(pending.cardCode, action[2])}
              </span>
            </button>
          ))}
        </div>

        <div className="mt-5 flex justify-end">
          <button
            type="button"
            className="rounded-md bg-slate-600 px-4 py-2 text-white transition-colors hover:bg-slate-500"
            onClick={clear}
          >
            Cancel
          </button>
        </div>
      </section>
    </div>
  );
}
