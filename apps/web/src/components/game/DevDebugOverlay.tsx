"use client";

import { useState } from "react";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import { useDragStore } from "@/stores/dragStore";

// Action type index to name mapping (from actionValidation.ts)
const ACTION_NAMES: Record<number, string> = {
  0: "NOOP",
  1: "PLAY_ENTITY_GARDEN",
  2: "PLAY_ENTITY_ALLEY",
  6: "ATTACK",
  7: "ATTACH_WEAPON_FROM_HAND",
  8: "PLAY_SPELL_FROM_HAND",
  9: "DECLARE_DEFENDER",
  10: "GATE_PORTAL",
  11: "ACTIVATE_GARDEN_OR_LEADER_ABILITY",
  12: "ACTIVATE_ALLEY_ABILITY",
  13: "SELECT_COST_TARGET",
  14: "SELECT_EFFECT_TARGET",
  16: "CONFIRM_ABILITY",
  18: "SELECT_FROM_SELECTION",
  19: "BOTTOM_DECK_CARD",
  20: "BOTTOM_DECK_ALL",
  21: "SELECT_TO_ALLEY",
  22: "SELECT_TO_EQUIP",
  23: "MULLIGAN_SHUFFLE",
};

function getActionName(index: number): string {
  return ACTION_NAMES[index] ?? `UNKNOWN_${index}`;
}

function setToString(set: Set<number>): string {
  if (set.size === 0) return "{}";
  return `{${[...set].join(", ")}}`;
}

export function DevDebugOverlay() {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const { gameState } = useGameState();
  const { activeRoom, connectionStatus } = useRoom();
  const dragStore = useDragStore();

  // Compute unique legal action types
  const legalActionTypes = gameState?.actionMask?.legalPrimary
    ? [...new Set(gameState.actionMask.legalPrimary)]
    : [];
  const legalActionNames = legalActionTypes.map(getActionName);

  // Compute board statistics
  const myGardenFilled = gameState?.myBoard?.garden.filter((c) => c !== null).length ?? 0;
  const myAlleyFilled = gameState?.myBoard?.alley.filter((c) => c !== null).length ?? 0;
  const oppGardenFilled = gameState?.opponentBoard?.garden.filter((c) => c !== null).length ?? 0;
  const oppAlleyFilled = gameState?.opponentBoard?.alley.filter((c) => c !== null).length ?? 0;

  const isMyTurn = gameState
    ? gameState.activePlayer === activeRoom?.playerSlot
    : false;

  if (isCollapsed) {
    return (
      <button
        onClick={() => setIsCollapsed(false)}
        className="fixed top-2 left-2 z-[100] bg-black/80 text-white px-2 py-1 rounded text-xs font-mono hover:bg-black/90"
      >
        [Dev]
      </button>
    );
  }

  return (
    <div className="fixed top-2 left-2 z-[100] bg-black/85 text-white rounded-lg shadow-xl font-mono text-xs max-w-[320px] border border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700 bg-gray-800/50 rounded-t-lg">
        <span className="text-gray-300 font-semibold">Dev Debug</span>
        <button
          onClick={() => setIsCollapsed(true)}
          className="text-gray-400 hover:text-white px-1"
        >
          [-]
        </button>
      </div>

      <div className="p-3 space-y-3">
        {/* Phase & Turn Info */}
        <Section title="Phase">
          <Row label="Phase" value={gameState?.phase ?? "-"} />
          <Row label="Subphase" value={gameState?.abilitySubphase ?? "-"} />
          <Row
            label="Turn"
            value={`${gameState?.turnNumber ?? "-"} | Active: P${gameState?.activePlayer ?? "-"}`}
          />
          <Row
            label="My Turn"
            value={isMyTurn ? "Yes" : "No"}
            valueClass={isMyTurn ? "text-green-400" : "text-gray-400"}
          />
        </Section>

        {/* Drag Store State */}
        <Section title="Drag State">
          <Row
            label="Phase"
            value={dragStore.dragPhase}
            valueClass={dragStore.dragPhase !== "idle" ? "text-yellow-400" : ""}
          />
          <Row
            label="Card Idx"
            value={dragStore.draggedCardIndex !== null ? String(dragStore.draggedCardIndex) : "-"}
          />
          <Row label="Card Code" value={dragStore.draggedCardCode ?? "-"} />
          <Row label="Hover Zone" value={dragStore.hoveredZone ?? "-"} />
          <Row
            label="Hover Slot"
            value={dragStore.hoveredSlotIndex !== null ? String(dragStore.hoveredSlotIndex) : "-"}
          />
          <Row label="Valid Garden" value={setToString(dragStore.validGardenSlots)} />
          <Row label="Valid Alley" value={setToString(dragStore.validAlleySlots)} />
        </Section>

        {/* Action Mask Info */}
        <Section title="Actions">
          <Row
            label="Has Mask"
            value={gameState?.actionMask ? "Yes" : "No"}
            valueClass={gameState?.actionMask ? "text-green-400" : "text-gray-400"}
          />
          <Row
            label="Legal Count"
            value={String(gameState?.actionMask?.legalActionCount ?? 0)}
          />
          <div className="mt-1">
            <span className="text-gray-400">Types: </span>
            <span className="text-blue-300 break-words">
              {legalActionNames.length > 0 ? legalActionNames.join(", ") : "-"}
            </span>
          </div>
        </Section>

        {/* Board Summary */}
        <Section title="Board - Me">
          <Row
            label="Leader HP"
            value={String(gameState?.myBoard?.leader.curHp ?? "-")}
          />
          <Row label="Garden" value={`${myGardenFilled}/5`} />
          <Row label="Alley" value={`${myAlleyFilled}/5`} />
          <Row label="Hand" value={String(gameState?.myHand?.length ?? "-")} />
          <Row label="Deck" value={String(gameState?.myBoard?.deckCount ?? "-")} />
          <Row
            label="IKZ"
            value={`${gameState?.myBoard?.ikzArea.length ?? 0}/${gameState?.myBoard?.ikzPileCount ?? 0}`}
          />
        </Section>

        <Section title="Board - Opp">
          <Row
            label="Leader HP"
            value={String(gameState?.opponentBoard?.leader.curHp ?? "-")}
          />
          <Row label="Garden" value={`${oppGardenFilled}/5`} />
          <Row label="Alley" value={`${oppAlleyFilled}/5`} />
          <Row label="Hand" value={String(gameState?.opponentBoard?.handCount ?? "-")} />
          <Row label="Deck" value={String(gameState?.opponentBoard?.deckCount ?? "-")} />
          <Row
            label="IKZ"
            value={`${gameState?.opponentBoard?.ikzArea.length ?? 0}/${gameState?.opponentBoard?.ikzPileCount ?? 0}`}
          />
        </Section>

        {/* Selection & Connection */}
        <Section title="Connection">
          <Row
            label="Status"
            value={connectionStatus}
            valueClass={
              connectionStatus === "connected"
                ? "text-green-400"
                : connectionStatus === "error"
                  ? "text-red-400"
                  : "text-yellow-400"
            }
          />
          <Row
            label="Player Slot"
            value={activeRoom?.playerSlot !== undefined ? `P${activeRoom.playerSlot}` : "-"}
          />
          <Row
            label="Selection"
            value={`${gameState?.selectionCards?.length ?? 0} cards`}
          />
        </Section>
      </div>
    </div>
  );
}

// Section component for grouping related debug info
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-gray-400 font-semibold mb-1 text-[10px] uppercase tracking-wider">
        {title}
      </div>
      <div className="bg-gray-900/50 rounded px-2 py-1.5 space-y-0.5">{children}</div>
    </div>
  );
}

// Row component for key-value pairs
function Row({
  label,
  value,
  valueClass = "",
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-500">{label}:</span>
      <span className={`text-gray-200 ${valueClass}`}>{value}</span>
    </div>
  );
}
