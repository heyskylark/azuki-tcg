import { create } from "zustand";
import type { SnapshotActionMask } from "@tcg/backend-core/types/ws";
import type { SpellAction } from "@/lib/game/actionValidation";
interface SpellSelectionGameState {
  actionMask: SnapshotActionMask | null;
  phase: string;
  abilitySubphase: string;
  activePlayer: 0 | 1;
  turnNumber: number;
  myHand: Array<{ cardCode: string }>;
}

export interface PendingSpellModeSelection {
  cardCode: string;
  cardName: string;
  handIndex: number;
  actions: SpellAction[];
  actionMask: SnapshotActionMask;
  phase: string;
  abilitySubphase: string;
  activePlayer: 0 | 1;
  turnNumber: number;
}

interface SpellModeState {
  pending: PendingSpellModeSelection | null;
  open: (selection: PendingSpellModeSelection) => void;
  clear: () => void;
}

export function isSpellModeSelectionCurrent(
  selection: PendingSpellModeSelection,
  gameState: SpellSelectionGameState | null,
  playerSlot: 0 | 1 | null
): boolean {
  if (
    !gameState ||
    playerSlot === null ||
    gameState.actionMask !== selection.actionMask ||
    gameState.phase !== selection.phase ||
    gameState.abilitySubphase !== selection.abilitySubphase ||
    gameState.activePlayer !== selection.activePlayer ||
    gameState.activePlayer !== playerSlot ||
    gameState.turnNumber !== selection.turnNumber
  ) {
    return false;
  }

  const handCard = gameState.myHand[selection.handIndex];
  return handCard?.cardCode === selection.cardCode;
}

export const useSpellModeStore = create<SpellModeState>((set) => ({
  pending: null,
  open: (selection) => set({ pending: selection }),
  clear: () => set({ pending: null }),
}));
