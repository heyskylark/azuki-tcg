import { WinType } from "@tcg/backend-core/types";

export interface GameOutcome {
  winnerSlot: 0 | 1 | null;
  winType: WinType;
}

/** Normalize the engine's winner sentinel (2) before persistence and broadcast. */
export function classifyGameOutcome(winner: number | null): GameOutcome {
  if (winner === null || winner === 2) {
    return { winnerSlot: null, winType: WinType.DRAW };
  }
  if (winner === 0 || winner === 1) {
    return { winnerSlot: winner, winType: WinType.WIN };
  }
  throw new Error(`Invalid game winner: ${winner}`);
}
