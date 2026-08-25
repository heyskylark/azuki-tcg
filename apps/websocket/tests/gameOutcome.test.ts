import { describe, expect, test } from "bun:test";
import { WinType } from "@tcg/backend-core/types";
import { classifyGameOutcome } from "@/engine/gameOutcome";

describe("classifyGameOutcome", () => {
  test("maps the simultaneous leader defeat sentinel to a draw", () => {
    expect(classifyGameOutcome(2)).toEqual({
      winnerSlot: null,
      winType: WinType.DRAW,
    });
  });

  test("preserves player winners", () => {
    expect(classifyGameOutcome(0)).toEqual({
      winnerSlot: 0,
      winType: WinType.WIN,
    });
    expect(classifyGameOutcome(1)).toEqual({
      winnerSlot: 1,
      winType: WinType.WIN,
    });
  });
});
