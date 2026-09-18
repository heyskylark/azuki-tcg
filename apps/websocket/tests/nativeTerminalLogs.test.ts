import { afterEach, expect, test } from "bun:test";
import { createRequire } from "node:module";

interface NativeLog {
  type: string;
  data: { winner?: number; reason?: string };
}

interface NativeActionResult {
  success: boolean;
  invalid: boolean;
  error?: string;
  gameOver: boolean;
  winner: number | null;
  logs: NativeLog[];
}

interface NativeBinding {
  createWorld(seed: number): { success: boolean; worldId: string };
  destroyWorld(worldId: string): void;
  isGameOver(worldId: string): boolean;
  getActivePlayer(worldId: string): number;
  submitAction(
    worldId: string,
    player: number,
    action: [number, number, number, number]
  ): NativeActionResult;
}

const require = createRequire(import.meta.url);
const binding: NativeBinding = require("../native/build/Release/azuki_engine.node");
let worldId: string | null = null;

afterEach(() => {
  if (worldId !== null) {
    binding.destroyWorld(worldId);
    worldId = null;
  }
});

test("native terminal result includes one persisted-compatible deck-out log", () => {
  const created = binding.createWorld(424242);
  expect(created.success).toBe(true);
  worldId = created.worldId;

  let finalResult: NativeActionResult | null = null;
  for (
    let actionCount = 0;
    actionCount < 200 && !binding.isGameOver(created.worldId);
    ++actionCount
  ) {
    const player = binding.getActivePlayer(created.worldId);
    const result = binding.submitAction(created.worldId, player, [0, 0, 0, 0]);
    if (!result.success || result.invalid) {
      throw new Error(result.error ?? "Native action failed");
    }
    if (result.gameOver) {
      finalResult = result;
    }
  }

  expect(finalResult).not.toBeNull();
  if (finalResult === null) {
    throw new Error("Expected a terminal native action result");
  }
  expect(finalResult.gameOver).toBe(true);
  expect([0, 1]).toContain(finalResult.winner);

  const terminalLogs = finalResult.logs.filter(
    (log) => log.type === "GAME_ENDED"
  );
  expect(terminalLogs).toHaveLength(1);
  expect(terminalLogs[0]?.data).toEqual({
    winner: finalResult.winner,
    reason: "DECK_OUT",
  });
  expect(() => JSON.parse(JSON.stringify(finalResult.logs))).not.toThrow();
});
