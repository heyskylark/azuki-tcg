import assert from "node:assert/strict";
import { describe, test } from "node:test";
import type { SnapshotActionMask } from "@tcg/backend-core/types/ws";
import {
  ACTION_PLAY_SPELL_FROM_HAND,
  getValidSpellActions,
  isSpellActionLegal,
} from "@/lib/game/actionValidation";
import {
  isSpellModeSelectionCurrent,
  type PendingSpellModeSelection,
} from "@/stores/spellModeStore";

function actionMask(rows: Array<[number, number, number, number]>): SnapshotActionMask {
  return {
    primaryActionMask: [],
    legalActionCount: rows.length,
    legalPrimary: rows.map((row) => row[0]),
    legalSub1: rows.map((row) => row[1]),
    legalSub2: rows.map((row) => row[2]),
    legalSub3: rows.map((row) => row[3]),
  };
}

describe("getValidSpellActions", () => {
  test("preserves two distinct spell modes", () => {
    const mask = actionMask([
      [ACTION_PLAY_SPELL_FROM_HAND, 2, 0, 0],
      [ACTION_PLAY_SPELL_FROM_HAND, 2, 1, 0],
    ]);

    assert.deepEqual(getValidSpellActions(mask, 2), [
      [ACTION_PLAY_SPELL_FROM_HAND, 2, 0, 0],
      [ACTION_PLAY_SPELL_FROM_HAND, 2, 1, 0],
    ]);
  });

  test("returns a single legal nonzero ability index unchanged", () => {
    const mask = actionMask([[ACTION_PLAY_SPELL_FROM_HAND, 1, 4, 1]]);

    assert.deepEqual(getValidSpellActions(mask, 1), [
      [ACTION_PLAY_SPELL_FROM_HAND, 1, 4, 1],
    ]);
  });

  test("groups payment alternatives by ability and prefers no token per mode", () => {
    const mask = actionMask([
      [ACTION_PLAY_SPELL_FROM_HAND, 3, 1, 1],
      [ACTION_PLAY_SPELL_FROM_HAND, 3, 0, 1],
      [ACTION_PLAY_SPELL_FROM_HAND, 3, 1, 0],
      [ACTION_PLAY_SPELL_FROM_HAND, 3, 0, 0],
    ]);

    assert.deepEqual(getValidSpellActions(mask, 3), [
      [ACTION_PLAY_SPELL_FROM_HAND, 3, 0, 0],
      [ACTION_PLAY_SPELL_FROM_HAND, 3, 1, 0],
    ]);
  });

  test("does not combine modes or cards", () => {
    const mask = actionMask([
      [ACTION_PLAY_SPELL_FROM_HAND, 0, 0, 0],
      [ACTION_PLAY_SPELL_FROM_HAND, 1, 1, 0],
      [1, 0, 2, 0],
    ]);

    assert.deepEqual(getValidSpellActions(mask, 0), [
      [ACTION_PLAY_SPELL_FROM_HAND, 0, 0, 0],
    ]);
  });

  test("safely ignores incomplete and malformed rows", () => {
    const mask = actionMask([
      [ACTION_PLAY_SPELL_FROM_HAND, 0, 0, 0],
      [ACTION_PLAY_SPELL_FROM_HAND, 0, 1, 0],
    ]);
    mask.legalSub3.pop();
    mask.legalSub2[0] = -1;

    assert.deepEqual(getValidSpellActions(mask, 0), []);
    assert.deepEqual(getValidSpellActions(null, 0), []);
  });
});

describe("spell selection freshness", () => {
  const mask = actionMask([
    [ACTION_PLAY_SPELL_FROM_HAND, 0, 0, 0],
    [ACTION_PLAY_SPELL_FROM_HAND, 0, 1, 0],
  ]);
  const selection: PendingSpellModeSelection = {
    cardCode: "STT03-017",
    cardName: "Sprout of Fortune",
    handIndex: 0,
    actions: getValidSpellActions(mask, 0),
    actionMask: mask,
    phase: "MAIN",
    abilitySubphase: "NONE",
    activePlayer: 0,
    turnNumber: 3,
  };
  const currentState = {
    actionMask: mask,
    phase: "MAIN",
    abilitySubphase: "NONE",
    activePlayer: selection.activePlayer,
    turnNumber: 3,
    myHand: [{ cardCode: "STT03-017" }],
  };

  test("accepts an original tuple while its exact mask row remains legal", () => {
    assert.equal(isSpellModeSelectionCurrent(selection, currentState, 0), true);
    assert.equal(isSpellActionLegal(mask, selection.actions[1]), true);
  });

  test("rejects a replaced mask even when it contains equivalent rows", () => {
    const replacedMask = actionMask([
      [ACTION_PLAY_SPELL_FROM_HAND, 0, 0, 0],
      [ACTION_PLAY_SPELL_FROM_HAND, 0, 1, 0],
    ]);

    assert.equal(isSpellModeSelectionCurrent(selection, { ...currentState, actionMask: replacedMask }, 0), false);
  });

  test("rejects turn, player, phase, and hand changes", () => {
    assert.equal(isSpellModeSelectionCurrent(selection, { ...currentState, turnNumber: 4 }, 0), false);
    assert.equal(isSpellModeSelectionCurrent(selection, currentState, 1), false);
    assert.equal(isSpellModeSelectionCurrent(selection, { ...currentState, abilitySubphase: "CONFIRMATION" }, 0), false);
    assert.equal(isSpellModeSelectionCurrent(
      selection,
      { ...currentState, myHand: [{ cardCode: "OTHER" }] },
      0
    ), false);
  });

  test("rejects a tuple missing from the current mask", () => {
    assert.equal(isSpellActionLegal(mask, [ACTION_PLAY_SPELL_FROM_HAND, 0, 9, 0]), false);
    assert.equal(isSpellActionLegal(null, [ACTION_PLAY_SPELL_FROM_HAND, 0, 0, 0]), false);
  });

});
