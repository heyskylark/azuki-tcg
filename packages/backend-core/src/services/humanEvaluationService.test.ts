import assert from "node:assert/strict";
import { test } from "node:test";
import {
  canMaterializeHumanEvaluationDeck,
  copyHumanDeckSnapshotJunctions,
  createHumanEvaluationRuntimeSessionKey,
} from "@core/services/humanEvaluationService";
import { HumanEvaluationMatchStatus } from "@core/types/humanEvaluations";

test("human evaluation snapshot quantities do not follow later source deck mutations", () => {
  const sourceJunctions = [
    { cardId: "leader", quantity: 1 },
    { cardId: "gate", quantity: 1 },
    { cardId: "main-card", quantity: 4 },
  ];

  const claimedRoomDeckJunctions = copyHumanDeckSnapshotJunctions(
    "evaluation-snapshot",
    sourceJunctions
  );

  const sourceMainCard = sourceJunctions[2];
  if (!sourceMainCard) {
    throw new Error("Test source deck is missing its main card");
  }
  sourceMainCard.quantity = 1;

  assert.deepEqual(claimedRoomDeckJunctions, [
    { deckId: "evaluation-snapshot", cardId: "leader", quantity: 1 },
    { deckId: "evaluation-snapshot", cardId: "gate", quantity: 1 },
    { deckId: "evaluation-snapshot", cardId: "main-card", quantity: 4 },
  ]);
});

test("runtime session keys are random opaque values unrelated to public ids", () => {
  const sessionId = "0198f782-cb6d-7eb5-b5cf-6ce7c625c94f";
  const matchId = "0198f782-d12e-7b96-a92a-70182049f117";
  const firstKey = createHumanEvaluationRuntimeSessionKey();
  const secondKey = createHumanEvaluationRuntimeSessionKey();

  assert.match(firstKey, /^human-eval:[a-f0-9]{64}$/);
  assert.notEqual(firstKey, secondKey);
  assert.equal(firstKey.includes(sessionId), false);
  assert.equal(firstKey.includes(matchId), false);
});

test("deck materialization is forbidden after evaluation finalization", () => {
  assert.equal(canMaterializeHumanEvaluationDeck(HumanEvaluationMatchStatus.CLAIMED), true);
  assert.equal(canMaterializeHumanEvaluationDeck(HumanEvaluationMatchStatus.IN_PROGRESS), true);
  assert.equal(canMaterializeHumanEvaluationDeck(HumanEvaluationMatchStatus.COMPLETED), false);
  assert.equal(canMaterializeHumanEvaluationDeck(HumanEvaluationMatchStatus.ABORTED), false);
});
