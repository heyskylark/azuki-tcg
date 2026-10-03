import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildHumanEvaluationModelAssignments,
  canMaterializeHumanEvaluationDeck,
  copyHumanDeckSnapshotJunctions,
  createHumanEvaluationRuntimeSessionKey,
} from "@core/services/humanEvaluationService";
import {
  HumanEvaluationDeckSource,
  HumanEvaluationMatchStatus,
  type HumanEvaluationPlan,
} from "@core/types/humanEvaluations";

const EARTH_PLAN: HumanEvaluationPlan = {
  premadeFraction: 0.5,
  premadeDecks: [
    { slug: "cat", name: "Cat", gateCardCode: "STT03-002", leaderCardCode: "STT03-001" },
    { slug: "goro", name: "Goro", gateCardCode: "STT03-002", leaderCardCode: "AZK01-123" },
  ],
  draftGates: [{ gateCardCode: "STT03-002", leaderCardCodes: ["STT03-001", "AZK01-123"] }],
};

function matchIds(count: number, prefix: string): string[] {
  return Array.from({ length: count }, (_, index) => `${prefix}-${index}`);
}

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

test("planned assignments split deck sources exactly and stay inside the plan", () => {
  for (const gamesPerModel of [8, 16] as const) {
    const assignments = buildHumanEvaluationModelAssignments({
      seed: `seed-${gamesPerModel}`,
      modelId: "model",
      gamesPerModel,
      plan: EARTH_PLAN,
      matchIds: matchIds(gamesPerModel, "match"),
    });
    const premade = assignments.filter(
      ({ deckSource }) => deckSource === HumanEvaluationDeckSource.PREMADE
    );
    assert.equal(premade.length, gamesPerModel / 2);
    for (const assignment of assignments) {
      if (assignment.deckSource === HumanEvaluationDeckSource.PREMADE) {
        const deck = EARTH_PLAN.premadeDecks.find(
          ({ slug }) => slug === assignment.premadeDeckSlug
        );
        assert.ok(deck);
        assert.equal(assignment.gateCardCode, deck.gateCardCode);
        assert.equal(assignment.leaderCardCode, deck.leaderCardCode);
      } else {
        assert.equal(assignment.premadeDeckSlug, null);
        const gate = EARTH_PLAN.draftGates.find(
          ({ gateCardCode }) => gateCardCode === assignment.gateCardCode
        );
        assert.ok(gate?.leaderCardCodes.includes(assignment.leaderCardCode));
      }
    }
    const seatCells = new Set(assignments.map((a) => `${a.aiSlot}:${a.startingPlayer}`));
    assert.equal(seatCells.size, 4);
  }
});

test("planned assignments are reproducible from the stored seed and ids", () => {
  const input = {
    seed: "stored-seed",
    modelId: "model",
    gamesPerModel: 16 as const,
    plan: EARTH_PLAN,
    matchIds: matchIds(16, "match"),
  };
  assert.deepEqual(
    buildHumanEvaluationModelAssignments(input),
    buildHumanEvaluationModelAssignments({ ...input, matchIds: [...input.matchIds] })
  );
  assert.notDeepEqual(
    buildHumanEvaluationModelAssignments(input),
    buildHumanEvaluationModelAssignments({ ...input, seed: "other-seed" })
  );
});

test("premade fraction bounds produce single-source schedules", () => {
  for (const [premadeFraction, expected] of [
    [0, HumanEvaluationDeckSource.DRAFT],
    [1, HumanEvaluationDeckSource.PREMADE],
  ] as const) {
    const assignments = buildHumanEvaluationModelAssignments({
      seed: "seed",
      modelId: "model",
      gamesPerModel: 8,
      plan: { ...EARTH_PLAN, premadeFraction },
      matchIds: matchIds(8, "match"),
    });
    assert.ok(assignments.every(({ deckSource }) => deckSource === expected));
  }
});
