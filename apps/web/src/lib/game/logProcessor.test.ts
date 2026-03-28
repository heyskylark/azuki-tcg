import assert from "node:assert/strict";
import { describe, test } from "node:test";

import type { GameState, CardMapping, ResolvedCard } from "@/types/game";
import type { ProcessedGameLog } from "@/types/gameLogs";
import { applySingleLog, createBatchIndexRebaseContext } from "@/lib/game/logProcessor";

function createCardMapping(cardCode: string, cardDefId: number, name: string): CardMapping {
  return {
    cardCode,
    imageKey: `${cardCode}.jpg`,
    imageUrl: `https://example.test/${cardCode}.jpg`,
    name,
    cardType: "ENTITY",
    attack: 1,
    health: 1,
    ikzCost: 0,
  };
}

function createBoardCard(
  cardCode: string,
  cardDefId: number,
  zoneIndex: number,
  overrides: Partial<ResolvedCard> = {}
): ResolvedCard {
  return {
    cardCode,
    cardDefId,
    imageUrl: `https://example.test/${cardCode}.jpg`,
    name: cardCode,
    curAtk: 1,
    curHp: 1,
    tapped: false,
    cooldown: false,
    isFrozen: false,
    isRooted: false,
    isShocked: false,
    isEffectImmune: false,
    hasCharge: false,
    hasDefender: false,
    hasInfiltrate: false,
    zoneIndex,
    ...overrides,
  };
}

function createTestState(): GameState {
  return {
    phase: "MAIN",
    abilitySubphase: "NONE",
    activePlayer: 0,
    turnNumber: 1,
    myBoard: {
      leader: {
        cardCode: "LEADER-0",
        cardDefId: 1,
        imageUrl: "https://example.test/leader-0.jpg",
        name: "Leader 0",
        curAtk: 0,
        curHp: 20,
        tapped: false,
        cooldown: false,
        isFrozen: false,
        isRooted: false,
        isShocked: false,
        isEffectImmune: false,
        hasCharge: false,
        hasDefender: false,
        hasInfiltrate: false,
      },
      gate: {
        cardCode: "GATE-0",
        cardDefId: 2,
        imageUrl: "https://example.test/gate-0.jpg",
        name: "Gate 0",
        tapped: false,
        cooldown: false,
      },
      garden: [
        createBoardCard("AZK01-097", 129, 0, { curAtk: 2, curHp: 2, tapped: true }),
        null,
        null,
        null,
        null,
      ],
      alley: [null, createBoardCard("AZK01-096", 128, 1), null, null, null],
      ikzArea: [],
      handCount: 0,
      deckCount: 30,
      discardCount: 0,
      ikzPileCount: 0,
      hasIkzToken: false,
    },
    opponentBoard: {
      leader: {
        cardCode: "LEADER-1",
        cardDefId: 3,
        imageUrl: "https://example.test/leader-1.jpg",
        name: "Leader 1",
        curAtk: 0,
        curHp: 20,
        tapped: false,
        cooldown: false,
        isFrozen: false,
        isRooted: false,
        isShocked: false,
        isEffectImmune: false,
        hasCharge: false,
        hasDefender: false,
        hasInfiltrate: false,
      },
      gate: {
        cardCode: "GATE-1",
        cardDefId: 4,
        imageUrl: "https://example.test/gate-1.jpg",
        name: "Gate 1",
        tapped: false,
        cooldown: false,
      },
      garden: [null, null, null, null, null],
      alley: [null, null, null, null, null],
      ikzArea: [],
      handCount: 0,
      deckCount: 30,
      discardCount: 0,
      ikzPileCount: 0,
      hasIkzToken: false,
    },
    myHand: [
      { cardCode: "H-0", cardDefId: 200, imageUrl: "", name: "H-0", type: "SPELL", ikzCost: 0 },
      { cardCode: "H-1", cardDefId: 201, imageUrl: "", name: "H-1", type: "SPELL", ikzCost: 0 },
      { cardCode: "H-2", cardDefId: 202, imageUrl: "", name: "H-2", type: "SPELL", ikzCost: 0 },
      { cardCode: "H-3", cardDefId: 203, imageUrl: "", name: "H-3", type: "SPELL", ikzCost: 0 },
      { cardCode: "H-4", cardDefId: 204, imageUrl: "", name: "H-4", type: "SPELL", ikzCost: 0 },
      { cardCode: "AZK01-096", cardDefId: 127, imageUrl: "", name: "Spell", type: "SPELL", ikzCost: 0 },
    ],
    selectionCards: undefined,
    actionMask: null,
    combatStack: [],
  };
}

describe("applySingleLog", () => {
  test("keeps both cards visible when a garden/alley swap is logged in one batch", () => {
    const state = createTestState();
    const cardDefIdMap = new Map<number, CardMapping>([
      [127, createCardMapping("AZK01-096-SPELL", 127, "Spell")],
      [128, createCardMapping("AZK01-096", 128, "Alley Card")],
      [129, createCardMapping("AZK01-097", 129, "Garden Card")],
    ]);
    const cardMappings = new Map<string, CardMapping>(
      Array.from(cardDefIdMap.values()).map((mapping) => [mapping.cardCode, mapping])
    );
    const batchContext = createBatchIndexRebaseContext();
    const logs: ProcessedGameLog[] = [
      {
        type: "ZONE_MOVED",
        data: {
          card: { player: 0, cardDefId: 127, zone: "DISCARD", zoneIndex: -1 },
          fromZone: "HAND",
          fromIndex: 5,
          toZone: "DISCARD",
          toIndex: -1,
          metadata: {
            curAtk: 0,
            curHp: 0,
            tapped: false,
            cooldown: false,
            hasCharge: false,
            hasDefender: false,
            hasInfiltrate: false,
            isFrozen: false,
            isRooted: false,
            isEffectImmune: false,
          },
        },
      },
      {
        type: "ZONE_MOVED",
        data: {
          card: { player: 0, cardDefId: 129, zone: "ALLEY", zoneIndex: 1 },
          fromZone: "GARDEN",
          fromIndex: 0,
          toZone: "ALLEY",
          toIndex: 1,
          metadata: {
            curAtk: 2,
            curHp: 2,
            tapped: true,
            cooldown: false,
            hasCharge: false,
            hasDefender: false,
            hasInfiltrate: false,
            isFrozen: false,
            isRooted: false,
            isEffectImmune: false,
          },
        },
      },
      {
        type: "ZONE_MOVED",
        data: {
          card: { player: 0, cardDefId: 128, zone: "GARDEN", zoneIndex: 0 },
          fromZone: "ALLEY",
          fromIndex: 1,
          toZone: "GARDEN",
          toIndex: 0,
          metadata: {
            curAtk: 1,
            curHp: 1,
            tapped: false,
            cooldown: false,
            hasCharge: false,
            hasDefender: false,
            hasInfiltrate: false,
            isFrozen: false,
            isRooted: false,
            isEffectImmune: false,
          },
        },
      },
    ];

    const finalState = logs.reduce(
      (currentState, log) =>
        applySingleLog(currentState, log, 0, cardMappings, cardDefIdMap, batchContext),
      state
    );

    assert.equal(finalState.myHand.length, 5);
    assert.equal(finalState.myBoard.discardCount, 1);
    assert.equal(finalState.myBoard.garden[0]?.cardDefId, 128);
    assert.equal(finalState.myBoard.alley[1]?.cardDefId, 129);
    assert.equal(finalState.myBoard.garden[0]?.curAtk, 1);
    assert.equal(finalState.myBoard.alley[1]?.curAtk, 2);
  });
});
