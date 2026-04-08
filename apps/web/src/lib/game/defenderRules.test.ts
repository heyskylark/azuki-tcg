import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { countGardenDefenders, countsAsGardenDefender } from "@/lib/game/defenderRules";
import type { ResolvedCard, ResolvedPlayerBoard } from "@/types/game";

function createBoardCard(
  cardCode: string,
  zoneIndex: number,
  overrides: Partial<ResolvedCard> = {}
): ResolvedCard {
  return {
    cardCode,
    cardDefId: zoneIndex + 100,
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

function createBoard(garden: (ResolvedCard | null)[]): ResolvedPlayerBoard {
  return {
    leader: {
      cardCode: "LEADER",
      cardDefId: 1,
      imageUrl: "https://example.test/leader.jpg",
      name: "Leader",
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
      cardCode: "GATE",
      cardDefId: 2,
      imageUrl: "https://example.test/gate.jpg",
      name: "Gate",
      tapped: false,
      cooldown: false,
    },
    garden,
    alley: [null, null, null, null, null],
    ikzArea: [],
    handCount: 0,
    deckCount: 30,
    discardCount: 0,
    ikzPileCount: 0,
    hasIkzToken: false,
  };
}

describe("defenderRules", () => {
  test("does not count Yojin as a defender when its controller has at least as many entities", () => {
    const yojin = createBoardCard("AZK01-052", 3, { hasDefender: true });
    const myBoard = createBoard([
      createBoardCard("AZK01-001", 0, { hasDefender: true }),
      createBoardCard("AZK01-025", 1, { hasDefender: true }),
      createBoardCard("AZK01-035", 2, { hasDefender: true }),
      yojin,
      null,
    ]);
    const opponentBoard = createBoard([
      createBoardCard("AZK01-070", 0),
      createBoardCard("AZK01-071", 1),
      createBoardCard("AZK01-072", 2),
      null,
      null,
    ]);

    assert.equal(countsAsGardenDefender(yojin, myBoard, opponentBoard), false);
    assert.equal(countGardenDefenders(myBoard, opponentBoard), 3);
  });

  test("counts Yojin as a defender when the opponent has more garden entities", () => {
    const yojin = createBoardCard("AZK01-052", 1, { hasDefender: false });
    const myBoard = createBoard([createBoardCard("AZK01-070", 0), yojin, null, null, null]);
    const opponentBoard = createBoard([
      createBoardCard("AZK01-071", 0),
      createBoardCard("AZK01-072", 1),
      createBoardCard("AZK01-073", 2),
      null,
      null,
    ]);

    assert.equal(countsAsGardenDefender(yojin, myBoard, opponentBoard), true);
  });

  test("still counts other cards that currently have defender from a temporary effect", () => {
    const temporaryDefender = createBoardCard("AZK01-070", 0, { hasDefender: true });
    const myBoard = createBoard([temporaryDefender, null, null, null, null]);
    const opponentBoard = createBoard([null, null, null, null, null]);

    assert.equal(countGardenDefenders(myBoard, opponentBoard), 1);
  });
});
