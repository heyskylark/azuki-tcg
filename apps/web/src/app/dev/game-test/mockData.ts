/**
 * Mock game state generator for testing the 3D game renderer.
 * Uses real deck card data to create test scenarios.
 */

import type { GameState, DeckCard, ResolvedCard, ResolvedLeader, ResolvedGate, ResolvedIkz, ResolvedHandCard, ResolvedPlayerBoard } from "@/types/game";
import { buildImageUrl } from "@/types/game";

// ============================================
// Helper functions to create resolved cards
// ============================================

function findCardByType(cards: DeckCard[], cardType: string): DeckCard | undefined {
  return cards.find((c) => c.cardType === cardType);
}

function findCardsByType(cards: DeckCard[], cardType: string): DeckCard[] {
  return cards.filter((c) => c.cardType === cardType);
}

function createResolvedCard(
  card: DeckCard,
  zoneIndex: number,
  overrides: Partial<ResolvedCard> = {}
): ResolvedCard {
  return {
    cardCode: card.cardCode,
    cardDefId: zoneIndex + 1,
    imageUrl: buildImageUrl(card.imageKey),
    name: card.name,
    curAtk: card.attack,
    curHp: card.health,
    tapped: false,
    cooldown: false,
    isFrozen: false,
    isShocked: false,
    isEffectImmune: false,
    hasCharge: false,
    hasDefender: false,
    hasInfiltrate: false,
    zoneIndex,
    ...overrides,
  };
}

function createResolvedLeader(card: DeckCard, hpOverride?: number): ResolvedLeader {
  return {
    cardCode: card.cardCode,
    cardDefId: 1,
    imageUrl: buildImageUrl(card.imageKey),
    name: card.name,
    curAtk: card.attack ?? 0,
    curHp: hpOverride ?? card.health ?? 25,
    tapped: false,
    cooldown: false,
    isFrozen: false,
    isShocked: false,
    isEffectImmune: false,
    hasCharge: false,
    hasDefender: false,
    hasInfiltrate: false,
  };
}

function createResolvedGate(card: DeckCard): ResolvedGate {
  return {
    cardCode: card.cardCode,
    cardDefId: 2,
    imageUrl: buildImageUrl(card.imageKey),
    name: card.name,
    tapped: false,
    cooldown: false,
  };
}

function createResolvedIkz(card: DeckCard, tapped = false): ResolvedIkz {
  return {
    cardCode: card.cardCode,
    cardDefId: 0,
    imageUrl: buildImageUrl(card.imageKey),
    name: card.name,
    tapped,
    cooldown: false,
  };
}

function createResolvedHandCard(card: DeckCard): ResolvedHandCard {
  return {
    cardCode: card.cardCode,
    cardDefId: 0,
    imageUrl: buildImageUrl(card.imageKey),
    name: card.name,
    type: card.cardType,
    ikzCost: card.ikzCost ?? 0,
  };
}

// ============================================
// Create board from deck cards
// ============================================

function createPlayerBoard(
  deckCards: DeckCard[],
  gardenCards: (ResolvedCard | null)[],
  alleyCards: (ResolvedCard | null)[],
  ikzCount: number,
  tappedIkzCount: number,
  leaderHp: number
): ResolvedPlayerBoard {
  const leader = findCardByType(deckCards, "LEADER");
  const gate = findCardByType(deckCards, "GATE");
  const ikzCard = findCardByType(deckCards, "IKZ") ?? findCardByType(deckCards, "EXTRA_IKZ");

  // Create IKZ area
  const ikzArea: ResolvedIkz[] = [];
  for (let i = 0; i < ikzCount; i++) {
    if (ikzCard) {
      ikzArea.push(createResolvedIkz(ikzCard, i < tappedIkzCount));
    }
  }

  return {
    leader: leader ? createResolvedLeader(leader, leaderHp) : createResolvedLeader({
      cardCode: "UNKNOWN",
      cardDefId: 0,
      imageKey: "unknown.png",
      name: "Unknown Leader",
      cardType: "LEADER",
      attack: 2,
      health: 25,
      ikzCost: null,
      quantity: 1,
    }, leaderHp),
    gate: gate ? createResolvedGate(gate) : createResolvedGate({
      cardCode: "UNKNOWN",
      cardDefId: 0,
      imageKey: "unknown.png",
      name: "Unknown Gate",
      cardType: "GATE",
      attack: null,
      health: null,
      ikzCost: null,
      quantity: 1,
    }),
    garden: gardenCards,
    alley: alleyCards,
    ikzArea,
    handCount: 5,
    deckCount: 20,
    discardCount: 0,
    ikzPileCount: 5,
    hasIkzToken: true,
  };
}

// ============================================
// Scenario generators
// ============================================

function generateEarlyGameState(deckCards: DeckCard[]): GameState {
  const entities = findCardsByType(deckCards, "ENTITY");
  const spells = findCardsByType(deckCards, "SPELL");

  // My garden: 2 entities
  const myGarden: (ResolvedCard | null)[] = [
    entities[0] ? createResolvedCard(entities[0], 0) : null,
    null,
    entities[1] ? createResolvedCard(entities[1], 2, { tapped: true }) : null,
    null,
    null,
  ];

  // Opponent garden: 2 entities
  const oppGarden: (ResolvedCard | null)[] = [
    null,
    entities[2] ? createResolvedCard(entities[2], 1) : null,
    null,
    entities[3] ? createResolvedCard(entities[3], 3) : null,
    null,
  ];

  // My hand (entities and spells only, no IKZ cards)
  const handCards: ResolvedHandCard[] = [];
  if (entities[4]) handCards.push(createResolvedHandCard(entities[4]));
  if (spells[0]) handCards.push(createResolvedHandCard(spells[0]));
  if (entities[0]) handCards.push(createResolvedHandCard(entities[0]));

  return {
    phase: "MAIN",
    abilitySubphase: "",
    activePlayer: 0,
    turnNumber: 3,
    myBoard: createPlayerBoard(deckCards, myGarden, [null, null, null, null, null], 3, 1, 23),
    opponentBoard: createPlayerBoard(deckCards, oppGarden, [null, null, null, null, null], 2, 0, 25),
    myHand: handCards,
    actionMask: null,
    combatStack: [],
  };
}

function generateCombatState(deckCards: DeckCard[]): GameState {
  const baseState = generateEarlyGameState(deckCards);
  return {
    ...baseState,
    phase: "COMBAT_DECLARED",
    turnNumber: 5,
  };
}

function generateStatusEffectsState(deckCards: DeckCard[]): GameState {
  const entities = findCardsByType(deckCards, "ENTITY");
  const spells = findCardsByType(deckCards, "SPELL");

  // My garden: entities with status effects
  const myGarden: (ResolvedCard | null)[] = [
    entities[0] ? createResolvedCard(entities[0], 0, { isFrozen: true }) : null,
    entities[1] ? createResolvedCard(entities[1], 1, { isShocked: true, cooldown: true }) : null,
    null,
    null,
    null,
  ];

  // Opponent garden
  const oppGarden: (ResolvedCard | null)[] = [
    null,
    entities[2] ? createResolvedCard(entities[2], 1) : null,
    null,
    entities[3] ? createResolvedCard(entities[3], 3) : null,
    null,
  ];

  // My hand (entities and spells only, no IKZ cards)
  const handCards: ResolvedHandCard[] = [];
  if (entities[4]) handCards.push(createResolvedHandCard(entities[4]));
  if (spells[0]) handCards.push(createResolvedHandCard(spells[0]));
  if (entities[0]) handCards.push(createResolvedHandCard(entities[0]));

  return {
    phase: "MAIN",
    abilitySubphase: "",
    activePlayer: 0,
    turnNumber: 7,
    myBoard: createPlayerBoard(deckCards, myGarden, [null, null, null, null, null], 3, 1, 23),
    opponentBoard: createPlayerBoard(deckCards, oppGarden, [null, null, null, null, null], 2, 0, 25),
    myHand: handCards,
    actionMask: null,
    combatStack: [],
  };
}

function generateFullBoardState(deckCards: DeckCard[]): GameState {
  const entities = findCardsByType(deckCards, "ENTITY");
  const spells = findCardsByType(deckCards, "SPELL");

  // My garden: 4 entities
  const myGarden: (ResolvedCard | null)[] = [
    entities[0] ? createResolvedCard(entities[0], 0) : null,
    entities[1] ? createResolvedCard(entities[1], 1) : null,
    entities[2] ? createResolvedCard(entities[2], 2, { tapped: true }) : null,
    entities[3] ? createResolvedCard(entities[3], 3, { curHp: 1 }) : null,
    null,
  ];

  // My alley: 1 entity
  const myAlley: (ResolvedCard | null)[] = [
    entities[4] ? createResolvedCard(entities[4], 0) : null,
    null,
    null,
    null,
    null,
  ];

  // Opponent garden: 4 entities
  const oppGarden: (ResolvedCard | null)[] = [
    entities[0] ? createResolvedCard(entities[0], 0) : null,
    entities[1] ? createResolvedCard(entities[1], 1) : null,
    entities[2] ? createResolvedCard(entities[2], 2, { tapped: true }) : null,
    null,
    entities[3] ? createResolvedCard(entities[3], 4, { curHp: 3 }) : null,
  ];

  // My hand (entities and spells only, no IKZ cards)
  const handCards: ResolvedHandCard[] = [];
  if (entities[5]) handCards.push(createResolvedHandCard(entities[5]));
  if (spells[0]) handCards.push(createResolvedHandCard(spells[0]));
  if (spells[1]) handCards.push(createResolvedHandCard(spells[1]));

  return {
    phase: "MAIN",
    abilitySubphase: "",
    activePlayer: 0,
    turnNumber: 10,
    myBoard: createPlayerBoard(deckCards, myGarden, myAlley, 5, 2, 18),
    opponentBoard: createPlayerBoard(deckCards, oppGarden, [null, null, null, null, null], 4, 1, 20),
    myHand: handCards,
    actionMask: null,
    combatStack: [],
  };
}

// ============================================
// Scenario list for UI selector
// ============================================

export type MockScenario =
  | "early"
  | "combat"
  | "statusEffects"
  | "fullBoard";

export const mockScenarios: { id: MockScenario; name: string; description: string }[] = [
  {
    id: "early",
    name: "Early Game",
    description: "Turn 3 with a few entities on board",
  },
  {
    id: "combat",
    name: "Combat Phase",
    description: "Combat has been declared",
  },
  {
    id: "statusEffects",
    name: "Status Effects",
    description: "Cards with frozen, shocked, and cooldown",
  },
  {
    id: "fullBoard",
    name: "Full Board",
    description: "Both gardens filled with entities",
  },
];

export function getMockGameState(scenario: MockScenario, deckCards: DeckCard[]): GameState {
  switch (scenario) {
    case "early":
      return generateEarlyGameState(deckCards);
    case "combat":
      return generateCombatState(deckCards);
    case "statusEffects":
      return generateStatusEffectsState(deckCards);
    case "fullBoard":
      return generateFullBoardState(deckCards);
    default:
      return generateEarlyGameState(deckCards);
  }
}
