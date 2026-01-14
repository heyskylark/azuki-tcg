/**
 * Mock game state data for testing the 3D game renderer.
 * These represent different game scenarios that can be tested in isolation.
 */

import type { GameState, DeckCard } from "@/types/game";
import { buildImageUrl } from "@/types/game";

// ============================================
// Mock Deck Cards (for asset loading)
// ============================================

export const mockDeckCards: DeckCard[] = [
  // Leaders
  {
    cardCode: "STT01-001",
    imageKey: "stt01_001.png",
    name: "Bobu Leader",
    cardType: "LEADER",
    attack: 3,
    health: 25,
    ikzCost: null,
    quantity: 1,
  },
  {
    cardCode: "STT02-001",
    imageKey: "stt02_001.png",
    name: "Beanz Leader",
    cardType: "LEADER",
    attack: 2,
    health: 27,
    ikzCost: null,
    quantity: 1,
  },
  // Gates
  {
    cardCode: "STT01-002",
    imageKey: "stt01_002.png",
    name: "Bobu Gate",
    cardType: "GATE",
    attack: null,
    health: null,
    ikzCost: null,
    quantity: 1,
  },
  {
    cardCode: "STT02-002",
    imageKey: "stt02_002.png",
    name: "Beanz Gate",
    cardType: "GATE",
    attack: null,
    health: null,
    ikzCost: null,
    quantity: 1,
  },
  // Entities
  {
    cardCode: "STT01-003",
    imageKey: "stt01_003.png",
    name: "Red Bean Warrior",
    cardType: "ENTITY",
    attack: 3,
    health: 2,
    ikzCost: 2,
    quantity: 4,
  },
  {
    cardCode: "STT01-004",
    imageKey: "stt01_004.png",
    name: "Spirit Guardian",
    cardType: "ENTITY",
    attack: 2,
    health: 4,
    ikzCost: 3,
    quantity: 3,
  },
  {
    cardCode: "STT01-005",
    imageKey: "stt01_005.png",
    name: "Fire Elemental",
    cardType: "ENTITY",
    attack: 4,
    health: 3,
    ikzCost: 4,
    quantity: 2,
  },
  {
    cardCode: "STT02-003",
    imageKey: "stt02_003.png",
    name: "Beanz Scout",
    cardType: "ENTITY",
    attack: 2,
    health: 2,
    ikzCost: 1,
    quantity: 4,
  },
  {
    cardCode: "STT02-004",
    imageKey: "stt02_004.png",
    name: "Forest Protector",
    cardType: "ENTITY",
    attack: 1,
    health: 5,
    ikzCost: 2,
    quantity: 3,
  },
  // IKZ
  {
    cardCode: "IKZ-001",
    imageKey: "ikz_001.png",
    name: "Basic IKZ",
    cardType: "IKZ",
    attack: null,
    health: null,
    ikzCost: null,
    quantity: 5,
  },
  {
    cardCode: "IKZ-002",
    imageKey: "ikz_002.png",
    name: "Enhanced IKZ",
    cardType: "EXTRA_IKZ",
    attack: null,
    health: null,
    ikzCost: null,
    quantity: 3,
  },
  // Spells
  {
    cardCode: "STT01-010",
    imageKey: "stt01_010.png",
    name: "Flame Burst",
    cardType: "SPELL",
    attack: null,
    health: null,
    ikzCost: 2,
    quantity: 3,
  },
  {
    cardCode: "STT02-010",
    imageKey: "stt02_010.png",
    name: "Nature's Blessing",
    cardType: "SPELL",
    attack: null,
    health: null,
    ikzCost: 1,
    quantity: 3,
  },
];

/**
 * Build a CardMapping from DeckCard for use in GameStateContext.
 */
export function buildMockCardMappings(): Map<
  string,
  {
    cardCode: string;
    imageKey: string;
    imageUrl: string;
    name: string;
    cardType: string;
    attack: number | null;
    health: number | null;
    ikzCost: number | null;
  }
> {
  const mappings = new Map();
  for (const card of mockDeckCards) {
    mappings.set(card.cardCode, {
      cardCode: card.cardCode,
      imageKey: card.imageKey,
      imageUrl: buildImageUrl(card.imageKey),
      name: card.name,
      cardType: card.cardType,
      attack: card.attack,
      health: card.health,
      ikzCost: card.ikzCost,
    });
  }
  return mappings;
}

// ============================================
// Mock Game States
// ============================================

/**
 * Early game state - turn 3, some entities on board, both players have resources.
 */
export const mockEarlyGameState: GameState = {
  phase: "MAIN",
  abilitySubphase: "",
  activePlayer: 0,
  turnNumber: 3,

  myBoard: {
    leader: {
      cardCode: "STT01-001",
      cardDefId: 2,
      imageUrl: buildImageUrl("stt01_001.png"),
      name: "Bobu Leader",
      curAtk: 3,
      curHp: 23,
      tapped: false,
      cooldown: false,
    },
    gate: {
      cardCode: "STT01-002",
      cardDefId: 3,
      imageUrl: buildImageUrl("stt01_002.png"),
      name: "Bobu Gate",
      tapped: false,
      cooldown: false,
    },
    garden: [
      {
        cardCode: "STT01-003",
        cardDefId: 4,
        imageUrl: buildImageUrl("stt01_003.png"),
        name: "Red Bean Warrior",
        curAtk: 3,
        curHp: 2,
        tapped: false,
        cooldown: false,
        isFrozen: false,
        isShocked: false,
        isEffectImmune: false,
        zoneIndex: 0,
      },
      null,
      {
        cardCode: "STT01-004",
        cardDefId: 5,
        imageUrl: buildImageUrl("stt01_004.png"),
        name: "Spirit Guardian",
        curAtk: 2,
        curHp: 4,
        tapped: true, // Already attacked
        cooldown: false,
        isFrozen: false,
        isShocked: false,
        isEffectImmune: false,
        zoneIndex: 2,
      },
      null,
      null,
    ],
    alley: [null, null, null, null, null],
    ikzArea: [
      {
        cardCode: "IKZ-001",
        cardDefId: 0,
        imageUrl: buildImageUrl("ikz_001.png"),
        name: "Basic IKZ",
        tapped: true,
        cooldown: false,
      },
      {
        cardCode: "IKZ-001",
        cardDefId: 0,
        imageUrl: buildImageUrl("ikz_001.png"),
        name: "Basic IKZ",
        tapped: false,
        cooldown: false,
      },
      {
        cardCode: "IKZ-001",
        cardDefId: 0,
        imageUrl: buildImageUrl("ikz_001.png"),
        name: "Basic IKZ",
        tapped: false,
        cooldown: false,
      },
    ],
    handCount: 4,
    deckCount: 18,
    discardCount: 2,
    ikzPileCount: 2,
    hasIkzToken: true,
  },

  opponentBoard: {
    leader: {
      cardCode: "STT02-001",
      cardDefId: 19,
      imageUrl: buildImageUrl("stt02_001.png"),
      name: "Beanz Leader",
      curAtk: 2,
      curHp: 25,
      tapped: false,
      cooldown: false,
    },
    gate: {
      cardCode: "STT02-002",
      cardDefId: 20,
      imageUrl: buildImageUrl("stt02_002.png"),
      name: "Beanz Gate",
      tapped: false,
      cooldown: false,
    },
    garden: [
      null,
      {
        cardCode: "STT02-003",
        cardDefId: 21,
        imageUrl: buildImageUrl("stt02_003.png"),
        name: "Beanz Scout",
        curAtk: 2,
        curHp: 2,
        tapped: false,
        cooldown: false,
        isFrozen: false,
        isShocked: false,
        isEffectImmune: false,
        zoneIndex: 1,
      },
      null,
      {
        cardCode: "STT02-004",
        cardDefId: 22,
        imageUrl: buildImageUrl("stt02_004.png"),
        name: "Forest Protector",
        curAtk: 1,
        curHp: 5,
        tapped: false,
        cooldown: false,
        isFrozen: false,
        isShocked: false,
        isEffectImmune: false,
        zoneIndex: 3,
      },
      null,
    ],
    alley: [null, null, null, null, null],
    ikzArea: [
      {
        cardCode: "IKZ-001",
        cardDefId: 0,
        imageUrl: buildImageUrl("ikz_001.png"),
        name: "Basic IKZ",
        tapped: false,
        cooldown: false,
      },
      {
        cardCode: "IKZ-001",
        cardDefId: 0,
        imageUrl: buildImageUrl("ikz_001.png"),
        name: "Basic IKZ",
        tapped: false,
        cooldown: false,
      },
    ],
    handCount: 5,
    deckCount: 17,
    discardCount: 1,
    ikzPileCount: 3,
    hasIkzToken: true,
  },

  myHand: [
    {
      cardCode: "STT01-005",
      cardDefId: 6,
      imageUrl: buildImageUrl("stt01_005.png"),
      name: "Fire Elemental",
      type: "ENTITY",
      ikzCost: 4,
    },
    {
      cardCode: "STT01-010",
      cardDefId: 11,
      imageUrl: buildImageUrl("stt01_010.png"),
      name: "Flame Burst",
      type: "SPELL",
      ikzCost: 2,
    },
    {
      cardCode: "STT01-003",
      cardDefId: 4,
      imageUrl: buildImageUrl("stt01_003.png"),
      name: "Red Bean Warrior",
      type: "ENTITY",
      ikzCost: 2,
    },
    {
      cardCode: "IKZ-001",
      cardDefId: 0,
      imageUrl: buildImageUrl("ikz_001.png"),
      name: "Basic IKZ",
      type: "IKZ",
      ikzCost: 0,
    },
  ],

  actionMask: null,
  combatStack: [],
};

/**
 * Combat phase state - entities are attacking.
 */
export const mockCombatState: GameState = {
  ...mockEarlyGameState,
  phase: "COMBAT_DECLARED",
  turnNumber: 5,
};

/**
 * State with status effects - frozen and shocked entities.
 */
export const mockStatusEffectsState: GameState = {
  ...mockEarlyGameState,
  phase: "MAIN",
  turnNumber: 7,
  myBoard: {
    ...mockEarlyGameState.myBoard,
    garden: [
      {
        cardCode: "STT01-003",
        cardDefId: 4,
        imageUrl: buildImageUrl("stt01_003.png"),
        name: "Red Bean Warrior",
        curAtk: 3,
        curHp: 2,
        tapped: false,
        cooldown: false,
        isFrozen: true, // Frozen!
        isShocked: false,
        isEffectImmune: false,
        zoneIndex: 0,
      },
      {
        cardCode: "STT01-004",
        cardDefId: 5,
        imageUrl: buildImageUrl("stt01_004.png"),
        name: "Spirit Guardian",
        curAtk: 2,
        curHp: 3,
        tapped: false,
        cooldown: true, // On cooldown
        isFrozen: false,
        isShocked: true, // Shocked!
        isEffectImmune: false,
        zoneIndex: 1,
      },
      null,
      null,
      null,
    ],
  },
};

/**
 * Full board state - both gardens have multiple entities.
 */
export const mockFullBoardState: GameState = {
  ...mockEarlyGameState,
  phase: "MAIN",
  turnNumber: 10,
  myBoard: {
    ...mockEarlyGameState.myBoard,
    garden: [
      {
        cardCode: "STT01-003",
        cardDefId: 4,
        imageUrl: buildImageUrl("stt01_003.png"),
        name: "Red Bean Warrior",
        curAtk: 3,
        curHp: 2,
        tapped: false,
        cooldown: false,
        isFrozen: false,
        isShocked: false,
        isEffectImmune: false,
        zoneIndex: 0,
      },
      {
        cardCode: "STT01-004",
        cardDefId: 5,
        imageUrl: buildImageUrl("stt01_004.png"),
        name: "Spirit Guardian",
        curAtk: 2,
        curHp: 4,
        tapped: false,
        cooldown: false,
        isFrozen: false,
        isShocked: false,
        isEffectImmune: false,
        zoneIndex: 1,
      },
      {
        cardCode: "STT01-005",
        cardDefId: 6,
        imageUrl: buildImageUrl("stt01_005.png"),
        name: "Fire Elemental",
        curAtk: 4,
        curHp: 3,
        tapped: true,
        cooldown: false,
        isFrozen: false,
        isShocked: false,
        isEffectImmune: false,
        zoneIndex: 2,
      },
      {
        cardCode: "STT01-003",
        cardDefId: 4,
        imageUrl: buildImageUrl("stt01_003.png"),
        name: "Red Bean Warrior",
        curAtk: 3,
        curHp: 1,
        tapped: false,
        cooldown: false,
        isFrozen: false,
        isShocked: false,
        isEffectImmune: false,
        zoneIndex: 3,
      },
      null,
    ],
    alley: [
      {
        cardCode: "STT01-004",
        cardDefId: 5,
        imageUrl: buildImageUrl("stt01_004.png"),
        name: "Spirit Guardian",
        curAtk: 2,
        curHp: 4,
        tapped: false,
        cooldown: false,
        isFrozen: false,
        isShocked: false,
        isEffectImmune: false,
        zoneIndex: 0,
      },
      null,
      null,
      null,
      null,
    ],
  },
  opponentBoard: {
    ...mockEarlyGameState.opponentBoard,
    garden: [
      {
        cardCode: "STT02-003",
        cardDefId: 21,
        imageUrl: buildImageUrl("stt02_003.png"),
        name: "Beanz Scout",
        curAtk: 2,
        curHp: 2,
        tapped: false,
        cooldown: false,
        isFrozen: false,
        isShocked: false,
        isEffectImmune: false,
        zoneIndex: 0,
      },
      {
        cardCode: "STT02-004",
        cardDefId: 22,
        imageUrl: buildImageUrl("stt02_004.png"),
        name: "Forest Protector",
        curAtk: 1,
        curHp: 5,
        tapped: false,
        cooldown: false,
        isFrozen: false,
        isShocked: false,
        isEffectImmune: false,
        zoneIndex: 1,
      },
      {
        cardCode: "STT02-003",
        cardDefId: 21,
        imageUrl: buildImageUrl("stt02_003.png"),
        name: "Beanz Scout",
        curAtk: 2,
        curHp: 2,
        tapped: true,
        cooldown: false,
        isFrozen: false,
        isShocked: false,
        isEffectImmune: false,
        zoneIndex: 2,
      },
      null,
      {
        cardCode: "STT02-004",
        cardDefId: 22,
        imageUrl: buildImageUrl("stt02_004.png"),
        name: "Forest Protector",
        curAtk: 1,
        curHp: 3,
        tapped: false,
        cooldown: false,
        isFrozen: false,
        isShocked: false,
        isEffectImmune: false,
        zoneIndex: 4,
      },
    ],
  },
};

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

export function getMockGameState(scenario: MockScenario): GameState {
  switch (scenario) {
    case "early":
      return mockEarlyGameState;
    case "combat":
      return mockCombatState;
    case "statusEffects":
      return mockStatusEffectsState;
    case "fullBoard":
      return mockFullBoardState;
    default:
      return mockEarlyGameState;
  }
}
