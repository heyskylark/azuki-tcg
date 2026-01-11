export interface StarterCardInfo {
  cardCode: string;
  quantity: number;
}

export interface StarterDeckConfig {
  name: string;
  cards: StarterCardInfo[];
}

// Shao Starter deck configuration (from world.c shaoDeckCardInfo)
const shaoStarterCards: StarterCardInfo[] = [
  { cardCode: "STT02-001", quantity: 1 },
  { cardCode: "STT02-002", quantity: 1 },
  { cardCode: "STT02-003", quantity: 4 },
  { cardCode: "STT02-004", quantity: 4 },
  { cardCode: "STT02-005", quantity: 4 },
  { cardCode: "STT02-006", quantity: 4 },
  { cardCode: "STT02-007", quantity: 4 },
  { cardCode: "STT02-008", quantity: 4 },
  { cardCode: "STT02-009", quantity: 4 },
  { cardCode: "STT02-010", quantity: 2 },
  { cardCode: "STT02-011", quantity: 4 },
  { cardCode: "STT02-012", quantity: 4 },
  { cardCode: "STT02-013", quantity: 2 },
  { cardCode: "STT02-014", quantity: 2 },
  { cardCode: "STT02-015", quantity: 4 },
  { cardCode: "STT02-016", quantity: 2 },
  { cardCode: "STT02-017", quantity: 2 },
  { cardCode: "IKZ-001", quantity: 10 },
];

// Raizan Starter deck configuration (from world.c raizenDeckCardInfo)
const raizanStarterCards: StarterCardInfo[] = [
  { cardCode: "STT01-001", quantity: 1 },
  { cardCode: "STT01-002", quantity: 1 },
  { cardCode: "STT01-003", quantity: 4 },
  { cardCode: "STT01-004", quantity: 4 },
  { cardCode: "STT01-005", quantity: 4 },
  { cardCode: "STT01-006", quantity: 2 },
  { cardCode: "STT01-007", quantity: 4 },
  { cardCode: "STT01-008", quantity: 4 },
  { cardCode: "STT01-009", quantity: 4 },
  { cardCode: "STT01-010", quantity: 2 },
  { cardCode: "STT01-011", quantity: 2 },
  { cardCode: "STT01-012", quantity: 4 },
  { cardCode: "STT01-013", quantity: 4 },
  { cardCode: "STT01-014", quantity: 4 },
  { cardCode: "STT01-015", quantity: 2 },
  { cardCode: "STT01-016", quantity: 2 },
  { cardCode: "STT01-017", quantity: 4 },
  { cardCode: "IKZ-001", quantity: 10 },
];

export const starterDecks: StarterDeckConfig[] = [
  { name: "Shao Starter", cards: shaoStarterCards },
  { name: "Raizan Starter", cards: raizanStarterCards },
];
