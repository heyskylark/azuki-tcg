// Card rarity (matches C engine)
export enum CardRarity {
  L = "L",
  L_S = "L_S",
  G = "G",
  G_S = "G_S",
  C = "C",
  UC = "UC",
  R = "R",
  SR = "SR",
  SR_S = "SR_S",
  SR_SS = "SR_SS",
  IKZ = "IKZ",
  IKZ_S = "IKZ_S",
}

export const RarityOrdering: Record<CardRarity, number> = {
  [CardRarity.C]: 1,
  [CardRarity.UC]: 2,
  [CardRarity.R]: 3,
  [CardRarity.SR]: 4,
  [CardRarity.SR_S]: 5,
  [CardRarity.SR_SS]: 6,
  [CardRarity.IKZ]: 7,
  [CardRarity.IKZ_S]: 8,
  [CardRarity.L]: 9,
  [CardRarity.L_S]: 10,
  [CardRarity.G]: 11,
  [CardRarity.G_S]: 12,
} as const;

export enum SpecialCardRarity {
  INV26_WINNER = "INV26_WINNER",
  INV26_SECOND = "INV26_SECOND",
  INV26_TOP8 = "INV26_TOP8",
  TOKEN_AX_WINNER = "TOKEN_AX_WINNER",
}

// Card element (matches C engine)
export enum CardElement {
  NORMAL = "NORMAL",
  LIGHTNING = "LIGHTNING",
  WATER = "WATER",
  EARTH = "EARTH",
  FIRE = "FIRE",
}

// Card type (matches C engine)
export enum CardType {
  LEADER = "LEADER",
  GATE = "GATE",
  ENTITY = "ENTITY",
  WEAPON = "WEAPON",
  SPELL = "SPELL",
  IKZ = "IKZ",
  EXTRA_IKZ = "EXTRA_IKZ",
}
