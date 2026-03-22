import { and, eq, inArray, isNull } from "drizzle-orm";
import db, { type IDatabase, type ITransaction } from "@core/database";
import { Cards, Decks, DeckCardJunctions } from "@core/drizzle/schemas";
import { CardRarity, CardType, RarityOrdering } from "@core/types/cards";

type Database = IDatabase | ITransaction;

/**
 * CardDefId values matching include/generated/card_defs.h
 * These must stay in sync with the C engine card definitions.
 */
export enum CardDefId {
  IKZ_001 = 0,
  IKZ_002 = 1,
  STT01_001 = 2,
  STT01_002 = 3,
  STT01_003 = 4,
  STT01_004 = 5,
  STT01_005 = 6,
  STT01_006 = 7,
  STT01_007 = 8,
  STT01_008 = 9,
  STT01_009 = 10,
  STT01_010 = 11,
  STT01_011 = 12,
  STT01_012 = 13,
  STT01_013 = 14,
  STT01_014 = 15,
  STT01_015 = 16,
  STT01_016 = 17,
  STT01_017 = 18,
  STT02_001 = 19,
  STT02_002 = 20,
  STT02_003 = 21,
  STT02_004 = 22,
  STT02_005 = 23,
  STT02_006 = 24,
  STT02_007 = 25,
  STT02_008 = 26,
  STT02_009 = 27,
  STT02_010 = 28,
  STT02_011 = 29,
  STT02_012 = 30,
  STT02_013 = 31,
  STT02_014 = 32,
  STT02_015 = 33,
  STT02_016 = 34,
  STT02_017 = 35,
  AZK01_001 = 36,
  AZK01_002 = 37,
  AZK01_003 = 38,
  AZK01_004 = 39,
  AZK01_005 = 40,
  AZK01_006 = 41,
  AZK01_007 = 42,
  AZK01_008 = 43,
  AZK01_009 = 44,
  AZK01_010 = 45,
  AZK01_011 = 46,
  AZK01_012 = 47,
  AZK01_014 = 48,
  AZK01_015 = 49,
  AZK01_016 = 50,
  AZK01_017 = 51,
  AZK01_018 = 52,
  AZK01_019 = 53,
  AZK01_020 = 54,
  AZK01_021 = 55,
  AZK01_022 = 56,
  AZK01_023 = 57,
  AZK01_024 = 58,
  AZK01_025 = 59,
  AZK01_026 = 60,
  AZK01_027 = 61,
  AZK01_028 = 62,
  AZK01_029 = 63,
  AZK01_030 = 64,
  AZK01_031 = 65,
  AZK01_032 = 66,
  AZK01_033 = 67,
  AZK01_034 = 68,
  AZK01_035 = 69,
  AZK01_036 = 70,
  AZK01_037 = 71,
  AZK01_038 = 72,
  AZK01_039 = 73,
  AZK01_040 = 74,
  AZK01_041 = 75,
  AZK01_042 = 76,
  AZK01_043 = 77,
  AZK01_044 = 78,
  AZK01_045 = 79,
  AZK01_046 = 80,
  AZK01_047 = 81,
  AZK01_048 = 82,
  AZK01_049 = 83,
  AZK01_050 = 84,
  AZK01_051 = 85,
  AZK01_052 = 86,
  AZK01_053 = 87,
  AZK01_054 = 88,
  AZK01_055 = 89,
  AZK01_056 = 90,
  AZK01_057 = 91,
  AZK01_058 = 92,
  AZK01_059 = 93,
  AZK01_060 = 94,
  AZK01_061 = 95,
  AZK01_062 = 96,
  AZK01_063 = 97,
  AZK01_064 = 98,
  AZK01_065 = 99,
  AZK01_066 = 100,
  AZK01_067 = 101,
  AZK01_068 = 102,
  AZK01_069 = 103,
  AZK01_070 = 104,
  AZK01_071 = 105,
  AZK01_072 = 106,
  AZK01_073 = 107,
  AZK01_074 = 108,
  AZK01_075 = 109,
  AZK01_077 = 110,
  AZK01_078 = 111,
  AZK01_080 = 112,
  AZK01_081 = 113,
  AZK01_082 = 114,
  AZK01_084 = 115,
  AZK01_085 = 116,
  AZK01_086 = 117,
  AZK01_087 = 118,
  AZK01_088 = 119,
  AZK01_089 = 120,
  AZK01_090 = 121,
  AZK01_091 = 122,
  AZK01_092 = 123,
  AZK01_093 = 124,
  AZK01_094 = 125,
  AZK01_095 = 126,
  AZK01_096 = 127,
  AZK01_097 = 128,
  AZK01_098 = 129,
  AZK01_100 = 130,
  AZK01_101 = 131,
  AZK01_102 = 132,
  AZK01_103 = 133,
  AZK01_104 = 134,
  AZK01_105 = 135,
  AZK01_106 = 136,
  AZK01_107 = 137,
  AZK01_108 = 138,
  AZK01_109 = 139,
  AZK01_110 = 140,
  AZK01_111 = 141,
  AZK01_112 = 142,
  AZK01_113 = 143,
  AZK01_114 = 144,
  AZK01_115 = 145,
  AZK01_116 = 146,
  AZK01_117 = 147,
  AZK01_118 = 148,
  AZK01_119 = 149,
  AZK01_120 = 150,
  AZK01_121 = 151,
  AZK01_122 = 152,
  AZK01_123 = 153,
  AZK01_124 = 154,
  AZK01_125 = 155,
  AZK01_126 = 156,
  AZK01_127 = 157,
  AZK01_128 = 158,
  AZK01_129 = 159,
  STT03_001 = 160,
  STT03_002 = 161,
  STT03_003 = 162,
  STT03_004 = 163,
  STT03_005 = 164,
  STT03_006 = 165,
  STT03_007 = 166,
  STT03_008 = 167,
  STT03_009 = 168,
  STT03_010 = 169,
  STT03_011 = 170,
  STT03_012 = 171,
  STT03_013 = 172,
  STT03_014 = 173,
  STT03_015 = 174,
  STT03_016 = 175,
  STT04_001 = 176,
  STT04_002 = 177,
  STT04_003 = 178,
  STT04_004 = 179,
  STT04_005 = 180,
  STT04_006 = 181,
  STT04_007 = 182,
  STT04_008 = 183,
  STT04_009 = 184,
  STT04_010 = 185,
  STT04_011 = 186,
  STT04_012 = 187,
  STT04_013 = 188,
  STT04_014 = 189,
  STT04_015 = 190,
  STT04_016 = 191,
  STT04_017 = 192,
}

/**
 * Mapping from database cardCode to C engine CardDefId.
 * Generated from include/generated/card_defs.h CardDefId enum.
 */
const CARD_CODE_TO_DEF_ID: Record<string, CardDefId> = {
  "IKZ-001": CardDefId.IKZ_001,
  "IKZ-002": CardDefId.IKZ_002,
  "STT01-001": CardDefId.STT01_001,
  "STT01-002": CardDefId.STT01_002,
  "STT01-003": CardDefId.STT01_003,
  "STT01-004": CardDefId.STT01_004,
  "STT01-005": CardDefId.STT01_005,
  "STT01-006": CardDefId.STT01_006,
  "STT01-007": CardDefId.STT01_007,
  "STT01-008": CardDefId.STT01_008,
  "STT01-009": CardDefId.STT01_009,
  "STT01-010": CardDefId.STT01_010,
  "STT01-011": CardDefId.STT01_011,
  "STT01-012": CardDefId.STT01_012,
  "STT01-013": CardDefId.STT01_013,
  "STT01-014": CardDefId.STT01_014,
  "STT01-015": CardDefId.STT01_015,
  "STT01-016": CardDefId.STT01_016,
  "STT01-017": CardDefId.STT01_017,
  "STT02-001": CardDefId.STT02_001,
  "STT02-002": CardDefId.STT02_002,
  "STT02-003": CardDefId.STT02_003,
  "STT02-004": CardDefId.STT02_004,
  "STT02-005": CardDefId.STT02_005,
  "STT02-006": CardDefId.STT02_006,
  "STT02-007": CardDefId.STT02_007,
  "STT02-008": CardDefId.STT02_008,
  "STT02-009": CardDefId.STT02_009,
  "STT02-010": CardDefId.STT02_010,
  "STT02-011": CardDefId.STT02_011,
  "STT02-012": CardDefId.STT02_012,
  "STT02-013": CardDefId.STT02_013,
  "STT02-014": CardDefId.STT02_014,
  "STT02-015": CardDefId.STT02_015,
  "STT02-016": CardDefId.STT02_016,
  "STT02-017": CardDefId.STT02_017,
  "AZK01-001": CardDefId.AZK01_001,
  "AZK01-002": CardDefId.AZK01_002,
  "AZK01-003": CardDefId.AZK01_003,
  "AZK01-004": CardDefId.AZK01_004,
  "AZK01-005": CardDefId.AZK01_005,
  "AZK01-006": CardDefId.AZK01_006,
  "AZK01-007": CardDefId.AZK01_007,
  "AZK01-008": CardDefId.AZK01_008,
  "AZK01-009": CardDefId.AZK01_009,
  "AZK01-010": CardDefId.AZK01_010,
  "AZK01-011": CardDefId.AZK01_011,
  "AZK01-012": CardDefId.AZK01_012,
  "AZK01-014": CardDefId.AZK01_014,
  "AZK01-015": CardDefId.AZK01_015,
  "AZK01-016": CardDefId.AZK01_016,
  "AZK01-017": CardDefId.AZK01_017,
  "AZK01-018": CardDefId.AZK01_018,
  "AZK01-019": CardDefId.AZK01_019,
  "AZK01-020": CardDefId.AZK01_020,
  "AZK01-021": CardDefId.AZK01_021,
  "AZK01-022": CardDefId.AZK01_022,
  "AZK01-023": CardDefId.AZK01_023,
  "AZK01-024": CardDefId.AZK01_024,
  "AZK01-025": CardDefId.AZK01_025,
  "AZK01-026": CardDefId.AZK01_026,
  "AZK01-027": CardDefId.AZK01_027,
  "AZK01-028": CardDefId.AZK01_028,
  "AZK01-029": CardDefId.AZK01_029,
  "AZK01-030": CardDefId.AZK01_030,
  "AZK01-031": CardDefId.AZK01_031,
  "AZK01-032": CardDefId.AZK01_032,
  "AZK01-033": CardDefId.AZK01_033,
  "AZK01-034": CardDefId.AZK01_034,
  "AZK01-035": CardDefId.AZK01_035,
  "AZK01-036": CardDefId.AZK01_036,
  "AZK01-037": CardDefId.AZK01_037,
  "AZK01-038": CardDefId.AZK01_038,
  "AZK01-039": CardDefId.AZK01_039,
  "AZK01-040": CardDefId.AZK01_040,
  "AZK01-041": CardDefId.AZK01_041,
  "AZK01-042": CardDefId.AZK01_042,
  "AZK01-043": CardDefId.AZK01_043,
  "AZK01-044": CardDefId.AZK01_044,
  "AZK01-045": CardDefId.AZK01_045,
  "AZK01-046": CardDefId.AZK01_046,
  "AZK01-047": CardDefId.AZK01_047,
  "AZK01-048": CardDefId.AZK01_048,
  "AZK01-049": CardDefId.AZK01_049,
  "AZK01-050": CardDefId.AZK01_050,
  "AZK01-051": CardDefId.AZK01_051,
  "AZK01-052": CardDefId.AZK01_052,
  "AZK01-053": CardDefId.AZK01_053,
  "AZK01-054": CardDefId.AZK01_054,
  "AZK01-055": CardDefId.AZK01_055,
  "AZK01-056": CardDefId.AZK01_056,
  "AZK01-057": CardDefId.AZK01_057,
  "AZK01-058": CardDefId.AZK01_058,
  "AZK01-059": CardDefId.AZK01_059,
  "AZK01-060": CardDefId.AZK01_060,
  "AZK01-061": CardDefId.AZK01_061,
  "AZK01-062": CardDefId.AZK01_062,
  "AZK01-063": CardDefId.AZK01_063,
  "AZK01-064": CardDefId.AZK01_064,
  "AZK01-065": CardDefId.AZK01_065,
  "AZK01-066": CardDefId.AZK01_066,
  "AZK01-067": CardDefId.AZK01_067,
  "AZK01-068": CardDefId.AZK01_068,
  "AZK01-069": CardDefId.AZK01_069,
  "AZK01-070": CardDefId.AZK01_070,
  "AZK01-071": CardDefId.AZK01_071,
  "AZK01-072": CardDefId.AZK01_072,
  "AZK01-073": CardDefId.AZK01_073,
  "AZK01-074": CardDefId.AZK01_074,
  "AZK01-075": CardDefId.AZK01_075,
  "AZK01-077": CardDefId.AZK01_077,
  "AZK01-078": CardDefId.AZK01_078,
  "AZK01-080": CardDefId.AZK01_080,
  "AZK01-081": CardDefId.AZK01_081,
  "AZK01-082": CardDefId.AZK01_082,
  "AZK01-084": CardDefId.AZK01_084,
  "AZK01-085": CardDefId.AZK01_085,
  "AZK01-086": CardDefId.AZK01_086,
  "AZK01-087": CardDefId.AZK01_087,
  "AZK01-088": CardDefId.AZK01_088,
  "AZK01-089": CardDefId.AZK01_089,
  "AZK01-090": CardDefId.AZK01_090,
  "AZK01-091": CardDefId.AZK01_091,
  "AZK01-092": CardDefId.AZK01_092,
  "AZK01-093": CardDefId.AZK01_093,
  "AZK01-094": CardDefId.AZK01_094,
  "AZK01-095": CardDefId.AZK01_095,
  "AZK01-096": CardDefId.AZK01_096,
  "AZK01-097": CardDefId.AZK01_097,
  "AZK01-098": CardDefId.AZK01_098,
  "AZK01-100": CardDefId.AZK01_100,
  "AZK01-101": CardDefId.AZK01_101,
  "AZK01-102": CardDefId.AZK01_102,
  "AZK01-103": CardDefId.AZK01_103,
  "AZK01-104": CardDefId.AZK01_104,
  "AZK01-105": CardDefId.AZK01_105,
  "AZK01-106": CardDefId.AZK01_106,
  "AZK01-107": CardDefId.AZK01_107,
  "AZK01-108": CardDefId.AZK01_108,
  "AZK01-109": CardDefId.AZK01_109,
  "AZK01-110": CardDefId.AZK01_110,
  "AZK01-111": CardDefId.AZK01_111,
  "AZK01-112": CardDefId.AZK01_112,
  "AZK01-113": CardDefId.AZK01_113,
  "AZK01-114": CardDefId.AZK01_114,
  "AZK01-115": CardDefId.AZK01_115,
  "AZK01-116": CardDefId.AZK01_116,
  "AZK01-117": CardDefId.AZK01_117,
  "AZK01-118": CardDefId.AZK01_118,
  "AZK01-119": CardDefId.AZK01_119,
  "AZK01-120": CardDefId.AZK01_120,
  "AZK01-121": CardDefId.AZK01_121,
  "AZK01-122": CardDefId.AZK01_122,
  "AZK01-123": CardDefId.AZK01_123,
  "AZK01-124": CardDefId.AZK01_124,
  "AZK01-125": CardDefId.AZK01_125,
  "AZK01-126": CardDefId.AZK01_126,
  "AZK01-127": CardDefId.AZK01_127,
  "AZK01-128": CardDefId.AZK01_128,
  "AZK01-129": CardDefId.AZK01_129,
  "STT03-001": CardDefId.STT03_001,
  "STT03-002": CardDefId.STT03_002,
  "STT03-003": CardDefId.STT03_003,
  "STT03-004": CardDefId.STT03_004,
  "STT03-005": CardDefId.STT03_005,
  "STT03-006": CardDefId.STT03_006,
  "STT03-007": CardDefId.STT03_007,
  "STT03-008": CardDefId.STT03_008,
  "STT03-009": CardDefId.STT03_009,
  "STT03-010": CardDefId.STT03_010,
  "STT03-011": CardDefId.STT03_011,
  "STT03-012": CardDefId.STT03_012,
  "STT03-013": CardDefId.STT03_013,
  "STT03-014": CardDefId.STT03_014,
  "STT03-015": CardDefId.STT03_015,
  "STT03-016": CardDefId.STT03_016,
  "STT04-001": CardDefId.STT04_001,
  "STT04-002": CardDefId.STT04_002,
  "STT04-003": CardDefId.STT04_003,
  "STT04-004": CardDefId.STT04_004,
  "STT04-005": CardDefId.STT04_005,
  "STT04-006": CardDefId.STT04_006,
  "STT04-007": CardDefId.STT04_007,
  "STT04-008": CardDefId.STT04_008,
  "STT04-009": CardDefId.STT04_009,
  "STT04-010": CardDefId.STT04_010,
  "STT04-011": CardDefId.STT04_011,
  "STT04-012": CardDefId.STT04_012,
  "STT04-013": CardDefId.STT04_013,
  "STT04-014": CardDefId.STT04_014,
  "STT04-015": CardDefId.STT04_015,
  "STT04-016": CardDefId.STT04_016,
  "STT04-017": CardDefId.STT04_017,
};

/**
 * Reverse mapping from CardDefId to database cardCode.
 */
const DEF_ID_TO_CARD_CODE: Record<CardDefId, string> = Object.fromEntries(
  Object.entries(CARD_CODE_TO_DEF_ID).map(([code, id]) => [id, code])
) as Record<CardDefId, string>;

/**
 * Convert a database cardCode (e.g., "STT01-001") to C engine CardDefId.
 * Returns null if the cardCode is not recognized.
 */
export function cardCodeToDefId(cardCode: string): CardDefId | null {
  return CARD_CODE_TO_DEF_ID[cardCode] ?? null;
}

/**
 * Convert a C engine CardDefId to database cardCode.
 * Returns null if the defId is not recognized.
 */
export function defIdToCardCode(defId: CardDefId): string | null {
  return DEF_ID_TO_CARD_CODE[defId] ?? null;
}

/**
 * Deck structure with CardDefIds ready for the C engine.
 */
export interface DeckAsDefIds {
  leader: CardDefId;
  gate: CardDefId;
  mainDeck: CardDefId[]; // Flattened with quantities (e.g., 4 copies = 4 entries)
  ikzPile: CardDefId[]; // Flattened with quantities
}

interface CardWithType {
  id: string;
  cardCode: string;
  cardType: CardType;
  rarity: CardRarity;
  quantity: number;
}

/**
 * Select the least rare card for each cardCode.
 * Excludes cards with specialRarity set (non-null).
 */
function selectLeastRareCards(
  cards: Array<{ id: string; cardCode: string; rarity: CardRarity }>
): Map<string, string> {
  const cardsByCode = new Map<
    string,
    Array<{ id: string; cardCode: string; rarity: CardRarity }>
  >();
  for (const card of cards) {
    const existing = cardsByCode.get(card.cardCode) ?? [];
    existing.push(card);
    cardsByCode.set(card.cardCode, existing);
  }

  const cardCodeToId = new Map<string, string>();
  for (const [cardCode, variants] of cardsByCode) {
    variants.sort((a, b) => RarityOrdering[a.rarity] - RarityOrdering[b.rarity]);
    const leastRare = variants[0];
    if (leastRare) {
      cardCodeToId.set(cardCode, leastRare.id);
    }
  }

  return cardCodeToId;
}

/**
 * Load a deck from the database and convert all cards to CardDefIds.
 * Separates cards into leader, gate, main deck, and IKZ pile based on CardType.
 *
 * @throws Error if deck not found or if any card code cannot be mapped to CardDefId
 */
export async function loadDeckAsDefIds(
  deckId: string,
  database: Database = db
): Promise<DeckAsDefIds> {
  // Fetch deck with all card junctions
  const deckExists = await database
    .select({ id: Decks.id })
    .from(Decks)
    .where(eq(Decks.id, deckId))
    .limit(1)
    .then((rows) => rows[0]);

  if (!deckExists) {
    throw new Error(`Deck not found: ${deckId}`);
  }

  // Get all cards in the deck with their types
  const deckCards = await database
    .select({
      cardId: DeckCardJunctions.cardId,
      quantity: DeckCardJunctions.quantity,
      cardCode: Cards.cardCode,
      cardType: Cards.cardType,
      rarity: Cards.rarity,
    })
    .from(DeckCardJunctions)
    .innerJoin(Cards, eq(DeckCardJunctions.cardId, Cards.id))
    .where(eq(DeckCardJunctions.deckId, deckId));

  if (deckCards.length === 0) {
    throw new Error(`Deck has no cards: ${deckId}`);
  }

  let leader: CardDefId | null = null;
  let gate: CardDefId | null = null;
  const mainDeck: CardDefId[] = [];
  const ikzPile: CardDefId[] = [];

  for (const card of deckCards) {
    const defId = cardCodeToDefId(card.cardCode);
    if (defId === null) {
      throw new Error(`Unknown cardCode: ${card.cardCode} - not found in CardDefId mapping`);
    }

    switch (card.cardType) {
      case CardType.LEADER:
        if (leader !== null) {
          throw new Error(`Deck has multiple leaders: ${deckId}`);
        }
        leader = defId;
        break;

      case CardType.GATE:
        if (gate !== null) {
          throw new Error(`Deck has multiple gates: ${deckId}`);
        }
        gate = defId;
        break;

      case CardType.IKZ:
      case CardType.EXTRA_IKZ:
        // Flatten by quantity
        for (let i = 0; i < card.quantity; i++) {
          ikzPile.push(defId);
        }
        break;

      case CardType.ENTITY:
      case CardType.WEAPON:
      case CardType.SPELL:
        // Flatten by quantity
        for (let i = 0; i < card.quantity; i++) {
          mainDeck.push(defId);
        }
        break;

      default:
        throw new Error(`Unknown card type: ${card.cardType}`);
    }
  }

  if (leader === null) {
    throw new Error(`Deck missing leader: ${deckId}`);
  }

  if (gate === null) {
    throw new Error(`Deck missing gate: ${deckId}`);
  }

  return {
    leader,
    gate,
    mainDeck,
    ikzPile,
  };
}

/**
 * Flatten a DeckAsDefIds structure into a single array suitable for engine initialization.
 * Format: [leader, gate, ...mainDeck, ...ikzPile]
 */
export function flattenDeckForEngine(deck: DeckAsDefIds): number[] {
  return [deck.leader, deck.gate, ...deck.mainDeck, ...deck.ikzPile];
}

/**
 * Entry format for native module deck initialization.
 */
export interface DeckCardEntry {
  cardId: number;
  count: number;
}

/**
 * Convert a DeckAsDefIds structure into DeckCardEntry[] for native module.
 * Groups duplicate CardDefIds and sets their counts.
 * Includes leader, gate, main deck, and IKZ pile cards.
 */
export function deckAsDefIdsToDeckEntries(deck: DeckAsDefIds): DeckCardEntry[] {
  const entries: DeckCardEntry[] = [];
  const cardCounts = new Map<number, number>();

  // Add leader (always count 1)
  entries.push({ cardId: deck.leader, count: 1 });

  // Add gate (always count 1)
  entries.push({ cardId: deck.gate, count: 1 });

  // Count main deck cards (already flattened, so we need to aggregate)
  for (const cardId of deck.mainDeck) {
    cardCounts.set(cardId, (cardCounts.get(cardId) ?? 0) + 1);
  }

  // Count IKZ pile cards
  for (const cardId of deck.ikzPile) {
    cardCounts.set(cardId, (cardCounts.get(cardId) ?? 0) + 1);
  }

  // Add aggregated counts
  for (const [cardId, count] of cardCounts) {
    entries.push({ cardId, count });
  }

  return entries;
}
