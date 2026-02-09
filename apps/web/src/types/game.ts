/**
 * Game state types for the 3D card game renderer.
 * These types transform WebSocket snapshot data into a format suitable for rendering.
 */

import type { SnapshotActionMask } from "@tcg/backend-core/types/ws";

/**
 * Resolved card with image URL and display data.
 * Created by transforming snapshot cards with deck card mappings.
 */
export interface ResolvedCard {
  cardCode: string; // From snapshot cardId field (e.g., "STT01-001")
  cardDefId: number;
  imageUrl: string; // Full URL: https://azuki-tcg.s3.us-east-1.amazonaws.com/{imageKey}
  name: string;
  curAtk: number | null;
  curHp: number | null;
  tapped: boolean;
  cooldown: boolean;
  isFrozen: boolean;
  isShocked: boolean;
  isEffectImmune: boolean;
  hasCharge: boolean;
  hasDefender: boolean;
  hasInfiltrate: boolean;
  zoneIndex: number;
}

/**
 * Leader card with resolved image data.
 */
export interface ResolvedLeader {
  cardCode: string;
  cardDefId: number;
  imageUrl: string;
  name: string;
  curAtk: number;
  curHp: number;
  tapped: boolean;
  cooldown: boolean;
  isFrozen: boolean;
  isShocked: boolean;
  isEffectImmune: boolean;
  hasCharge: boolean;
  hasDefender: boolean;
  hasInfiltrate: boolean;
}

/**
 * Gate card with resolved image data.
 */
export interface ResolvedGate {
  cardCode: string;
  cardDefId: number;
  imageUrl: string;
  name: string;
  tapped: boolean;
  cooldown: boolean;
}

/**
 * IKZ resource card with resolved image data.
 */
export interface ResolvedIkz {
  cardCode: string;
  cardDefId: number;
  imageUrl: string;
  name: string;
  tapped: boolean;
  cooldown: boolean;
}

/**
 * A player's board state with all zones resolved.
 */
export interface ResolvedPlayerBoard {
  leader: ResolvedLeader;
  gate: ResolvedGate;
  garden: (ResolvedCard | null)[]; // 5 slots (front row)
  alley: (ResolvedCard | null)[]; // 5 slots (back row)
  ikzArea: ResolvedIkz[];
  handCount: number;
  deckCount: number;
  discardCount: number;
  ikzPileCount: number;
  hasIkzToken: boolean;
}

/**
 * A card in the player's hand with resolved image data.
 */
export interface ResolvedHandCard {
  cardCode: string;
  cardDefId: number;
  imageUrl: string;
  name: string;
  type: string; // ENTITY, SPELL, WEAPON, etc.
  ikzCost: number;
}

/**
 * A card in the selection zone with resolved image data.
 * Used during SELECTION_PICK and BOTTOM_DECK ability phases.
 */
export interface ResolvedSelectionCard {
  cardCode: string;
  cardDefId: number;
  zoneIndex: number;
  imageUrl: string;
  name: string;
  type: string;
  ikzCost: number;
  curAtk: number | null;
  curHp: number | null;
}

/**
 * Complete game state for rendering.
 */
export interface GameState {
  phase: string;
  abilitySubphase: string;
  activePlayer: 0 | 1;
  turnNumber: number;
  pendingConfirmationCount?: number;
  abilitySourceCardDefId?: number;
  abilityCostTargetType?: number;
  abilityEffectTargetType?: number;

  // Player boards - myBoard is always the viewing player
  myBoard: ResolvedPlayerBoard;
  opponentBoard: ResolvedPlayerBoard;

  // Only the viewing player's hand is visible
  myHand: ResolvedHandCard[];

  // Selection zone cards (for SELECTION_PICK and BOTTOM_DECK ability phases)
  selectionCards?: ResolvedSelectionCard[];

  // Action mask for legal moves (only when it's your turn)
  actionMask: SnapshotActionMask | null;

  // Combat stack (for future combat visualization)
  combatStack: unknown[];
}

// ============================================
// Card data types from deck API
// ============================================

/**
 * Card data returned from the deck API.
 */
export interface DeckCard {
  cardCode: string;
  cardDefId: number;
  imageKey: string;
  name: string;
  cardType: string;
  attack: number | null;
  health: number | null;
  ikzCost: number | null;
  quantity: number;
}

/**
 * Deck with full card details from GET /api/decks/[deckId].
 */
export interface DeckWithCards {
  id: string;
  name: string;
  isSystemDeck: boolean;
  cards: DeckCard[];
}

// ============================================
// Asset loading types
// ============================================

/**
 * Card mapping for texture resolution.
 */
export interface CardMapping {
  cardCode: string;
  imageKey: string;
  imageUrl: string;
  name: string;
  cardType: string;
  attack: number | null;
  health: number | null;
  ikzCost: number | null;
}

/**
 * Asset loading progress state.
 */
export interface AssetLoadingState {
  isLoading: boolean;
  progress: number; // 0-100
  loadedCount: number;
  totalCount: number;
  error: string | null;
}

// ============================================
// Constants
// ============================================

export const STATIC_CDN_BASE = "https://azuki-tcg.s3.us-east-1.amazonaws.com";

/**
 * Build full image URL from imageKey.
 */
export function buildImageUrl(imageKey: string): string {
  return `${STATIC_CDN_BASE}/${imageKey}`;
}

/**
 * Build a cardDefId -> CardMapping map from deck cards.
 * Used to pre-populate the map with all cards in both players' decks
 * so that drawn cards can be resolved correctly.
 */
export function buildCardDefIdMapFromDeckCards(
  cards: DeckCard[]
): Map<number, CardMapping> {
  const map = new Map<number, CardMapping>();

  for (const card of cards) {
    if (!map.has(card.cardDefId)) {
      map.set(card.cardDefId, {
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
  }

  return map;
}
