/**
 * Game log types for client-side processing.
 * These types mirror the server-side processed log structures.
 */

// ============================================
// Enums matching C engine values
// ============================================

export type ZoneType =
  | "DECK"
  | "HAND"
  | "LEADER"
  | "GATE"
  | "GARDEN"
  | "ALLEY"
  | "IKZ_PILE"
  | "IKZ_AREA"
  | "DISCARD"
  | "SELECTION"
  | "EQUIPPED";

export type TapStateValue = "TAPPED" | "UNTAPPED" | "COOLDOWN";

export type StatusEffect = "FROZEN" | "SHOCKED" | "EFFECT_IMMUNE";

export type DeathCause = "COMBAT" | "ABILITY" | "EFFECT";

export type ShuffleReason = "MULLIGAN" | "EFFECT" | "GAME_START";

export type GameEndReason = "LEADER_DEFEATED" | "DECK_OUT" | "CONCEDE";

// ============================================
// Card reference in logs
// ============================================

export interface LogCardRef {
  player: 0 | 1;
  cardDefId: number | null; // null when hidden (opponent's private zones)
  zone: ZoneType;
  zoneIndex: number;
}

// ============================================
// Zone moved metadata
// ============================================

export interface ZoneMovedMetadata {
  curAtk: number;
  curHp: number;
  tapped: boolean;
  cooldown: boolean;
  hasCharge: boolean;
  hasDefender: boolean;
  hasInfiltrate: boolean;
  isFrozen: boolean;
  isEffectImmune: boolean;
}

// ============================================
// Log data types
// ============================================

export interface CardZoneMovedData {
  card: LogCardRef;
  fromZone: ZoneType;
  fromIndex: number;
  toZone: ZoneType;
  toIndex: number;
  metadata: ZoneMovedMetadata | null; // null when hidden
}

export interface CardStatChangeData {
  card: LogCardRef;
  atkDelta: number;
  hpDelta: number;
  newAtk: number;
  newHp: number;
}

export interface KeywordsChangedData {
  card: LogCardRef;
  hasCharge: boolean;
  hasDefender: boolean;
  hasInfiltrate: boolean;
}

export interface CardTapChangeData {
  card: LogCardRef;
  newState: TapStateValue;
}

export interface StatusAppliedData {
  card: LogCardRef;
  effect: StatusEffect;
  duration: number;
}

export interface StatusExpiredData {
  card: LogCardRef;
  effect: StatusEffect;
}

export interface CombatDeclaredData {
  attacker: LogCardRef;
  target: LogCardRef;
}

export interface DefenderDeclaredData {
  defender: LogCardRef;
}

export interface CombatDamageData {
  attacker: LogCardRef;
  defender: LogCardRef;
  attackerDamageDealt: number;
  attackerDamageTaken: number;
  defenderDamageDealt: number;
  defenderDamageTaken: number;
}

export interface EntityDiedData {
  card: LogCardRef;
  cause: DeathCause;
}

export interface EffectQueuedData {
  card: LogCardRef;
  abilityIndex: number;
  triggerTag: number;
}

export interface EffectEnabledData {
  card: LogCardRef;
  abilityIndex: number;
}

export interface DeckShuffledData {
  player: 0 | 1;
  reason: ShuffleReason;
}

export interface TurnStartedData {
  player: 0 | 1;
  turnNumber: number;
}

export interface TurnEndedData {
  player: 0 | 1;
  turnNumber: number;
}

export interface GameEndedData {
  winner: number;
  reason: GameEndReason;
}

// ============================================
// Processed game log union type
// ============================================

export type ProcessedGameLog =
  | { type: "ZONE_MOVED"; data: CardZoneMovedData }
  | { type: "STAT_CHANGE"; data: CardStatChangeData }
  | { type: "KEYWORDS_CHANGED"; data: KeywordsChangedData }
  | { type: "TAP_CHANGED"; data: CardTapChangeData }
  | { type: "STATUS_EFFECT_APPLIED"; data: StatusAppliedData }
  | { type: "STATUS_APPLIED"; data: StatusAppliedData }
  | { type: "STATUS_EFFECT_EXPIRED"; data: StatusExpiredData }
  | { type: "STATUS_EXPIRED"; data: StatusExpiredData }
  | { type: "COMBAT_DECLARED"; data: CombatDeclaredData }
  | { type: "DEFENDER_DECLARED"; data: DefenderDeclaredData }
  | { type: "COMBAT_DAMAGE"; data: CombatDamageData }
  | { type: "ENTITY_DIED"; data: EntityDiedData }
  | { type: "EFFECT_QUEUED"; data: EffectQueuedData }
  | { type: "EFFECT_ENABLED"; data: EffectEnabledData }
  | { type: "DECK_SHUFFLED"; data: DeckShuffledData }
  | { type: "TURN_STARTED"; data: TurnStartedData }
  | { type: "TURN_ENDED"; data: TurnEndedData }
  | { type: "GAME_ENDED"; data: GameEndedData };
