/**
 * Engine types matching the native module structures.
 * These mirror the C engine types defined in include/components/
 */

export type Phase =
  | "PREGAME_MULLIGAN"
  | "START_OF_TURN"
  | "MAIN"
  | "RESPONSE_WINDOW"
  | "COMBAT_RESOLVE"
  | "END_TURN_ACTION"
  | "END_TURN"
  | "END_MATCH";

export type CardTypeString =
  | "LEADER"
  | "GATE"
  | "ENTITY"
  | "WEAPON"
  | "SPELL"
  | "IKZ"
  | "EXTRA_IKZ";

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
  | "SELECTION";

export type TapStateValue = "TAPPED" | "UNTAPPED" | "COOLDOWN";

export type StatusEffect = "FROZEN" | "SHOCKED" | "EFFECT_IMMUNE";

export type DeathCause = "COMBAT" | "ABILITY" | "EFFECT";

export type ShuffleReason = "MULLIGAN" | "EFFECT" | "GAME_START";

export type GameEndReason = "LEADER_DEFEATED" | "DECK_OUT" | "CONCEDE";

export type AbilityPhase =
  | "NONE"
  | "CONFIRMATION"
  | "COST_SELECTION"
  | "EFFECT_SELECTION"
  | "SELECTION_PICK"
  | "BOTTOM_DECK";

export interface CardObservation {
  cardCode: string | null;
  cardDefId: number;
  type: CardTypeString;
  zoneIndex: number;
  ikzCost: number;
  tapped: boolean;
  cooldown: boolean;
  curAtk: number | null;
  curHp: number | null;
  gatePoints: number | null;
  isFrozen: boolean;
  isShocked: boolean;
  isEffectImmune: boolean;
  hasCharge: boolean;
  hasDefender: boolean;
  hasInfiltrate: boolean;
  weapons: WeaponObservation[];
}

export interface WeaponObservation {
  cardCode: string | null;
  cardDefId: number;
  curAtk: number;
}

export interface LeaderObservation {
  cardCode: string | null;
  cardDefId: number;
  type: CardTypeString;
  tapped: boolean;
  cooldown: boolean;
  curAtk: number;
  curHp: number;
  hasCharge: boolean;
  hasDefender: boolean;
  hasInfiltrate: boolean;
}

export interface GateObservation {
  cardCode: string | null;
  cardDefId: number;
  type: CardTypeString;
  tapped: boolean;
  cooldown: boolean;
}

export interface IKZObservation {
  cardCode: string | null;
  cardDefId: number;
  type: CardTypeString;
  tapped: boolean;
  cooldown: boolean;
}

export interface MyObservation {
  leader: LeaderObservation;
  gate: GateObservation;
  hand: CardObservation[];
  garden: (CardObservation | null)[];
  alley: (CardObservation | null)[];
  ikzArea: IKZObservation[];
  discard: CardObservation[];
  selection: (CardObservation | null)[];
  deckCount: number;
  ikzPileCount: number;
  selectionCount: number;
  hasIkzToken: boolean;
}

export interface OpponentObservation {
  leader: LeaderObservation;
  gate: GateObservation;
  garden: (CardObservation | null)[];
  alley: (CardObservation | null)[];
  ikzArea: IKZObservation[];
  discard: CardObservation[];
  handCount: number;
  deckCount: number;
  ikzPileCount: number;
  hasIkzToken: boolean;
}

export interface ActionMask {
  primaryActionMask: boolean[];
  legalActionCount: number;
  legalPrimary: number[];
  legalSub1: number[];
  legalSub2: number[];
  legalSub3: number[];
}

export interface ObservationData {
  myObservationData: MyObservation;
  opponentObservationData: OpponentObservation;
  phase: Phase;
  actionMask: ActionMask;
}

export interface StateContext {
  phase: Phase;
  abilityPhase: AbilityPhase;
  /** @deprecated Use abilityPhase - this is for client compatibility */
  abilitySubphase?: string;
  // Number of optional confirmation prompts remaining for active player
  pendingConfirmationCount?: number;
  abilitySourceCardDefId?: number;
  abilityCostTargetType?: number;
  abilityEffectTargetType?: number;
  activePlayer: 0 | 1;
  turnNumber: number;
  responseWindow: number;
  winner: number | null;
}

export interface GameLogCardRef {
  player: 0 | 1;
  cardDefId: number;
  zone: ZoneType;
  zoneIndex: number;
}

export interface GameLogZoneMoved {
  card: GameLogCardRef;
  fromZone: ZoneType;
  fromIndex: number;
  toZone: ZoneType;
  toIndex: number;
  metadata: {
    curAtk: number;
    curHp: number;
    tapped: boolean;
    cooldown: boolean;
    hasCharge: boolean;
    hasDefender: boolean;
    hasInfiltrate: boolean;
    isFrozen: boolean;
    isEffectImmune: boolean;
  };
}

export interface GameLogStatChange {
  card: GameLogCardRef;
  atkDelta: number;
  hpDelta: number;
  newAtk: number;
  newHp: number;
}

export interface GameLogKeywordsChanged {
  card: GameLogCardRef;
  hasCharge: boolean;
  hasDefender: boolean;
  hasInfiltrate: boolean;
}

export interface GameLogTapChange {
  card: GameLogCardRef;
  newState: TapStateValue;
}

export interface GameLogStatusApplied {
  card: GameLogCardRef;
  effect: StatusEffect;
  duration: number;
}

export interface GameLogStatusExpired {
  card: GameLogCardRef;
  effect: StatusEffect;
}

export interface GameLogCombatDeclared {
  attacker: GameLogCardRef;
  target: GameLogCardRef;
}

export interface GameLogDefenderDeclared {
  defender: GameLogCardRef;
}

export interface GameLogCombatDamage {
  attacker: GameLogCardRef;
  defender: GameLogCardRef;
  attackerDamageDealt: number;
  attackerDamageTaken: number;
  defenderDamageDealt: number;
  defenderDamageTaken: number;
}

export interface GameLogEntityDied {
  card: GameLogCardRef;
  cause: DeathCause;
}

export interface GameLogEffectQueued {
  card: GameLogCardRef;
  abilityIndex: number;
  triggerTag: number;
}

export interface GameLogEffectEnabled {
  card: GameLogCardRef;
  abilityIndex: number;
}

export interface GameLogDeckShuffled {
  player: 0 | 1;
  reason: ShuffleReason;
}

export interface GameLogTurnStarted {
  player: 0 | 1;
  turnNumber: number;
}

export interface GameLogTurnEnded {
  player: 0 | 1;
  turnNumber: number;
}

export interface GameLogGameEnded {
  winner: number;
  reason: GameEndReason;
}

export type GameLogType =
  | "CARD_ZONE_MOVED"
  | "CARD_STAT_CHANGE"
  | "KEYWORDS_CHANGED"
  | "CARD_TAP_STATE_CHANGED"
  | "STATUS_EFFECT_APPLIED"
  | "STATUS_EFFECT_EXPIRED"
  | "COMBAT_DECLARED"
  | "DEFENDER_DECLARED"
  | "COMBAT_DAMAGE"
  | "ENTITY_DIED"
  | "EFFECT_QUEUED"
  | "CARD_EFFECT_ENABLED"
  | "DECK_SHUFFLED"
  | "TURN_STARTED"
  | "TURN_ENDED"
  | "GAME_ENDED";

export interface GameLog {
  type: GameLogType;
  data:
    | GameLogZoneMoved
    | GameLogStatChange
    | GameLogKeywordsChanged
    | GameLogTapChange
    | GameLogStatusApplied
    | GameLogStatusExpired
    | GameLogCombatDeclared
    | GameLogDefenderDeclared
    | GameLogCombatDamage
    | GameLogEntityDied
    | GameLogEffectQueued
    | GameLogEffectEnabled
    | GameLogDeckShuffled
    | GameLogTurnStarted
    | GameLogTurnEnded
    | GameLogGameEnded;
}

export interface CreateWorldResult {
  worldId: string;
  success: boolean;
  error?: string;
}

export interface ActionResult {
  success: boolean;
  invalid: boolean;
  error?: string;
  gameOver: boolean;
  winner: number | null;
  logs: GameLog[];
  stateContext: StateContext;
}

export interface DeckCardEntry {
  cardId: number;
  count: number;
}

export type ActionTuple = [number, number, number, number];
