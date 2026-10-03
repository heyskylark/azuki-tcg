export enum HumanEvaluationSessionStatus {
  ACTIVE = "ACTIVE",
  REVEALED = "REVEALED",
}

export enum HumanEvaluationMatchStatus {
  SCHEDULED = "SCHEDULED",
  CLAIMED = "CLAIMED",
  IN_PROGRESS = "IN_PROGRESS",
  COMPLETED = "COMPLETED",
  ABORTED = "ABORTED",
}

export enum HumanEvaluationActorSource {
  HUMAN = "HUMAN",
  AI = "AI",
}

export enum HumanEvaluationDeckSource {
  DRAFT = "DRAFT",
  PREMADE = "PREMADE",
}

export interface HumanEvaluationPremadeDeck {
  slug: string;
  name: string;
  gateCardCode: string;
  leaderCardCode: string;
}

export interface HumanEvaluationDraftGate {
  gateCardCode: string;
  leaderCardCodes: string[];
}

/**
 * Per-model opponent deck plan. Each scheduled match is PREMADE (a fixed pool
 * deck resolved by slug in the inference sidecar's deck pool) or DRAFT (the
 * model drafts under a uniformly random gate, then a uniformly random leader
 * valid for it). Models without a plan use the legacy all-element contexts.
 */
export interface HumanEvaluationPlan {
  premadeFraction: number;
  premadeDecks: HumanEvaluationPremadeDeck[];
  draftGates: HumanEvaluationDraftGate[];
}

export type HumanEvaluationRating = 1 | 2 | 3 | 4 | 5 | 6 | 7;
export type HumanEvaluationObservationKind = "ISSUE" | "OPPORTUNITY";
export type HumanEvaluationObservationSeverity = "MINOR" | "MODERATE" | "MAJOR" | "CRITICAL";
export type HumanEvaluationObservationTag =
  | "MISPLAY"
  | "SEQUENCING"
  | "RESOURCE_WASTE"
  | "TARGETING"
  | "ATTACK_CHOICE"
  | "DEFENSE_CHOICE"
  | "TEMPO"
  | "DECK_SYNERGY"
  | "REPETITIVE"
  | "TIMEOUT_OR_STALL"
  | "OTHER";

export interface HumanEvaluationRatings {
  opponentStrength: HumanEvaluationRating;
  decisionQuality: HumanEvaluationRating;
  deckCoherence: HumanEvaluationRating;
  humanLikeness: HumanEvaluationRating;
  matchEnjoyment: HumanEvaluationRating;
}

export interface HumanEvaluationObservationInput {
  kind: HumanEvaluationObservationKind;
  tag: HumanEvaluationObservationTag;
  severity: HumanEvaluationObservationSeverity;
  actionNumber: number | null;
  detail: string | null;
}

export interface HumanEvaluationAnnotationInput {
  ratings: HumanEvaluationRatings;
  modelGuess: string | null;
  guessConfidence: HumanEvaluationRating | null;
  observations: HumanEvaluationObservationInput[];
  notes: string | null;
}

export interface HumanEvaluationDraftPick {
  ordinal: number;
  candidateCardCodes: string[];
  selectedIndex: number;
  selectedCardCode: string;
}

export interface HumanEvaluationDeckArtifactInput {
  gateCardCode: string;
  leaderCardCode: string;
  premadeDeckSlug: string | null;
  orderedMainCardCodes: string[];
  cardCounts: Record<string, number>;
  picks: HumanEvaluationDraftPick[];
  deckHash: string;
  catalogHash: string;
  checkpointSha256: string;
}

export interface HumanEvaluationRuntimeMatch {
  matchId: string;
  sessionId: string;
  ordinal: number;
  totalMatches: number;
  roomId: string;
  modelKey: string;
  checkpointSha256: string;
  sessionKey: string;
  draftSeed: number;
  battleSeed: number;
  aiSlot: 0 | 1;
  humanSlot: 0 | 1;
  startingPlayer: 0 | 1;
  gateCardCode: string;
  leaderCardCode: string;
  deckSource: HumanEvaluationDeckSource;
  premadeDeckSlug: string | null;
  generatedDeckId: string | null;
  status: HumanEvaluationMatchStatus;
}

export interface HumanEvaluationSessionSummaryMatch {
  matchId: string;
  ordinal: number;
  status: HumanEvaluationMatchStatus;
  opponentLabel: string;
  roomId: string | null;
  outcome: "WIN" | "LOSS" | "DRAW" | "ABORTED" | null;
  hasAnnotation: boolean;
  revealed: {
    modelDisplayName: string;
    checkpointSha256: string;
    deckSource: HumanEvaluationDeckSource;
    premadeDeckSlug: string | null;
    gateCardCode: string;
    leaderCardCode: string;
  } | null;
}

export interface HumanEvaluationSessionSummary {
  id: string;
  humanDeckId: string;
  gamesPerModel: 8 | 16;
  totalMatches: number;
  completedMatches: number;
  annotatedMatches: number;
  revealReady: boolean;
  createdAt: string;
  revealedAt: string | null;
  matches: HumanEvaluationSessionSummaryMatch[];
}

export interface HumanEvaluationAnnotation {
  matchId: string;
  ratings: HumanEvaluationRatings;
  modelGuess: string | null;
  guessConfidence: HumanEvaluationRating | null;
  observations: HumanEvaluationObservationInput[];
  notes: string | null;
  createdAt: string;
}

export interface HumanEvaluationReviewAction {
  actionNumber: number;
  actorSlot: 0 | 1;
  actorSource: HumanEvaluationActorSource;
  action: [number, number, number, number];
  accepted: boolean;
  error: string | null;
  observation: unknown;
  legalActionMask: unknown;
  stateHash: string;
  receivedAt: string;
  resolvedAt: string;
}

export interface HumanEvaluationReviewGameLog {
  batchNumber: number;
  sequenceNumber: number;
  logType: string;
  player: number | null;
  logData: unknown;
  createdAt: string;
}

export interface HumanEvaluationReviewResult {
  winnerId: string | null;
  winType: string;
  totalTurns: number;
  durationSeconds: number;
}

export interface HumanEvaluationReview {
  matchId: string;
  sessionId: string;
  ordinal: number;
  model: {
    displayName: string;
    modelKey: string;
    checkpointSha256: string;
  };
  assignment: {
    aiSlot: 0 | 1;
    startingPlayer: 0 | 1;
    gateCardCode: string;
    leaderCardCode: string;
    deckSource: HumanEvaluationDeckSource;
    premadeDeckSlug: string | null;
    battleSeed: number;
  };
  draft: {
    orderedMainCardCodes: string[];
    cardCounts: Record<string, number>;
    picks: HumanEvaluationDraftPick[];
    deckHash: string;
    catalogHash: string;
    checkpointSha256: string;
  } | null;
  actions: HumanEvaluationReviewAction[];
  gameLogs: HumanEvaluationReviewGameLog[];
  result: HumanEvaluationReviewResult | null;
  annotation: HumanEvaluationAnnotation | null;
}
