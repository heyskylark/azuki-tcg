/**
 * Browser-side client for the blind human-evaluation API.
 *
 * Response schemas intentionally strip unknown keys: anything the API adds later
 * (including anything that would identify a checkpoint before reveal) can never
 * reach a component by accident. Request bodies are built here so every caller
 * sends exactly the fields the route accepts.
 */

import { z } from "zod";
import { authenticatedFetch } from "@/lib/api/authenticatedFetch";
import { HumanEvaluationMatchStatus } from "@tcg/backend-core/types/humanEvaluations";
import type {
  HumanEvaluationAnnotation,
  HumanEvaluationAnnotationInput,
  HumanEvaluationObservationInput,
  HumanEvaluationRatings,
  HumanEvaluationSessionSummary,
  HumanEvaluationSessionSummaryMatch,
} from "@tcg/backend-core/types/humanEvaluations";

export const GAMES_PER_MODEL_OPTIONS = [8, 16] as const;

export const EVALUATION_PROTOCOL = "human-eval-v1";

export const OBSERVATION_KINDS = ["ISSUE", "OPPORTUNITY"] as const;

export const OBSERVATION_SEVERITIES = ["MINOR", "MODERATE", "MAJOR", "CRITICAL"] as const;

export const OBSERVATION_TAGS = [
  "MISPLAY",
  "SEQUENCING",
  "RESOURCE_WASTE",
  "TARGETING",
  "ATTACK_CHOICE",
  "DEFENSE_CHOICE",
  "TEMPO",
  "DECK_SYNERGY",
  "REPETITIVE",
  "TIMEOUT_OR_STALL",
  "OTHER",
] as const;

export const RATING_SCALE = [1, 2, 3, 4, 5, 6, 7] as const;

const ratingSchema = z.literal(RATING_SCALE);

const ratingsSchema = z.object({
  opponentStrength: ratingSchema,
  decisionQuality: ratingSchema,
  deckCoherence: ratingSchema,
  humanLikeness: ratingSchema,
  matchEnjoyment: ratingSchema,
});

const observationSchema = z.object({
  kind: z.enum(OBSERVATION_KINDS),
  tag: z.enum(OBSERVATION_TAGS),
  severity: z.enum(OBSERVATION_SEVERITIES),
  actionNumber: z.number().int().nullable(),
  detail: z.string().nullable(),
});

const annotationSchema = z.object({
  matchId: z.string(),
  ratings: ratingsSchema,
  modelGuess: z.string().nullable(),
  guessConfidence: ratingSchema.nullable(),
  observations: z.array(observationSchema),
  notes: z.string().nullable(),
  createdAt: z.string(),
});

const sessionMatchSchema = z.object({
  matchId: z.string(),
  ordinal: z.number().int(),
  status: z.enum(HumanEvaluationMatchStatus),
  opponentLabel: z.string(),
  roomId: z.string().nullable(),
  outcome: z.enum(["WIN", "LOSS", "DRAW", "ABORTED"]).nullable(),
  hasAnnotation: z.boolean(),
  revealed: z
    .object({
      modelDisplayName: z.string(),
      checkpointSha256: z.string(),
    })
    .nullable(),
});

const sessionSummarySchema = z.object({
  id: z.string(),
  humanDeckId: z.string(),
  gamesPerModel: z.literal(GAMES_PER_MODEL_OPTIONS),
  totalMatches: z.number().int(),
  completedMatches: z.number().int(),
  annotatedMatches: z.number().int(),
  revealReady: z.boolean(),
  createdAt: z.string(),
  revealedAt: z.string().nullable(),
  matches: z.array(sessionMatchSchema),
});

const sessionResponseSchema = z.object({ session: sessionSummarySchema });

const nextMatchSchema = z.object({
  roomId: z.string(),
  matchId: z.string(),
  ordinal: z.number().int(),
  totalMatches: z.number().int(),
  resumed: z.boolean(),
});

const nextMatchResponseSchema = z.object({ match: nextMatchSchema });

const blindMatchSchema = z.object({
  matchId: z.string(),
  sessionId: z.string(),
  ordinal: z.number().int(),
  totalMatches: z.number().int(),
  status: z.enum(HumanEvaluationMatchStatus),
  opponentLabel: z.string(),
  roomId: z.string().nullable(),
  outcome: z.enum(["WIN", "LOSS", "DRAW", "ABORTED"]).nullable(),
  hasAnnotation: z.boolean(),
});

const blindMatchResponseSchema = z.object({ match: blindMatchSchema });

const annotationResponseSchema = z.object({ annotation: annotationSchema });

const errorResponseSchema = z.object({ message: z.string() });

export type EvaluationSessionSummary = z.infer<typeof sessionSummarySchema>;
export type EvaluationSessionMatch = z.infer<typeof sessionMatchSchema>;
export type EvaluationNextMatch = z.infer<typeof nextMatchSchema>;
export type EvaluationBlindMatch = z.infer<typeof blindMatchSchema>;
export type EvaluationAnnotation = z.infer<typeof annotationSchema>;
export type EvaluationObservation = z.infer<typeof observationSchema>;
export type EvaluationRatings = z.infer<typeof ratingsSchema>;
export type EvaluationObservationKind = (typeof OBSERVATION_KINDS)[number];
export type EvaluationObservationSeverity = (typeof OBSERVATION_SEVERITIES)[number];
export type EvaluationObservationTag = (typeof OBSERVATION_TAGS)[number];
export type EvaluationGamesPerModel = (typeof GAMES_PER_MODEL_OPTIONS)[number];

// Compile-time proof that the parsed payloads are the backend contract, not a
// parallel definition of it.
type AssertExtends<TValue extends TContract, TContract> = TValue;
type SessionSummaryMatchesContract = AssertExtends<
  EvaluationSessionSummary,
  HumanEvaluationSessionSummary
>;
type SessionMatchMatchesContract = AssertExtends<
  EvaluationSessionMatch,
  HumanEvaluationSessionSummaryMatch
>;
type ObservationMatchesContract = AssertExtends<
  EvaluationObservation,
  HumanEvaluationObservationInput
>;
type RatingsMatchContract = AssertExtends<EvaluationRatings, HumanEvaluationRatings>;
type AnnotationMatchesContract = AssertExtends<EvaluationAnnotation, HumanEvaluationAnnotation>;
export type EvaluationContractAssertions = [
  SessionSummaryMatchesContract,
  SessionMatchMatchesContract,
  ObservationMatchesContract,
  RatingsMatchContract,
  AnnotationMatchesContract,
];

async function requestJson<TSchema extends z.ZodType>(
  input: string,
  schema: TSchema,
  fallbackMessage: string,
  init?: RequestInit
): Promise<z.infer<TSchema>> {
  const response = await authenticatedFetch(input, init);
  const payload: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    const parsedError = errorResponseSchema.safeParse(payload);
    throw new Error(parsedError.success ? parsedError.data.message : fallbackMessage);
  }

  return schema.parse(payload);
}

function jsonBody(body: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export async function createEvaluationSession(input: {
  humanDeckId: string;
  gamesPerModel: EvaluationGamesPerModel;
}): Promise<EvaluationSessionSummary> {
  const payload = await requestJson(
    "/api/evaluations/sessions",
    sessionResponseSchema,
    "Failed to start evaluation session",
    jsonBody({ humanDeckId: input.humanDeckId, gamesPerModel: input.gamesPerModel })
  );

  return payload.session;
}

export async function fetchEvaluationSession(sessionId: string): Promise<EvaluationSessionSummary> {
  const payload = await requestJson(
    `/api/evaluations/sessions/${sessionId}`,
    sessionResponseSchema,
    "Failed to load evaluation session"
  );

  return payload.session;
}

export async function claimNextEvaluationMatch(sessionId: string): Promise<EvaluationNextMatch> {
  const payload = await requestJson(
    `/api/evaluations/sessions/${sessionId}/next`,
    nextMatchResponseSchema,
    "Failed to claim the next match",
    jsonBody({})
  );

  return payload.match;
}

export async function revealEvaluationSession(
  sessionId: string
): Promise<EvaluationSessionSummary> {
  const payload = await requestJson(
    `/api/evaluations/sessions/${sessionId}/reveal`,
    sessionResponseSchema,
    "Failed to reveal the session",
    jsonBody({})
  );

  return payload.session;
}

export async function fetchEvaluationMatch(matchId: string): Promise<EvaluationBlindMatch> {
  const payload = await requestJson(
    `/api/evaluations/matches/${matchId}`,
    blindMatchResponseSchema,
    "Failed to load match"
  );

  return payload.match;
}

export async function submitEvaluationAnnotation(
  matchId: string,
  input: HumanEvaluationAnnotationInput
): Promise<EvaluationAnnotation> {
  const payload = await requestJson(
    `/api/evaluations/matches/${matchId}/annotations`,
    annotationResponseSchema,
    "Failed to save annotation",
    jsonBody(input)
  );

  return payload.annotation;
}

// ============================================
// Presentation helpers shared by the evaluation surfaces
// ============================================

interface RatingDescriptor {
  key: keyof EvaluationRatings;
  label: string;
  question: string;
  lowAnchor: string;
  highAnchor: string;
}

export const RATING_DESCRIPTORS: readonly RatingDescriptor[] = [
  {
    key: "opponentStrength",
    label: "Opponent strength",
    question: "How hard was this opponent to beat?",
    lowAnchor: "Trivial",
    highAnchor: "Overwhelming",
  },
  {
    key: "decisionQuality",
    label: "Decision quality",
    question: "How sound were its individual plays?",
    lowAnchor: "Random",
    highAnchor: "Near-optimal",
  },
  {
    key: "deckCoherence",
    label: "Deck coherence",
    question: "Did its 50 cards work together as a plan?",
    lowAnchor: "Incoherent",
    highAnchor: "Purpose-built",
  },
  {
    key: "humanLikeness",
    label: "Human-likeness",
    question: "Could a human have made these choices?",
    lowAnchor: "Obviously machine",
    highAnchor: "Indistinguishable",
  },
  {
    key: "matchEnjoyment",
    label: "Match quality",
    question: "Was this a match worth playing?",
    lowAnchor: "Miserable",
    highAnchor: "Excellent",
  },
] as const;

export const OBSERVATION_TAG_LABELS: Record<EvaluationObservationTag, string> = {
  MISPLAY: "Outright misplay",
  SEQUENCING: "Bad sequencing",
  RESOURCE_WASTE: "Wasted resources",
  TARGETING: "Wrong target",
  ATTACK_CHOICE: "Attack choice",
  DEFENSE_CHOICE: "Defense choice",
  TEMPO: "Tempo loss",
  DECK_SYNERGY: "Deck synergy",
  REPETITIVE: "Repetitive behaviour",
  TIMEOUT_OR_STALL: "Stall or timeout",
  OTHER: "Other",
};

export const OBSERVATION_KIND_LABELS: Record<EvaluationObservationKind, string> = {
  ISSUE: "Issue",
  OPPORTUNITY: "Missed opportunity",
};

export const OBSERVATION_SEVERITY_LABELS: Record<EvaluationObservationSeverity, string> = {
  MINOR: "Minor",
  MODERATE: "Moderate",
  MAJOR: "Major",
  CRITICAL: "Critical",
};

export const MATCH_STATUS_LABELS: Record<HumanEvaluationMatchStatus, string> = {
  [HumanEvaluationMatchStatus.SCHEDULED]: "Scheduled",
  [HumanEvaluationMatchStatus.CLAIMED]: "Room open",
  [HumanEvaluationMatchStatus.IN_PROGRESS]: "In progress",
  [HumanEvaluationMatchStatus.COMPLETED]: "Played",
  [HumanEvaluationMatchStatus.ABORTED]: "Technical abort",
};

/** A match that is over and therefore owes an annotation. */
export function needsAnnotation(match: EvaluationSessionMatch): boolean {
  return (
    !match.hasAnnotation &&
    (match.status === HumanEvaluationMatchStatus.COMPLETED ||
      match.status === HumanEvaluationMatchStatus.ABORTED)
  );
}
