import {
  boolean,
  check,
  index,
  integer,
  jsonb,
  pgEnum,
  pgTable,
  text,
  timestamp,
  unique,
  uniqueIndex,
  uuid,
} from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";
import {
  createdAtTimestampField,
  enumToPgEnum,
  updatedAtTimestampField,
  uuidv7PrimaryKeyField,
} from "@core/drizzle/helpers";
import {
  HumanEvaluationActorSource,
  HumanEvaluationDeckSource,
  HumanEvaluationMatchStatus,
  HumanEvaluationSessionStatus,
} from "@core/types/humanEvaluations";
import type { HumanEvaluationDraftPick, HumanEvaluationRating } from "@core/types/humanEvaluations";
import { AiModels } from "@core/drizzle/schemas/ai_models";
import { Decks } from "@core/drizzle/schemas/decks";
import { MatchResults } from "@core/drizzle/schemas/match_results";
import { Rooms } from "@core/drizzle/schemas/rooms";
import { Users } from "@core/drizzle/schemas/users";

export const humanEvaluationSessionStatusEnum = pgEnum(
  "human_evaluation_session_status",
  enumToPgEnum(HumanEvaluationSessionStatus)
);
export const humanEvaluationMatchStatusEnum = pgEnum(
  "human_evaluation_match_status",
  enumToPgEnum(HumanEvaluationMatchStatus)
);
export const humanEvaluationActorSourceEnum = pgEnum(
  "human_evaluation_actor_source",
  enumToPgEnum(HumanEvaluationActorSource)
);
export const humanEvaluationDeckSourceEnum = pgEnum(
  "human_evaluation_deck_source",
  enumToPgEnum(HumanEvaluationDeckSource)
);
export const humanEvaluationObservationKindEnum = pgEnum("human_evaluation_observation_kind", [
  "ISSUE",
  "OPPORTUNITY",
]);
export const humanEvaluationObservationTagEnum = pgEnum("human_evaluation_observation_tag", [
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
]);
export const humanEvaluationObservationSeverityEnum = pgEnum(
  "human_evaluation_observation_severity",
  ["MINOR", "MODERATE", "MAJOR", "CRITICAL"]
);

export const HumanEvaluationSessions = pgTable(
  "human_evaluation_sessions",
  {
    id: uuidv7PrimaryKeyField(),
    reviewerId: uuid("reviewer_id")
      .notNull()
      .references(() => Users.id),
    humanDeckId: uuid("human_deck_id")
      .notNull()
      .references(() => Decks.id),
    sourceHumanDeckId: uuid("source_human_deck_id")
      .notNull()
      .references(() => Decks.id),
    gamesPerModel: integer("games_per_model").notNull(),
    protocol: text("protocol").notNull(),
    scheduleSeed: text("schedule_seed").notNull(),
    status: humanEvaluationSessionStatusEnum("status")
      .notNull()
      .default(HumanEvaluationSessionStatus.ACTIVE),
    revealedAt: timestamp("revealed_at", { withTimezone: true }),
    createdAt: createdAtTimestampField(),
    updatedAt: updatedAtTimestampField(),
  },
  (table) => [
    index("human_evaluation_sessions_reviewer_idx").on(table.reviewerId, table.createdAt),
    uniqueIndex("human_evaluation_sessions_active_reviewer_unique")
      .on(table.reviewerId)
      .where(sql`${table.status} = 'ACTIVE'`),
    check(
      "human_evaluation_sessions_games_per_model_check",
      sql`${table.gamesPerModel} IN (8, 16)`
    ),
    check("human_evaluation_sessions_protocol_check", sql`${table.protocol} = 'human-eval-v1'`),
  ]
);

export const HumanEvaluationMatches = pgTable(
  "human_evaluation_matches",
  {
    id: uuidv7PrimaryKeyField(),
    sessionId: uuid("session_id")
      .notNull()
      .references(() => HumanEvaluationSessions.id),
    ordinal: integer("ordinal").notNull(),
    modelId: uuid("model_id")
      .notNull()
      .references(() => AiModels.id),
    modelKeySnapshot: text("model_key_snapshot").notNull(),
    modelDisplayNameSnapshot: text("model_display_name_snapshot").notNull(),
    checkpointSha256Snapshot: text("checkpoint_sha256_snapshot").notNull(),
    sessionKey: text("session_key").notNull().unique(),
    draftSeed: integer("draft_seed").notNull(),
    battleSeed: integer("battle_seed").notNull(),
    aiSlot: integer("ai_slot").notNull().$type<0 | 1>(),
    startingPlayer: integer("starting_player").notNull().$type<0 | 1>(),
    gateCardCode: text("gate_card_code").notNull(),
    leaderCardCode: text("leader_card_code").notNull(),
    deckSource: humanEvaluationDeckSourceEnum("deck_source")
      .notNull()
      .default(HumanEvaluationDeckSource.DRAFT),
    premadeDeckSlug: text("premade_deck_slug"),
    status: humanEvaluationMatchStatusEnum("status")
      .notNull()
      .default(HumanEvaluationMatchStatus.SCHEDULED),
    roomId: uuid("room_id").references(() => Rooms.id),
    matchResultId: uuid("match_result_id").references(() => MatchResults.id),
    claimedAt: timestamp("claimed_at", { withTimezone: true }),
    completedAt: timestamp("completed_at", { withTimezone: true }),
    createdAt: createdAtTimestampField(),
    updatedAt: updatedAtTimestampField(),
  },
  (table) => [
    unique("human_evaluation_matches_session_ordinal_unique").on(table.sessionId, table.ordinal),
    unique("human_evaluation_matches_room_unique").on(table.roomId),
    unique("human_evaluation_matches_result_unique").on(table.matchResultId),
    index("human_evaluation_matches_session_status_idx").on(table.sessionId, table.status),
    check("human_evaluation_matches_ai_slot_check", sql`${table.aiSlot} IN (0, 1)`),
    check("human_evaluation_matches_starting_player_check", sql`${table.startingPlayer} IN (0, 1)`),
    check("human_evaluation_matches_ordinal_check", sql`${table.ordinal} > 0`),
    check(
      "human_evaluation_matches_premade_deck_check",
      sql`(${table.deckSource} = 'PREMADE') = (${table.premadeDeckSlug} IS NOT NULL)`
    ),
  ]
);

export const HumanEvaluationDeckArtifacts = pgTable(
  "human_evaluation_deck_artifacts",
  {
    id: uuidv7PrimaryKeyField(),
    matchId: uuid("match_id")
      .notNull()
      .unique()
      .references(() => HumanEvaluationMatches.id),
    deckId: uuid("deck_id")
      .notNull()
      .unique()
      .references(() => Decks.id),
    gateCardCode: text("gate_card_code").notNull(),
    leaderCardCode: text("leader_card_code").notNull(),
    orderedMainCardCodes: text("ordered_main_card_codes").array().notNull(),
    cardCounts: jsonb("card_counts").notNull().$type<Record<string, number>>(),
    picks: jsonb("picks").notNull().$type<HumanEvaluationDraftPick[]>(),
    deckHash: text("deck_hash").notNull(),
    catalogHash: text("catalog_hash").notNull(),
    checkpointSha256: text("checkpoint_sha256").notNull(),
    createdAt: createdAtTimestampField(),
  },
  (table) => [
    check(
      "human_evaluation_deck_artifacts_main_count_check",
      sql`cardinality(${table.orderedMainCardCodes}) = 50`
    ),
  ]
);

export const HumanEvaluationActions = pgTable(
  "human_evaluation_actions",
  {
    id: uuidv7PrimaryKeyField(),
    matchId: uuid("match_id")
      .notNull()
      .references(() => HumanEvaluationMatches.id),
    roomId: uuid("room_id")
      .notNull()
      .references(() => Rooms.id),
    actionNumber: integer("action_number").notNull(),
    actorSlot: integer("actor_slot").notNull().$type<0 | 1>(),
    actorSource: humanEvaluationActorSourceEnum("actor_source").notNull(),
    action: integer("action").array().notNull().$type<[number, number, number, number]>(),
    accepted: boolean("accepted").notNull(),
    error: text("error"),
    observation: jsonb("observation").notNull(),
    legalActionMask: jsonb("legal_action_mask").notNull(),
    stateHash: text("state_hash").notNull(),
    receivedAt: timestamp("received_at", { withTimezone: true }).notNull(),
    resolvedAt: timestamp("resolved_at", { withTimezone: true }).notNull(),
    createdAt: createdAtTimestampField(),
  },
  (table) => [
    unique("human_evaluation_actions_match_number_unique").on(table.matchId, table.actionNumber),
    index("human_evaluation_actions_room_idx").on(table.roomId, table.actionNumber),
    check("human_evaluation_actions_number_check", sql`${table.actionNumber} > 0`),
    check("human_evaluation_actions_actor_slot_check", sql`${table.actorSlot} IN (0, 1)`),
    check("human_evaluation_actions_tuple_check", sql`cardinality(${table.action}) = 4`),
    check(
      "human_evaluation_actions_timestamps_check",
      sql`${table.resolvedAt} >= ${table.receivedAt}`
    ),
  ]
);

export const HumanEvaluationAnnotations = pgTable(
  "human_evaluation_annotations",
  {
    id: uuidv7PrimaryKeyField(),
    matchId: uuid("match_id")
      .notNull()
      .unique()
      .references(() => HumanEvaluationMatches.id),
    reviewerId: uuid("reviewer_id")
      .notNull()
      .references(() => Users.id),
    opponentStrengthRating: integer("opponent_strength_rating")
      .notNull()
      .$type<HumanEvaluationRating>(),
    decisionQualityRating: integer("decision_quality_rating")
      .notNull()
      .$type<HumanEvaluationRating>(),
    deckCoherenceRating: integer("deck_coherence_rating").notNull().$type<HumanEvaluationRating>(),
    humanLikenessRating: integer("human_likeness_rating").notNull().$type<HumanEvaluationRating>(),
    matchEnjoymentRating: integer("match_enjoyment_rating")
      .notNull()
      .$type<HumanEvaluationRating>(),
    modelGuess: text("model_guess"),
    guessConfidence: integer("guess_confidence").$type<HumanEvaluationRating>(),
    notes: text("notes"),
    createdAt: createdAtTimestampField(),
  },
  (table) => [
    check(
      "human_evaluation_annotations_ratings_check",
      sql`${table.opponentStrengthRating} BETWEEN 1 AND 7
        AND ${table.decisionQualityRating} BETWEEN 1 AND 7
        AND ${table.deckCoherenceRating} BETWEEN 1 AND 7
        AND ${table.humanLikenessRating} BETWEEN 1 AND 7
        AND ${table.matchEnjoymentRating} BETWEEN 1 AND 7`
    ),
    check(
      "human_evaluation_annotations_confidence_check",
      sql`${table.guessConfidence} IS NULL OR ${table.guessConfidence} BETWEEN 1 AND 7`
    ),
  ]
);

export const HumanEvaluationAnnotationObservations = pgTable(
  "human_evaluation_annotation_observations",
  {
    id: uuidv7PrimaryKeyField(),
    annotationId: uuid("annotation_id")
      .notNull()
      .references(() => HumanEvaluationAnnotations.id, { onDelete: "cascade" }),
    ordinal: integer("ordinal").notNull(),
    kind: humanEvaluationObservationKindEnum("kind").notNull(),
    tag: humanEvaluationObservationTagEnum("tag").notNull(),
    severity: humanEvaluationObservationSeverityEnum("severity").notNull(),
    actionNumber: integer("action_number"),
    detail: text("detail"),
    createdAt: createdAtTimestampField(),
  },
  (table) => [
    unique("human_evaluation_annotation_observations_ordinal_unique").on(
      table.annotationId,
      table.ordinal
    ),
  ]
);
