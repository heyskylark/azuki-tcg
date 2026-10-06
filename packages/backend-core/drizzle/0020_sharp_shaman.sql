CREATE TYPE "public"."human_evaluation_actor_source" AS ENUM('HUMAN', 'AI');--> statement-breakpoint
CREATE TYPE "public"."human_evaluation_match_status" AS ENUM('SCHEDULED', 'CLAIMED', 'IN_PROGRESS', 'COMPLETED', 'ABORTED');--> statement-breakpoint
CREATE TYPE "public"."human_evaluation_observation_kind" AS ENUM('ISSUE', 'OPPORTUNITY');--> statement-breakpoint
CREATE TYPE "public"."human_evaluation_observation_severity" AS ENUM('MINOR', 'MODERATE', 'MAJOR', 'CRITICAL');--> statement-breakpoint
CREATE TYPE "public"."human_evaluation_observation_tag" AS ENUM('MISPLAY', 'SEQUENCING', 'RESOURCE_WASTE', 'TARGETING', 'ATTACK_CHOICE', 'DEFENSE_CHOICE', 'TEMPO', 'DECK_SYNERGY', 'REPETITIVE', 'TIMEOUT_OR_STALL', 'OTHER');--> statement-breakpoint
CREATE TYPE "public"."human_evaluation_session_status" AS ENUM('ACTIVE', 'REVEALED');--> statement-breakpoint
CREATE TABLE "human_evaluation_actions" (
	"id" uuid PRIMARY KEY NOT NULL,
	"match_id" uuid NOT NULL,
	"room_id" uuid NOT NULL,
	"action_number" integer NOT NULL,
	"actor_slot" integer NOT NULL,
	"actor_source" "human_evaluation_actor_source" NOT NULL,
	"action" integer[] NOT NULL,
	"accepted" boolean NOT NULL,
	"error" text,
	"observation" jsonb NOT NULL,
	"legal_action_mask" jsonb NOT NULL,
	"state_hash" text NOT NULL,
	"received_at" timestamp with time zone NOT NULL,
	"resolved_at" timestamp with time zone NOT NULL,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "human_evaluation_actions_match_number_unique" UNIQUE("match_id","action_number"),
	CONSTRAINT "human_evaluation_actions_number_check" CHECK ("human_evaluation_actions"."action_number" > 0),
	CONSTRAINT "human_evaluation_actions_actor_slot_check" CHECK ("human_evaluation_actions"."actor_slot" IN (0, 1)),
	CONSTRAINT "human_evaluation_actions_tuple_check" CHECK (cardinality("human_evaluation_actions"."action") = 4),
	CONSTRAINT "human_evaluation_actions_timestamps_check" CHECK ("human_evaluation_actions"."resolved_at" >= "human_evaluation_actions"."received_at")
);
--> statement-breakpoint
CREATE TABLE "human_evaluation_annotation_observations" (
	"id" uuid PRIMARY KEY NOT NULL,
	"annotation_id" uuid NOT NULL,
	"ordinal" integer NOT NULL,
	"kind" "human_evaluation_observation_kind" NOT NULL,
	"tag" "human_evaluation_observation_tag" NOT NULL,
	"severity" "human_evaluation_observation_severity" NOT NULL,
	"action_number" integer,
	"detail" text,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "human_evaluation_annotation_observations_ordinal_unique" UNIQUE("annotation_id","ordinal")
);
--> statement-breakpoint
CREATE TABLE "human_evaluation_annotations" (
	"id" uuid PRIMARY KEY NOT NULL,
	"match_id" uuid NOT NULL,
	"reviewer_id" uuid NOT NULL,
	"opponent_strength_rating" integer NOT NULL,
	"decision_quality_rating" integer NOT NULL,
	"deck_coherence_rating" integer NOT NULL,
	"human_likeness_rating" integer NOT NULL,
	"match_enjoyment_rating" integer NOT NULL,
	"model_guess" text,
	"guess_confidence" integer,
	"notes" text,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "human_evaluation_annotations_match_id_unique" UNIQUE("match_id"),
	CONSTRAINT "human_evaluation_annotations_ratings_check" CHECK ("human_evaluation_annotations"."opponent_strength_rating" BETWEEN 1 AND 7
        AND "human_evaluation_annotations"."decision_quality_rating" BETWEEN 1 AND 7
        AND "human_evaluation_annotations"."deck_coherence_rating" BETWEEN 1 AND 7
        AND "human_evaluation_annotations"."human_likeness_rating" BETWEEN 1 AND 7
        AND "human_evaluation_annotations"."match_enjoyment_rating" BETWEEN 1 AND 7),
	CONSTRAINT "human_evaluation_annotations_confidence_check" CHECK ("human_evaluation_annotations"."guess_confidence" IS NULL OR "human_evaluation_annotations"."guess_confidence" BETWEEN 1 AND 7)
);
--> statement-breakpoint
CREATE TABLE "human_evaluation_deck_artifacts" (
	"id" uuid PRIMARY KEY NOT NULL,
	"match_id" uuid NOT NULL,
	"deck_id" uuid NOT NULL,
	"gate_card_code" text NOT NULL,
	"leader_card_code" text NOT NULL,
	"ordered_main_card_codes" text[] NOT NULL,
	"card_counts" jsonb NOT NULL,
	"picks" jsonb NOT NULL,
	"deck_hash" text NOT NULL,
	"catalog_hash" text NOT NULL,
	"checkpoint_sha256" text NOT NULL,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "human_evaluation_deck_artifacts_match_id_unique" UNIQUE("match_id"),
	CONSTRAINT "human_evaluation_deck_artifacts_deck_id_unique" UNIQUE("deck_id"),
	CONSTRAINT "human_evaluation_deck_artifacts_main_count_check" CHECK (cardinality("human_evaluation_deck_artifacts"."ordered_main_card_codes") = 50)
);
--> statement-breakpoint
CREATE TABLE "human_evaluation_matches" (
	"id" uuid PRIMARY KEY NOT NULL,
	"session_id" uuid NOT NULL,
	"ordinal" integer NOT NULL,
	"model_id" uuid NOT NULL,
	"model_key_snapshot" text NOT NULL,
	"model_display_name_snapshot" text NOT NULL,
	"checkpoint_sha256_snapshot" text NOT NULL,
	"session_key" text NOT NULL,
	"draft_seed" integer NOT NULL,
	"battle_seed" integer NOT NULL,
	"ai_slot" integer NOT NULL,
	"starting_player" integer NOT NULL,
	"gate_card_code" text NOT NULL,
	"leader_card_code" text NOT NULL,
	"status" "human_evaluation_match_status" DEFAULT 'SCHEDULED' NOT NULL,
	"room_id" uuid,
	"match_result_id" uuid,
	"claimed_at" timestamp with time zone,
	"completed_at" timestamp with time zone,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "human_evaluation_matches_session_key_unique" UNIQUE("session_key"),
	CONSTRAINT "human_evaluation_matches_session_ordinal_unique" UNIQUE("session_id","ordinal"),
	CONSTRAINT "human_evaluation_matches_room_unique" UNIQUE("room_id"),
	CONSTRAINT "human_evaluation_matches_result_unique" UNIQUE("match_result_id"),
	CONSTRAINT "human_evaluation_matches_ai_slot_check" CHECK ("human_evaluation_matches"."ai_slot" IN (0, 1)),
	CONSTRAINT "human_evaluation_matches_starting_player_check" CHECK ("human_evaluation_matches"."starting_player" IN (0, 1)),
	CONSTRAINT "human_evaluation_matches_ordinal_check" CHECK ("human_evaluation_matches"."ordinal" > 0)
);
--> statement-breakpoint
CREATE TABLE "human_evaluation_sessions" (
	"id" uuid PRIMARY KEY NOT NULL,
	"reviewer_id" uuid NOT NULL,
	"human_deck_id" uuid NOT NULL,
	"games_per_model" integer NOT NULL,
	"protocol" text NOT NULL,
	"schedule_seed" text NOT NULL,
	"status" "human_evaluation_session_status" DEFAULT 'ACTIVE' NOT NULL,
	"revealed_at" timestamp with time zone,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "human_evaluation_sessions_games_per_model_check" CHECK ("human_evaluation_sessions"."games_per_model" IN (8, 16)),
	CONSTRAINT "human_evaluation_sessions_protocol_check" CHECK ("human_evaluation_sessions"."protocol" = 'human-eval-v1')
);
--> statement-breakpoint
ALTER TABLE "ai_models" ADD COLUMN "checkpoint_sha256" text DEFAULT '' NOT NULL;--> statement-breakpoint
ALTER TABLE "ai_models" ADD COLUMN "human_evaluation_enabled" boolean DEFAULT false NOT NULL;--> statement-breakpoint
ALTER TABLE "decks" ADD COLUMN "is_evaluation_generated" boolean DEFAULT false NOT NULL;--> statement-breakpoint
ALTER TABLE "human_evaluation_actions" ADD CONSTRAINT "human_evaluation_actions_match_id_human_evaluation_matches_id_fk" FOREIGN KEY ("match_id") REFERENCES "public"."human_evaluation_matches"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "human_evaluation_actions" ADD CONSTRAINT "human_evaluation_actions_room_id_rooms_id_fk" FOREIGN KEY ("room_id") REFERENCES "public"."rooms"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "human_evaluation_annotation_observations" ADD CONSTRAINT "human_evaluation_annotation_observations_annotation_id_human_evaluation_annotations_id_fk" FOREIGN KEY ("annotation_id") REFERENCES "public"."human_evaluation_annotations"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "human_evaluation_annotations" ADD CONSTRAINT "human_evaluation_annotations_match_id_human_evaluation_matches_id_fk" FOREIGN KEY ("match_id") REFERENCES "public"."human_evaluation_matches"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "human_evaluation_annotations" ADD CONSTRAINT "human_evaluation_annotations_reviewer_id_users_id_fk" FOREIGN KEY ("reviewer_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "human_evaluation_deck_artifacts" ADD CONSTRAINT "human_evaluation_deck_artifacts_match_id_human_evaluation_matches_id_fk" FOREIGN KEY ("match_id") REFERENCES "public"."human_evaluation_matches"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "human_evaluation_deck_artifacts" ADD CONSTRAINT "human_evaluation_deck_artifacts_deck_id_decks_id_fk" FOREIGN KEY ("deck_id") REFERENCES "public"."decks"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "human_evaluation_matches" ADD CONSTRAINT "human_evaluation_matches_session_id_human_evaluation_sessions_id_fk" FOREIGN KEY ("session_id") REFERENCES "public"."human_evaluation_sessions"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "human_evaluation_matches" ADD CONSTRAINT "human_evaluation_matches_model_id_ai_models_id_fk" FOREIGN KEY ("model_id") REFERENCES "public"."ai_models"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "human_evaluation_matches" ADD CONSTRAINT "human_evaluation_matches_room_id_rooms_id_fk" FOREIGN KEY ("room_id") REFERENCES "public"."rooms"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "human_evaluation_matches" ADD CONSTRAINT "human_evaluation_matches_match_result_id_match_results_id_fk" FOREIGN KEY ("match_result_id") REFERENCES "public"."match_results"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "human_evaluation_sessions" ADD CONSTRAINT "human_evaluation_sessions_reviewer_id_users_id_fk" FOREIGN KEY ("reviewer_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "human_evaluation_sessions" ADD CONSTRAINT "human_evaluation_sessions_human_deck_id_decks_id_fk" FOREIGN KEY ("human_deck_id") REFERENCES "public"."decks"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "human_evaluation_actions_room_idx" ON "human_evaluation_actions" USING btree ("room_id","action_number");--> statement-breakpoint
CREATE INDEX "human_evaluation_matches_session_status_idx" ON "human_evaluation_matches" USING btree ("session_id","status");--> statement-breakpoint
CREATE INDEX "human_evaluation_sessions_reviewer_idx" ON "human_evaluation_sessions" USING btree ("reviewer_id","created_at");--> statement-breakpoint
CREATE UNIQUE INDEX "human_evaluation_sessions_active_reviewer_unique" ON "human_evaluation_sessions" USING btree ("reviewer_id") WHERE "human_evaluation_sessions"."status" = 'ACTIVE';--> statement-breakpoint
CREATE INDEX "ai_models_human_evaluation_idx" ON "ai_models" USING btree ("status","human_evaluation_enabled");--> statement-breakpoint
CREATE INDEX "idx_decks_user_evaluation" ON "decks" USING btree ("user_id","is_evaluation_generated");--> statement-breakpoint
ALTER TABLE "match_results" ADD CONSTRAINT "match_results_room_id_unique" UNIQUE("room_id");