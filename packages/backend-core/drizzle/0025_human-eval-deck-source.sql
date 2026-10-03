CREATE TYPE "public"."human_evaluation_deck_source" AS ENUM('DRAFT', 'PREMADE');--> statement-breakpoint
ALTER TABLE "ai_models" ADD COLUMN "human_evaluation_plan" jsonb;--> statement-breakpoint
ALTER TABLE "human_evaluation_matches" ADD COLUMN "deck_source" "human_evaluation_deck_source" DEFAULT 'DRAFT' NOT NULL;--> statement-breakpoint
ALTER TABLE "human_evaluation_matches" ADD COLUMN "premade_deck_slug" text;--> statement-breakpoint
ALTER TABLE "human_evaluation_matches" ADD CONSTRAINT "human_evaluation_matches_premade_deck_check" CHECK (("human_evaluation_matches"."deck_source" = 'PREMADE') = ("human_evaluation_matches"."premade_deck_slug" IS NOT NULL));