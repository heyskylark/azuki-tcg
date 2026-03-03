CREATE TYPE "public"."ai_model_status" AS ENUM('ENABLED', 'ADMIN', 'DISABLED');--> statement-breakpoint
CREATE TABLE "ai_models" (
	"id" uuid PRIMARY KEY NOT NULL,
	"display_name" text NOT NULL,
	"model_key" text NOT NULL,
	"status" "ai_model_status" DEFAULT 'ENABLED' NOT NULL,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "ai_models_model_key_unique" UNIQUE("model_key")
);
--> statement-breakpoint
CREATE INDEX "ai_models_status_idx" ON "ai_models" USING btree ("status");--> statement-breakpoint
CREATE INDEX "ai_models_display_name_idx" ON "ai_models" USING btree ("display_name");