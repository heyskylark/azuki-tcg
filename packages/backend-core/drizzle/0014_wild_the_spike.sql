ALTER TABLE "rooms" ADD COLUMN "ai_model_id" uuid;--> statement-breakpoint
ALTER TABLE "match_results" ADD COLUMN "ai_model_id" uuid;--> statement-breakpoint
ALTER TABLE "rooms" ADD CONSTRAINT "rooms_ai_model_id_ai_models_id_fk" FOREIGN KEY ("ai_model_id") REFERENCES "public"."ai_models"("id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "match_results" ADD CONSTRAINT "match_results_ai_model_id_ai_models_id_fk" FOREIGN KEY ("ai_model_id") REFERENCES "public"."ai_models"("id") ON DELETE set null ON UPDATE no action;