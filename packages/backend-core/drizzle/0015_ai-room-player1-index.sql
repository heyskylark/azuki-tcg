DROP INDEX IF EXISTS "rooms_player1_active_idx";
--> statement-breakpoint
CREATE UNIQUE INDEX "rooms_player1_active_idx" ON "rooms" ("player1_id")
WHERE
  "status" NOT IN ('COMPLETED', 'ABORTED', 'CLOSED')
  AND "player1_id" IS NOT NULL
  AND "ai_model_id" IS NULL;
