INSERT INTO "ai_models" ("id", "display_name", "model_key", "status")
VALUES (
  gen_random_uuid()::uuid,
  'Mid Level Prototype',
  'model_009646.pt',
  'ENABLED'
)
ON CONFLICT ("model_key") DO UPDATE
SET
  "display_name" = EXCLUDED."display_name",
  "status" = EXCLUDED."status",
  "updated_at" = now();
