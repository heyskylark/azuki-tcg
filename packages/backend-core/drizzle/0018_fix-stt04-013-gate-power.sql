-- Custom SQL migration file, put your code below! --
UPDATE "cards"
SET "gate_points" = 1,
    "updated_at" = now()
WHERE "card_code" = 'STT04-013';
