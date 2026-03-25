-- Custom SQL migration file, put your code below! --
UPDATE "cards"
SET "ikz_cost" = 4,
    "updated_at" = now()
WHERE "card_code" = 'STT03-015';
