UPDATE "cards"
SET "ikz_cost" = 0,
    "updated_at" = now()
WHERE "card_code" = 'AZK01-107';
