-- Correct Sundering Strike's element across all printings.
UPDATE "cards"
SET "element" = 'NORMAL'
WHERE "card_code" = 'AZK01-127';