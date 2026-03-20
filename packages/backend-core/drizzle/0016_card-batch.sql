-- Custom SQL migration file, put your code below! --
-- Seed card batch

INSERT INTO "cards" (
  "id", "card_code", "name", "rarity", "special_rarity", "element", "card_type",
  "attack", "health", "gate_points", "ikz_cost", "keywords", "subtypes",
  "effect_text", "flavor_text", "image_url"
) VALUES
(
  gen_random_uuid()::uuid,
  'AZK01-001',
  'Penny',
  'C',
  NULL,
  'NORMAL',
  'ENTITY',
  0,
  1,
  0,
  1,
  ARRAY['Defender'],
  ARRAY['Beanz'],
  '[Defender] (If this card is in the Garden, you may tap it to redirect an attack to this card)',
  NULL,
  'S1-AZK01-001_Penny_E_C_die.jpg'
),
(
  gen_random_uuid()::uuid,
  'AZK01-002',
  'Healing Flutter',
  'UC',
  NULL,
  'NORMAL',
  'SPELL',
  NULL,
  NULL,
  NULL,
  1,
  ARRAY[]::text[],
  ARRAY['Beanz'],
  '[Main] Heal 2 to your leader',
  NULL,
  'S1-AZK01-002_Healing-Flutter_S_UC_die.jpg'
),
(
  gen_random_uuid()::uuid,
  'AZK01-003',
  'Black Jade Courier',
  'C',
  NULL,
  'NORMAL',
  'ENTITY',
  1,
  1,
  0,
  1,
  ARRAY[]::text[],
  ARRAY['Black Jade', 'Strider'],
  '[On Play] Look at the top 5 cards of your deck, reveal up to 1 Black Jade subtype card other than Black Jade Courier and add it to your hand, then bottom deck the rest in any order.',
  NULL,
  'S1-AZK01-003_Black-Jade-Courier_E_C_die.jpg'
);
