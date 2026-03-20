-- Custom SQL migration file, put your code below! --
-- Seed card batch

INSERT INTO "cards" (
  "id", "card_code", "name", "rarity", "special_rarity", "element", "card_type",
  "attack", "health", "gate_points", "ikz_cost", "keywords", "subtypes",
  "effect_text", "flavor_text", "image_url"
) VALUES (
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
);
