INSERT INTO "cards" (
  "id", "card_code", "name", "rarity", "special_rarity", "element", "card_type",
  "attack", "health", "gate_points", "ikz_cost", "keywords", "subtypes",
  "effect_text", "flavor_text", "image_url"
) VALUES (
  gen_random_uuid()::uuid,
  'STT03-017',
  'Sprout of Fortune',
  'C',
  NULL,
  'EARTH',
  'SPELL',
  NULL,
  NULL,
  NULL,
  3,
  ARRAY[]::text[],
  ARRAY['Verdant'],
  '[Main] Choose 1 - add 1 IKZ from your IKZ pile to your IKZ area; the IKZ enters the field tapped. Then, heal up to 1 to your leader. - Draw 1.',
  NULL,
  '/cards/S1-STT03-017_Sprout-of-Fortune_S_C_die.jpg'
)
ON CONFLICT ("card_code", "rarity", "special_rarity") DO NOTHING;
