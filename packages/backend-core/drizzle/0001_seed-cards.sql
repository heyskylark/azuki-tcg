-- Custom SQL migration file, put your code below! --
-- Seed card data for Azuki TCG
-- This migration inserts all 45 cards (36 base + 9 special rarity variants)

INSERT INTO "cards" (
  "id", "card_code", "name", "rarity", "special_rarity", "element", "card_type",
  "attack", "health", "gate_points", "ikz_cost", "keywords", "subtypes",
  "effect_text", "flavor_text", "image_url"
) VALUES
  -- IKZ Cards
  (gen_random_uuid()::uuid, 'IKZ-001', 'IKZ Card', 'IKZ', NULL, 'NORMAL', 'IKZ', NULL, NULL, NULL, NULL, '{}', '{}', NULL, NULL, 'S1-IKZ-001_IKZ!_IKZ_die.jpg'),
  (gen_random_uuid()::uuid, 'IKZ-002', 'Extra IKZ Card', 'IKZ', NULL, 'NORMAL', 'IKZ', NULL, NULL, NULL, NULL, '{}', '{}', NULL, NULL, 'S1-IKZ-002_IKZ!_IKZ-Token_die.jpg'),

  -- STT01 (Lightning Deck) - Leaders and Gates
  (gen_random_uuid()::uuid, 'STT01-001', 'Raizan', 'L', NULL, 'LIGHTNING', 'LEADER', 0, 20, NULL, NULL, '{}', ARRAY['Raizan', 'Steelborn'], '[Once/Turn] *Pay 1 IKZ*: Give an entity equipped with a (Weapon) card "Charge" (Can attack the same turn it enters the Garden).', NULL, 'S1-STT01-001_Raizan_L_L_die.jpg'),
  (gen_random_uuid()::uuid, 'STT01-002', 'Surge', 'G', NULL, 'LIGHTNING', 'GATE', NULL, NULL, NULL, NULL, '{}', '{}', 'TAP: Portal an untapped entity from the Alley into the Garden, then you may play from your discard pile a Weapon card with a cost equal to or less than the (Gate Power) of the portaled entity.', NULL, 'S1-STT01-002_Surge-Gate_G_G_die.jpg'),

  -- STT01 Entities
  (gen_random_uuid()::uuid, 'STT01-003', 'Crate Rat Kurobo', 'C', NULL, 'LIGHTNING', 'ENTITY', 1, 1, 1, 1, '{}', ARRAY['Black Jade'], '[On Play] Put 3 cards from the top of your deck into your discard pile. If you have no (Weapon) cards in your discard pile when you activate this ability, put 5 cards instead.', NULL, 'S1-STT01-003_Crate-Rat-Kurobo_E_C_die.jpg'),
  (gen_random_uuid()::uuid, 'STT01-004', 'Black Jade Recruit', 'C', NULL, 'NORMAL', 'ENTITY', 1, 1, 1, 1, '{}', ARRAY['Black Jade', 'Dawnling'], '[On Play] You may discard a (Weapon) card: Look at the top 5 cards of your deck, reveal up to 1 (Weapon) card and add it to your hand, then bottom deck the rest in any order.', NULL, 'S1-STT01-004_Black-Jade-Recruit_E_C_die.jpg'),
  (gen_random_uuid()::uuid, 'STT01-005', 'Alpine Prowler', 'C', NULL, 'LIGHTNING', 'ENTITY', 1, 1, 2, 2, '{}', ARRAY['Bandit'], '[Alley Only Ability] You may sacrifice this card: Draw 3 and discard 2.', NULL, 'S1-STT01-005_Alpine-Prowler_E_C_die.jpg'),
  (gen_random_uuid()::uuid, 'STT01-006', 'Silver Current, Haruhi', 'R', NULL, 'LIGHTNING', 'ENTITY', 1, 2, 1, 2, '{}', ARRAY['Elder', 'Monk'], '[Once/Turn][When Attacking] Deal up to 1 damage to a leader or entity in your opponent''s Garden.', NULL, 'S1-STT01-006_Silver-Current-Haruhi_E_R_die.jpg'),
  (gen_random_uuid()::uuid, 'STT01-007', 'Alley Guy', 'C', NULL, 'NORMAL', 'ENTITY', 1, 2, 0, 2, '{}', ARRAY['Dawnling', 'Alley Dweller'], '[On Play] You may discard 1: Draw 1.', NULL, 'S1-STT01-007_Alley-Guy_E_C_die.jpg'),
  (gen_random_uuid()::uuid, 'STT01-008', 'Black Jade Crewleader', 'UC', NULL, 'LIGHTNING', 'ENTITY', 2, 2, 1, 3, '{}', ARRAY['Black Jade', 'Steelborn'], 'When equipped with a (Weapon) card, this card has +1 attack.', NULL, 'S1-STT01-008_Black-Jade-Crewleader_E_UC_die.jpg'),
  (gen_random_uuid()::uuid, 'STT01-009', 'Weapon Master Yamada', 'UC', NULL, 'LIGHTNING', 'ENTITY', 2, 2, 2, 4, '{}', ARRAY['Steelborn'], 'If there are 6 or more (Weapon) cards in your discard pile, this card has +2 attack.', NULL, 'S1-STT01-009_Mastersmith-Yamada_E_UC_die.jpg'),
  (gen_random_uuid()::uuid, 'STT01-010', 'Indra', 'R', NULL, 'LIGHTNING', 'ENTITY', 3, 2, 3, 5, ARRAY['Charge'], ARRAY['Monk', 'Steelborn'], 'Charge (Can attack the same turn it enters the Garden)', NULL, 'S1-STT01-010_Indra_E_R_die.jpg'),
  (gen_random_uuid()::uuid, 'STT01-011', 'Raizan', 'SR', NULL, 'LIGHTNING', 'ENTITY', 4, 3, 4, 6, '{}', ARRAY['Raizan', 'Steelborn'], 'As long as this card is in play, the card (Raizan''s Zanbatou) has +5 attack instead of +4.', NULL, 'S1-STT01-011_Raizan_E_SR_die.jpg'),

  -- STT01 Weapons
  (gen_random_uuid()::uuid, 'STT01-012', 'Lightning Shuriken', 'C', NULL, 'LIGHTNING', 'WEAPON', 1, NULL, NULL, 1, '{}', ARRAY['Shuriken', 'Shadowfang'], '[When Attacking] Put the top card of your deck into your discard pile.', NULL, 'S1-STT01-012_Lightning-Shuriken_W_C_die.jpg'),
  (gen_random_uuid()::uuid, 'STT01-013', 'Black Jade Dagger', 'C', NULL, 'NORMAL', 'WEAPON', 1, NULL, NULL, 1, '{}', ARRAY['Black Jade', 'Sword'], '[When Equipping] You may deal 1 damage to your leader: This card gives an additional +1 attack.', NULL, 'S1-STT01-013_Black-Jade-Dagger_W_C_die.jpg'),
  (gen_random_uuid()::uuid, 'STT01-014', 'Tenshin', 'C', NULL, 'NORMAL', 'WEAPON', 2, NULL, NULL, 2, '{}', ARRAY['Sword'], '[When Equipping] Deal up to 1 damage to a leader.', NULL, 'S1-STT01-014_Tenshin_W_C_die.jpg'),
  (gen_random_uuid()::uuid, 'STT01-015', 'Tenraku', 'UC', NULL, 'NORMAL', 'WEAPON', 3, NULL, NULL, 3, '{}', ARRAY['Sword'], 'If you have 15 or more cards in your discard pile, this card gives an additional +1 attack.', NULL, 'S1-STT01-015_Tenraku_W_UC_die.jpg'),
  (gen_random_uuid()::uuid, 'STT01-016', 'Raizan''s Zanbato', 'SR', NULL, 'LIGHTNING', 'WEAPON', 4, NULL, NULL, 4, '{}', ARRAY['Raizan', 'Sword'], '[When Attacking] If equipped to a (Raizan) card, deal 1 damage to all entities in your opponent''s Garden.', NULL, 'S1-STT01-016_Ikazuchi_W_SR_die.jpg'),

  -- STT01 Spells
  (gen_random_uuid()::uuid, 'STT01-017', 'Lightning Orb', 'UC', NULL, 'LIGHTNING', 'SPELL', NULL, NULL, NULL, 1, '{}', ARRAY['Stormcaller', 'Orb'], '[Response] Deal up to 1 damage each to 2 different entities in your opponent''s Garden.', NULL, 'S1-STT01-017_Lightning-Orb_S_UC_die.jpg'),

  -- STT02 (Water Deck) - Leaders and Gates
  (gen_random_uuid()::uuid, 'STT02-001', 'Shao', 'L', NULL, 'WATER', 'LEADER', 0, 20, NULL, NULL, '{}', ARRAY['Shao', 'Driftward'], '[Once/Turn][Response] *Pay 1 IKZ*: Reduce a leader''s or entity''s attack by 1 until the end of the turn.', NULL, 'S1-STT02-001_Shao_L_L_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-002', 'Hydromancy', 'G', NULL, 'WATER', 'GATE', NULL, NULL, NULL, NULL, '{}', '{}', 'TAP Portal an untapped entity from the Alley into the Garden: You may untap IKZ up to the (Gate Power) of the portaled card.', NULL, 'S1-STT02-002_Hydromancy-Gate_G_G_die.jpg'),

  -- STT02 Entities
  (gen_random_uuid()::uuid, 'STT02-003', 'Hayabusa Itto', 'C', NULL, 'WATER', 'ENTITY', 1, 1, 1, 1, '{}', ARRAY['Elder', 'Sushi Chef'], '[On Play] Look at the top 5 cards of your deck, reveal up to 1 (Watercrafting) card and add it to your hand, then bottom deck the rest in any order.', NULL, 'S1-STT02-003_Hayabusa-Itto_E_C_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-004', 'Rei', 'C', NULL, 'NORMAL', 'ENTITY', 1, 2, 0, 1, '{}', ARRAY['Dawnling'], NULL, NULL, 'S1-STT02-004_Rei_E_C_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-005', 'Hayabusa Saburo', 'UC', NULL, 'WATER', 'ENTITY', 1, 1, 1, 1, '{}', ARRAY['Sushi Chef'], '[On Play] If you played 2 other entities this turn, draw 1.', NULL, 'S1-STT02-005_Hayabusa-Saburo_E_UC_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-006', 'Foamback Crab', 'C', NULL, 'WATER', 'ENTITY', 0, 1, 1, 2, ARRAY['Defender', 'Effect Immune'], ARRAY['Crab', 'Rippleborn', 'Driftward'], '"Defender" (If this card is in the Garden, you may tap it to redirect an attack to this card). This card cannot take damage from card effects.', NULL, 'S1-STT02-006_Foamback-Crab_E_C_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-007', 'Benzai the Merchant', 'C', NULL, 'WATER', 'ENTITY', 1, 2, 0, 2, '{}', ARRAY['Frog', 'Wavecaller'], '[On Play] Draw 1.', NULL, 'S1-STT02-007_Benzai-the-Merchant_E_C_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-008', 'Serene Fist, Misaki', 'UC', NULL, 'WATER', 'ENTITY', 2, 1, 0, 2, ARRAY['Effect Immune'], ARRAY['Driftward'], 'This card cannot take damage from card effects.', NULL, 'S1-STT02-008_Serene-Fist-Misaki_E_UC_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-009', 'Aya', 'C', NULL, 'WATER', 'ENTITY', 1, 2, 1, 3, '{}', ARRAY['Watercrafting', 'Water Painter'], '[On Play] You may return an entity with a cost of 2 more in your Garden to your hand: Return up to 1 entity with a cost of 2 or less in your opponent''s Garden to its owner''s hand.', NULL, 'S1-STT02-009_Aya_E_C_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-010', 'Selis of the Shore', 'R', NULL, 'WATER', 'ENTITY', 1, 3, 1, 3, '{}', ARRAY['Wavecaller'], '[Garden Only Ability] When any entity is returned to its owner''s hand, you may TAP this card, then draw 1. (This ability is not affected by Fatigue)', NULL, 'S1-STT02-010_Selis-of-the-Shore_E_R_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-011', 'Bubblemancer', 'C', NULL, 'WATER', 'ENTITY', 2, 2, 0, 3, '{}', ARRAY['Driftward'], '[Garden Only Ability] You may sacrifice this card: Choose an entity in your Garden; it cannot take damage from card effects until the start of your next turn.', NULL, 'S1-STT02-011_Bubblemancer_E_C_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-012', 'Young Shao', 'UC', NULL, 'NORMAL', 'ENTITY', 2, 2, 2, 4, '{}', ARRAY['Shao', 'Dawnling'], 'If the number of entities in your Garden is 2 or more than the number of entities in your opponent''s Garden, this card has +1 health and +1 attack.', NULL, 'S1-STT02-012_Young-Shao_E_UC_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-013', 'Mizuki', 'SR', NULL, 'WATER', 'ENTITY', 3, 4, 2, 5, '{}', ARRAY['Wavecaller'], '[On Play] Look at the top 3 cards of your deck, reveal up to 1 (Water) type card with a cost of 2 or less and add it to your hand, then bottom deck the rest in any order. You may play the card in the Alley if it is an entity card.', NULL, 'S1-STT02-013_Mizuki_E_SR_die.jpg'),

  -- STT02 Spells
  (gen_random_uuid()::uuid, 'STT02-014', 'Chilling Water', 'C', NULL, 'WATER', 'SPELL', NULL, NULL, NULL, 1, '{}', ARRAY['Subzero', 'Watercrafting'], '[Main] Choose an entity with a cost of 2 or less in your opponent''s Garden; until the start of your next turn, it is Frozen (abilities are disabled and cannot attack or be damaged).', NULL, 'S1-STT02-014_Chilling-Water_S_C_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-015', 'Commune with Water', 'UC', NULL, 'WATER', 'SPELL', NULL, NULL, NULL, 2, '{}', ARRAY['Watercrafting'], '[Response] Return an entity with a cost of 3 or less in any Garden to its owner''s hand.', NULL, 'S1-STT02-015_Commune-with-Water_S_UC_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-016', 'Water Orb', 'R', NULL, 'WATER', 'SPELL', NULL, NULL, NULL, 1, '{}', ARRAY['Watercrafting', 'Orb'], '[Response] Discard 1: Reduce a leader''s or entity''s attack by 2 until the end of the turn.', NULL, 'S1-STT02-016_Water-Orb_S_R_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-017', 'Shao''s Perseverance', 'SR', NULL, 'WATER', 'SPELL', NULL, NULL, NULL, 6, '{}', ARRAY['Watercrafting', 'Shao'], '[Main] If your leader is (Shao), return all entities with a cost of 6 or less in your opponent''s Garden to their owner''s hand.', NULL, 'S1-STT02-017_Shaos-Perseverance_S_SR_die.jpg'),

  -- Special Rarity Variants (alternate art versions)
  -- L_S variants (Leader Special)
  (gen_random_uuid()::uuid, 'STT01-001', 'Raizan', 'L_S', NULL, 'LIGHTNING', 'LEADER', 0, 20, NULL, NULL, '{}', ARRAY['Raizan', 'Steelborn'], '[Once/Turn] *Pay 1 IKZ*: Give an entity equipped with a (Weapon) card "Charge" (Can attack the same turn it enters the Garden).', NULL, 'STT01-001A_Raizan_L_AA_Die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-001', 'Shao', 'L_S', NULL, 'WATER', 'LEADER', 0, 20, NULL, NULL, '{}', ARRAY['Shao', 'Driftward'], '[Once/Turn][Response] *Pay 1 IKZ*: Reduce a leader''s or entity''s attack by 1 until the end of the turn.', NULL, 'STT02-001A_Shao_L_AA_Die.jpg'),

  -- G_S variants (Gate Special)
  (gen_random_uuid()::uuid, 'STT01-002', 'Surge', 'G_S', NULL, 'LIGHTNING', 'GATE', NULL, NULL, NULL, NULL, '{}', '{}', 'TAP: Portal an untapped entity from the Alley into the Garden, then you may play from your discard pile a Weapon card with a cost equal to or less than the (Gate Power) of the portaled entity.', NULL, 'STT01-002A_Surge-Gate_G_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-002', 'Hydromancy', 'G_S', NULL, 'WATER', 'GATE', NULL, NULL, NULL, NULL, '{}', '{}', 'TAP Portal an untapped entity from the Alley into the Garden: You may untap IKZ up to the (Gate Power) of the portaled card.', NULL, 'STT02-002A_Hydromancy-Gate_G_die.jpg'),

  -- SR_S variants (Super Rare Special)
  (gen_random_uuid()::uuid, 'STT01-011', 'Raizan', 'SR_S', NULL, 'LIGHTNING', 'ENTITY', 4, 3, 4, 6, '{}', ARRAY['Raizan', 'Steelborn'], 'As long as this card is in play, the card (Raizan''s Zanbatou) has +5 attack instead of +4.', NULL, 'STT01-011A_Raizan_E_SR_die.jpg'),
  (gen_random_uuid()::uuid, 'STT01-016', 'Raizan''s Zanbato', 'SR_S', NULL, 'LIGHTNING', 'WEAPON', 4, NULL, NULL, 4, '{}', ARRAY['Raizan', 'Sword'], '[When Attacking] If equipped to a (Raizan) card, deal 1 damage to all entities in your opponent''s Garden.', NULL, 'STT01-016A_Ikazuchi_W_SR_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-013', 'Mizuki', 'SR_S', NULL, 'WATER', 'ENTITY', 3, 4, 2, 5, '{}', ARRAY['Wavecaller'], '[On Play] Look at the top 3 cards of your deck, reveal up to 1 (Water) type card with a cost of 2 or less and add it to your hand, then bottom deck the rest in any order. You may play the card in the Alley if it is an entity card.', NULL, 'STT02-013A_Mizuki_E_SR_die.jpg'),
  (gen_random_uuid()::uuid, 'STT02-017', 'Shao''s Perseverance', 'SR_S', NULL, 'WATER', 'SPELL', NULL, NULL, NULL, 6, '{}', ARRAY['Watercrafting', 'Shao'], '[Main] If your leader is (Shao), return all entities with a cost of 6 or less in your opponent''s Garden to their owner''s hand.', NULL, 'STT02-017A_Shaos-Perseverance_S_SR_die.jpg'),

  -- SR_SS variant (Super Rare Super Special)
  (gen_random_uuid()::uuid, 'STT02-013', 'Mizuki', 'SR_SS', NULL, 'WATER', 'ENTITY', 3, 4, 2, 5, '{}', ARRAY['Wavecaller'], '[On Play] Look at the top 3 cards of your deck, reveal up to 1 (Water) type card with a cost of 2 or less and add it to your hand, then bottom deck the rest in any order. You may play the card in the Alley if it is an entity card.', NULL, 'STT02-013ASN_Mizuki_E_SR_die.jpg');
