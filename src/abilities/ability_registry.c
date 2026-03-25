#include "abilities/ability_registry.h"

#include "abilities/cards/azk01_002.h"
#include "abilities/cards/azk01_003.h"
#include "abilities/cards/azk01_004.h"
#include "abilities/cards/azk01_005.h"
#include "abilities/cards/azk01_006.h"
#include "abilities/cards/azk01_007.h"
#include "abilities/cards/azk01_008.h"
#include "abilities/cards/azk01_009.h"
#include "abilities/cards/azk01_010.h"
#include "abilities/cards/azk01_011.h"
#include "abilities/cards/azk01_014.h"
#include "abilities/cards/azk01_015.h"
#include "abilities/cards/azk01_016.h"
#include "abilities/cards/azk01_017.h"
#include "abilities/cards/azk01_019.h"
#include "abilities/cards/azk01_020.h"
#include "abilities/cards/azk01_021.h"
#include "abilities/cards/azk01_022.h"
#include "abilities/cards/azk01_023.h"
#include "abilities/cards/azk01_024.h"
#include "abilities/cards/azk01_026.h"
#include "abilities/cards/azk01_027.h"
#include "abilities/cards/azk01_028.h"
#include "abilities/cards/azk01_029.h"
#include "abilities/cards/azk01_030.h"
#include "abilities/cards/azk01_031.h"
#include "abilities/cards/azk01_032.h"
#include "abilities/cards/azk01_033.h"
#include "abilities/cards/azk01_034.h"
#include "abilities/cards/azk01_045.h"
#include "abilities/cards/azk01_046.h"
#include "abilities/cards/azk01_047.h"
#include "abilities/cards/azk01_048.h"
#include "abilities/cards/azk01_050.h"
#include "abilities/cards/azk01_051.h"
#include "abilities/cards/azk01_052.h"
#include "abilities/cards/azk01_053.h"
#include "abilities/cards/azk01_055.h"
#include "abilities/cards/azk01_058.h"
#include "abilities/cards/azk01_059.h"
#include "abilities/cards/azk01_056.h"
#include "abilities/cards/azk01_057.h"
#include "abilities/cards/azk01_060.h"
#include "abilities/cards/azk01_061.h"
#include "abilities/cards/azk01_062.h"
#include "abilities/cards/azk01_063.h"
#include "abilities/cards/azk01_064.h"
#include "abilities/cards/azk01_065.h"
#include "abilities/cards/azk01_066.h"
#include "abilities/cards/azk01_068.h"
#include "abilities/cards/azk01_069.h"
#include "abilities/cards/azk01_070.h"
#include "abilities/cards/azk01_071.h"
#include "abilities/cards/azk01_072.h"
#include "abilities/cards/azk01_073.h"
#include "abilities/cards/azk01_074.h"
#include "abilities/cards/azk01_075.h"
#include "abilities/cards/azk01_078.h"
#include "abilities/cards/azk01_080.h"
#include "abilities/cards/azk01_084.h"
#include "abilities/cards/azk01_085.h"
#include "abilities/cards/azk01_086.h"
#include "abilities/cards/azk01_087.h"
#include "abilities/cards/azk01_088.h"
#include "abilities/cards/azk01_089.h"
#include "abilities/cards/azk01_090.h"
#include "abilities/cards/azk01_091.h"
#include "abilities/cards/azk01_092.h"
#include "abilities/cards/azk01_093.h"
#include "abilities/cards/azk01_096.h"
#include "abilities/cards/azk01_097.h"
#include "abilities/cards/azk01_098.h"
#include "abilities/cards/azk01_100.h"
#include "abilities/cards/azk01_101.h"
#include "abilities/cards/azk01_102.h"
#include "abilities/cards/azk01_103.h"
#include "abilities/cards/azk01_104.h"
#include "abilities/cards/azk01_105.h"
#include "abilities/cards/azk01_107.h"
#include "abilities/cards/azk01_108.h"
#include "abilities/cards/azk01_110.h"
#include "abilities/cards/azk01_111.h"
#include "abilities/cards/azk01_112.h"
#include "abilities/cards/azk01_113.h"
#include "abilities/cards/azk01_114.h"
#include "abilities/cards/azk01_115.h"
#include "abilities/cards/azk01_116.h"
#include "abilities/cards/azk01_117.h"
#include "abilities/cards/azk01_118.h"
#include "abilities/cards/azk01_119.h"
#include "abilities/cards/azk01_120.h"
#include "abilities/cards/azk01_121.h"
#include "abilities/cards/azk01_122.h"
#include "abilities/cards/azk01_123.h"
#include "abilities/cards/azk01_124.h"
#include "abilities/cards/azk01_125.h"
#include "abilities/cards/azk01_126.h"
#include "abilities/cards/azk01_127.h"
#include "abilities/cards/azk01_128.h"
#include "abilities/cards/azk01_129.h"
#include "abilities/cards/azk01_036.h"
#include "abilities/cards/azk01_039.h"
#include "abilities/cards/azk01_040.h"
#include "abilities/cards/azk01_041.h"
#include "abilities/cards/azk01_042.h"
#include "abilities/cards/st01_007.h"
#include "abilities/cards/stt01_001.h"
#include "abilities/cards/stt01_002.h"
#include "abilities/cards/stt01_003.h"
#include "abilities/cards/stt01_004.h"
#include "abilities/cards/stt01_005.h"
#include "abilities/cards/stt01_006.h"
#include "abilities/cards/stt01_008.h"
#include "abilities/cards/stt01_009.h"
#include "abilities/cards/stt01_011.h"
#include "abilities/cards/stt01_012.h"
#include "abilities/cards/stt01_017.h"
#include "abilities/cards/stt01_013.h"
#include "abilities/cards/stt01_014.h"
#include "abilities/cards/stt01_015.h"
#include "abilities/cards/stt01_016.h"
#include "abilities/cards/stt02_001.h"
#include "abilities/cards/stt02_002.h"
#include "abilities/cards/stt02_003.h"
#include "abilities/cards/stt02_005.h"
#include "abilities/cards/stt02_007.h"
#include "abilities/cards/stt02_009.h"
#include "abilities/cards/stt02_010.h"
#include "abilities/cards/stt02_011.h"
#include "abilities/cards/stt02_012.h"
#include "abilities/cards/stt02_013.h"
#include "abilities/cards/stt02_014.h"
#include "abilities/cards/stt02_015.h"
#include "abilities/cards/stt02_016.h"
#include "abilities/cards/stt02_017.h"
#include "abilities/cards/stt03_001.h"
#include "abilities/cards/stt03_002.h"
#include "abilities/cards/stt03_003.h"
#include "abilities/cards/stt03_004.h"
#include "abilities/cards/stt03_005.h"
#include "abilities/cards/stt03_006.h"
#include "abilities/cards/stt03_009.h"
#include "abilities/cards/stt03_010.h"
#include "abilities/cards/stt03_011.h"
#include "abilities/cards/stt03_012.h"
#include "abilities/cards/stt03_013.h"
#include "abilities/cards/stt03_014.h"
#include "abilities/cards/stt03_015.h"
#include "abilities/cards/stt03_016.h"
#include "abilities/cards/stt04_001.h"
#include "abilities/cards/stt04_002.h"
#include "abilities/cards/stt04_003.h"
#include "abilities/cards/stt04_004.h"
#include "abilities/cards/stt04_005.h"
#include "abilities/cards/stt04_007.h"
#include "abilities/cards/stt04_008.h"
#include "abilities/cards/stt04_009.h"
#include "abilities/cards/stt04_010.h"
#include "abilities/cards/stt04_012.h"
#include "abilities/cards/stt04_014.h"
#include "abilities/cards/stt04_015.h"
#include "abilities/cards/stt04_016.h"
#include "abilities/cards/stt04_017.h"
#include "components/abilities.h"

// Static registry table - most entries are empty (no ability)
// Entries are populated in azk_init_ability_registry() after tags are
// registered
static AbilityDef kAbilityRegistry[CARD_DEF_COUNT] = {0};

// Flag to track if registry has been initialized
static bool kRegistryInitialized = false;

const AbilityDef *azk_get_ability_def(CardDefId id) {
  if ((size_t)id >= CARD_DEF_COUNT) {
    return NULL;
  }
  return &kAbilityRegistry[id];
}

bool azk_has_ability(CardDefId id) {
  if ((size_t)id >= CARD_DEF_COUNT) {
    return false;
  }
  return kAbilityRegistry[id].has_ability;
}

bool azk_has_ability_with_timing(CardDefId id, ecs_id_t timing_tag) {
  if ((size_t)id >= CARD_DEF_COUNT) {
    return false;
  }
  const AbilityDef *def = &kAbilityRegistry[id];
  return def->has_ability &&
         (def->timing_tag == timing_tag ||
          def->secondary_timing_tag == timing_tag);
}

void azk_init_ability_registry(ecs_world_t *world) {
  (void)world;

  if (kRegistryInitialized) {
    return;
  }

  // STT01-001: [Main] [Once/Turn] Pay 1 IKZ: Give a friendly garden entity
  // equipped with a weapon Charge.
  kAbilityRegistry[CARD_DEF_STT01_001] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .is_once_per_turn = true,
      .ikz_cost = 1,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = stt01_001_validate,
      .validate_effect_target = stt01_001_validate_effect_target,
      .apply_effects = stt01_001_apply_effects,
  };

  // STT02-001 "Shao": [Response] [Once/Turn] Pay 1 IKZ: Reduce a leader's or
  // entity's attack by 1 until the end of the turn.
  kAbilityRegistry[CARD_DEF_STT02_001] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .is_once_per_turn = true,
      .ikz_cost = 1,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AResponse),
      .validate = stt02_001_validate,
      .validate_effect_target = stt02_001_validate_effect_target,
      .apply_effects = stt02_001_apply_effects,
  };

  // ST01-007 "Alley Guy": On Play; You may discard 1:Draw 1
  kAbilityRegistry[CARD_DEF_STT01_007] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_HAND, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = st01_007_validate,
      .validate_cost_target = st01_007_validate_cost_target,
      .validate_effect_target = NULL,
      .apply_costs = st01_007_apply_costs,
      .apply_effects = st01_007_apply_effects,
  };

  // STT01-008: When equipped with a weapon card, this card has +1 attack.
  // This is a passive observer-based ability - no timing tag needed.
  kAbilityRegistry[CARD_DEF_STT01_008] = (AbilityDef){
      .has_ability = true,
      .init_passive_observers = stt01_008_init_passive_observers,
      .cleanup_passive_observers = stt01_008_cleanup_passive_observers,
  };

  // STT01-009: If there are 6 or more weapon cards in your discard pile,
  // this card has +2 attack.
  kAbilityRegistry[CARD_DEF_STT01_009] = (AbilityDef){
      .has_ability = true,
      .init_passive_observers = stt01_009_init_passive_observers,
      .cleanup_passive_observers = stt01_009_cleanup_passive_observers,
  };

  // STT01-011 "Raizan": As long as this card is in play, the card Ikazuchi
  // (STT01-016 "Raizan's Zanbato") has +5 attack instead of +4.
  kAbilityRegistry[CARD_DEF_STT01_011] = (AbilityDef){
      .has_ability = true,
      .init_passive_observers = stt01_011_init_passive_observers,
      .cleanup_passive_observers = stt01_011_cleanup_passive_observers,
  };

  // STT01-003 "Crate Rat Kurobo": On Play; Put 3 cards from the top of your
  // deck into your discard pile. If you have no weapon cards in your discard
  // pile when you activate this ability, put 5 cards instead.
  kAbilityRegistry[CARD_DEF_STT01_003] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt01_003_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = NULL,
      .apply_effects = stt01_003_apply_effects,
  };

  // STT01-004: "On Play; You may discard a weapon card: look at the top 5 cards
  // of your deck, reveal up to 1 weapon card and add it to your hand, then
  // bottom deck the rest in any order"
  kAbilityRegistry[CARD_DEF_STT01_004] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .selection_pick_is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_HAND_WEAPON,
                   .min = 1,
                   .max = 1},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt01_004_validate,
      .validate_cost_target = stt01_004_validate_cost_target,
      .validate_effect_target = NULL,
      .apply_costs = stt01_004_apply_costs,
      .on_cost_paid = stt01_004_on_cost_paid,
      .validate_selection_target = stt01_004_validate_selection_target,
      .on_selection_complete = stt01_004_on_selection_complete,
      .apply_effects = NULL,
  };

  // STT02-007 "Benzai the Merchant": On Play; Draw 1
  kAbilityRegistry[CARD_DEF_STT02_007] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt02_007_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = NULL,
      .apply_effects = stt02_007_apply_effects,
  };

  // STT02-003 "Hayabusa Itto": [On Play] Look at the top 5 cards of your deck,
  // reveal up to 1 (Watercrafting) card and add it to your hand, then bottom
  // deck the rest in any order
  kAbilityRegistry[CARD_DEF_STT02_003] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .selection_pick_is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt02_003_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = NULL,
      .on_cost_paid = stt02_003_on_cost_paid,
      .validate_selection_target = stt02_003_validate_selection_target,
      .on_selection_complete = stt02_003_on_selection_complete,
      .apply_effects = NULL,
  };

  // STT02-014 "Chilling Water": [Main] Freeze an entity with cost <= 2 in
  // opponent's garden for 2 turns
  kAbilityRegistry[CARD_DEF_STT02_014] = (AbilityDef){
      .has_ability = true,
      .is_optional = false, // Spells are not optional once cast
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = stt02_014_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = stt02_014_validate_effect_target,
      .apply_costs = NULL,
      .apply_effects = stt02_014_apply_effects,
  };

  // STT02-015 "Commune with Water": [Response] Return an entity with cost <= 3
  // in any Garden to its owner's hand
  kAbilityRegistry[CARD_DEF_STT02_015] = (AbilityDef){
      .has_ability = true,
      .is_optional = false, // Spells are not optional once cast
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AResponse),
      .validate = stt02_015_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = stt02_015_validate_effect_target,
      .apply_costs = NULL,
      .apply_effects = stt02_015_apply_effects,
  };

  // AZK01-003 "Black Jade Courier": [On Play] Look at the top 5 cards of your
  // deck, reveal up to 1 Black Jade subtype card other than Black Jade Courier
  // and add it to your hand, then bottom deck the rest in any order.
  kAbilityRegistry[CARD_DEF_AZK01_003] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_003_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = NULL,
      .on_cost_paid = azk01_003_on_cost_paid,
      .validate_selection_target = azk01_003_validate_selection_target,
      .on_selection_complete = azk01_003_on_selection_complete,
      .apply_effects = NULL,
  };

  // STT01-005: "Main; Alley Only; You may sacrifice this card: Draw 3 cards and
  // discard 2"
  kAbilityRegistry[CARD_DEF_STT01_005] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, // Sacrifice self is automatic
                                                // (no user selection)
                   .min = 0,
                   .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_HAND, // Select cards from
                                                           // hand to discard
                     .min = 2,
                     .max = 2},
      .timing_tag = ecs_id(AMain),
      .validate = stt01_005_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = stt01_005_validate_effect_target,
      .apply_costs = stt01_005_apply_costs,
      .apply_effects = stt01_005_apply_effects,
  };

  // STT02-009 "Aya": [On Play] You may return an entity with cost >= 2 in your
  // Garden to your hand: Return up to 1 entity with cost <= 2 in opponent's
  // Garden to its owner's hand.
  kAbilityRegistry[CARD_DEF_STT02_009] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY,
                   .min = 1,
                   .max = 1},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY,
                     .min = 0, // "up to 1"
                     .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt02_009_validate,
      .validate_cost_target = stt02_009_validate_cost_target,
      .validate_effect_target = stt02_009_validate_effect_target,
      .apply_costs = stt02_009_apply_costs,
      .apply_effects = stt02_009_apply_effects,
  };

  // STT01-013 "Black Jade Dagger": On Play; You may deal damage to your leader:
  // this card gives an additional +1 attack
  kAbilityRegistry[CARD_DEF_STT01_013] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt01_013_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = stt01_013_apply_costs,
      .apply_effects = stt01_013_apply_effects,
  };

  // STT02-010: "Garden only; whenever an entity is returned to its owner's
  // hand, you may tap this card, then draw 1. (this ability is not affected by
  // cooldown)"
  kAbilityRegistry[CARD_DEF_STT02_010] = (AbilityDef){
      .has_ability = true,
      .is_optional = true, // "you may"
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenReturnedToHand),
      .validate = stt02_010_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = stt02_010_apply_costs,
      .apply_effects = stt02_010_apply_effects,
  };

  // STT02-011: "Garden only; Main; You may sacrifice this card: choose an
  // entity in your garden; it cannot take damage from card effects until the
  // start of your next turn."
  kAbilityRegistry[CARD_DEF_STT02_011] = (AbilityDef){
      .has_ability = true,
      .is_optional = false, // Once activated, must select a target
      .cost_req = {.type = ABILITY_TARGET_NONE, // Sacrifice self is automatic
                   .min = 0,
                   .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = stt02_011_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = stt02_011_validate_effect_target,
      .apply_costs = stt02_011_apply_costs,
      .apply_effects = stt02_011_apply_effects,
  };

  // STT02-012: If the number of entities in your garden is 2 or more than the
  // number of entities in your opponent's garden, this card has +1/+1.
  // This is a passive observer-based ability - no timing tag needed.
  kAbilityRegistry[CARD_DEF_STT02_012] = (AbilityDef){
      .has_ability = true,
      .init_passive_observers = stt02_012_init_passive_observers,
      .cleanup_passive_observers = stt02_012_cleanup_passive_observers,
  };

  // STT02-005: On Play; If you played 2 other entities this turn, draw 1
  kAbilityRegistry[CARD_DEF_STT02_005] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt02_005_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = NULL,
      .apply_effects = stt02_005_apply_effects,
  };

  // STT02-013: [On Play] Look at top 3 cards, reveal up to 1 <=2 cost water
  // card and add to hand OR play to alley if entity, bottom deck rest
  kAbilityRegistry[CARD_DEF_STT02_013] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .can_select_to_alley = true,
      .can_select_to_hand = true, // Allows both add to hand AND play to alley
      .selection_pick_is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt02_013_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = NULL,
      .on_cost_paid = stt02_013_on_cost_paid,
      .validate_selection_target = stt02_013_validate_selection_target,
      .on_selection_complete = stt02_013_on_selection_complete,
      .apply_effects = NULL,
  };

  // STT02-016: [Response] Discard 1: Reduce a leader's or entity's attack by 2
  // until the end of the turn.
  kAbilityRegistry[CARD_DEF_STT02_016] = (AbilityDef){
      .has_ability = true,
      .is_optional = false, // Spells are not optional once cast
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_HAND, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AResponse),
      .validate = stt02_016_validate,
      .validate_cost_target = stt02_016_validate_cost_target,
      .validate_effect_target = stt02_016_validate_effect_target,
      .apply_costs = stt02_016_apply_costs,
      .apply_effects = stt02_016_apply_effects,
  };

  // STT02-002 "Hydromancy": On Gate Portal; untap IKZ up to portaled card's
  // gate points
  kAbilityRegistry[CARD_DEF_STT02_002] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnGatePortal),
      .validate = stt02_002_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = NULL,
      .apply_effects = stt02_002_apply_effects,
  };

  // STT01-002 "Surge": On Gate Portal; you may play from your discard pile
  // a weapon card with cost <= gate points of the portaled entity
  // Note: is_optional=false skips confirmation; the "may" is handled by
  // allowing ACT_NOOP during selection pick phase
  kAbilityRegistry[CARD_DEF_STT01_002] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .can_select_to_equip = true,
      .can_select_to_hand = false, // Only allow equip, not add to hand
      .selection_pick_is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnGatePortal),
      .validate = stt01_002_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = NULL,
      .apply_effects = NULL,
      .on_cost_paid = stt01_002_on_cost_paid,
      .validate_selection_target = stt01_002_validate_selection_target,
      .on_selection_complete = stt01_002_on_selection_complete,
  };

  // STT01-006 "Silver Current, Haruhi": [Once/Turn][When Attacking] Deal 1
  // damage to a leader or entity in your opponent's garden.
  kAbilityRegistry[CARD_DEF_STT01_006] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AWhenAttacking),
      .validate = stt01_006_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = stt01_006_validate_effect_target,
      .apply_costs = NULL,
      .apply_effects = stt01_006_apply_effects,
  };

  // AZK01-004 "Alley Thug": [When Attacking] This card gets +1 attack until
  // the end of the turn.
  kAbilityRegistry[CARD_DEF_AZK01_004] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenAttacking),
      .validate = azk01_004_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = NULL,
      .apply_effects = azk01_004_apply_effects,
  };

  // STT01-012 "Lightning Shuriken": [When Attacking] Put the top card of your
  // deck into your discard pile.
  kAbilityRegistry[CARD_DEF_STT01_012] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenAttacking),
      .validate = stt01_012_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = NULL,
      .apply_effects = stt01_012_apply_effects,
  };

  // STT01-014 "Tenshin": [On Play] Deal up to 1 damage to a leader.
  kAbilityRegistry[CARD_DEF_STT01_014] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER,
                     .min = 0, // "up to" means optional
                     .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt01_014_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = stt01_014_validate_effect_target,
      .apply_costs = NULL,
      .apply_effects = stt01_014_apply_effects,
  };

  // STT01-015 "Tenraku": [When Equipped] If you have 15 or more cards in your
  // discard pile, this card gives an additional +1 attack.
  kAbilityRegistry[CARD_DEF_STT01_015] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenEquipped),
      .validate = stt01_015_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = NULL,
      .apply_effects = stt01_015_apply_effects,
  };

  // STT01-016 "Raizan's Zanbato": [When Attacking] If equipped to a (Raizan)
  // card, deal 1 damage to all entities in your opponent's garden.
  kAbilityRegistry[CARD_DEF_STT01_016] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenAttacking),
      .validate = stt01_016_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = NULL,
      .apply_effects = stt01_016_apply_effects,
  };

  // STT02-017 "Shao's Perseverance": [Main] If your leader's Shao, return all
  // entities with cost <= 4 in opponent's garden to their owner's hand
  kAbilityRegistry[CARD_DEF_STT02_017] = (AbilityDef){
      .has_ability = true,
      .is_optional = false, // Spells are not optional once cast
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = stt02_017_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = NULL,
      .apply_effects = stt02_017_apply_effects,
  };

  // STT01-017 "Lightning Orb": [Response] Deal 1 damage to an entity in your
  // opponent's garden and 1 damage to another entity in your opponent's garden.
  kAbilityRegistry[CARD_DEF_STT01_017] = (AbilityDef){
      .has_ability = true,
      .is_optional = false, // Spells are not optional once cast
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY,
                     .min = 1,
                     .max = 2},
      .timing_tag = ecs_id(AResponse),
      .validate = stt01_017_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = stt01_017_validate_effect_target,
      .apply_costs = NULL,
      .apply_effects = stt01_017_apply_effects,
  };

  // AZK01-002 "Healing Flutter": [Main] Heal 2 to your leader.
  kAbilityRegistry[CARD_DEF_AZK01_002] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_002_validate,
      .validate_cost_target = NULL,
      .validate_effect_target = NULL,
      .apply_costs = NULL,
      .apply_effects = azk01_002_apply_effects,
  };

  // AZK01-005 "Caravan Guard": [On Play] Deal up to 1 damage to an entity in
  // your opponent's Garden.
  kAbilityRegistry[CARD_DEF_AZK01_005] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY,
                     .min = 0,
                     .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_005_validate,
      .validate_effect_target = azk01_005_validate_effect_target,
      .apply_effects = azk01_005_apply_effects,
  };

  // AZK01-006 "Gus": [When Attacked] You may return this card to your hand.
  kAbilityRegistry[CARD_DEF_AZK01_006] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenAttacked),
      .validate = azk01_006_validate,
      .apply_effects = azk01_006_apply_effects,
  };

  // AZK01-007 "Johnny": [On Play] Give an entity in your Garden +1 attack
  // until the end of the turn.
  kAbilityRegistry[CARD_DEF_AZK01_007] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_007_validate,
      .validate_effect_target = azk01_007_validate_effect_target,
      .apply_effects = azk01_007_apply_effects,
  };

  // AZK01-008 "Rainy Day Assassin": [On Play] You may sacrifice this card:
  // destroy an entity with a cost of 3 or less in your opponent's Garden.
  kAbilityRegistry[CARD_DEF_AZK01_008] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_008_validate,
      .validate_effect_target = azk01_008_validate_effect_target,
      .apply_costs = azk01_008_apply_costs,
      .apply_effects = azk01_008_apply_effects,
  };

  // AZK01-009 "The Red Bean": [Main] Give an entity with a cost of 4 or less
  // Charge until the end of the turn.
  kAbilityRegistry[CARD_DEF_AZK01_009] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_009_validate,
      .validate_effect_target = azk01_009_validate_effect_target,
      .apply_effects = azk01_009_apply_effects,
  };

  // AZK01-010 "JD": If you only have (Normal) entities in your Garden, this
  // card has +2 attack.
  kAbilityRegistry[CARD_DEF_AZK01_010] = (AbilityDef){
      .has_ability = true,
      .init_passive_observers = azk01_010_init_passive_observers,
      .cleanup_passive_observers = azk01_010_cleanup_passive_observers,
  };

  // AZK01-011 "Rooftop Hunter": [Garden Only] If this card is untapped at the
  // end of your turn, sacrifice it.
  kAbilityRegistry[CARD_DEF_AZK01_011] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AEndOfTurn),
      .validate = azk01_011_validate,
      .apply_effects = azk01_011_apply_effects,
  };

  // AZK01-014 "Trade Guild Cavalry": [When Attacking] Give another entity in
  // your Garden +2 attack until the end of the turn.
  kAbilityRegistry[CARD_DEF_AZK01_014] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AWhenAttacking),
      .validate = azk01_014_validate,
      .validate_effect_target = azk01_014_validate_effect_target,
      .apply_effects = azk01_014_apply_effects,
  };

  // AZK01-015 "Mo": Element-based on-play effect keyed off your leader.
  kAbilityRegistry[CARD_DEF_AZK01_015] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER, .min = 0, .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_015_validate,
      .validate_effect_target = azk01_015_validate_effect_target,
      .on_cost_paid = azk01_015_on_cost_paid,
      .apply_effects = azk01_015_apply_effects,
  };

  // AZK01-016 "Sleight of Hand": [Main] Draw 2, then discard 2.
  kAbilityRegistry[CARD_DEF_AZK01_016] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_HAND, .min = 0, .max = 2},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_016_validate,
      .validate_effect_target = azk01_016_validate_effect_target,
      .apply_costs = azk01_016_apply_costs,
      .on_cost_paid = azk01_016_on_cost_paid,
      .apply_effects = azk01_016_apply_effects,
  };

  // AZK01-017 "Hook Sword Strike": [Main] Deal up to 1 damage to an entity in
  // any Garden and up to 1 damage to a leader.
  kAbilityRegistry[CARD_DEF_AZK01_017] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY,
                     .min = 0,
                     .max = 2},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_017_validate,
      .validate_effect_target = azk01_017_validate_effect_target,
      .on_cost_paid = azk01_017_on_cost_paid,
      .apply_effects = azk01_017_apply_effects,
  };

  // AZK01-018 "Monk Staff of Warding": weapon grants its host -1 incoming
  // combat damage while equipped to a leader.
  ecs_entity_t monk_staff_prefab = azk_prefab_from_id(CARD_DEF_AZK01_018);
  ecs_assert(monk_staff_prefab != 0, ECS_INVALID_PARAMETER,
             "Prefab missing for AZK01-018");
  ecs_set(world, monk_staff_prefab, EquippedCombatModifier,
          {.incoming_modifier = -1,
           .outgoing_modifier = 0,
           .requires_leader = true});

  // AZK01-019 "Jay": If you only have (Normal) entities in your Garden, this
  // card has +2 health.
  kAbilityRegistry[CARD_DEF_AZK01_019] = (AbilityDef){
      .has_ability = true,
      .init_passive_observers = azk01_019_init_passive_observers,
      .cleanup_passive_observers = azk01_019_cleanup_passive_observers,
  };

  // AZK01-020 "Power of Friendship": Main and Response spell on one card.
  kAbilityRegistry[CARD_DEF_AZK01_020] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY,
                     .min = 2,
                     .max = 2},
      .timing_tag = ecs_id(AMain),
      .secondary_timing_tag = ecs_id(AResponse),
      .validate = azk01_020_validate,
      .validate_effect_target = azk01_020_validate_effect_target,
      .apply_effects = azk01_020_apply_effects,
  };

  // AZK01-021 "Mizuto": [On Play] Look at the top 5 cards of your deck,
  // reveal up to 1 Driftward card and add it to your hand, then bottom deck
  // the rest in any order.
  kAbilityRegistry[CARD_DEF_AZK01_021] = (AbilityDef){
      .has_ability = true,
      .selection_pick_is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_021_validate,
      .on_cost_paid = azk01_021_on_cost_paid,
      .validate_selection_target = azk01_021_validate_selection_target,
      .on_selection_complete = azk01_021_on_selection_complete,
  };

  // AZK01-022 "Mirage Frog": [On Play] You may discard 1: Return an entity
  // with cost <= 2 in any Garden to its owner's hand.
  kAbilityRegistry[CARD_DEF_AZK01_022] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_HAND, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_ANY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_022_validate,
      .validate_cost_target = azk01_022_validate_cost_target,
      .validate_effect_target = azk01_022_validate_effect_target,
      .apply_costs = azk01_022_apply_costs,
      .apply_effects = azk01_022_apply_effects,
  };

  // AZK01-023 "Maho": [End of Your Turn] If you have 2 or less cards in your
  // hand, draw 1.
  kAbilityRegistry[CARD_DEF_AZK01_023] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AEndOfTurn),
      .validate = azk01_023_validate,
      .apply_effects = azk01_023_apply_effects,
  };

  // AZK01-024 "Fumiko": [On Play] You may return an entity in your Garden to
  // your hand: Play an entity with cost <= 2 from your hand.
  kAbilityRegistry[CARD_DEF_AZK01_024] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .can_select_to_garden = true,
      .can_select_to_alley = true,
      .can_select_to_hand = false,
      .clear_selection_if_still_active = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_024_validate,
      .validate_cost_target = azk01_024_validate_cost_target,
      .apply_costs = azk01_024_apply_costs,
      .on_cost_paid = azk01_024_on_cost_paid,
      .validate_selection_target = azk01_024_validate_selection_target,
      .on_selection_complete = azk01_024_on_selection_complete,
  };

  // AZK01-026 "Moonlit Crane": [Response] [Once/Turn] You may discard 1:
  // Reduce a leader's or entity's attack by 1 until end of turn.
  kAbilityRegistry[CARD_DEF_AZK01_026] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_HAND, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AResponse),
      .validate = azk01_026_validate,
      .validate_cost_target = azk01_026_validate_cost_target,
      .validate_effect_target = azk01_026_validate_effect_target,
      .apply_costs = azk01_026_apply_costs,
      .apply_effects = azk01_026_apply_effects,
  };

  // AZK01-027 "Kaiya Mizumi": [On Play] You may discard up to 4 cards: Draw
  // that many cards.
  kAbilityRegistry[CARD_DEF_AZK01_027] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_HAND, .min = 1, .max = 4},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_027_validate,
      .validate_cost_target = azk01_027_validate_cost_target,
      .apply_costs = azk01_027_apply_costs,
      .apply_effects = azk01_027_apply_effects,
  };

  // AZK01-028 "Soryu no Rin": [On Play] You must discard your hand: Return
  // all other entities in each player's Garden to their owner's hand.
  kAbilityRegistry[CARD_DEF_AZK01_028] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_028_validate,
      .apply_costs = azk01_028_apply_costs,
      .apply_effects = azk01_028_apply_effects,
  };

  // AZK01-029 "Aquatic Veil": [Response] Discard 2: Reduce a leader's or
  // entity's attack by 3 until end of turn.
  kAbilityRegistry[CARD_DEF_AZK01_029] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_HAND, .min = 2, .max = 2},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AResponse),
      .validate = azk01_029_validate,
      .validate_cost_target = azk01_029_validate_cost_target,
      .validate_effect_target = azk01_029_validate_effect_target,
      .apply_costs = azk01_029_apply_costs,
      .apply_effects = azk01_029_apply_effects,
  };

  // AZK01-030 "Lotus of Paradise": [Response] Untap up to 2 IKZ.
  kAbilityRegistry[CARD_DEF_AZK01_030] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AResponse),
      .validate = azk01_030_validate,
      .apply_effects = azk01_030_apply_effects,
  };

  // AZK01-031 "Tidal Insight": [Main] Look at the top 3 cards of your deck,
  // reveal up to 1 Water card and add it to your hand, then put the rest on
  // the top or bottom of your deck in any order.
  kAbilityRegistry[CARD_DEF_AZK01_031] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .selection_pick_is_optional = true,
      .can_topdeck_selection = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_031_validate,
      .on_cost_paid = azk01_031_on_cost_paid,
      .validate_selection_target = azk01_031_validate_selection_target,
      .on_selection_complete = azk01_031_on_selection_complete,
  };

  // AZK01-032 "Rippling Recall": [Main] Return an entity with cost >= 2 in
  // your Garden to your hand: Return up to 1 entity with cost <= 4 in your
  // opponent's Garden to its owner's hand.
  kAbilityRegistry[CARD_DEF_AZK01_032] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 0, .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_032_validate,
      .validate_cost_target = azk01_032_validate_cost_target,
      .validate_effect_target = azk01_032_validate_effect_target,
      .apply_costs = azk01_032_apply_costs,
      .apply_effects = azk01_032_apply_effects,
  };

  // AZK01-033 "Elder Hoshin": [On Play] Look at the top 5 cards of your deck,
  // reveal up to 1 Steelborn card and add it to your hand, then bottom deck
  // the rest.
  kAbilityRegistry[CARD_DEF_AZK01_033] = (AbilityDef){
      .has_ability = true,
      .selection_pick_is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_033_validate,
      .on_cost_paid = azk01_033_on_cost_paid,
      .validate_selection_target = azk01_033_validate_selection_target,
      .on_selection_complete = azk01_033_on_selection_complete,
  };

  // AZK01-034 "Kira": [In Alley Only Ability] Whenever an entity in your
  // Garden is attacked, you may swap that entity and this card, then make this
  // card the new attack target.
  kAbilityRegistry[CARD_DEF_AZK01_034] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenAttacked),
      .validate = azk01_034_validate,
      .apply_effects = azk01_034_apply_effects,
  };

  // AZK01-035 "Raimaru the Stolen": This entity can be played as a response.
  kAbilityRegistry[CARD_DEF_AZK01_035] = (AbilityDef){
      .can_play_as_response_from_hand = true,
  };

  // AZK01-036 "Denmu": [When Attacked] The attacking leader or entity becomes
  // Shocked.
  kAbilityRegistry[CARD_DEF_AZK01_036] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenAttacked),
      .validate = azk01_036_validate,
      .apply_effects = azk01_036_apply_effects,
  };

  // AZK01-039 "Piko": [When Equipped] Gain Charge.
  kAbilityRegistry[CARD_DEF_AZK01_039] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenEquipped),
      .validate = azk01_039_validate,
      .apply_effects = azk01_039_apply_effects,
  };

  // AZK01-040 "Black Jade Vault Master": [When Attacked] Deal up to 1 damage
  // to a leader.
  kAbilityRegistry[CARD_DEF_AZK01_040] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER, .min = 0, .max = 1},
      .timing_tag = ecs_id(AWhenAttacked),
      .validate = azk01_040_validate,
      .validate_effect_target = azk01_040_validate_effect_target,
      .apply_effects = azk01_040_apply_effects,
  };

  // AZK01-041 "Wu Cha": [In Garden Only Ability] [Once/Turn] [Main] Pay 1
  // IKZ: Equip up to 2 weapons with cost <= 2 from your discard pile to this
  // card.
  kAbilityRegistry[CARD_DEF_AZK01_041] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .selection_pick_is_optional = true,
      .clear_selection_if_still_active = true,
      .ikz_cost = 1,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_041_validate,
      .on_cost_paid = azk01_041_on_cost_paid,
      .validate_selection_target = azk01_041_validate_selection_target,
      .on_selection_complete = azk01_041_on_selection_complete,
  };

  // AZK01-042 "Thunderclap": [Main] Deal 3, 2, and 1 damage to 3 different
  // entities in your opponent's Garden.
  kAbilityRegistry[CARD_DEF_AZK01_042] = (AbilityDef){
      .has_ability = true,
      .is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 3, .max = 3},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_042_validate,
      .validate_effect_target = azk01_042_validate_effect_target,
      .apply_effects = azk01_042_apply_effects,
  };

  // AZK01-044 "Lightning Kanabo": [Once/Turn] Whenever the equipped host deals
  // combat damage to an opponent's card, that card becomes Shocked.
  kAbilityRegistry[CARD_DEF_AZK01_044] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
  };

  // AZK01-045 "Treetop Scout": [On Play] Look at the top 5 cards of your
  // deck, reveal up to 1 Obsidian card and add it to your hand, then bottom
  // deck the rest.
  kAbilityRegistry[CARD_DEF_AZK01_045] = (AbilityDef){
      .has_ability = true,
      .selection_pick_is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_045_validate,
      .on_cost_paid = azk01_045_on_cost_paid,
      .validate_selection_target = azk01_045_validate_selection_target,
      .on_selection_complete = azk01_045_on_selection_complete,
  };

  // AZK01-046 "Mina the Geomancer": [Start of Your Turn] Deal up to 1 damage
  // to a leader while this card is in your Garden.
  kAbilityRegistry[CARD_DEF_AZK01_046] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER, .min = 0, .max = 1},
      .timing_tag = ecs_id(AStartOfTurn),
      .validate = azk01_046_validate,
      .validate_effect_target = azk01_046_validate_effect_target,
      .apply_effects = azk01_046_apply_effects,
  };

  // AZK01-047 "Shiko the Priestess": [Once/Turn] [When Attacking] Heal 1 to
  // your leader.
  kAbilityRegistry[CARD_DEF_AZK01_047] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenAttacking),
      .validate = azk01_047_validate,
      .apply_effects = azk01_047_apply_effects,
  };

  // AZK01-048 "Kale": Carapace 1.
  kAbilityRegistry[CARD_DEF_AZK01_048] = (AbilityDef){
      .has_ability = true,
      .init_passive_observers = azk01_048_init_passive_observers,
  };

  // AZK01-050 "Shroom Tender": [On Play] Heal 2 to your leader.
  kAbilityRegistry[CARD_DEF_AZK01_050] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_050_validate,
      .apply_effects = azk01_050_apply_effects,
  };

  // AZK01-051 "Chillax": [When Attacked] You may give another entity in your
  // Garden +1 health until the end of the turn.
  kAbilityRegistry[CARD_DEF_AZK01_051] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 0, .max = 1},
      .timing_tag = ecs_id(AWhenAttacked),
      .validate = azk01_051_validate,
      .validate_effect_target = azk01_051_validate_effect_target,
      .apply_effects = azk01_051_apply_effects,
  };

  // AZK01-052 "Yojin": Conditional Defender while in the Garden.
  kAbilityRegistry[CARD_DEF_AZK01_052] = (AbilityDef){
      .has_ability = true,
      .init_passive_observers = azk01_052_init_passive_observers,
      .cleanup_passive_observers = azk01_052_cleanup_passive_observers,
  };

  // AZK01-053 "Geodust Smuggler": Other entities in your Garden have +1
  // health while this card is in play.
  kAbilityRegistry[CARD_DEF_AZK01_053] = (AbilityDef){
      .has_ability = true,
      .init_passive_observers = azk01_053_init_passive_observers,
      .cleanup_passive_observers = azk01_053_cleanup_passive_observers,
  };

  // AZK01-055 "Earth Orb": [Response] Deal up to 1 damage to a leader or an
  // entity in your opponent's Garden and reduce that card's attack by 1 until
  // end of turn.
  kAbilityRegistry[CARD_DEF_AZK01_055] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY, .min = 0, .max = 1},
      .timing_tag = ecs_id(AResponse),
      .validate = azk01_055_validate,
      .validate_effect_target = azk01_055_validate_effect_target,
      .apply_effects = azk01_055_apply_effects,
  };

  // AZK01-056 "Glass Blower, Hokuto": [On Play] Look at the top 5 cards of
  // your deck, reveal up to 1 Scorchweaver card and add it to your hand, then
  // bottom deck the rest.
  kAbilityRegistry[CARD_DEF_AZK01_056] = (AbilityDef){
      .has_ability = true,
      .selection_pick_is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_056_validate,
      .on_cost_paid = azk01_056_on_cost_paid,
      .validate_selection_target = azk01_056_validate_selection_target,
      .on_selection_complete = azk01_056_on_selection_complete,
  };

  // AZK01-058 "Black Jade Warlord": [After Attacking] You may sacrifice this
  // card: Give a leader or an entity in your Garden +2 attack until end of
  // turn.
  kAbilityRegistry[CARD_DEF_AZK01_058] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AAfterAttacking),
      .validate = azk01_058_validate,
      .validate_effect_target = azk01_058_validate_effect_target,
      .apply_costs = azk01_058_apply_costs,
      .apply_effects = azk01_058_apply_effects,
  };

  // AZK01-059 "Spice": [Once/Turn] Whenever this card takes damage, give
  // another entity in your Garden +1 attack until the end of the turn.
  kAbilityRegistry[CARD_DEF_AZK01_059] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AWhenTakesDamage),
      .validate = azk01_059_validate,
      .validate_effect_target = azk01_059_validate_effect_target,
      .apply_effects = azk01_059_apply_effects,
  };

  // AZK01-057 "Lounge Siren, Saeko": [Start of Your Turn] Deal 1 damage to an
  // entity in your Garden and 1 damage to an entity in your opponent's Garden.
  kAbilityRegistry[CARD_DEF_AZK01_057] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_GARDEN_ENTITY, .min = 2, .max = 2},
      .timing_tag = ecs_id(AStartOfTurn),
      .validate = azk01_057_validate,
      .validate_effect_target = azk01_057_validate_effect_target,
      .apply_effects = azk01_057_apply_effects,
  };

  // AZK01-060 "Scarlett": [When Attacking] You may give this card Infiltrate
  // and +1 attack until end of turn. If you do, sacrifice it at end of turn.
  kAbilityRegistry[CARD_DEF_AZK01_060] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenAttacking),
      .validate = azk01_060_validate,
      .apply_effects = azk01_060_apply_effects,
  };

  // AZK01-061 "Firebrand Renji": [Once/Turn] Whenever this card takes
  // damage from 3 different sources in one turn, deal up to 3 damage to a
  // leader or entity in any Garden.
  kAbilityRegistry[CARD_DEF_AZK01_061] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY,
                     .min = 0,
                     .max = 1},
      .timing_tag = ecs_id(AWhenTakesDamage),
      .validate = azk01_061_validate,
      .apply_effects = azk01_061_apply_effects,
  };

  // AZK01-062 "Pekiro": [Once/Turn] Whenever this card would take damage
  // from an ability or spell, you may redirect that damage to another entity
  // in any Garden.
  kAbilityRegistry[CARD_DEF_AZK01_062] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_GARDEN_ENTITY, .min = 0, .max = 1},
      .timing_tag = ecs_id(AWhenTakesDamage),
      .validate = azk01_062_validate,
      .validate_effect_target = azk01_062_validate_effect_target,
      .apply_effects = azk01_062_apply_effects,
  };

  // AZK01-063 "Enzo": [On Play] If this card is played in the Garden, deal up
  // to 3 damage to a leader or entity in any Garden.
  kAbilityRegistry[CARD_DEF_AZK01_063] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY, .min = 0, .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_063_validate,
      .validate_effect_target = azk01_063_validate_effect_target,
      .apply_effects = azk01_063_apply_effects,
  };

  // AZK01-064 "Zero": When this card enters the Garden, deal 2 damage to all
  // entities in each player's Garden.
  kAbilityRegistry[CARD_DEF_AZK01_064] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_064_validate,
      .apply_effects = azk01_064_apply_effects,
  };

  // AZK01-065 "Fire Orb": [Main] Deal 3 damage to your leader: Deal up to 5
  // damage to a leader or an entity in any Garden.
  kAbilityRegistry[CARD_DEF_AZK01_065] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_065_validate,
      .validate_effect_target = azk01_065_validate_effect_target,
      .apply_costs = azk01_065_apply_costs,
      .apply_effects = azk01_065_apply_effects,
  };

  // AZK01-066 "Firestorm": [Main] Deal 2 damage to all leaders and entities
  // in each player's Garden.
  kAbilityRegistry[CARD_DEF_AZK01_066] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_066_validate,
      .apply_effects = azk01_066_apply_effects,
  };

  // AZK01-068 "Pip": [On Play] When this card is played in the Alley, draw 1,
  // then discard 1.
  kAbilityRegistry[CARD_DEF_AZK01_068] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_HAND, .min = 1, .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_068_validate,
      .validate_effect_target = azk01_068_validate_effect_target,
      .apply_costs = azk01_068_apply_costs,
      .apply_effects = azk01_068_apply_effects,
  };

  // AZK01-069 "Link": [On Play] Look at the top 5 cards of your deck, reveal
  // up to 1 Beanz card and add it to your hand, then bottom deck the rest.
  kAbilityRegistry[CARD_DEF_AZK01_069] = (AbilityDef){
      .has_ability = true,
      .selection_pick_is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_069_validate,
      .on_cost_paid = azk01_069_on_cost_paid,
      .validate_selection_target = azk01_069_validate_selection_target,
      .on_selection_complete = azk01_069_on_selection_complete,
  };

  // AZK01-070 "Mocking Dummy": [In Garden Only Ability][Response] Tap this
  // card and deal 1 damage to it: Reduce an opponent entity's attack by 1
  // until end of turn.
  kAbilityRegistry[CARD_DEF_AZK01_070] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AResponse),
      .validate = azk01_070_validate,
      .validate_effect_target = azk01_070_validate_effect_target,
      .apply_costs = azk01_070_apply_costs,
      .apply_effects = azk01_070_apply_effects,
  };

  // AZK01-071 "Alley Fetchduck": [End of Your Turn] You may bottom deck this
  // card: Draw 1.
  kAbilityRegistry[CARD_DEF_AZK01_071] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AEndOfTurn),
      .validate = azk01_071_validate,
      .apply_costs = azk01_071_apply_costs,
      .apply_effects = azk01_071_apply_effects,
  };

  // AZK01-072 "Beanz Mentor": [When Attacking] Give another Beanz entity in
  // your Garden +1 attack until the end of the turn.
  kAbilityRegistry[CARD_DEF_AZK01_072] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AWhenAttacking),
      .validate = azk01_072_validate,
      .validate_effect_target = azk01_072_validate_effect_target,
      .apply_effects = azk01_072_apply_effects,
  };

  // AZK01-073 "Top Beanz": Passive +1/+1 while your Garden only contains
  // Beanz subtype entities.
  kAbilityRegistry[CARD_DEF_AZK01_073] = (AbilityDef){
      .has_ability = true,
      .init_passive_observers = azk01_073_init_passive_observers,
      .cleanup_passive_observers = azk01_073_cleanup_passive_observers,
  };

  // AZK01-074 "Gurugumi Vanguard": [Once/Turn][Main] This card's attack
  // becomes the attack of an entity in your opponent's Garden until end of
  // turn.
  kAbilityRegistry[CARD_DEF_AZK01_074] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_074_validate,
      .validate_effect_target = azk01_074_validate_effect_target,
      .apply_effects = azk01_074_apply_effects,
  };

  // AZK01-075 "Drunken Brewmaster": [Once/Turn] You may sacrifice 2 Beanz
  // entities in your Garden: This card gets +2/+2 until end of turn.
  kAbilityRegistry[CARD_DEF_AZK01_075] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 2, .max = 2},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_075_validate,
      .validate_cost_target = azk01_075_validate_cost_target,
      .apply_costs = azk01_075_apply_costs,
      .apply_effects = azk01_075_apply_effects,
  };

  // AZK01-078 "Fermented Beanz": [When Destroyed] Deal 1 damage to a leader
  // or entity in your opponent's Garden.
  kAbilityRegistry[CARD_DEF_AZK01_078] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AWhenDestroyed),
      .validate = azk01_078_validate,
      .validate_effect_target = azk01_078_validate_effect_target,
      .apply_effects = azk01_078_apply_effects,
  };

  // AZK01-080 "Bladebound Ally": [On Play] If this card is played in the
  // Garden, give another entity in your Garden +2 attack until end of turn.
  kAbilityRegistry[CARD_DEF_AZK01_080] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_080_validate,
      .validate_effect_target = azk01_080_validate_effect_target,
      .apply_effects = azk01_080_apply_effects,
  };

  // AZK01-084 "Good Enough Replica": [Main] Return a Normal entity card with
  // cost <= 6 from your discard pile to your hand.
  kAbilityRegistry[CARD_DEF_AZK01_084] = (AbilityDef){
      .has_ability = true,
      .can_select_to_hand = true,
      .clear_selection_if_still_active = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_084_validate,
      .on_cost_paid = azk01_084_on_cost_paid,
      .validate_selection_target = azk01_084_validate_selection_target,
      .on_selection_complete = azk01_084_on_selection_complete,
  };

  // AZK01-085 "Invigorating Concoction": [Main][Response] Give an entity in
  // your Garden +2 attack until the end of the turn.
  kAbilityRegistry[CARD_DEF_AZK01_085] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AMain),
      .secondary_timing_tag = ecs_id(AResponse),
      .validate = azk01_085_validate,
      .validate_effect_target = azk01_085_validate_effect_target,
      .apply_effects = azk01_085_apply_effects,
  };

  // AZK01-086 "Forging Tricks": [Main] You may place up to 5 Weapon cards
  // from your discard pile to the bottom of your deck in any order: Until end
  // of turn, give your leader +1 attack for each card you bottom decked.
  kAbilityRegistry[CARD_DEF_AZK01_086] = (AbilityDef){
      .has_ability = true,
      .selection_pick_is_optional = true,
      .clear_selection_if_still_active = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_086_validate,
      .on_cost_paid = azk01_086_on_cost_paid,
      .validate_selection_target = azk01_086_validate_selection_target,
      .on_selection_complete = azk01_086_on_selection_complete,
  };

  // AZK01-087 "Mizuryuu's Torrent": [Main] Put up to 2 entities with a
  // combined cost of 5 or less in your opponent's Garden to the bottom of
  // their owner's deck.
  kAbilityRegistry[CARD_DEF_AZK01_087] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 0, .max = 2},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_087_validate,
      .validate_effect_target = azk01_087_validate_effect_target,
      .apply_effects = azk01_087_apply_effects,
  };

  // AZK01-088 "Pulled Under": [Main] Put an entity with cost <= 7 in your
  // opponent's Garden to the bottom of its owner's deck.
  kAbilityRegistry[CARD_DEF_AZK01_088] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_088_validate,
      .validate_effect_target = azk01_088_validate_effect_target,
      .apply_effects = azk01_088_apply_effects,
  };

  // AZK01-089 "Mizuryuu Fist Master": [In Alley Only Ability][Main] You may
  // bottom deck this card: Put up to 2 entity cards with a combined cost of 5
  // or less in your opponent's Garden to the bottom of their owner's deck.
  kAbilityRegistry[CARD_DEF_AZK01_089] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 0, .max = 2},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_089_validate,
      .validate_effect_target = azk01_089_validate_effect_target,
      .apply_costs = azk01_089_apply_costs,
      .apply_effects = azk01_089_apply_effects,
  };

  // AZK01-090 "Priestess of the Mists": [On Play] If played in the Garden,
  // you may return a friendly Garden entity with cost <= 2 to your hand.
  kAbilityRegistry[CARD_DEF_AZK01_090] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_090_validate,
      .validate_effect_target = azk01_090_validate_effect_target,
      .apply_effects = azk01_090_apply_effects,
  };

  // AZK01-091 "Bubble Adept": [In Garden Only Ability][Response] You may
  // sacrifice this card: Reduce an entity's attack by 1 until the end of your
  // opponent's turn.
  kAbilityRegistry[CARD_DEF_AZK01_091] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AResponse),
      .validate = azk01_091_validate,
      .validate_effect_target = azk01_091_validate_effect_target,
      .apply_costs = azk01_091_apply_costs,
      .apply_effects = azk01_091_apply_effects,
  };

  // AZK01-092 "Lotus of Reflection": [Main] Look at the top 5 cards of your
  // deck, reveal up to 1 Water card with cost <= 2, add it to your hand, then
  // bottom deck the rest. You may play the revealed card.
  kAbilityRegistry[CARD_DEF_AZK01_092] = (AbilityDef){
      .has_ability = true,
      .selection_pick_is_optional = true,
      .can_select_to_garden = true,
      .can_select_to_alley = true,
      .can_select_to_equip = true,
      .can_select_to_hand = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_092_validate,
      .on_cost_paid = azk01_092_on_cost_paid,
      .validate_selection_target = azk01_092_validate_selection_target,
      .on_selection_complete = azk01_092_on_selection_complete,
  };

  // AZK01-093 "Naiyara the Tideweaver": [On Play] If this card is played in
  // the Garden, put an opposing Garden entity with cost <= 4 on the bottom of
  // its owner's deck.
  kAbilityRegistry[CARD_DEF_AZK01_093] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_093_validate,
      .validate_effect_target = azk01_093_validate_effect_target,
      .apply_effects = azk01_093_apply_effects,
  };

  // AZK01-096 "Ninpo: Thunderstep": [Response] Swap an entity in your Garden
  // with an entity in your Alley. If the Garden entity was being attacked, the
  // Alley entity becomes the new target.
  kAbilityRegistry[CARD_DEF_AZK01_096] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_ALLEY_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AResponse),
      .validate = azk01_096_validate,
      .validate_cost_target = azk01_096_validate_cost_target,
      .validate_effect_target = azk01_096_validate_effect_target,
      .apply_effects = azk01_096_apply_effects,
  };

  // AZK01-097 "Black Jade Pawnbroker": [On Play] Mill 5. If any milled cards
  // are weapons, you may add 1 of them to your hand.
  kAbilityRegistry[CARD_DEF_AZK01_097] = (AbilityDef){
      .has_ability = true,
      .selection_pick_is_optional = true,
      .clear_selection_if_still_active = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_097_validate,
      .on_cost_paid = azk01_097_on_cost_paid,
      .validate_selection_target = azk01_097_validate_selection_target,
      .on_selection_complete = azk01_097_on_selection_complete,
  };

  // AZK01-098 "Arms Dealer, Kin": [On Play] If this card is played in the
  // Alley, you may tap this card: Play a weapon card with cost <= 3 from your
  // hand.
  kAbilityRegistry[CARD_DEF_AZK01_098] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .can_select_to_equip = true,
      .can_select_to_hand = false,
      .clear_selection_if_still_active = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_098_validate,
      .apply_costs = azk01_098_apply_costs,
      .on_cost_paid = azk01_098_on_cost_paid,
      .validate_selection_target = azk01_098_validate_selection_target,
      .on_selection_complete = azk01_098_on_selection_complete,
  };

  // AZK01-100 "Raizan's Riposte": [Response] Play from your discard pile a
  // weapon card with cost <= 2.
  kAbilityRegistry[CARD_DEF_AZK01_100] = (AbilityDef){
      .has_ability = true,
      .can_select_to_equip = true,
      .can_select_to_hand = false,
      .clear_selection_if_still_active = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AResponse),
      .validate = azk01_100_validate,
      .on_cost_paid = azk01_100_on_cost_paid,
      .validate_selection_target = azk01_100_validate_selection_target,
      .on_selection_complete = azk01_100_on_selection_complete,
  };

  // AZK01-101 "Sand Stands Still": [Main][Response] Give an Earth entity in
  // your Garden +3 health until the end of turn.
  kAbilityRegistry[CARD_DEF_AZK01_101] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AMain),
      .secondary_timing_tag = ecs_id(AResponse),
      .validate = azk01_101_validate,
      .validate_effect_target = azk01_101_validate_effect_target,
      .apply_effects = azk01_101_apply_effects,
  };

  // AZK01-102 "Oathstone": [Response] Give an entity with cost <= 4 Carapace
  // 1 until the end of turn.
  kAbilityRegistry[CARD_DEF_AZK01_102] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AResponse),
      .validate = azk01_102_validate,
      .validate_effect_target = azk01_102_validate_effect_target,
      .apply_effects = azk01_102_apply_effects,
  };

  // AZK01-103 "Dropline Station": [In Garden Only Ability][Main] Sacrifice an
  // untapped Earth entity in your Garden: Deal damage equal to the sacrificed
  // card's health to a leader, capped at 5 damage. If it had 3+ health, draw 1.
  kAbilityRegistry[CARD_DEF_AZK01_103] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER, .min = 1, .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_103_validate,
      .validate_cost_target = azk01_103_validate_cost_target,
      .validate_effect_target = azk01_103_validate_effect_target,
      .apply_costs = azk01_103_apply_costs,
      .apply_effects = azk01_103_apply_effects,
  };

  // AZK01-104 "Sanzu's Envoy": [On Play] If this card is played in the
  // Garden, you may discard 1: Heal 2 to your leader.
  kAbilityRegistry[CARD_DEF_AZK01_104] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_HAND, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_104_validate,
      .validate_cost_target = azk01_104_validate_cost_target,
      .apply_costs = azk01_104_apply_costs,
      .apply_effects = azk01_104_apply_effects,
  };

  // AZK01-105 "Prickly Tumbleweed": [In Garden Only Ability][Main] Sacrifice
  // this card: Deal damage equal to this card's health to a leader or an
  // entity in your opponent's Garden.
  kAbilityRegistry[CARD_DEF_AZK01_105] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_105_validate,
      .validate_effect_target = azk01_105_validate_effect_target,
      .apply_costs = azk01_105_apply_costs,
      .apply_effects = azk01_105_apply_effects,
  };

  // AZK01-107 "Offering to Stillstone": [Main] Discard 1: If you do not
  // have an IKZ token, add 1. The IKZ token expires at the end of the turn.
  kAbilityRegistry[CARD_DEF_AZK01_107] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_HAND, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_107_validate,
      .validate_cost_target = azk01_107_validate_cost_target,
      .apply_costs = azk01_107_apply_costs,
      .apply_effects = azk01_107_apply_effects,
  };

  // AZK01-108 "Crushing Weight": [Main] Deal damage equal to the health of an
  // Earth entity in your Garden to a leader or an entity in your opponent's
  // Garden, capped at 5 damage.
  kAbilityRegistry[CARD_DEF_AZK01_108] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_108_validate,
      .validate_cost_target = azk01_108_validate_cost_target,
      .validate_effect_target = azk01_108_validate_effect_target,
      .apply_effects = azk01_108_apply_effects,
  };

  // AZK01-110 "Gluttonous Devourer, Kasha": [When Attacking] You may
  // sacrifice all other entities in your Garden: Until the end of the turn,
  // this entity gains +1 attack for each entity sacrificed this way.
  kAbilityRegistry[CARD_DEF_AZK01_110] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenAttacking),
      .validate = azk01_110_validate,
      .apply_effects = azk01_110_apply_effects,
  };

  // AZK01-111 "Black Jade Decoy": [In Alley Only Ability][Main] You may
  // sacrifice this card: Deal up to 2 damage to an entity in your opponent's
  // Garden, then you may play an entity with a cost of 2 or less from your
  // hand in the Garden.
  kAbilityRegistry[CARD_DEF_AZK01_111] = (AbilityDef){
      .has_ability = true,
      .can_select_to_garden = true,
      .selection_pick_is_optional = true,
      .clear_selection_if_still_active = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 0, .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_111_validate,
      .validate_effect_target = azk01_111_validate_effect_target,
      .apply_costs = azk01_111_apply_costs,
      .on_cost_paid = azk01_111_on_cost_paid,
      .apply_effects = azk01_111_apply_effects,
      .validate_selection_target = azk01_111_validate_selection_target,
      .on_selection_complete = azk01_111_on_selection_complete,
  };

  // AZK01-112 "Enrai Shakunetsu": [On Play] You may sacrifice an untapped
  // entity in your Garden: If you control no entities in the Garden, this
  // entity gains Charge until the end of the turn.
  kAbilityRegistry[CARD_DEF_AZK01_112] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_112_validate,
      .validate_cost_target = azk01_112_validate_cost_target,
      .apply_costs = azk01_112_apply_costs,
      .apply_effects = azk01_112_apply_effects,
  };

  // AZK01-113 "Cinderwake Pursuer": [On Play] If you played 2 other cards
  // this turn, this entity gains Charge until the end of the turn.
  kAbilityRegistry[CARD_DEF_AZK01_113] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_113_validate,
      .apply_effects = azk01_113_apply_effects,
  };

  // AZK01-114 "Omen Peddler": [On Play] If this card is played in the Alley,
  // deal 2 damage to an entity in your opponent's Garden.
  kAbilityRegistry[CARD_DEF_AZK01_114] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_114_validate,
      .validate_effect_target = azk01_114_validate_effect_target,
      .apply_effects = azk01_114_apply_effects,
  };

  // AZK01-115 "Crazed Arsonist": [When Attacking] and [When Sacrificed]:
  // Deal 1 damage to all leaders.
  kAbilityRegistry[CARD_DEF_AZK01_115] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenAttacking),
      .secondary_timing_tag = ecs_id(AWhenSacrificed),
      .validate = azk01_115_validate,
      .apply_effects = azk01_115_apply_effects,
  };

  // AZK01-116 "Tenmoku Daiki": [On Play] You must deal 3 damage to your
  // leader.
  kAbilityRegistry[CARD_DEF_AZK01_116] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_116_validate,
      .apply_effects = azk01_116_apply_effects,
  };

  // AZK01-117 "Ignition Pact": [Main] Deal 2 damage to your leader: Give an
  // entity with a cost of 5 or less Charge until the end of the turn.
  kAbilityRegistry[CARD_DEF_AZK01_117] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_117_validate,
      .validate_effect_target = azk01_117_validate_effect_target,
      .apply_costs = azk01_117_apply_costs,
      .apply_effects = azk01_117_apply_effects,
  };

  // AZK01-118 "Bandit Ringleader": [On Play] You may deal 1 damage to an
  // entity in your Garden: If you played 2 other cards this turn, deal up to
  // 2 damage to an entity in your opponent's Garden.
  kAbilityRegistry[CARD_DEF_AZK01_118] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 0, .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = azk01_118_validate,
      .validate_cost_target = azk01_118_validate_cost_target,
      .validate_effect_target = azk01_118_validate_effect_target,
      .apply_costs = azk01_118_apply_costs,
      .apply_effects = azk01_118_apply_effects,
  };

  // AZK01-119 "Piko of Thousand Blades": [Once/Turn][Main] Pay 3 IKZ:
  // Friendly equipped entity gets +1 attack for each Weapon in your discard,
  // capped at +3 until end of turn.
  kAbilityRegistry[CARD_DEF_AZK01_119] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .ikz_cost = 3,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_119_validate,
      .validate_effect_target = azk01_119_validate_effect_target,
      .apply_effects = azk01_119_apply_effects,
  };

  // AZK01-120 "Stormchain Gate": On Gate Portal; you may re-equip a weapon
  // with cost <= the portaled entity's gate power to a different friendly
  // leader or Garden entity.
  kAbilityRegistry[CARD_DEF_AZK01_120] = (AbilityDef){
      .has_ability = true,
      .can_select_to_equip = true,
      .selection_to_equip_is_reequip = true,
      .selection_pick_is_optional = true,
      .clear_selection_if_still_active = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnGatePortal),
      .validate = azk01_120_validate,
      .on_cost_paid = azk01_120_on_cost_paid,
      .validate_selection_target = azk01_120_validate_selection_target,
      .on_selection_complete = azk01_120_on_selection_complete,
  };

  // AZK01-121 "Kagoro of the Burnt Path": [Once/Turn][Main] Pay 1 IKZ:
  // This card gains +1 attack for each entity you played this turn, capped at
  // +2, until the end of the turn.
  kAbilityRegistry[CARD_DEF_AZK01_121] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .ikz_cost = 1,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_121_validate,
      .apply_effects = azk01_121_apply_effects,
  };

  // AZK01-122 "Rushfire Gate": On Gate Portal; you may play an entity with
  // cost <= the portaled entity's gate power from your hand into the Garden.
  // It gains Charge until end of turn and is sacrificed at end of turn.
  kAbilityRegistry[CARD_DEF_AZK01_122] = (AbilityDef){
      .has_ability = true,
      .can_select_to_garden = true,
      .selection_pick_is_optional = true,
      .clear_selection_if_still_active = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnGatePortal),
      .validate = azk01_122_validate,
      .on_cost_paid = azk01_122_on_cost_paid,
      .validate_selection_target = azk01_122_validate_selection_target,
      .on_selection_complete = azk01_122_on_selection_complete,
  };

  // AZK01-123 "Goro Graveloth": [Once/Turn][Main] Pay 1 IKZ: Give an
  // entity in your Garden +1 health until end of turn.
  kAbilityRegistry[CARD_DEF_AZK01_123] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .ikz_cost = 1,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = azk01_123_validate,
      .validate_effect_target = azk01_123_validate_effect_target,
      .apply_effects = azk01_123_apply_effects,
  };

  // AZK01-124 "Gate of Devotion": On Gate Portal; you may sacrifice another
  // untapped entity in your Garden with cost <= the portaled entity's gate
  // power. If you do, you may deal damage equal to its health to an entity in
  // your opponent's Garden.
  kAbilityRegistry[CARD_DEF_AZK01_124] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 0, .max = 1},
      .timing_tag = ecs_id(AOnGatePortal),
      .validate = azk01_124_validate,
      .validate_cost_target = azk01_124_validate_cost_target,
      .validate_effect_target = azk01_124_validate_effect_target,
      .apply_costs = azk01_124_apply_costs,
      .apply_effects = azk01_124_apply_effects,
  };

  // AZK01-125 "Benzai the Sly": [Once/Turn][Main][Response] Pay 1 IKZ: If
  // you discarded a card this turn, the next card you play this turn costs 2
  // less IKZ.
  kAbilityRegistry[CARD_DEF_AZK01_125] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .ikz_cost = 1,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .secondary_timing_tag = ecs_id(AResponse),
      .validate = azk01_125_validate,
      .apply_effects = azk01_125_apply_effects,
  };

  // AZK01-126 "Gate of Echoed Waves": On Gate Portal; return a spell card
  // with cost <= the portaled entity's gate power from your discard pile to
  // your hand.
  kAbilityRegistry[CARD_DEF_AZK01_126] = (AbilityDef){
      .has_ability = true,
      .clear_selection_if_still_active = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnGatePortal),
      .validate = azk01_126_validate,
      .on_cost_paid = azk01_126_on_cost_paid,
      .validate_selection_target = azk01_126_validate_selection_target,
      .on_selection_complete = azk01_126_on_selection_complete,
  };

  // AZK01-127 "Sundering Strike": [Response] Deal 1 damage to an entity in
  // your opponent's Garden.
  kAbilityRegistry[CARD_DEF_AZK01_127] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AResponse),
      .validate = azk01_127_validate,
      .validate_effect_target = azk01_127_validate_effect_target,
      .apply_effects = azk01_127_apply_effects,
  };

  // AZK01-128 "Wrong Step": [Response] Destroy an attacking entity with a
  // health of 2 or less.
  kAbilityRegistry[CARD_DEF_AZK01_128] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AResponse),
      .validate = azk01_128_validate,
      .validate_effect_target = azk01_128_validate_effect_target,
      .apply_effects = azk01_128_apply_effects,
  };

  // AZK01-129 "Silk Tongue Veyla": [Once/Turn] Whenever this entity takes
  // or deals damage, deal 1 damage to all leaders.
  kAbilityRegistry[CARD_DEF_AZK01_129] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenTakesDamage),
      .secondary_timing_tag = ecs_id(AWhenDealsDamage),
      .validate = azk01_129_validate,
      .apply_effects = azk01_129_apply_effects,
  };

  // STT03-001 "Bobu": [Once/Turn][Main] Pay 1 IKZ: Until the start of your
  // next turn, the first time an Earth entity in your Garden or Alley is
  // destroyed or sacrificed, heal 1 to your leader.
  kAbilityRegistry[CARD_DEF_STT03_001] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .ikz_cost = 1,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = stt03_001_validate,
      .apply_effects = stt03_001_apply_effects,
  };

  // STT03-002 "Stonehaven Gate": On Gate Portal; you may give a friendly
  // Garden entity with base health <= gate power Defender until the start of
  // your next turn.
  kAbilityRegistry[CARD_DEF_STT03_002] = (AbilityDef){
      .has_ability = true,
      .can_select_to_hand = false,
      .selection_pick_is_optional = false,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 0, .max = 1},
      .timing_tag = ecs_id(AOnGatePortal),
      .validate = stt03_002_validate,
      .validate_effect_target = stt03_002_validate_effect_target,
      .apply_effects = stt03_002_apply_effects,
  };

  // STT03-003 "Koyama Farm Potter": [On Play] Look at the top 5 cards of your
  // deck, reveal up to 1 Verdant card and add it to your hand, then bottom deck
  // the rest.
  kAbilityRegistry[CARD_DEF_STT03_003] = (AbilityDef){
      .has_ability = true,
      .selection_pick_is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt03_003_validate,
      .on_cost_paid = stt03_003_on_cost_paid,
      .validate_selection_target = stt03_003_validate_selection_target,
      .on_selection_complete = stt03_003_on_selection_complete,
  };

  // STT03-004 "Sloth Scarecrow": [Main] Sacrifice this card: Heal 1 to your
  // leader.
  kAbilityRegistry[CARD_DEF_STT03_004] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = stt03_004_validate,
      .apply_costs = stt03_004_apply_costs,
      .apply_effects = stt03_004_apply_effects,
  };

  // STT03-005 "Wobbly Cabbage Cart": [When Destroyed] You may destroy an
  // entity with 1 health in your opponent's Garden.
  kAbilityRegistry[CARD_DEF_STT03_005] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AWhenDestroyed),
      .validate = stt03_005_validate,
      .validate_effect_target = stt03_005_validate_effect_target,
      .apply_effects = stt03_005_apply_effects,
  };

  // STT03-006 "Cactus Farmer": [When Destroyed] Draw 1, then discard 1.
  kAbilityRegistry[CARD_DEF_STT03_006] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_HAND, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenDestroyed),
      .validate = stt03_006_validate,
      .on_cost_paid = stt03_006_on_cost_paid,
      .validate_effect_target = stt03_006_validate_effect_target,
      .apply_effects = stt03_006_apply_effects,
  };

  // STT03-009 "Warding Totem": [On Play] Move 1 IKZ from pile to area tapped.
  kAbilityRegistry[CARD_DEF_STT03_009] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt03_009_validate,
      .apply_effects = stt03_009_apply_effects,
  };

  // STT03-010 "Shroommancer": [Once/Turn][After Attacking] If this card
  // destroyed an opposing Garden entity, heal 1 to your leader.
  kAbilityRegistry[CARD_DEF_STT03_010] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AAfterAttacking),
      .validate = stt03_010_validate,
      .apply_effects = stt03_010_apply_effects,
  };

  // STT03-011 "Koyama Farm Plowman": [On Play] If played in the Garden, you
  // may destroy an opposing Garden entity with base health <= 2.
  kAbilityRegistry[CARD_DEF_STT03_011] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt03_011_validate,
      .validate_effect_target = stt03_011_validate_effect_target,
      .apply_effects = stt03_011_apply_effects,
  };

  // STT03-012 "Miharu of the White Bloom": [On Play] You may play an entity
  // with cost <= 2 from your hand into your Garden.
  kAbilityRegistry[CARD_DEF_STT03_012] = (AbilityDef){
      .has_ability = true,
      .can_select_to_garden = true,
      .selection_pick_is_optional = true,
      .clear_selection_if_still_active = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt03_012_validate,
      .on_cost_paid = stt03_012_on_cost_paid,
      .validate_selection_target = stt03_012_validate_selection_target,
      .on_selection_complete = stt03_012_on_selection_complete,
  };

  // STT03-013 "Stone Masked Ancient": [When Enters Garden] You may tap it.
  kAbilityRegistry[CARD_DEF_STT03_013] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenEntersGarden),
      .validate = stt03_013_validate,
      .apply_effects = stt03_013_apply_effects,
  };

  // STT03-014 "Sandcoil Python": [On Play] Root opposing Garden entities with
  // base attack <= 3 until the start of your next turn.
  kAbilityRegistry[CARD_DEF_STT03_014] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt03_014_validate,
      .apply_effects = stt03_014_apply_effects,
  };

  // STT03-015 "Jar of Beans": [Main] Heal 3 to your leader, or 5 if you have
  // 7 or more IKZ in your IKZ area.
  kAbilityRegistry[CARD_DEF_STT03_015] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = stt03_015_validate,
      .apply_effects = stt03_015_apply_effects,
  };

  // STT03-016 "Quicksand": [Main] Destroy all opposing Garden entities with 2
  // health or less.
  kAbilityRegistry[CARD_DEF_STT03_016] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = stt03_016_validate,
      .apply_effects = stt03_016_apply_effects,
  };

  // STT04-001 "Zero": [Once/Turn][Main] Deal 1 damage to this card: Deal 1
  // damage to a friendly Garden or Alley entity, then give it +1 attack until
  // end of turn.
  kAbilityRegistry[CARD_DEF_STT04_001] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_OR_ALLEY_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = stt04_001_validate,
      .validate_effect_target = stt04_001_validate_effect_target,
      .apply_costs = stt04_001_apply_costs,
      .apply_effects = stt04_001_apply_effects,
  };

  // STT04-002 "Ragefire Gate": On Gate Portal; you may give a friendly Garden
  // entity that took damage this turn +Attack equal to gate power until end of
  // turn.
  kAbilityRegistry[CARD_DEF_STT04_002] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 0, .max = 1},
      .timing_tag = ecs_id(AOnGatePortal),
      .validate = stt04_002_validate,
      .validate_effect_target = stt04_002_validate_effect_target,
      .apply_effects = stt04_002_apply_effects,
  };

  // STT04-003 "Cinderwake Seer": [Start of Each Turn] Deal 1 damage to this
  // card.
  kAbilityRegistry[CARD_DEF_STT04_003] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AStartOfEachTurn),
      .validate = stt04_003_validate,
      .apply_effects = stt04_003_apply_effects,
  };

  // STT04-004 "Fanatic Kindler": [On Play] You may sacrifice this card to
  // deal 1 damage to an entity in any Garden.
  kAbilityRegistry[CARD_DEF_STT04_004] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt04_004_validate,
      .validate_effect_target = stt04_004_validate_effect_target,
      .apply_costs = stt04_004_apply_costs,
      .apply_effects = stt04_004_apply_effects,
  };

  // STT04-005 "Ruby": [On Play] Look at the top 5 cards of your deck, reveal
  // up to 1 Pyreskin card and add it to your hand, then bottom deck the rest.
  kAbilityRegistry[CARD_DEF_STT04_005] = (AbilityDef){
      .has_ability = true,
      .selection_pick_is_optional = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt04_005_validate,
      .on_cost_paid = stt04_005_on_cost_paid,
      .validate_selection_target = stt04_005_validate_selection_target,
      .on_selection_complete = stt04_005_on_selection_complete,
  };

  // STT04-007 "Enraged Howler": [Once/Turn] Whenever this entity takes damage,
  // it gets +1 attack until end of turn.
  kAbilityRegistry[CARD_DEF_STT04_007] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AWhenTakesDamage),
      .validate = stt04_007_validate,
      .apply_effects = stt04_007_apply_effects,
  };

  // STT04-008 "Lady Emberheart": [Once/Turn][After Attacking] On your turn,
  // after this card attacks an entity, you may untap it.
  kAbilityRegistry[CARD_DEF_STT04_008] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AAfterAttacking),
      .validate = stt04_008_validate,
      .apply_effects = stt04_008_apply_effects,
  };

  // STT04-009 "Cinderwake Ritualist": [In Garden Only Ability][Once/Turn]
  // Whenever this card takes damage from card effects, you may deal that much
  // damage, capped at 2, to another leader or Garden entity.
  kAbilityRegistry[CARD_DEF_STT04_009] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AWhenTakesDamage),
      .validate = stt04_009_validate,
      .validate_effect_target = stt04_009_validate_effect_target,
      .apply_effects = stt04_009_apply_effects,
  };

  // STT04-010 "Reckless Tinkerer": [On Play] Deal 1 damage to this card.
  kAbilityRegistry[CARD_DEF_STT04_010] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt04_010_validate,
      .apply_effects = stt04_010_apply_effects,
  };

  // STT04-012 "Spiteful Raider": [Once/Turn] Whenever this card takes damage,
  // deal 1 damage to a leader or entity in any Garden.
  kAbilityRegistry[CARD_DEF_STT04_012] = (AbilityDef){
      .has_ability = true,
      .is_once_per_turn = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY, .min = 1, .max = 1},
      .timing_tag = ecs_id(AWhenTakesDamage),
      .validate = stt04_012_validate,
      .validate_effect_target = stt04_012_validate_effect_target,
      .apply_effects = stt04_012_apply_effects,
  };

  // STT04-013 "Kurai the Volcano": once per turn, when an opposing Garden
  // entity is destroyed, untap this card. The actual trigger is handled from
  // the discard path.
  kAbilityRegistry[CARD_DEF_STT04_013] = (AbilityDef){
      .has_ability = true,
  };

  // STT04-014 "Scorchveil Shinobi, Suzuka": [On Play] If your leader has the
  // Scorchweaver subtype, deal 1 damage to all Garden entities, then give all
  // other friendly Garden entities +1 attack until end of turn.
  kAbilityRegistry[CARD_DEF_STT04_014] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AOnPlay),
      .validate = stt04_014_validate,
      .apply_effects = stt04_014_apply_effects,
  };

  // STT04-015 "Detonation Pact": [Main] Deal 1 damage to your leader: deal 2
  // damage to your opponent's leader.
  kAbilityRegistry[CARD_DEF_STT04_015] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .effect_req = {.type = ABILITY_TARGET_NONE, .min = 0, .max = 0},
      .timing_tag = ecs_id(AMain),
      .validate = stt04_015_validate,
      .apply_costs = stt04_015_apply_costs,
      .apply_effects = stt04_015_apply_effects,
  };

  // STT04-016 "Collateral Burst": [Main] Deal 1 damage to a friendly Garden
  // entity: deal up to 2 damage to an enemy leader or Garden entity.
  kAbilityRegistry[CARD_DEF_STT04_016] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1},
      .effect_req = {.type = ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY,
                     .min = 0,
                     .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = stt04_016_validate,
      .validate_cost_target = stt04_016_validate_cost_target,
      .validate_effect_target = stt04_016_validate_effect_target,
      .apply_costs = stt04_016_apply_costs,
      .apply_effects = stt04_016_apply_effects,
  };

  // STT04-017 "Wrath of Sinder": [Main] Sacrifice one or more entities in
  // your Garden: deal damage equal to the number sacrificed to a leader or an
  // entity in any Garden.
  kAbilityRegistry[CARD_DEF_STT04_017] = (AbilityDef){
      .has_ability = true,
      .cost_req = {.type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY,
                   .min = 1,
                   .max = GARDEN_SIZE},
      .effect_req = {.type = ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY,
                     .min = 1,
                     .max = 1},
      .timing_tag = ecs_id(AMain),
      .validate = stt04_017_validate,
      .validate_cost_target = stt04_017_validate_cost_target,
      .validate_effect_target = stt04_017_validate_effect_target,
      .apply_costs = stt04_017_apply_costs,
      .apply_effects = stt04_017_apply_effects,
  };

  kRegistryInitialized = true;
}
