#include "abilities/ability_registry.h"

#include "abilities/cards/st01_007.h"
#include "abilities/cards/stt01_004.h"
#include "abilities/cards/stt01_005.h"
#include "abilities/cards/stt01_013.h"
#include "abilities/cards/stt02_007.h"
#include "abilities/cards/stt02_009.h"
#include "abilities/cards/stt02_014.h"
#include "abilities/cards/stt02_015.h"
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

void azk_init_ability_registry(ecs_world_t *world) {
  (void)world;

  if (kRegistryInitialized) {
    return;
  }

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

  // STT01-004: "On Play; You may discard a weapon card: look at the top 5 cards
  // of your deck, reveal up to 1 weapon card and add it to your hand, then
  // bottom deck the rest in any order"
  kAbilityRegistry[CARD_DEF_STT01_004] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
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

  kRegistryInitialized = true;
}
