#include "abilities/ability_registry.h"

#include "abilities/cards/st01_007.h"
#include "abilities/cards/stt01_001.h"
#include "abilities/cards/stt01_003.h"
#include "abilities/cards/stt01_004.h"
#include "abilities/cards/stt01_005.h"
#include "abilities/cards/stt01_006.h"
#include "abilities/cards/stt01_008.h"
#include "abilities/cards/stt01_012.h"
#include "abilities/cards/stt01_017.h"
#include "abilities/cards/stt01_013.h"
#include "abilities/cards/stt01_014.h"
#include "abilities/cards/stt01_016.h"
#include "abilities/cards/stt02_001.h"
#include "abilities/cards/stt02_002.h"
#include "abilities/cards/stt02_003.h"
#include "abilities/cards/stt02_005.h"
#include "abilities/cards/stt02_007.h"
#include "abilities/cards/stt02_009.h"
#include "abilities/cards/stt02_010.h"
#include "abilities/cards/stt02_011.h"
#include "abilities/cards/stt02_013.h"
#include "abilities/cards/stt02_014.h"
#include "abilities/cards/stt02_015.h"
#include "abilities/cards/stt02_016.h"
#include "abilities/cards/stt02_017.h"
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
  return def->has_ability && def->timing_tag == timing_tag;
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

  // STT01-003 "Crate Rat Kurobo": On Play; You may put 3 cards from the top of
  // your deck into your discard pile. If you have no weapon cards in your
  // discard pile, put 5 cards instead.
  kAbilityRegistry[CARD_DEF_STT01_003] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
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
      .is_optional = true,
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

  // STT02-005: On Play; If you played 2 other entities this turn, draw 1
  kAbilityRegistry[CARD_DEF_STT02_005] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
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
      .is_optional = true,
      .can_select_to_alley = true,
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

  // STT01-012 "Lightning Shuriken": [When Attacking] Put the top card of your
  // deck into your discard pile.
  kAbilityRegistry[CARD_DEF_STT01_012] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
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
      .is_optional = true,
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
  // STT01-016 "Raizan's Zanbato": [When Attacking] If equipped to a (Raizan)
  // card, deal 1 damage to all entities in your opponent's garden.
  kAbilityRegistry[CARD_DEF_STT01_016] = (AbilityDef){
      .has_ability = true,
      .is_optional = true,
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

  kRegistryInitialized = true;
}
