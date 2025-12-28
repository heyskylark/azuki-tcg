#include "components/abilities.h"
#include "abilities/ability_registry.h"
#include "generated/card_defs.h"

ECS_COMPONENT_DECLARE(AbilityRepeatContext);
ECS_COMPONENT_DECLARE(AbilityCostRequirements);
ECS_COMPONENT_DECLARE(AbilityEffectRequirements);
ECS_COMPONENT_DECLARE(AbilityFunctions);

ECS_TAG_DECLARE(AOnPlay);
ECS_TAG_DECLARE(AStartOfTurn);
ECS_TAG_DECLARE(AEndOfTurn);
ECS_TAG_DECLARE(AWhenEquipping);
ECS_TAG_DECLARE(AWhenEquipped);
ECS_TAG_DECLARE(AMain);
ECS_TAG_DECLARE(AWhenAttacking);
ECS_TAG_DECLARE(AWhenAttacked);
ECS_TAG_DECLARE(AResponse);
ECS_TAG_DECLARE(AAlleyOnly);
ECS_TAG_DECLARE(AGardenOnly);
ECS_TAG_DECLARE(AOnceTurn);

ECS_TAG_DECLARE(Charge);
ECS_TAG_DECLARE(Defender);
ECS_TAG_DECLARE(Infiltrate);
ECS_TAG_DECLARE(Godmode);

ECS_TAG_DECLARE(Frozen);
ECS_TAG_DECLARE(Shocked);

ECS_TAG_DECLARE(EffectImmune);

ECS_COMPONENT_DECLARE(CardConditionCountdown);

void azk_register_ability_components(ecs_world_t *world) {
  ECS_COMPONENT_DEFINE(world, AbilityRepeatContext);
  ECS_COMPONENT_DEFINE(world, AbilityCostRequirements);
  ECS_COMPONENT_DEFINE(world, AbilityEffectRequirements);
  ECS_COMPONENT_DEFINE(world, AbilityFunctions);

  ECS_TAG_DEFINE(world, AOnPlay);
  ECS_TAG_DEFINE(world, AStartOfTurn);
  ECS_TAG_DEFINE(world, AEndOfTurn);
  ECS_TAG_DEFINE(world, AWhenEquipping);
  ECS_TAG_DEFINE(world, AWhenEquipped);
  ECS_TAG_DEFINE(world, AMain);
  ECS_TAG_DEFINE(world, AWhenAttacking);
  ECS_TAG_DEFINE(world, AWhenAttacked);
  ECS_TAG_DEFINE(world, AResponse);
  ECS_TAG_DEFINE(world, AAlleyOnly);
  ECS_TAG_DEFINE(world, AGardenOnly);
  ECS_TAG_DEFINE(world, AOnceTurn);

  ECS_TAG_DEFINE(world, Charge);
  ECS_TAG_DEFINE(world, Defender);
  ECS_TAG_DEFINE(world, Infiltrate);
  ECS_TAG_DEFINE(world, Godmode);

  ECS_TAG_DEFINE(world, Frozen);
  ECS_TAG_DEFINE(world, Shocked);

  ECS_TAG_DEFINE(world, EffectImmune);

  ECS_COMPONENT_DEFINE(world, CardConditionCountdown);

  // Ensure CardConditionCountdown is copied to each instance on instantiation
  // (EcsOverride gives each instance its own mutable copy, unlike EcsInherit)
  ecs_add_pair(world, ecs_id(CardConditionCountdown), EcsOnInstantiate,
               EcsOverride);
}

void attach_ability_components(ecs_world_t* world, ecs_entity_t card) {
  const CardId* card_id = ecs_get(world, card, CardId);
  ecs_assert(card_id != 0, ECS_INVALID_PARAMETER, "CardId component not found for card %d", card);

  // Look up ability definition from registry
  const AbilityDef* ability_def = azk_get_ability_def(card_id->id);
  if (ability_def == NULL || !ability_def->has_ability) {
    return;
  }

  // Attach the timing tag if one is defined
  if (ability_def->timing_tag != 0) {
    ecs_add_id(world, card, ability_def->timing_tag);
  }
}