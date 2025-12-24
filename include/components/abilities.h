#ifndef AZUKI_ECS_ABILITIES_H
#define AZUKI_ECS_ABILITIES_H

#include <stdint.h>
#include <flecs.h>

typedef enum {
  ABILITY_TARGET_NONE = 0,
  ABILITY_TARGET_SELF = 1,
  ABILITY_TARGET_FRIENDLY_HAND = 2,
  ABILITY_TARGET_FRIENDLY_IKZ = 3,
  ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY = 4,
  ABILITY_TARGET_FRIENDLY_ALLEY_ENTITY = 5,
  ABILITY_TARGET_FRIENDLY_ENTITY_WITH_WEAPON = 6,
  ABILITY_TARGET_FRIENDLY_LEADER = 7,
  ABILITY_TARGET_ENEMY_GARDEN_ENTITY = 8,
  ABILITY_TARGET_ENEMY_LEADER = 9,
  ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY = 10,
  ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY = 11,
  ABILITY_TARGET_ANY_GARDEN_ENTITY = 12,  // 0-4 = self garden, 5-9 = opponent garden
} AbilityTargetType;

typedef struct {
  AbilityTargetType type;
  uint8_t min;
  uint8_t max;
} AbilityCostRequirements;

typedef struct {
  AbilityTargetType type;
  uint8_t min;
  uint8_t max;
} AbilityEffectRequirements;

typedef struct {
  bool is_once_per_turn;
  bool was_applied;
} AbilityRepeatContext;

typedef struct {
  void (*init_observer)(ecs_world_t,ecs_entity_t,ecs_entity_t);
  bool (*validate_all)(ecs_world_t); // informs if the ability can be run
  void (*validate_cost)(uint8_t); // validate then pass target to AbilityContext
  void (*apply_all_costs)(); // Take all targets from ability context, run through validate to get intents, apply mutations
  void (*validate_effect)(uint8_t); // Same as cost routine
  void (*apply_all_effects)();
} AbilityFunctions;

extern ECS_COMPONENT_DECLARE(AbilityRepeatContext);
extern ECS_COMPONENT_DECLARE(AbilityCostRequirements);
extern ECS_COMPONENT_DECLARE(AbilityEffectRequirements);
extern ECS_COMPONENT_DECLARE(AbilityFunctions);

/* Ability Timing Tags */
extern ECS_TAG_DECLARE(AOnPlay);
extern ECS_TAG_DECLARE(AStartOfTurn);
extern ECS_TAG_DECLARE(AEndOfTurn);
extern ECS_TAG_DECLARE(AWhenEquipping);
extern ECS_TAG_DECLARE(AWhenEquipped);
extern ECS_TAG_DECLARE(AMain);
extern ECS_TAG_DECLARE(AWhenAttacking);
extern ECS_TAG_DECLARE(AWhenAttacked);
extern ECS_TAG_DECLARE(AResponse);
extern ECS_TAG_DECLARE(AAlleyOnly);
extern ECS_TAG_DECLARE(AGardenOnly);
extern ECS_TAG_DECLARE(AOnceTurn);

/* Keyword Ability Tags */
extern ECS_TAG_DECLARE(Charge);
extern ECS_TAG_DECLARE(Defender);
extern ECS_TAG_DECLARE(Infiltrate);
extern ECS_TAG_DECLARE(Godmode);

/* Negative Condition Tags */
extern ECS_TAG_DECLARE(Frozen);
extern ECS_TAG_DECLARE(Shocked);

void azk_register_ability_components(ecs_world_t *world);
void attach_ability_components(ecs_world_t* world, ecs_entity_t card);

#endif
