#ifndef AZUKI_ABILITY_TARGETING_H
#define AZUKI_ABILITY_TARGETING_H

#include <flecs.h>
#include <stdint.h>

#include "abilities/ability_registry.h"

typedef enum {
  ABILITY_TARGET_SCOPE_COST = 0,
  ABILITY_TARGET_SCOPE_EFFECT = 1,
} AbilityTargetScope;

// Visits every valid target in deterministic action order and returns the total
// count. A NULL visitor performs the same traversal without emitting targets.
typedef void (*AbilityTargetVisitorFn)(int action_index, ecs_entity_t entity,
                                       void *user_data);

int azk_visit_ability_target_choices(ecs_world_t *world, const AbilityDef *def,
                                     AbilityTargetScope scope,
                                     ecs_entity_t source_card,
                                     ecs_entity_t owner,
                                     AbilityTargetVisitorFn visitor,
                                     void *user_data);

uint8_t azk_count_ability_target_choices(ecs_world_t *world,
                                         const AbilityDef *def,
                                         AbilityTargetScope scope,
                                         ecs_entity_t source_card,
                                         ecs_entity_t owner);

ecs_entity_t azk_resolve_ability_target_choice_entity(ecs_world_t *world,
                                                      const AbilityDef *def,
                                                      AbilityTargetScope scope,
                                                      ecs_entity_t owner,
                                                      int action_index);

#endif // AZUKI_ABILITY_TARGETING_H
