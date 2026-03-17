#ifndef AZUKI_ABILITY_TARGETING_H
#define AZUKI_ABILITY_TARGETING_H

#include <flecs.h>
#include <stdint.h>

#include "abilities/ability_registry.h"
#include "constants/game.h"

#define AZK_MAX_ABILITY_TARGET_CHOICES (MAX_HAND_SIZE + GARDEN_SIZE + 1)

typedef enum {
  ABILITY_TARGET_SCOPE_COST = 0,
  ABILITY_TARGET_SCOPE_EFFECT = 1,
} AbilityTargetScope;

typedef struct {
  int action_index;
  ecs_entity_t entity;
} AbilityTargetChoice;

int azk_collect_ability_target_choices(ecs_world_t *world,
                                       const AbilityDef *def,
                                       AbilityTargetScope scope,
                                       ecs_entity_t source_card,
                                       ecs_entity_t owner,
                                       AbilityTargetChoice *out, int out_cap);

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
