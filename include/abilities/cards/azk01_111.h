#ifndef AZUKI_ABILITIES_AZK01_111_H
#define AZUKI_ABILITIES_AZK01_111_H

#include "abilities/ability_registry.h"

bool azk01_111_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool azk01_111_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void azk01_111_apply_costs(ecs_world_t *world, const AbilityContext *ctx);
void azk01_111_on_cost_paid(ecs_world_t *world, AbilityContext *ctx);
void azk01_111_apply_effects(ecs_world_t *world, const AbilityContext *ctx);
bool azk01_111_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target);
void azk01_111_on_selection_complete(ecs_world_t *world, AbilityContext *ctx);

#endif
