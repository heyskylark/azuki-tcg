#ifndef AZUKI_ABILITIES_AZK01_120_H
#define AZUKI_ABILITIES_AZK01_120_H

#include "abilities/ability_registry.h"

bool azk01_120_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void azk01_120_on_cost_paid(ecs_world_t *world, AbilityContext *ctx);
bool azk01_120_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target);
void azk01_120_on_selection_complete(ecs_world_t *world, AbilityContext *ctx);

#endif
