#ifndef AZUKI_ABILITIES_AZK01_126_H
#define AZUKI_ABILITIES_AZK01_126_H

#include "abilities/ability_registry.h"

bool azk01_126_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void azk01_126_on_cost_paid(ecs_world_t *world, AbilityContext *ctx);
bool azk01_126_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target);
void azk01_126_on_selection_complete(ecs_world_t *world, AbilityContext *ctx);

#endif
