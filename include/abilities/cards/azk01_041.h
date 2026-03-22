#ifndef AZUKI_ABILITIES_AZK01_041_H
#define AZUKI_ABILITIES_AZK01_041_H

#include "abilities/ability_registry.h"

bool azk01_041_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void azk01_041_on_cost_paid(ecs_world_t *world, AbilityContext *ctx);
bool azk01_041_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target);
void azk01_041_on_selection_complete(ecs_world_t *world, AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_041_H
