#ifndef AZUKI_ABILITIES_STT04_005_H
#define AZUKI_ABILITIES_STT04_005_H

#include "abilities/ability_registry.h"

bool stt04_005_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void stt04_005_on_cost_paid(ecs_world_t *world, AbilityContext *ctx);
bool stt04_005_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target);
void stt04_005_on_selection_complete(ecs_world_t *world, AbilityContext *ctx);

#endif
