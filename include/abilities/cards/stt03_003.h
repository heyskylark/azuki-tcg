#ifndef AZUKI_ABILITIES_STT03_003_H
#define AZUKI_ABILITIES_STT03_003_H

#include "abilities/ability_registry.h"

bool stt03_003_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void stt03_003_on_cost_paid(ecs_world_t *world, AbilityContext *ctx);
bool stt03_003_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target);
void stt03_003_on_selection_complete(ecs_world_t *world, AbilityContext *ctx);

#endif
