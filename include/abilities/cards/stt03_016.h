#ifndef AZUKI_ABILITIES_STT03_016_H
#define AZUKI_ABILITIES_STT03_016_H

#include "abilities/ability_registry.h"

bool stt03_016_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void stt03_016_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif
