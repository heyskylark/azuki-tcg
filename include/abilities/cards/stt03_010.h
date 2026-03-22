#ifndef AZUKI_ABILITIES_STT03_010_H
#define AZUKI_ABILITIES_STT03_010_H

#include "abilities/ability_registry.h"

bool stt03_010_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void stt03_010_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif
