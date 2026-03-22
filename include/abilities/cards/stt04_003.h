#ifndef AZUKI_ABILITIES_STT04_003_H
#define AZUKI_ABILITIES_STT04_003_H

#include "abilities/ability_registry.h"

bool stt04_003_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void stt04_003_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif
