#ifndef AZUKI_ABILITIES_STT04_010_H
#define AZUKI_ABILITIES_STT04_010_H

#include "abilities/ability_registry.h"

bool stt04_010_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void stt04_010_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif
