#ifndef AZUKI_ABILITIES_AZK01_036_H
#define AZUKI_ABILITIES_AZK01_036_H

#include "abilities/ability_registry.h"

bool azk01_036_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void azk01_036_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_036_H
