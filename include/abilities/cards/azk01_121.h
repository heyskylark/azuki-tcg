#ifndef AZUKI_ABILITIES_AZK01_121_H
#define AZUKI_ABILITIES_AZK01_121_H

#include "abilities/ability_registry.h"

bool azk01_121_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void azk01_121_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif
