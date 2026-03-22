#ifndef AZUKI_ABILITIES_AZK01_071_H
#define AZUKI_ABILITIES_AZK01_071_H

#include "abilities/ability_registry.h"

bool azk01_071_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void azk01_071_apply_costs(ecs_world_t *world, const AbilityContext *ctx);
void azk01_071_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_071_H
