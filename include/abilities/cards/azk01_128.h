#ifndef AZUKI_ABILITIES_AZK01_128_H
#define AZUKI_ABILITIES_AZK01_128_H

#include "abilities/ability_registry.h"

bool azk01_128_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool azk01_128_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void azk01_128_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif
