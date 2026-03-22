#ifndef AZUKI_ABILITIES_AZK01_005_H
#define AZUKI_ABILITIES_AZK01_005_H

#include "components/components.h"
#include <flecs.h>
#include <stdbool.h>

// AZK01-005 "Caravan Guard": [On Play] Deal up to 1 damage to an entity in
// your opponent's Garden.

bool azk01_005_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool azk01_005_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void azk01_005_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_005_H
