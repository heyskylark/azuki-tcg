#ifndef AZUKI_ABILITIES_AZK01_009_H
#define AZUKI_ABILITIES_AZK01_009_H

#include "components/components.h"
#include <flecs.h>
#include <stdbool.h>

// AZK01-009 "The Red Bean": [Main] Give an entity with a cost of 4 or less
// Charge until the end of the turn.

bool azk01_009_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool azk01_009_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void azk01_009_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_009_H
