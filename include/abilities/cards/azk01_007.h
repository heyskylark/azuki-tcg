#ifndef AZUKI_ABILITIES_AZK01_007_H
#define AZUKI_ABILITIES_AZK01_007_H

#include "components/components.h"
#include <flecs.h>
#include <stdbool.h>

// AZK01-007 "Johnny": [On Play] Give an entity in your Garden +1 attack until
// the end of the turn.

bool azk01_007_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool azk01_007_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void azk01_007_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_007_H
