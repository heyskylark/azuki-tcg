#ifndef AZUKI_ABILITIES_AZK01_014_H
#define AZUKI_ABILITIES_AZK01_014_H

#include "components/components.h"
#include <flecs.h>
#include <stdbool.h>

// AZK01-014 "Trade Guild Cavalry": [When Attacking] Give another entity in
// your Garden +2 attack until the end of the turn.

bool azk01_014_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool azk01_014_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void azk01_014_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_014_H
