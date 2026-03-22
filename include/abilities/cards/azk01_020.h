#ifndef AZUKI_ABILITIES_AZK01_020_H
#define AZUKI_ABILITIES_AZK01_020_H

#include "components/components.h"
#include <flecs.h>
#include <stdbool.h>

// AZK01-020 "Power of Friendship": [Main] Give 2 entities in your Garden each
// +1 attack until the end of the turn. [Response] Give 2 entities in your
// Garden each +1 health until the end of the turn.

bool azk01_020_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool azk01_020_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void azk01_020_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_020_H
