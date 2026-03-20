#ifndef AZUKI_ABILITY_AZK01_004_H
#define AZUKI_ABILITY_AZK01_004_H

#include <flecs.h>
#include <stdbool.h>

#include "components/components.h"

// AZK01-004 "Alley Thug": [When Attacking] This card gets +1 attack until the
// end of the turn.

// Validate if the attacking card can receive the temporary attack buff.
bool azk01_004_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);

// Apply effect: give this card +1 attack until end of turn.
void azk01_004_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITY_AZK01_004_H
