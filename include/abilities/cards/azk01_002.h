#ifndef AZUKI_ABILITY_AZK01_002_H
#define AZUKI_ABILITY_AZK01_002_H

#include <flecs.h>
#include <stdbool.h>

#include "components/components.h"

// AZK01-002 "Healing Flutter": [Main] Heal 2 to your leader.

// Validate if ability can be activated.
// Always returns true.
bool azk01_002_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);

// Apply effect: heal the owner's leader by up to 2 HP.
void azk01_002_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITY_AZK01_002_H
