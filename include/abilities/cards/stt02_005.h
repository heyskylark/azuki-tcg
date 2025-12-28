#ifndef AZUKI_ABILITY_STT02_005_H
#define AZUKI_ABILITY_STT02_005_H

#include <stdbool.h>
#include <flecs.h>

#include "components/components.h"

// STT02-005: On Play; If you played 2 other entities this turn, draw 1

// Validate if ability can be activated
// Returns true if player has played 2+ other entities this turn
bool stt02_005_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner);

// Apply effect: draw 1 card
void stt02_005_apply_effects(ecs_world_t* world, const AbilityContext* ctx);

#endif // AZUKI_ABILITY_STT02_005_H
