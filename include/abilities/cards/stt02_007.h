#ifndef AZUKI_ABILITY_STT02_007_H
#define AZUKI_ABILITY_STT02_007_H

#include <stdbool.h>
#include <flecs.h>

#include "components/components.h"

// STT02-007 "Benzai the Merchant": On Play; Draw 1

// Validate if ability can be activated
// Always returns true - no validation needed for this simple effect
bool stt02_007_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner);

// Apply effect: draw 1 card
void stt02_007_apply_effects(ecs_world_t* world, const AbilityContext* ctx);

#endif // AZUKI_ABILITY_STT02_007_H
