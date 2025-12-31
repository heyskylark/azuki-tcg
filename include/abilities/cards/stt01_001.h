#ifndef AZUKI_ABILITY_STT01_001_H
#define AZUKI_ABILITY_STT01_001_H

#include <stdbool.h>
#include <flecs.h>

#include "components/components.h"

// STT01-001: [Main] [Once/Turn] Pay 1 IKZ: Give a friendly garden entity
// equipped with a weapon Charge.

// Validate if ability can be activated
// Returns true if any garden entity has a weapon equipped and is on cooldown
bool stt01_001_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner);

// Validate an effect target (friendly garden entity with weapon and on cooldown)
bool stt01_001_validate_effect_target(ecs_world_t* world, ecs_entity_t card,
                                       ecs_entity_t owner, ecs_entity_t target);

// Apply effect: add Charge tag and clear cooldown
void stt01_001_apply_effects(ecs_world_t* world, const AbilityContext* ctx);

#endif // AZUKI_ABILITY_STT01_001_H
