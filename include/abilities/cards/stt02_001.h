#ifndef AZUKI_ABILITY_STT02_001_H
#define AZUKI_ABILITY_STT02_001_H

#include <stdbool.h>
#include <flecs.h>

#include "components/components.h"

// STT02-001 (Shao): [Response] [Once/Turn] Pay 1 IKZ: Reduce a leader's or
// entity's attack by 1 until the end of the turn.

// Validate if ability can be activated
// Returns true if enemy has at least one valid target (leader or garden entity)
bool stt02_001_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner);

// Validate an effect target (enemy leader or garden entity)
bool stt02_001_validate_effect_target(ecs_world_t* world, ecs_entity_t card,
                                       ecs_entity_t owner, ecs_entity_t target);

// Apply effect: reduce target's attack by 1 until end of turn
void stt02_001_apply_effects(ecs_world_t* world, const AbilityContext* ctx);

#endif // AZUKI_ABILITY_STT02_001_H
