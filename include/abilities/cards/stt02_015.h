#ifndef AZUKI_ABILITY_STT02_015_H
#define AZUKI_ABILITY_STT02_015_H

#include <stdbool.h>
#include <flecs.h>

#include "components/components.h"

// STT02-015 "Commune with Water": [Response] Return an entity with cost <= 3 in any Garden to its owner's hand

// Validate if ability can be activated
// Returns true if there's at least one entity with cost <= 3 in any garden
bool stt02_015_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner);

// Validate an effect target (entity in any garden with cost <= 3)
bool stt02_015_validate_effect_target(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner, ecs_entity_t target);

// Apply effect: return the target entity to its owner's hand
void stt02_015_apply_effects(ecs_world_t* world, const AbilityContext* ctx);

#endif // AZUKI_ABILITY_STT02_015_H
