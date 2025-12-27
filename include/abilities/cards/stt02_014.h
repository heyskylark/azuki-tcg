#ifndef AZUKI_ABILITY_STT02_014_H
#define AZUKI_ABILITY_STT02_014_H

#include <stdbool.h>
#include <flecs.h>

#include "components/components.h"

// STT02-014 "Chilling Water": [Main] Freeze an entity with cost <= 2 in opponent's garden for 2 turns

// Validate if ability can be activated
// Returns true if there's at least one entity with cost <= 2 in opponent's garden
bool stt02_014_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner);

// Validate an effect target (entity in opponent's garden with cost <= 2)
bool stt02_014_validate_effect_target(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner, ecs_entity_t target);

// Apply effect: freeze the target entity for 2 turns
void stt02_014_apply_effects(ecs_world_t* world, const AbilityContext* ctx);

#endif // AZUKI_ABILITY_STT02_014_H
