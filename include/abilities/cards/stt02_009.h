#ifndef AZUKI_ABILITY_STT02_009_H
#define AZUKI_ABILITY_STT02_009_H

#include <stdbool.h>
#include <flecs.h>

#include "components/components.h"

// STT02-009 "Aya": [On Play] You may return an entity with cost >= 2 in your Garden
// to your hand: Return up to 1 entity with cost <= 2 in your opponent's Garden to its owner's hand.

// Validate if ability can be activated
// Returns true if player has at least one entity with cost >= 2 in their garden
bool stt02_009_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner);

// Validate a cost target (friendly garden entity with cost >= 2)
bool stt02_009_validate_cost_target(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner, ecs_entity_t target);

// Validate an effect target (opponent garden entity with cost <= 2)
bool stt02_009_validate_effect_target(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner, ecs_entity_t target);

// Apply cost: return the cost target to owner's hand
void stt02_009_apply_costs(ecs_world_t* world, const AbilityContext* ctx);

// Apply effect: return the effect target to its owner's hand (if any)
void stt02_009_apply_effects(ecs_world_t* world, const AbilityContext* ctx);

#endif // AZUKI_ABILITY_STT02_009_H
