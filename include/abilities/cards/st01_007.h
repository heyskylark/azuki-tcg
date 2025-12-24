#ifndef AZUKI_ABILITY_ST01_007_H
#define AZUKI_ABILITY_ST01_007_H

#include <stdbool.h>
#include <flecs.h>

#include "components/components.h"

// ST01-007 "Alley Guy": On Play; You may discard 1:Draw 1

// Validate if ability can be activated
// Returns true if player has at least 1 card in hand to discard and 1 card in deck to draw
bool st01_007_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner);

// Validate a cost target (card to discard from hand)
bool st01_007_validate_cost_target(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner, ecs_entity_t target);

// Apply cost: discard the selected card
void st01_007_apply_costs(ecs_world_t* world, const AbilityContext* ctx);

// Apply effect: draw 1 card
void st01_007_apply_effects(ecs_world_t* world, const AbilityContext* ctx);

#endif // AZUKI_ABILITY_ST01_007_H
