#ifndef AZUKI_ABILITY_STT02_016_H
#define AZUKI_ABILITY_STT02_016_H

#include <stdbool.h>
#include <flecs.h>

#include "components/components.h"

// STT02-016: [Response] Discard 1: Reduce a leader's or entity's attack by 2 until the end of the turn.

// Validate if ability can be activated
// Returns true if player has at least 1 card in hand to discard
// AND enemy has at least one valid target (leader or garden entity)
bool stt02_016_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner);

// Validate a cost target (card to discard from hand)
bool stt02_016_validate_cost_target(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner, ecs_entity_t target);

// Validate an effect target (enemy leader or garden entity)
bool stt02_016_validate_effect_target(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner, ecs_entity_t target);

// Apply cost: discard the selected card from hand
void stt02_016_apply_costs(ecs_world_t* world, const AbilityContext* ctx);

// Apply effect: reduce target's attack by 2 until end of turn
void stt02_016_apply_effects(ecs_world_t* world, const AbilityContext* ctx);

#endif // AZUKI_ABILITY_STT02_016_H
