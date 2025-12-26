#ifndef AZUKI_ABILITY_STT01_013_H
#define AZUKI_ABILITY_STT01_013_H

#include <stdbool.h>
#include <flecs.h>

#include "components/components.h"

// STT01-013 "Black Jade Dagger": On Play; You may deal damage to your leader:
// This card gives an additional +1 attack

// Validate if ability can be activated
// Returns true if owner's leader has >= 1 HP
bool stt01_013_validate(ecs_world_t *world, ecs_entity_t card, ecs_entity_t owner);

// Apply cost: deal 1 damage to owner's leader
void stt01_013_apply_costs(ecs_world_t *world, const AbilityContext *ctx);

// Apply effect: +1 attack to weapon and attached entity
void stt01_013_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITY_STT01_013_H
