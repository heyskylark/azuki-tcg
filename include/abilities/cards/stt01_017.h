#ifndef AZUKI_ABILITY_STT01_017_H
#define AZUKI_ABILITY_STT01_017_H

#include <flecs.h>
#include <stdbool.h>

#include "components/components.h"

// STT01-017 "Lightning Orb": [Response] Deal 1 damage to an entity in your
// opponent's garden and 1 damage to another entity in your opponent's garden.

// Validate if ability can be activated
// Returns true if opponent has at least one entity in their garden
bool stt01_017_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);

// Validate effect target selection
// Returns true if target is in opponent's garden
bool stt01_017_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);

// Apply effect: deal 1 damage to each selected target (1 or 2 targets)
void stt01_017_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITY_STT01_017_H
