#ifndef AZUKI_ABILITY_STT01_014_H
#define AZUKI_ABILITY_STT01_014_H

#include <flecs.h>
#include <stdbool.h>

#include "components/components.h"

// STT01-014 "Tenshin": [On Play] Deal up to 1 damage to a leader.

// Validate if ability can be activated
// Always returns true - leaders always exist as valid targets
bool stt01_014_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);

// Validate effect target selection
// Returns true if target is a leader (friendly or enemy)
bool stt01_014_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);

// Apply effect: deal 1 damage to selected leader
void stt01_014_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITY_STT01_014_H
