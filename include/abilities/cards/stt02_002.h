#ifndef AZUKI_ABILITY_STT02_002_H
#define AZUKI_ABILITY_STT02_002_H

#include <flecs.h>
#include <stdbool.h>

#include "components/components.h"

// STT02-002 "Hydromancy": On Gate Portal; untap IKZ up to portaled card's gate
// points

// Validate if ability can be activated
// Always returns true - no validation needed for this effect
bool stt02_002_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);

// Apply effect: untap IKZ up to portaled card's gate points
void stt02_002_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITY_STT02_002_H
