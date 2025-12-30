#ifndef AZUKI_ABILITY_STT02_017_H
#define AZUKI_ABILITY_STT02_017_H

#include <flecs.h>
#include <stdbool.h>

#include "components/components.h"

// STT02-017 "Shao's Perseverance": [Main] If your leader's Shao, return all
// entities with cost <= 4 in opponent's garden to their owner's hand

// Validate if ability can be activated
// Returns true if player's leader has the Shao subtype
bool stt02_017_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);

// Apply effect: return all opponent's garden entities with cost <= 4 to hand
void stt02_017_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITY_STT02_017_H
