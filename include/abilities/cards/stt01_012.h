#ifndef AZUKI_ABILITY_STT01_012_H
#define AZUKI_ABILITY_STT01_012_H

#include <flecs.h>
#include <stdbool.h>

#include "components/components.h"

// STT01-012 "Lightning Shuriken": [When Attacking] Put the top card of your
// deck into your discard pile.

// Validate if ability can be activated
// Returns true if owner has at least 1 card in their deck
bool stt01_012_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);

// Apply effect: mill 1 card from top of deck to discard pile
void stt01_012_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITY_STT01_012_H
