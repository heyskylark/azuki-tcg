#ifndef AZUKI_ABILITY_STT01_015_H
#define AZUKI_ABILITY_STT01_015_H

#include <flecs.h>
#include <stdbool.h>

#include "components/components.h"

// STT01-015 "Tenraku": When Equipped; If you have 15 or more cards in your
// discard pile, this card gives an additional +1 attack.

// Validate if ability can be triggered (always returns true; condition checked
// in apply)
bool stt01_015_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);

// Apply effect: check discard pile count and give +1 attack if >= 15 cards
void stt01_015_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITY_STT01_015_H
