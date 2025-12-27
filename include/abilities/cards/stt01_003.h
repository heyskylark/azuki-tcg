#ifndef AZUKI_ABILITY_STT01_003_H
#define AZUKI_ABILITY_STT01_003_H

#include <flecs.h>
#include <stdbool.h>

#include "components/components.h"

// STT01-003 "Crate Rat Kurobo": On Play; You may put 3 cards from the top of
// your deck into your discard pile. If you have no weapon cards in your
// discard pile, put 5 cards instead.

// Validate if ability can be activated
// Always returns true - no validation needed for this simple effect
bool stt01_003_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);

// Apply effect: mill 3 cards (or 5 if no weapons in discard)
void stt01_003_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITY_STT01_003_H
