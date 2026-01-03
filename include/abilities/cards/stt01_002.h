#ifndef AZUKI_ABILITY_STT01_002_H
#define AZUKI_ABILITY_STT01_002_H

#include <flecs.h>
#include <stdbool.h>

#include "components/components.h"

// STT01-002 "Surge": On Gate Portal; you may play from your discard pile
// a weapon card with cost <= gate points of the portaled entity

// Validate if ability can be activated
// Always returns true - validation happens during selection
bool stt01_002_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);

// Called after confirmation to set up weapon selection from discard
void stt01_002_on_cost_paid(ecs_world_t *world, AbilityContext *ctx);

// Validate selection target (weapon cost <= gate points)
bool stt01_002_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target);

// Called when selection is complete to clean up
void stt01_002_on_selection_complete(ecs_world_t *world, AbilityContext *ctx);

#endif // AZUKI_ABILITY_STT01_002_H
