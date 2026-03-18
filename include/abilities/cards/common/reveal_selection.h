#ifndef AZUKI_REVEAL_SELECTION_H
#define AZUKI_REVEAL_SELECTION_H

#include <flecs.h>
#include <stdint.h>

#include "abilities/selection/ability_selection_helpers.h"

typedef struct {
  uint8_t revealed_count;
  uint8_t matching_count;
} AbilityRevealSelectionResult;

AbilityRevealSelectionResult azk_setup_reveal_top_cards_selection(
    ecs_world_t *world, AbilityContext *ctx, uint8_t reveal_count,
    uint8_t pick_max, AbilitySelectionCardPredicate predicate,
    const void *user_ctx);

#endif // AZUKI_REVEAL_SELECTION_H
