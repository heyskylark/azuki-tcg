#ifndef AZUKI_ABILITY_SELECTION_H
#define AZUKI_ABILITY_SELECTION_H

#include <flecs.h>
#include <stdbool.h>

#include "abilities/ability_registry.h"

typedef enum {
  AZK_SELECTION_COMPLETION_ALLOW_BOTTOM_DECK = 0,
  AZK_SELECTION_COMPLETION_CLEAR_IF_STILL_ACTIVE = 1,
} AbilitySelectionCompletionMode;

bool azk_finish_selection_resolution(ecs_world_t *world, AbilityContext *ctx,
                                     const AbilityDef *def,
                                     AbilitySelectionCompletionMode mode);

bool azk_bottom_deck_selection_card(ecs_world_t *world, AbilityContext *ctx,
                                    int selection_index);

bool azk_bottom_deck_all_selection_cards(ecs_world_t *world,
                                         AbilityContext *ctx);

#endif // AZUKI_ABILITY_SELECTION_H
