#ifndef AZUKI_ABILITY_SELECTION_HELPERS_H
#define AZUKI_ABILITY_SELECTION_HELPERS_H

#include <flecs.h>
#include <stdbool.h>
#include <stdint.h>

#include "components/components.h"

typedef bool (*AbilitySelectionCardPredicate)(ecs_world_t *world,
                                              ecs_entity_t card,
                                              const void *user_ctx);

void azk_init_selection_state(AbilityContext *ctx, const ecs_entity_t *cards,
                              uint8_t count, uint8_t pick_max);

uint8_t azk_move_matching_hand_cards_to_selection(
    ecs_world_t *world, AbilityContext *ctx, uint8_t pick_max,
    AbilitySelectionCardPredicate predicate, const void *user_ctx);

uint8_t azk_count_selection_cards_matching(
    ecs_world_t *world, const AbilityContext *ctx,
    AbilitySelectionCardPredicate predicate, const void *user_ctx);

uint8_t azk_begin_bottom_deck_for_remaining_selection(AbilityContext *ctx);

void azk_record_selection_pick(AbilityContext *ctx, int selection_index,
                               ecs_entity_t picked_card);

uint8_t azk_move_picked_selection_cards_to_hand(ecs_world_t *world,
                                                const AbilityContext *ctx);

uint8_t azk_move_picked_selection_cards_to_hand_if_still_in_selection(
    ecs_world_t *world, const AbilityContext *ctx);

uint8_t azk_return_remaining_selection_cards_to_hand(ecs_world_t *world,
                                                     AbilityContext *ctx);

uint8_t azk_return_remaining_selection_cards_to_discard(ecs_world_t *world,
                                                        AbilityContext *ctx);

#endif // AZUKI_ABILITY_SELECTION_HELPERS_H
