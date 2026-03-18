#include "abilities/cards/common/reveal_selection.h"

#include "abilities/selection/ability_selection_helpers.h"
#include "utils/deck_utils.h"

AbilityRevealSelectionResult azk_setup_reveal_top_cards_selection(
    ecs_world_t *world, AbilityContext *ctx, uint8_t reveal_count,
    uint8_t pick_max, AbilitySelectionCardPredicate predicate,
    const void *user_ctx) {
  AbilityRevealSelectionResult result = {0};
  if (!ctx || reveal_count == 0) {
    return result;
  }

  ecs_entity_t cards[MAX_SELECTION_ZONE_SIZE] = {0};
  const int count =
      look_at_top_n_cards(world, ctx->runtime.owner, reveal_count, cards);
  if (count <= 0) {
    return result;
  }

  result.revealed_count = (uint8_t)count;
  azk_init_selection_state(ctx, cards, result.revealed_count, pick_max);

  result.matching_count = azk_count_selection_cards_matching(
      world, ctx, predicate, user_ctx);
  ctx->runtime.phase = result.matching_count > 0 ? ABILITY_PHASE_SELECTION_PICK
                                                 : ABILITY_PHASE_BOTTOM_DECK;

  return result;
}
