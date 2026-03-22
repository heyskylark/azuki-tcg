#include "abilities/cards/azk01_031.h"

#include "abilities/cards/common/reveal_selection.h"
#include "abilities/selection/ability_selection_helpers.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"

static bool is_water_selection_card(ecs_world_t *world, ecs_entity_t card,
                                    const void *user_ctx) {
  (void)user_ctx;
  return is_water_element_card(world, card);
}

bool azk01_031_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

void azk01_031_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  const AbilityRevealSelectionResult result =
      azk_setup_reveal_top_cards_selection(world, ctx, 3, 1,
                                           is_water_selection_card, NULL);

  if (result.revealed_count == 0) {
    cli_render_logf("[AZK01-031] No cards in deck to look at");
    return;
  }

  if (result.matching_count > 0) {
    cli_render_logf("[AZK01-031] Looking at top %d cards, found %d Water "
                    "card(s)",
                    result.revealed_count, result.matching_count);
  } else {
    cli_render_logf("[AZK01-031] Looking at top %d cards, no Water cards found",
                    result.revealed_count);
  }
}

bool azk01_031_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  return is_water_selection_card(world, target, NULL);
}

void azk01_031_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  if (azk_move_picked_selection_cards_to_hand(world, ctx) > 0) {
    cli_render_logf("[AZK01-031] Added Water card to hand");
  }

  const uint8_t remaining = azk_begin_bottom_deck_for_remaining_selection(ctx);
  if (remaining > 0) {
    cli_render_logf("[AZK01-031] %d cards remaining to order",
                    (unsigned)remaining);
  } else {
    cli_render_logf("[AZK01-031] Ability complete");
  }
}
