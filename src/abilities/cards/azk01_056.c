#include "abilities/cards/azk01_056.h"

#include "abilities/cards/common/reveal_selection.h"
#include "abilities/selection/ability_selection_helpers.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"

static bool is_scorchweaver_selection_card(ecs_world_t *world,
                                           ecs_entity_t card,
                                           const void *user_ctx) {
  (void)user_ctx;
  return has_subtype(world, card, ecs_id(TSubtype_Scorchweaver));
}

bool azk01_056_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

void azk01_056_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  const AbilityRevealSelectionResult result =
      azk_setup_reveal_top_cards_selection(world, ctx, 5, 1,
                                           is_scorchweaver_selection_card,
                                           NULL);

  if (result.revealed_count == 0) {
    cli_render_logf("[AZK01-056] No cards in deck to look at");
    return;
  }

  cli_render_logf("[AZK01-056] Looking at top %d cards of deck",
                  result.revealed_count);
}

bool azk01_056_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;
  return target != 0 && is_scorchweaver_selection_card(world, target, NULL);
}

void azk01_056_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  if (azk_move_picked_selection_cards_to_hand(world, ctx) > 0) {
    cli_render_logf("[AZK01-056] Added Scorchweaver card to hand");
  }

  azk_begin_bottom_deck_for_remaining_selection(ctx);
}
