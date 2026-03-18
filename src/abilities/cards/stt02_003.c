#include "abilities/cards/stt02_003.h"

#include "abilities/cards/common/reveal_selection.h"
#include "abilities/selection/ability_selection_helpers.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"

// STT02-003: "[On Play] Look at the top 5 cards of your deck, reveal up to 1
// (Watercrafting) card and add it to your hand, then bottom deck the rest in
// any order"

static bool is_watercrafting_selection_card(ecs_world_t *world,
                                            ecs_entity_t card,
                                            const void *user_ctx) {
  (void)user_ctx;
  return is_watercrafting_card(world, card);
}

// Validate if ability can be activated
// This ability has no cost, so it can always be activated
bool stt02_003_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;

  // No cost requirement, always valid to trigger
  return true;
}

// Called after the ability is accepted: move top 5 cards from deck to
// selection zone
void stt02_003_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  const AbilityRevealSelectionResult result = azk_setup_reveal_top_cards_selection(
      world, ctx, 5, 1, is_watercrafting_selection_card, NULL);

  if (result.revealed_count == 0) {
    cli_render_logf("[STT02-003] No cards in deck to look at");
    return;
  }

  if (result.matching_count > 0) {
    cli_render_logf(
        "[STT02-003] Looking at top %d cards, found %d Watercrafting card(s)",
        result.revealed_count, result.matching_count);
  } else {
    cli_render_logf("[STT02-003] Looking at top %d cards, no Watercrafting "
                    "cards found - bottom decking",
                    result.revealed_count);
  }
}

// Validate selection target - must be a watercrafting card
bool stt02_003_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  // Target must be a watercrafting card
  return is_watercrafting_card(world, target);
}

// Called after selection pick is complete: move picked watercrafting card to
// hand
void stt02_003_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  if (azk_move_picked_selection_cards_to_hand(world, ctx) > 0) {
    cli_render_logf("[STT02-003] Added Watercrafting card to hand");
  }

  const uint8_t remaining = azk_begin_bottom_deck_for_remaining_selection(ctx);
  if (remaining > 0) {
    cli_render_logf("[STT02-003] %d cards remaining to bottom deck", remaining);
  } else {
    cli_render_logf("[STT02-003] Ability complete");
  }
}
