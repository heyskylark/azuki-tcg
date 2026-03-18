#include "abilities/cards/stt02_013.h"

#include "abilities/cards/common/reveal_selection.h"
#include "abilities/selection/ability_selection_helpers.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

// STT02-013: "[On Play] Look at the top 3 cards of your deck, reveal up to 1
// 2 cost or less water type card and add it to your hand, then bottom deck
// the rest in any order. You may play the card to the alley if it is an entity."

// Helper to check if a card is a valid selection target (<=2 cost AND water)
static bool is_valid_selection(ecs_world_t *world, ecs_entity_t card) {
  // Check IKZ cost <= 2
  const IKZCost *cost = ecs_get(world, card, IKZCost);
  if (!cost || cost->ikz_cost > 2) {
    return false;
  }

  // Check water element
  return is_water_element_card(world, card);
}

static bool is_valid_reveal_selection_card(ecs_world_t *world,
                                           ecs_entity_t card,
                                           const void *user_ctx) {
  (void)user_ctx;
  return is_valid_selection(world, card);
}

// Validate if ability can be activated
// This ability requires at least 3 cards in deck
bool stt02_013_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t deck = gs->zones[player_num].deck;

  ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck);
  if (deck_cards.count < 3) {
    cli_render_logf("[STT02-013] Requires 3+ cards in deck, have %d",
                    deck_cards.count);
    return false;
  }

  return true;
}

// Called after the ability is accepted: move top 3 cards from deck to
// selection zone
void stt02_013_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  const AbilityRevealSelectionResult result = azk_setup_reveal_top_cards_selection(
      world, ctx, 3, 1, is_valid_reveal_selection_card, NULL);

  if (result.revealed_count == 0) {
    cli_render_logf("[STT02-013] No cards in deck to look at");
    return;
  }

  if (result.matching_count > 0) {
    cli_render_logf(
        "[STT02-013] Looking at top %d cards, found %d valid card(s) "
        "(<=2 cost water type)",
        result.revealed_count, result.matching_count);
  } else {
    cli_render_logf("[STT02-013] Looking at top %d cards, no valid cards "
                    "found - bottom decking",
                    result.revealed_count);
  }
}

// Validate selection target - must be <=2 cost AND water element
bool stt02_013_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  return is_valid_selection(world, target);
}

// Called after selection pick is complete
// ACT_SELECT_TO_ALLEY moves the card to alley, but ACT_SELECT_FROM_SELECTION
// only stores it in selection.picked_cards - we need to move it to hand here
void stt02_013_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  if (azk_move_picked_selection_cards_to_hand_if_still_in_selection(world,
                                                                     ctx) > 0) {
    cli_render_logf("[STT02-013] Added card to hand");
  }

  const uint8_t remaining = azk_begin_bottom_deck_for_remaining_selection(ctx);
  if (remaining > 0) {
    cli_render_logf("[STT02-013] %d cards remaining to bottom deck", remaining);
  } else {
    cli_render_logf("[STT02-013] Ability complete");
  }
}
