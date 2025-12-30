#include "abilities/cards/stt02_013.h"

#include "components/components.h"
#include "constants/game.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/deck_utils.h"
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

// Called after confirmation: move top 3 cards from deck to selection zone
void stt02_013_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  // Look at top 3 cards
  ecs_entity_t cards[MAX_SELECTION_ZONE_SIZE];
  int count = look_at_top_n_cards(world, ctx->owner, 3, cards);

  if (count == 0) {
    cli_render_logf("[STT02-013] No cards in deck to look at");
    // Skip directly to done - no cards to process
    return;
  }

  // Store cards in selection context
  ctx->selection_count = count;
  for (int i = 0; i < count; i++) {
    ctx->selection_cards[i] = cards[i];
  }

  // Set up selection pick phase - "up to 1" valid card
  ctx->selection_pick_max = 1;
  ctx->selection_picked = 0;

  // Count how many valid cards (<=2 cost AND water element) are in the selection
  int valid_count = 0;
  for (int i = 0; i < count; i++) {
    if (is_valid_selection(world, cards[i])) {
      valid_count++;
    }
  }

  if (valid_count > 0) {
    ctx->phase = ABILITY_PHASE_SELECTION_PICK;
    cli_render_logf(
        "[STT02-013] Looking at top %d cards, found %d valid card(s) "
        "(<=2 cost water type)",
        count, valid_count);
  } else {
    // No valid cards to pick - go directly to bottom deck phase
    ctx->phase = ABILITY_PHASE_BOTTOM_DECK;
    cli_render_logf("[STT02-013] Looking at top %d cards, no valid cards "
                    "found - bottom decking",
                    count);
  }

  ecs_singleton_modified(world, AbilityContext);
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
// only stores it in effect_targets - we need to move it to hand here
void stt02_013_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->owner);
  ecs_entity_t selection_zone = gs->zones[player_num].selection;

  // Move any picked cards to hand if still in selection zone
  // (ACT_SELECT_TO_ALLEY already moved to alley, so skip those)
  for (int i = 0; i < ctx->selection_picked && i < MAX_ABILITY_SELECTION; i++) {
    ecs_entity_t picked = ctx->effect_targets[i];
    if (picked != 0) {
      ecs_entity_t parent = ecs_get_target(world, picked, EcsChildOf, 0);
      if (parent == selection_zone) {
        move_selection_to_hand(world, picked);
        cli_render_logf("[STT02-013] Added card to hand");
      }
    }
  }

  // Count remaining cards to bottom deck
  int remaining = 0;
  for (int i = 0; i < ctx->selection_count; i++) {
    if (ctx->selection_cards[i] != 0) {
      remaining++;
    }
  }

  if (remaining > 0) {
    ctx->phase = ABILITY_PHASE_BOTTOM_DECK;
    cli_render_logf("[STT02-013] %d cards remaining to bottom deck", remaining);
  } else {
    cli_render_logf("[STT02-013] Ability complete");
  }

  ecs_singleton_modified(world, AbilityContext);
}
