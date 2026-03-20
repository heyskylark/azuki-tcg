#include "abilities/cards/azk01_003.h"

#include "abilities/cards/common/reveal_selection.h"
#include "abilities/selection/ability_selection_helpers.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"

// AZK01-003: "[On Play] Look at the top 5 cards of your deck, reveal up to 1
// Black Jade subtype card other than Black Jade Courier and add it to your
// hand, then bottom deck the rest in any order."

static const CardId *get_selection_card_id(ecs_world_t *world,
                                           ecs_entity_t card) {
  const CardId *card_id = ecs_get(world, card, CardId);
  if (card_id != NULL) {
    return card_id;
  }

  const ecs_entity_t prefab = ecs_get_target(world, card, EcsIsA, 0);
  if (prefab == 0) {
    return NULL;
  }

  return ecs_get(world, prefab, CardId);
}

static bool is_black_jade_non_courier_selection_card(ecs_world_t *world,
                                                     ecs_entity_t card,
                                                     const void *user_ctx) {
  (void)user_ctx;

  if (!has_subtype(world, card, ecs_id(TSubtype_BlackJade))) {
    return false;
  }

  const CardId *card_id = get_selection_card_id(world, card);
  return card_id != NULL && card_id->id != CARD_DEF_AZK01_003;
}

bool azk01_003_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;

  return true;
}

void azk01_003_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  const AbilityRevealSelectionResult result =
      azk_setup_reveal_top_cards_selection(
          world, ctx, 5, 1, is_black_jade_non_courier_selection_card, NULL);

  if (result.revealed_count == 0) {
    cli_render_logf("[AZK01-003] No cards in deck to look at");
    return;
  }

  if (result.matching_count > 0) {
    cli_render_logf("[AZK01-003] Looking at top %d cards, found %d valid Black "
                    "Jade card(s)",
                    result.revealed_count, result.matching_count);
  } else {
    cli_render_logf("[AZK01-003] Looking at top %d cards, no valid Black Jade "
                    "cards found - bottom decking",
                    result.revealed_count);
  }
}

bool azk01_003_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  return is_black_jade_non_courier_selection_card(world, target, NULL);
}

void azk01_003_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  if (azk_move_picked_selection_cards_to_hand(world, ctx) > 0) {
    cli_render_logf("[AZK01-003] Added Black Jade card to hand");
  }

  const uint8_t remaining = azk_begin_bottom_deck_for_remaining_selection(ctx);
  if (remaining > 0) {
    cli_render_logf("[AZK01-003] %d cards remaining to bottom deck", remaining);
  } else {
    cli_render_logf("[AZK01-003] Ability complete");
  }
}
