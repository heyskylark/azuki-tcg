#include "abilities/cards/stt01_004.h"

#include "abilities/cards/common/reveal_selection.h"
#include "abilities/selection/ability_selection_helpers.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

// STT01-004: "On Play; You may discard a weapon card: look at the top 5 cards
// of your deck, reveal up to 1 weapon card and add it to your hand, then
// bottom deck the rest in any order"

static bool is_weapon_selection_card(ecs_world_t *world, ecs_entity_t card,
                                     const void *user_ctx) {
  (void)user_ctx;
  return is_weapon_card(world, card);
}

// Validate if ability can be activated
// Returns true if player has at least one weapon card in hand
bool stt01_004_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t hand = gs->zones[player_num].hand;

  // Check if player has at least one weapon in hand
  int weapon_count = count_weapons_in_zone(world, hand);
  return weapon_count > 0;
}

// Validate cost target - must be a weapon card in owner's hand
bool stt01_004_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t hand = gs->zones[player_num].hand;

  // Target must be in owner's hand
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  if (parent != hand) {
    return false;
  }

  // Target must be a weapon card
  return is_weapon_card(world, target);
}

// Apply cost: discard the selected weapon card
void stt01_004_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->cost.entities[0];

  if (target == 0) {
    cli_render_logf("[STT01-004] No cost target to discard");
    return;
  }

  sacrifice_card(world, target);
  cli_render_logf("[STT01-004] Discarded weapon card as cost");
}

// Called after cost is paid: move top 5 cards from deck to selection zone
void stt01_004_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  const AbilityRevealSelectionResult result = azk_setup_reveal_top_cards_selection(
      world, ctx, 5, 1, is_weapon_selection_card, NULL);

  if (result.revealed_count == 0) {
    cli_render_logf("[STT01-004] No cards in deck to look at");
    return;
  }

  if (result.matching_count > 0) {
    cli_render_logf("[STT01-004] Looking at top %d cards, found %d weapon(s)",
                    result.revealed_count, result.matching_count);
  } else {
    cli_render_logf("[STT01-004] Looking at top %d cards, no weapons found - "
                    "bottom decking",
                    result.revealed_count);
  }
}

// Validate selection target - must be a weapon card
bool stt01_004_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  // Target must be a weapon card
  return is_weapon_card(world, target);
}

// Called after selection pick is complete: move picked weapon to hand
void stt01_004_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  if (azk_move_picked_selection_cards_to_hand(world, ctx) > 0) {
    cli_render_logf("[STT01-004] Added weapon card to hand");
  }

  const uint8_t remaining = azk_begin_bottom_deck_for_remaining_selection(ctx);
  if (remaining > 0) {
    cli_render_logf("[STT01-004] %d cards remaining to bottom deck", remaining);
  } else {
    cli_render_logf("[STT01-004] Ability complete");
  }
}
