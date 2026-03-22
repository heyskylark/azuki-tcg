#include "abilities/cards/azk01_097.h"

#include "abilities/cards/common/reveal_selection.h"
#include "abilities/selection/ability_selection_helpers.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"

static bool is_weapon_selection_card(ecs_world_t *world, ecs_entity_t card,
                                     const void *user_ctx) {
  (void)user_ctx;
  return is_weapon_card(world, card);
}

bool azk01_097_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

void azk01_097_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  const AbilityRevealSelectionResult result = azk_setup_reveal_top_cards_selection(
      world, ctx, 5, 1, is_weapon_selection_card, NULL);

  if (result.revealed_count == 0) {
    cli_render_logf("[AZK01-097] No cards in deck to mill");
    return;
  }

  if (result.matching_count > 0) {
    cli_render_logf("[AZK01-097] Milled %d cards and found %d weapon card(s)",
                    result.revealed_count, result.matching_count);
  } else {
    cli_render_logf("[AZK01-097] Milled %d cards and found no weapon cards",
                    result.revealed_count);
  }
}

bool azk01_097_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  return is_weapon_card(world, target);
}

void azk01_097_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  if (azk_move_picked_selection_cards_to_hand_if_still_in_selection(world, ctx) >
      0) {
    cli_render_logf("[AZK01-097] Added milled weapon to hand");
  }

  azk_return_remaining_selection_cards_to_discard(world, ctx);
}
