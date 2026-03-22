#include "abilities/cards/azk01_098.h"

#include "abilities/selection/ability_selection_helpers.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

static bool is_valid_weapon_hand_card(ecs_world_t *world, ecs_entity_t card,
                                      const void *user_ctx) {
  (void)user_ctx;

  if (!is_weapon_card(world, card)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, card, IKZCost);
  return cost != NULL && cost->ikz_cost <= 3;
}

bool azk01_098_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  if (ecs_get_target(world, card, EcsChildOf, 0) != gs->zones[owner_num].alley) {
    return false;
  }

  if (!can_tap_card(world, card, false)) {
    return false;
  }

  ecs_entity_t hand = gs->zones[owner_num].hand;
  ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
  for (int32_t i = 0; i < hand_cards.count; ++i) {
    if (is_valid_weapon_hand_card(world, hand_cards.ids[i], NULL)) {
      return true;
    }
  }

  return false;
}

void azk01_098_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  tap_card(world, ctx->runtime.source_card);
}

void azk01_098_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  uint8_t selection_count = azk_move_matching_hand_cards_to_selection(
      world, ctx, 1, is_valid_weapon_hand_card, NULL);

  if (selection_count == 0) {
    ctx->runtime.phase = ABILITY_PHASE_NONE;
    cli_render_logf("[AZK01-098] No weapon card with cost <= 3 in hand");
    return;
  }

  cli_render_logf("[AZK01-098] Found %u weapon card(s) to play from hand",
                  (unsigned)selection_count);
}

bool azk01_098_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  return is_valid_weapon_hand_card(world, target, NULL);
}

void azk01_098_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  uint8_t returned = azk_return_remaining_selection_cards_to_hand(world, ctx);
  cli_render_logf("[AZK01-098] Returned %u unplayed weapon card(s) to hand",
                  (unsigned)returned);
}
