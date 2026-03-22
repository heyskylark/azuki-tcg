#include "abilities/cards/azk01_024.h"

#include "abilities/selection/ability_selection_helpers.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

static bool is_valid_cost_target(ecs_world_t *world, ecs_entity_t entity,
                                 ecs_entity_t owner) {
  if (!is_card_type(world, entity, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, entity, EcsChildOf, 0) ==
         gs->zones[owner_num].garden;
}

static bool is_valid_selection_card(ecs_world_t *world, ecs_entity_t card,
                                    const void *user_ctx) {
  (void)user_ctx;

  if (!is_card_type(world, card, CARD_TYPE_ENTITY)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, card, IKZCost);
  return cost != NULL && cost->ikz_cost <= 2;
}

bool azk01_024_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  return garden_cards.count > 0;
}

bool azk01_024_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  return is_valid_cost_target(world, target, owner);
}

void azk01_024_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->cost.entities[0];
  if (target == 0) {
    return;
  }

  return_card_to_hand(world, target);
  cli_render_logf("[AZK01-024] Returned cost target to hand");
}

void azk01_024_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  uint8_t selection_count = azk_move_matching_hand_cards_to_selection(
      world, ctx, 1, is_valid_selection_card, NULL);

  if (selection_count == 0) {
    ctx->runtime.phase = ABILITY_PHASE_NONE;
    cli_render_logf("[AZK01-024] No cost 2 or less entity available to play");
    return;
  }

  cli_render_logf("[AZK01-024] Found %u card(s) to play from hand",
                  (unsigned)selection_count);
}

bool azk01_024_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  return is_valid_selection_card(world, target, NULL);
}

void azk01_024_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  uint8_t returned = azk_return_remaining_selection_cards_to_hand(world, ctx);
  cli_render_logf("[AZK01-024] Returned %u unplayed selection card(s) to hand",
                  (unsigned)returned);
}
