#include "abilities/cards/azk01_122.h"

#include "abilities/selection/ability_selection_helpers.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

static uint8_t gate_power_from_ctx(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx == NULL || ctx->scratch.kind != ABILITY_SCRATCH_GATE_PORTAL) {
    return 0;
  }

  const GatePoints *gp =
      ecs_get(world, ctx->scratch.data.gate_portal.portaled_card, GatePoints);
  return gp != NULL ? gp->gate_points : 0;
}

static bool is_valid_rushfire_target(ecs_world_t *world, ecs_entity_t card,
                                     const void *user_ctx) {
  const uint8_t max_cost = *(const uint8_t *)user_ctx;
  if (!is_card_type(world, card, CARD_TYPE_ENTITY)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, card, IKZCost);
  return cost != NULL && cost->ikz_cost <= max_cost;
}

bool azk01_122_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  const uint8_t max_cost = gate_power_from_ctx(world, ctx);
  if (max_cost == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t hand_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].hand);
  for (int32_t i = 0; i < hand_cards.count; ++i) {
    if (is_valid_rushfire_target(world, hand_cards.ids[i], &max_cost)) {
      return true;
    }
  }

  return false;
}

void azk01_122_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  const uint8_t max_cost = gate_power_from_ctx(world, ctx);
  const uint8_t selection_count = azk_move_matching_hand_cards_to_selection(
      world, ctx, 1, is_valid_rushfire_target, &max_cost);
  if (selection_count == 0) {
    ctx->runtime.phase = ABILITY_PHASE_NONE;
  }
}

bool azk01_122_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  const uint8_t max_cost = gate_power_from_ctx(world, ctx);
  return target != 0 &&
         is_valid_rushfire_target(world, target, &max_cost);
}

void azk01_122_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  if (ctx->selection.picked_count > 0 && ctx->selection.picked_cards[0] != 0) {
    ecs_entity_t played = ctx->selection.picked_cards[0];
    apply_charge_grant(world, played, TAG_GRANT_TICK_NONE, -1);

    const TapState *tap = ecs_get(world, played, TapState);
    ecs_set(world, played, TapState,
            {.tapped = azk_card_enters_garden_tapped(world, played) ||
                       (tap != NULL && tap->tapped),
             .cooldown = false});
  }

  azk_return_remaining_selection_cards_to_hand(world, ctx);
}
