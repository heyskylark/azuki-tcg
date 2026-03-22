#include "abilities/cards/azk01_126.h"

#include "abilities/selection/ability_selection_helpers.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"

static uint8_t gate_power_from_ctx(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx == NULL || ctx->scratch.kind != ABILITY_SCRATCH_GATE_PORTAL) {
    return 0;
  }

  const GatePoints *gp =
      ecs_get(world, ctx->scratch.data.gate_portal.portaled_card, GatePoints);
  return gp != NULL ? gp->gate_points : 0;
}

static bool is_valid_echoed_waves_target(ecs_world_t *world, ecs_entity_t card,
                                         const void *user_ctx) {
  const uint8_t max_cost = *(const uint8_t *)user_ctx;
  if (!is_card_type(world, card, CARD_TYPE_SPELL)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, card, IKZCost);
  return cost != NULL && cost->ikz_cost <= max_cost;
}

static uint8_t move_matching_discard_spells_to_selection(ecs_world_t *world,
                                                         AbilityContext *ctx,
                                                         uint8_t max_cost) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  const ecs_entity_t discard_zone = gs->zones[owner_num].discard;
  const ecs_entity_t selection_zone = gs->zones[owner_num].selection;
  ecs_entities_t discard_cards = ecs_get_ordered_children(world, discard_zone);

  ecs_entity_t selection_cards[MAX_SELECTION_ZONE_SIZE] = {0};
  uint8_t selection_count = 0;
  for (int32_t i = 0;
       i < discard_cards.count && selection_count < MAX_SELECTION_ZONE_SIZE; ++i) {
    ecs_entity_t target = discard_cards.ids[i];
    if (!is_valid_echoed_waves_target(world, target, &max_cost)) {
      continue;
    }

    const int8_t from_index =
        azk_get_card_index_in_zone(world, target, discard_zone);
    ecs_add_pair(world, target, EcsChildOf, selection_zone);
    azk_log_card_zone_moved(world, target, GLOG_ZONE_DISCARD, from_index,
                            GLOG_ZONE_SELECTION, (int8_t)selection_count);
    selection_cards[selection_count++] = target;
  }

  azk_init_selection_state(ctx, selection_cards, selection_count, 1);
  ctx->runtime.phase =
      selection_count > 0 ? ABILITY_PHASE_SELECTION_PICK : ABILITY_PHASE_NONE;
  return selection_count;
}

bool azk01_126_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  const uint8_t max_cost = gate_power_from_ctx(world, ctx);
  if (max_cost == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t discard_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].discard);
  for (int32_t i = 0; i < discard_cards.count; ++i) {
    if (is_valid_echoed_waves_target(world, discard_cards.ids[i], &max_cost)) {
      return true;
    }
  }

  return false;
}

void azk01_126_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  move_matching_discard_spells_to_selection(world, ctx, gate_power_from_ctx(world, ctx));
}

bool azk01_126_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  const uint8_t max_cost = gate_power_from_ctx(world, ctx);
  return target != 0 &&
         is_valid_echoed_waves_target(world, target, &max_cost);
}

void azk01_126_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  azk_move_picked_selection_cards_to_hand_if_still_in_selection(world, ctx);
  azk_return_remaining_selection_cards_to_discard(world, ctx);
}
