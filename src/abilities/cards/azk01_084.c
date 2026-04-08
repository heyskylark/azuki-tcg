#include "abilities/cards/azk01_084.h"

#include "abilities/selection/ability_selection_helpers.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"

static bool is_valid_replica_target(ecs_world_t *world, ecs_entity_t card) {
  if (!is_card_type(world, card, CARD_TYPE_ENTITY) ||
      !is_normal_element_card(world, card)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, card, IKZCost);
  return cost != NULL && cost->ikz_cost <= 6;
}

static uint8_t move_matching_discard_entities_to_selection(ecs_world_t *world,
                                                           AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  const ecs_entity_t discard_zone = gs->zones[player_num].discard;
  const ecs_entity_t selection_zone = gs->zones[player_num].selection;
  ecs_entities_t discard_cards = ecs_get_ordered_children(world, discard_zone);

  ecs_entity_t selection_cards[MAX_SELECTION_ZONE_SIZE] = {0};
  uint8_t selection_count = 0;
  for (int32_t i = 0;
       i < discard_cards.count && selection_count < MAX_SELECTION_ZONE_SIZE;
       ++i) {
    ecs_entity_t target = discard_cards.ids[i];
    if (!is_valid_replica_target(world, target)) {
      continue;
    }
    selection_cards[selection_count++] = target;
  }

  for (uint8_t i = 0; i < selection_count; ++i) {
    ecs_entity_t target = selection_cards[i];
    int8_t from_index = azk_get_card_index_in_zone(world, target, discard_zone);
    ecs_add_pair(world, target, EcsChildOf, selection_zone);
    azk_log_card_zone_moved(world, target, GLOG_ZONE_DISCARD, from_index,
                            GLOG_ZONE_SELECTION, (int8_t)i);
  }

  azk_init_selection_state(ctx, selection_cards, selection_count, 1);
  ctx->runtime.phase =
      selection_count > 0 ? ABILITY_PHASE_SELECTION_PICK : ABILITY_PHASE_NONE;
  return selection_count;
}

bool azk01_084_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t discard_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].discard);
  for (int32_t i = 0; i < discard_cards.count; ++i) {
    if (is_valid_replica_target(world, discard_cards.ids[i])) {
      return true;
    }
  }

  return false;
}

void azk01_084_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  const uint8_t selection_count =
      move_matching_discard_entities_to_selection(world, ctx);
  if (selection_count == 0) {
    ctx->runtime.phase = ABILITY_PHASE_NONE;
    cli_render_logf("[AZK01-084] No valid Normal entities in discard");
    return;
  }

  cli_render_logf("[AZK01-084] Found %u valid entity card(s) in discard",
                  (unsigned)selection_count);
}

bool azk01_084_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;
  return target != 0 && is_valid_replica_target(world, target);
}

void azk01_084_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  if (azk_move_picked_selection_cards_to_hand_if_still_in_selection(world, ctx) >
      0) {
    cli_render_logf("[AZK01-084] Returned selected entity to hand");
  }

  azk_return_remaining_selection_cards_to_discard(world, ctx);
}
