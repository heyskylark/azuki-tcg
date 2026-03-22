#include "abilities/cards/stt03_012.h"

#include "abilities/selection/ability_selection_helpers.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"

static bool is_small_hand_entity(ecs_world_t *world, ecs_entity_t card,
                                 const void *user_ctx) {
  (void)user_ctx;
  if (!is_card_type(world, card, CARD_TYPE_ENTITY)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, card, IKZCost);
  return cost != NULL && cost->ikz_cost <= 2;
}

bool stt03_012_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t hand_cards = ecs_get_ordered_children(world, gs->zones[owner_num].hand);
  for (int32_t i = 0; i < hand_cards.count; ++i) {
    if (is_small_hand_entity(world, hand_cards.ids[i], NULL)) {
      return true;
    }
  }

  return false;
}

void stt03_012_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  if (azk_move_matching_hand_cards_to_selection(world, ctx, 1,
                                                is_small_hand_entity, NULL) == 0) {
    ctx->runtime.phase = ABILITY_PHASE_NONE;
  }
}

bool stt03_012_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;
  return target != 0 && is_small_hand_entity(world, target, NULL);
}

void stt03_012_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  azk_return_remaining_selection_cards_to_hand(world, ctx);
}
