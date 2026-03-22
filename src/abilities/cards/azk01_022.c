#include "abilities/cards/azk01_022.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

static bool is_valid_bounce_target(ecs_world_t *world, ecs_entity_t entity) {
  if (!is_card_type(world, entity, CARD_TYPE_ENTITY)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, entity, IKZCost);
  return cost != NULL && cost->ikz_cost <= 2;
}

bool azk01_022_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entity_t hand = gs->zones[owner_num].hand;
  ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
  if (hand_cards.count < 1) {
    return false;
  }

  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; ++p) {
    ecs_entities_t garden_cards =
        ecs_get_ordered_children(world, gs->zones[p].garden);
    for (int32_t i = 0; i < garden_cards.count; ++i) {
      if (is_valid_bounce_target(world, garden_cards.ids[i])) {
        return true;
      }
    }
  }

  return false;
}

bool azk01_022_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  if (target == 0 || target == card) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entity_t hand = gs->zones[owner_num].hand;
  return ecs_get_target(world, target, EcsChildOf, 0) == hand;
}

bool azk01_022_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0 || !is_valid_bounce_target(world, target)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; ++p) {
    if (parent == gs->zones[p].garden) {
      return true;
    }
  }

  return false;
}

void azk01_022_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t to_discard = ctx->cost.entities[0];
  if (to_discard == 0) {
    return;
  }

  discard_card(world, to_discard);
  cli_render_logf("[AZK01-022] Discarded 1 card as cost");
}

void azk01_022_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect.entities[0];
  if (target == 0) {
    return;
  }

  return_card_to_hand(world, target);
  cli_render_logf("[AZK01-022] Returned entity to owner's hand");
}
