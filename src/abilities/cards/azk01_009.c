#include "abilities/cards/azk01_009.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/status_util.h"

static bool is_valid_red_bean_target(ecs_world_t *world, ecs_entity_t target) {
  if (!is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, target, IKZCost);
  return cost != NULL && cost->ikz_cost <= 4;
}

bool azk01_009_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;
  (void)owner;

  const GameState *gs = ecs_singleton_get(world, GameState);
  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; p++) {
    ecs_entities_t cards =
        ecs_get_ordered_children(world, gs->zones[p].garden);
    for (int32_t i = 0; i < cards.count; i++) {
      if (is_valid_red_bean_target(world, cards.ids[i])) {
        return true;
      }
    }
  }

  return false;
}

bool azk01_009_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0 || !is_valid_red_bean_target(world, target)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  return parent == gs->zones[0].garden || parent == gs->zones[1].garden;
}

void azk01_009_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect.entities[0];
  if (target == 0) {
    return;
  }

  apply_charge_grant(world, target, TAG_GRANT_TICK_END_OF_TURN, 1);
}
