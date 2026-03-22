#include "abilities/cards/azk01_102.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

static bool is_valid_carapace_target(ecs_world_t *world, ecs_entity_t target) {
  if (!is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, target, IKZCost);
  return cost != NULL && cost->ikz_cost <= 4;
}

bool azk01_102_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;
  (void)owner;

  const GameState *gs = ecs_singleton_get(world, GameState);
  for (uint8_t player_num = 0; player_num < MAX_PLAYERS_PER_MATCH; ++player_num) {
    ecs_entities_t garden_cards =
        ecs_get_ordered_children(world, gs->zones[player_num].garden);
    for (int32_t i = 0; i < garden_cards.count; ++i) {
      if (is_valid_carapace_target(world, garden_cards.ids[i])) {
        return true;
      }
    }
  }

  return false;
}

bool azk01_102_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0 || !is_valid_carapace_target(world, target)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  for (uint8_t player_num = 0; player_num < MAX_PLAYERS_PER_MATCH; ++player_num) {
    if (parent == gs->zones[player_num].garden) {
      return true;
    }
  }

  return false;
}

void azk01_102_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0) {
    return;
  }

  ecs_entity_t target = ctx->effect.entities[0];
  if (target != 0) {
    apply_carapace_modifier(world, target, ctx->runtime.source_card, 1, true);
  }
}
