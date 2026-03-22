#include "abilities/cards/azk01_087.h"

#include "abilities/core/ability_context.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

static bool is_enemy_garden_entity_with_cost(ecs_world_t *world,
                                             ecs_entity_t target,
                                             ecs_entity_t enemy_garden,
                                             int max_cost) {
  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY) ||
      ecs_get_target(world, target, EcsChildOf, 0) != enemy_garden) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, target, IKZCost);
  return cost != NULL && cost->ikz_cost <= max_cost;
}

static int current_selected_combined_cost(ecs_world_t *world,
                                          const AbilityContext *ctx) {
  int total = 0;
  for (uint8_t i = 0; i < ctx->effect.selected_count; ++i) {
    const IKZCost *cost = ecs_get(world, ctx->effect.entities[i], IKZCost);
    if (cost != NULL) {
      total += cost->ikz_cost;
    }
  }
  return total;
}

bool azk01_087_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[opponent_num].garden);
  return garden_cards.count > 0;
}

bool azk01_087_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  if (!is_enemy_garden_entity_with_cost(world, target,
                                        gs->zones[opponent_num].garden, 5)) {
    return false;
  }

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  if (ctx == NULL) {
    return false;
  }

  for (uint8_t i = 0; i < ctx->effect.selected_count; ++i) {
    if (ctx->effect.entities[i] == target) {
      return false;
    }
  }

  const IKZCost *target_cost = ecs_get(world, target, IKZCost);
  return target_cost != NULL &&
         current_selected_combined_cost(world, ctx) + target_cost->ikz_cost <= 5;
}

void azk01_087_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  for (uint8_t i = 0; i < ctx->effect.selected_count; ++i) {
    ecs_entity_t target = ctx->effect.entities[i];
    if (target == 0) {
      continue;
    }

    ecs_entity_t owner = ecs_get_target(world, target, Rel_OwnedBy, 0);
    if (owner != 0) {
      add_card_to_bottom_of_deck(world, owner, target);
    }
  }
}
