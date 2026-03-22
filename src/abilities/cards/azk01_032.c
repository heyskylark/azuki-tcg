#include "abilities/cards/azk01_032.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

static bool is_valid_cost_target(ecs_world_t *world, ecs_entity_t entity) {
  if (!is_card_type(world, entity, CARD_TYPE_ENTITY)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, entity, IKZCost);
  return cost != NULL && cost->ikz_cost >= 2;
}

static bool is_valid_effect_target(ecs_world_t *world, ecs_entity_t entity) {
  if (!is_card_type(world, entity, CARD_TYPE_ENTITY)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, entity, IKZCost);
  return cost != NULL && cost->ikz_cost <= 4;
}

bool azk01_032_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);

  for (int32_t i = 0; i < garden_cards.count; ++i) {
    if (is_valid_cost_target(world, garden_cards.ids[i])) {
      return true;
    }
  }

  return false;
}

bool azk01_032_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0 || !is_valid_cost_target(world, target)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[owner_num].garden;
}

bool azk01_032_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0 || !is_valid_effect_target(world, target)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[opponent_num].garden;
}

void azk01_032_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->cost.entities[0];
  if (target == 0) {
    return;
  }

  return_card_to_hand(world, target);
  cli_render_logf("[AZK01-032] Returned cost target to hand");
}

void azk01_032_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect.entities[0];
  if (target == 0) {
    cli_render_logf("[AZK01-032] No opponent target selected");
    return;
  }

  return_card_to_hand(world, target);
  cli_render_logf("[AZK01-032] Returned opponent entity to hand");
}
