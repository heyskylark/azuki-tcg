#include "abilities/cards/azk01_118.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"

bool azk01_118_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, card, EcsChildOf, 0) == gs->zones[owner_num].garden;
}

bool azk01_118_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) == gs->zones[owner_num].garden;
}

bool azk01_118_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  if (gs->cards_played_this_turn[owner_num] < 3) {
    return false;
  }

  uint8_t enemy_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  return ecs_get_target(world, target, EcsChildOf, 0) == gs->zones[enemy_num].garden;
}

void azk01_118_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->cost.selected_count > 0 && ctx->cost.entities[0] != 0) {
    deal_effect_damage(world, ctx->cost.entities[0], 1);
  }
}

void azk01_118_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count > 0 && ctx->effect.entities[0] != 0) {
    deal_effect_damage(world, ctx->effect.entities[0], 2);
  }
}
