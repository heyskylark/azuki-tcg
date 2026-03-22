#include "abilities/cards/azk01_117.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"
#include "utils/zone_util.h"

static ecs_entity_t owner_leader(ecs_world_t *world, ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return find_leader_card_in_zone(world, gs->zones[owner_num].leader);
}

bool azk01_117_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  if (owner_leader(world, owner) == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  for (uint8_t player_num = 0; player_num < MAX_PLAYERS_PER_MATCH; ++player_num) {
    ecs_entities_t garden_cards =
        ecs_get_ordered_children(world, gs->zones[player_num].garden);
    for (int32_t i = 0; i < garden_cards.count; ++i) {
      const IKZCost *cost = ecs_get(world, garden_cards.ids[i], IKZCost);
      if (cost != NULL && cost->ikz_cost <= 5) {
        return true;
      }
    }
  }

  return false;
}

bool azk01_117_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;
  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  bool in_garden = false;
  for (uint8_t player_num = 0; player_num < MAX_PLAYERS_PER_MATCH; ++player_num) {
    if (parent == gs->zones[player_num].garden) {
      in_garden = true;
      break;
    }
  }
  if (!in_garden) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, target, IKZCost);
  return cost != NULL && cost->ikz_cost <= 5;
}

void azk01_117_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t leader = owner_leader(world, ctx->runtime.owner);
  if (leader != 0) {
    deal_effect_damage(world, leader, 2);
  }
}

void azk01_117_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count > 0 && ctx->effect.entities[0] != 0) {
    apply_charge_grant(world, ctx->effect.entities[0], TAG_GRANT_TICK_END_OF_TURN,
                       1);
  }
}
