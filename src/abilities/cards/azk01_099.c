#include "abilities/cards/azk01_099.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

#define AZK01_099_MAX_SHOCK_COST 5

static ecs_entity_t get_opponent_garden(ecs_world_t *world,
                                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  return gs->zones[(owner_num + 1) % MAX_PLAYERS_PER_MATCH].garden;
}

static bool is_shock_candidate(ecs_world_t *world, ecs_entity_t target) {
  if (!is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, target, IKZCost);
  return cost != NULL && cost->ikz_cost <= AZK01_099_MAX_SHOCK_COST;
}

bool azk01_099_validate_shock(ecs_world_t *world, ecs_entity_t card,
                              ecs_entity_t owner) {
  (void)card;
  ecs_entities_t cards =
      ecs_get_ordered_children(world, get_opponent_garden(world, owner));
  for (int32_t i = 0; i < cards.count; ++i) {
    if (is_shock_candidate(world, cards.ids[i])) {
      return true;
    }
  }
  return false;
}

bool azk01_099_validate_shock_target(ecs_world_t *world, ecs_entity_t card,
                                     ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  return target != 0 && is_shock_candidate(world, target) &&
         ecs_get_target(world, target, EcsChildOf, 0) ==
             get_opponent_garden(world, owner);
}

void azk01_099_apply_shock(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0 || ctx->effect.entities[0] == 0) {
    return;
  }

  apply_shocked(world, ctx->effect.entities[0], 1);
}

void azk01_099_apply_charge(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->runtime.source_card == 0) {
    return;
  }

  (void)apply_charge_grant(world, ctx->runtime.source_card,
                           TAG_GRANT_TICK_END_OF_TURN, 1);
}
