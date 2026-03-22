#include "abilities/cards/azk01_116.h"

#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

bool azk01_116_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

void azk01_116_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t leader = find_leader_card_in_zone(world, gs->zones[owner_num].leader);
  if (leader != 0) {
    deal_effect_damage(world, leader, 3);
  }
}
