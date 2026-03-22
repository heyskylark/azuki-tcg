#include "abilities/cards/azk01_115.h"

#include "components/components.h"
#include "utils/damage_util.h"
#include "utils/zone_util.h"

bool azk01_115_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)owner;
  return card != 0 && ecs_is_valid(world, card);
}

void azk01_115_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  for (uint8_t player_num = 0; player_num < MAX_PLAYERS_PER_MATCH; ++player_num) {
    ecs_entity_t leader = find_leader_card_in_zone(world, gs->zones[player_num].leader);
    if (leader != 0) {
      deal_effect_damage(world, leader, 1);
    }
  }
}
