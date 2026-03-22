#include "abilities/cards/stt04_015.h"

#include "components/components.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

bool stt04_015_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;
  (void)owner;
  return true;
}

void stt04_015_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t leader =
      find_leader_card_in_zone(world, gs->zones[owner_num].leader);
  if (leader != 0) {
    deal_effect_damage(world, leader, 1);
  }
}

void stt04_015_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t enemy_num =
      (get_player_number(world, ctx->runtime.owner) + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entity_t leader =
      find_leader_card_in_zone(world, gs->zones[enemy_num].leader);
  if (leader != 0) {
    deal_effect_damage(world, leader, 2);
  }
}
