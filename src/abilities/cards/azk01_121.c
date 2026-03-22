#include "abilities/cards/azk01_121.h"

#include "components/components.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_121_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL) {
    return false;
  }

  const uint8_t owner_num = get_player_number(world, owner);
  return gs->entities_played_garden_this_turn[owner_num] +
             gs->entities_played_alley_this_turn[owner_num] >
         0;
}

void azk01_121_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  int8_t attack_buff =
      (int8_t)(gs->entities_played_garden_this_turn[owner_num] +
               gs->entities_played_alley_this_turn[owner_num]);
  if (attack_buff > 2) {
    attack_buff = 2;
  }

  if (attack_buff > 0) {
    apply_attack_modifier(world, ctx->runtime.source_card,
                          ctx->runtime.source_card, attack_buff, true);
  }
}
