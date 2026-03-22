#include "abilities/cards/azk01_125.h"

#include "components/components.h"
#include "utils/player_util.h"

bool azk01_125_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL) {
    return false;
  }

  const uint8_t owner_num = get_player_number(world, owner);
  return gs->discarded_cards_this_turn[owner_num] > 0;
}

void azk01_125_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  if (gs == NULL) {
    return;
  }

  const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  gs->next_card_play_cost_reduction[owner_num] += 2;
  ecs_singleton_modified(world, GameState);
}
