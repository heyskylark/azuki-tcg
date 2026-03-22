#include "abilities/cards/azk01_113.h"

#include "components/components.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_113_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return gs->cards_played_this_turn[owner_num] >= 3;
}

void azk01_113_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  apply_charge_grant(world, ctx->runtime.source_card, TAG_GRANT_TICK_END_OF_TURN,
                     1);
}
