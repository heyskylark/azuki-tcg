#include "abilities/cards/azk01_006.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"

bool azk01_006_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  return parent == gs->zones[player_num].garden;
}

void azk01_006_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  return_card_to_hand(world, ctx->runtime.source_card);
}
