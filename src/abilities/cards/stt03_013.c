#include "abilities/cards/stt03_013.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"

bool stt03_013_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  if (ecs_get_target(world, card, EcsChildOf, 0) != gs->zones[owner_num].garden) {
    return false;
  }

  return can_tap_card(world, card, true);
}

void stt03_013_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  tap_card(world, ctx->runtime.source_card);
}
