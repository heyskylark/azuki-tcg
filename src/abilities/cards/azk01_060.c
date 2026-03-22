#include "abilities/cards/azk01_060.h"

#include "components/abilities.h"
#include "components/components.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_060_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, card, EcsChildOf, 0) ==
         gs->zones[owner_num].garden;
}

void azk01_060_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t card = ctx->runtime.source_card;
  apply_timed_tag_grant(world, card, ecs_id(Infiltrate),
                        TAG_GRANT_TICK_END_OF_TURN, 1);
  apply_timed_tag_grant(world, card, ecs_id(SacrificeAtEndOfTurn),
                        TAG_GRANT_TICK_END_OF_TURN, 1);
  apply_attack_modifier(world, card, card, 1, true);
}
