#include "abilities/cards/azk01_036.h"

#include "components/components.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_036_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  return parent == gs->zones[owner_num].garden ||
         parent == gs->zones[owner_num].alley;
}

void azk01_036_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t attacker = gs->combat_state.attacking_card;
  if (attacker == 0) {
    return;
  }

  apply_shocked(world, attacker, 1);
}
