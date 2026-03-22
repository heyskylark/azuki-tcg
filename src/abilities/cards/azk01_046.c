#include "abilities/cards/azk01_046.h"

#include "components/components.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

bool azk01_046_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, card, EcsChildOf, 0) ==
         gs->zones[owner_num].garden;
}

bool azk01_046_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;
  return target != 0 && ecs_has(world, target, TLeader);
}

void azk01_046_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0) {
    return;
  }

  ecs_entity_t target = ctx->effect.entities[0];
  if (target != 0) {
    deal_effect_damage(world, target, 1);
  }
}
