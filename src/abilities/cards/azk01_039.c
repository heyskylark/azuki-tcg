#include "abilities/cards/azk01_039.h"

#include "utils/status_util.h"

bool azk01_039_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

void azk01_039_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target =
      ecs_get_target(world, ctx->runtime.source_card, EcsChildOf, 0);
  if (target == 0) {
    return;
  }

  apply_charge_grant(world, target, TAG_GRANT_TICK_NONE, -1);
}
