#include "abilities/cards/azk01_040.h"

#include "utils/damage_util.h"

bool azk01_040_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

bool azk01_040_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;
  return target != 0 && ecs_has(world, target, TLeader);
}

void azk01_040_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect.entities[0];
  if (target == 0) {
    return;
  }

  deal_effect_damage(world, target, 1);
}
