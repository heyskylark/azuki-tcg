#include "abilities/cards/stt04_010.h"

#include "utils/damage_util.h"

bool stt04_010_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

void stt04_010_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  deal_effect_damage(world, ctx->runtime.source_card, 1);
}
