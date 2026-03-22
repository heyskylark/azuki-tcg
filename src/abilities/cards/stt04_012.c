#include "abilities/cards/stt04_012.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"

bool stt04_012_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)owner;
  const DamageTracker *tracker = ecs_get(world, card, DamageTracker);
  return tracker != NULL && tracker->took_damage_this_turn;
}

bool stt04_012_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;
  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  return parent == gs->zones[0].garden || parent == gs->zones[1].garden ||
         parent == gs->zones[0].leader || parent == gs->zones[1].leader;
}

void stt04_012_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count > 0 && ctx->effect.entities[0] != 0) {
    deal_effect_damage(world, ctx->effect.entities[0], 1);
  }
}
