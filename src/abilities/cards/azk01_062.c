#include "abilities/cards/azk01_062.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"

bool azk01_062_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)owner;
  return azk_has_pending_damage_redirect_for_target(world, card);
}

bool azk01_062_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)owner;

  if (target == 0 || target == card || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL) {
    return false;
  }

  const ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  return parent == gs->zones[0].garden || parent == gs->zones[1].garden;
}

void azk01_062_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  PendingDamageRedirect redirect = {0};
  if (!azk_consume_pending_damage_redirect(world, ctx->runtime.source_card,
                                           &redirect)) {
    return;
  }

  ecs_entity_t resolved_target = redirect.original_target;
  if (ctx->effect.selected_count > 0 && ctx->effect.entities[0] != 0) {
    resolved_target = ctx->effect.entities[0];
  }

  deal_effect_damage_from_source(world, redirect.source_card, resolved_target,
                                 redirect.damage);
}
