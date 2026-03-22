#include "abilities/cards/stt04_004.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"

bool stt04_004_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;
  (void)owner;

  const GameState *gs = ecs_singleton_get(world, GameState);
  return ecs_get_ordered_children(world, gs->zones[0].garden).count > 0 ||
         ecs_get_ordered_children(world, gs->zones[1].garden).count > 0;
}

bool stt04_004_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;
  return target != 0 && is_card_type(world, target, CARD_TYPE_ENTITY) &&
         (ecs_get_target(world, target, EcsChildOf, 0) ==
              ecs_singleton_get(world, GameState)->zones[0].garden ||
          ecs_get_target(world, target, EcsChildOf, 0) ==
              ecs_singleton_get(world, GameState)->zones[1].garden);
}

void stt04_004_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  sacrifice_card(world, ctx->runtime.source_card);
}

void stt04_004_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count > 0 && ctx->effect.entities[0] != 0) {
    deal_effect_damage(world, ctx->effect.entities[0], 1);
  }
}
