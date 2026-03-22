#include "abilities/cards/azk01_128.h"

#include "components/components.h"
#include "utils/card_utils.h"

bool azk01_128_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;
  (void)owner;

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL || gs->combat_state.attacking_card == 0) {
    return false;
  }

  const CurStats *stats = ecs_get(world, gs->combat_state.attacking_card, CurStats);
  return stats != NULL && stats->cur_hp <= 2;
}

bool azk01_128_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL || target == 0 || target != gs->combat_state.attacking_card ||
      !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const CurStats *stats = ecs_get(world, target, CurStats);
  return stats != NULL && stats->cur_hp <= 2;
}

void azk01_128_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0 || ctx->effect.entities[0] == 0) {
    return;
  }

  discard_card(world, ctx->effect.entities[0]);
}
