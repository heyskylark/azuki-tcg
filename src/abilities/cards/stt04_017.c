#include "abilities/cards/stt04_017.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"

bool stt04_017_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_ordered_children(world, gs->zones[owner_num].garden).count > 0;
}

bool stt04_017_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  if (ecs_get_target(world, target, EcsChildOf, 0) != gs->zones[owner_num].garden) {
    return false;
  }

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  for (uint8_t i = 0; ctx != NULL && i < ctx->cost.selected_count; ++i) {
    if (ctx->cost.entities[i] == target) {
      return false;
    }
  }

  return true;
}

bool stt04_017_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
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

void stt04_017_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  AbilityContext *ctx_mut = ecs_singleton_get_mut(world, AbilityContext);
  if (ctx_mut == NULL) {
    return;
  }

  ctx_mut->scratch.kind = ABILITY_SCRATCH_SACRIFICE_VALUE;
  ctx_mut->scratch.data.sacrifice_value.damage = ctx->cost.selected_count;
  ctx_mut->scratch.data.sacrifice_value.draw_after_effect = false;

  for (uint8_t i = 0; i < ctx->cost.selected_count; ++i) {
    if (ctx->cost.entities[i] != 0) {
      sacrifice_card(world, ctx->cost.entities[i]);
    }
  }
}

void stt04_017_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0 || ctx->effect.entities[0] == 0 ||
      ctx->scratch.kind != ABILITY_SCRATCH_SACRIFICE_VALUE ||
      ctx->scratch.data.sacrifice_value.damage <= 0) {
    return;
  }

  deal_effect_damage(world, ctx->effect.entities[0],
                     ctx->scratch.data.sacrifice_value.damage);
}
