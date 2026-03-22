#include "abilities/cards/azk01_105.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"

bool azk01_105_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, card, EcsChildOf, 0) == gs->zones[owner_num].garden;
}

bool azk01_105_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t enemy_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  return parent == gs->zones[enemy_num].leader ||
         parent == gs->zones[enemy_num].garden;
}

void azk01_105_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  const CurStats *stats = ecs_get(world, ctx->runtime.source_card, CurStats);
  int8_t damage = stats != NULL ? stats->cur_hp : 0;

  AbilityContext *ctx_mut = ecs_singleton_get_mut(world, AbilityContext);
  ctx_mut->scratch = (AbilityScratchState){
      .kind = ABILITY_SCRATCH_SACRIFICE_VALUE,
      .data.sacrifice_value = {.damage = damage, .draw_after_effect = false},
  };
  ecs_singleton_modified(world, AbilityContext);

  sacrifice_card(world, ctx->runtime.source_card);
}

void azk01_105_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0 ||
      ctx->scratch.kind != ABILITY_SCRATCH_SACRIFICE_VALUE) {
    return;
  }

  ecs_entity_t target = ctx->effect.entities[0];
  if (target != 0 && ctx->scratch.data.sacrifice_value.damage > 0) {
    deal_effect_damage(world, target, ctx->scratch.data.sacrifice_value.damage);
  }
}
