#include "abilities/cards/azk01_020.h"

#include "components/components.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_020_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entities_t cards =
      ecs_get_ordered_children(world, gs->zones[player_num].garden);
  return cards.count >= 2;
}

bool azk01_020_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  if (parent != gs->zones[player_num].garden) {
    return false;
  }

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  for (int i = 0; i < ctx->effect.selected_count; i++) {
    if (ctx->effect.entities[i] == target) {
      return false;
    }
  }

  return true;
}

void azk01_020_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  bool is_response = gs->phase == PHASE_RESPONSE_WINDOW;

  for (int i = 0; i < ctx->effect.selected_count; i++) {
    ecs_entity_t target = ctx->effect.entities[i];
    if (target == 0) {
      continue;
    }

    if (is_response) {
      apply_health_modifier(world, target, ctx->runtime.source_card, 1, true);
    } else {
      apply_attack_modifier(world, target, ctx->runtime.source_card, 1, true);
    }
  }
}
