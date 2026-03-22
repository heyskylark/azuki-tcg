#include "abilities/cards/azk01_014.h"

#include "components/components.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_014_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entities_t cards =
      ecs_get_ordered_children(world, gs->zones[player_num].garden);
  int others = 0;

  for (int32_t i = 0; i < cards.count; i++) {
    if (cards.ids[i] != card) {
      others++;
    }
  }

  return others > 0;
}

bool azk01_014_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  if (target == 0 || target == card) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  return parent == gs->zones[player_num].garden;
}

void azk01_014_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect.entities[0];
  if (target == 0) {
    return;
  }

  apply_attack_modifier(world, target, ctx->runtime.source_card, 2, true);
}
