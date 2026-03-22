#include "abilities/cards/azk01_007.h"

#include "components/components.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_007_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entities_t cards =
      ecs_get_ordered_children(world, gs->zones[player_num].garden);
  return cards.count > 0;
}

bool azk01_007_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  return parent == gs->zones[player_num].garden;
}

void azk01_007_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect.entities[0];
  if (target == 0) {
    return;
  }

  apply_attack_modifier(world, target, ctx->runtime.source_card, 1, true);
}
