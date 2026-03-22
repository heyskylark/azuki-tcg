#include "abilities/cards/azk01_042.h"

#include "abilities/core/ability_context.h"
#include "components/components.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"

bool azk01_042_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[opponent_num].garden);
  return garden_cards.count >= 3;
}

bool azk01_042_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  if (ecs_get_target(world, target, EcsChildOf, 0) !=
      gs->zones[opponent_num].garden) {
    return false;
  }

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  if (!ctx) {
    return false;
  }

  for (uint8_t i = 0; i < ctx->effect.selected_count; ++i) {
    if (ctx->effect.entities[i] == target) {
      return false;
    }
  }

  return true;
}

void azk01_042_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  static const int8_t damages[3] = {3, 2, 1};

  for (uint8_t i = 0; i < 3 && i < ctx->effect.selected_count; ++i) {
    ecs_entity_t target = ctx->effect.entities[i];
    if (target != 0) {
      deal_effect_damage(world, target, damages[i]);
    }
  }
}
