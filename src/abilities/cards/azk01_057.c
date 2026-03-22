#include "abilities/cards/azk01_057.h"

#include "abilities/core/ability_context.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"

static bool is_card_in_zone(ecs_world_t *world, ecs_entity_t card,
                            ecs_entity_t zone) {
  return ecs_get_target(world, card, EcsChildOf, 0) == zone;
}

bool azk01_057_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;

  if (!is_card_in_zone(world, card, gs->zones[owner_num].garden)) {
    return false;
  }

  ecs_entities_t own_garden =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  ecs_entities_t opponent_garden =
      ecs_get_ordered_children(world, gs->zones[opponent_num].garden);
  return own_garden.count > 0 && opponent_garden.count > 0;
}

bool azk01_057_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);

  bool is_own_target = parent == gs->zones[owner_num].garden;
  bool is_opponent_target = parent == gs->zones[opponent_num].garden;
  if (!is_own_target && !is_opponent_target) {
    return false;
  }

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  if (ctx == NULL) {
    return false;
  }

  for (uint8_t i = 0; i < ctx->effect.selected_count; ++i) {
    ecs_entity_t selected = ctx->effect.entities[i];
    if (selected == target) {
      return false;
    }

    ecs_entity_t selected_parent = ecs_get_target(world, selected, EcsChildOf, 0);
    if ((selected_parent == gs->zones[owner_num].garden && is_own_target) ||
        (selected_parent == gs->zones[opponent_num].garden &&
         is_opponent_target)) {
      return false;
    }
  }

  return true;
}

void azk01_057_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  for (uint8_t i = 0; i < ctx->effect.selected_count; ++i) {
    ecs_entity_t target = ctx->effect.entities[i];
    if (target != 0) {
      deal_effect_damage(world, target, 1);
    }
  }
}
