#include "abilities/cards/azk01_013.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"

#define AZK01_013_DISCARD_COUNT 2

static ecs_entity_t get_zone(ecs_world_t *world, ecs_entity_t player,
                             bool garden) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t player_num = get_player_number(world, player);
  return garden ? gs->zones[player_num].garden : gs->zones[player_num].hand;
}

bool azk01_013_validate_sacrifice(ecs_world_t *world, ecs_entity_t card,
                                  ecs_entity_t owner) {
  (void)card;
  return ecs_get_ordered_children(world, get_zone(world, owner, true)).count >
         0;
}

bool azk01_013_validate_sacrifice_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  return target != 0 && is_card_type(world, target, CARD_TYPE_ENTITY) &&
         ecs_get_target(world, target, EcsChildOf, 0) ==
             get_zone(world, owner, true);
}

void azk01_013_apply_sacrifice(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0 || ctx->effect.entities[0] == 0) {
    return;
  }

  sacrifice_card(world, ctx->effect.entities[0]);
}

bool azk01_013_validate_discard(ecs_world_t *world, ecs_entity_t card,
                                ecs_entity_t owner) {
  (void)card;
  return ecs_get_ordered_children(world, get_zone(world, owner, false)).count >
         0;
}

bool azk01_013_validate_discard_target(ecs_world_t *world, ecs_entity_t card,
                                       ecs_entity_t owner,
                                       ecs_entity_t target) {
  (void)card;
  if (target == 0 || ecs_get_target(world, target, EcsChildOf, 0) !=
                         get_zone(world, owner, false)) {
    return false;
  }

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  for (uint8_t i = 0; i < ctx->effect.selected_count; ++i) {
    if (ctx->effect.entities[i] == target) {
      return false;
    }
  }
  return true;
}

void azk01_013_begin_discard(ecs_world_t *world, AbilityContext *ctx) {
  const int32_t hand_count =
      ecs_get_ordered_children(world, get_zone(world, ctx->runtime.owner, false))
          .count;
  const uint8_t discard_count = hand_count < AZK01_013_DISCARD_COUNT
                                    ? (uint8_t)hand_count
                                    : AZK01_013_DISCARD_COUNT;
  if (discard_count == 0) {
    return;
  }

  ctx->effect.min_required = discard_count;
  ctx->effect.max_allowed = discard_count;
  ctx->runtime.phase = ABILITY_PHASE_EFFECT_SELECTION;
}

void azk01_013_apply_discard(ecs_world_t *world, const AbilityContext *ctx) {
  for (uint8_t i = 0; i < ctx->effect.selected_count; ++i) {
    if (ctx->effect.entities[i] != 0) {
      discard_card(world, ctx->effect.entities[i]);
    }
  }
}
