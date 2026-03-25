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

static bool is_entity_in_garden(ecs_world_t *world, ecs_entity_t entity,
                                ecs_entity_t garden) {
  return entity != 0 && is_card_type(world, entity, CARD_TYPE_ENTITY) &&
         ecs_get_target(world, entity, EcsChildOf, 0) == garden;
}

bool azk01_057_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);

  if (!is_card_in_zone(world, card, gs->zones[owner_num].garden)) {
    return false;
  }

  ecs_entities_t own_garden =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  return own_garden.count > 0;
}

void azk01_057_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  if (ctx == NULL) {
    return;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL || ctx->runtime.owner == 0) {
    return;
  }

  const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  const uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  const ecs_entities_t opponent_garden =
      ecs_get_ordered_children(world, gs->zones[opponent_num].garden);

  ctx->effect.min_required = opponent_garden.count > 0 ? 2 : 1;
  ctx->effect.max_allowed = ctx->effect.min_required;
  ctx->runtime.phase = ABILITY_PHASE_EFFECT_SELECTION;
}

bool azk01_057_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  if (ctx == NULL) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  const bool is_own_target =
      is_entity_in_garden(world, target, gs->zones[owner_num].garden);
  const bool is_opponent_target =
      is_entity_in_garden(world, target, gs->zones[opponent_num].garden);
  if (!is_own_target && !is_opponent_target) {
    return false;
  }

  if (ctx->effect.selected_count == 0) {
    return is_own_target;
  }

  if (ctx->effect.selected_count == 1) {
    return is_entity_in_garden(world, ctx->effect.entities[0],
                               gs->zones[owner_num].garden) &&
           is_opponent_target && target != ctx->effect.entities[0];
  }

  return false;
}

void azk01_057_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  for (uint8_t i = 0; i < ctx->effect.selected_count; ++i) {
    ecs_entity_t target = ctx->effect.entities[i];
    if (target != 0) {
      deal_effect_damage(world, target, 1);
    }
  }
}
