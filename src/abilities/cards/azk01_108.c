#include "abilities/cards/azk01_108.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"

static bool is_friendly_earth_garden_entity(ecs_world_t *world,
                                            ecs_entity_t owner,
                                            ecs_entity_t target) {
  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY) ||
      get_card_element(world, target) != CARD_ELEMENT_EARTH) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) == gs->zones[owner_num].garden;
}

bool azk01_108_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  for (int32_t i = 0; i < garden_cards.count; ++i) {
    if (is_friendly_earth_garden_entity(world, owner, garden_cards.ids[i])) {
      return true;
    }
  }

  return false;
}

bool azk01_108_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  return is_friendly_earth_garden_entity(world, owner, target);
}

bool azk01_108_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
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

void azk01_108_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->cost.selected_count == 0 || ctx->effect.selected_count == 0) {
    return;
  }

  const CurStats *stats = ecs_get(world, ctx->cost.entities[0], CurStats);
  int8_t damage = stats != NULL ? stats->cur_hp : 0;
  if (damage > 5) {
    damage = 5;
  }

  if (ctx->effect.entities[0] != 0 && damage > 0) {
    deal_effect_damage(world, ctx->effect.entities[0], damage);
  }
}
