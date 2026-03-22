#include "abilities/cards/azk01_101.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

static bool is_earth_garden_entity(ecs_world_t *world, ecs_entity_t entity) {
  return is_card_type(world, entity, CARD_TYPE_ENTITY) &&
         get_card_element(world, entity) == CARD_ELEMENT_EARTH;
}

bool azk01_101_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);

  for (int32_t i = 0; i < garden_cards.count; ++i) {
    if (is_earth_garden_entity(world, garden_cards.ids[i])) {
      return true;
    }
  }

  return false;
}

bool azk01_101_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0 || !is_earth_garden_entity(world, target)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[owner_num].garden;
}

void azk01_101_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0) {
    return;
  }

  ecs_entity_t target = ctx->effect.entities[0];
  if (target != 0) {
    apply_health_modifier(world, target, ctx->runtime.source_card, 3, true);
  }
}
