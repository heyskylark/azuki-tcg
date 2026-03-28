#include "abilities/cards/azk01_112.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

static bool is_untapped_friendly_garden_entity(ecs_world_t *world,
                                               ecs_entity_t owner,
                                               ecs_entity_t target) {
  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY) ||
      is_card_tapped(world, target)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) == gs->zones[owner_num].garden;
}

bool azk01_112_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  for (int32_t i = 0; i < garden_cards.count; ++i) {
    if (is_untapped_friendly_garden_entity(world, owner, garden_cards.ids[i])) {
      return true;
    }
  }

  return false;
}

bool azk01_112_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  return is_untapped_friendly_garden_entity(world, owner, target);
}

void azk01_112_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->cost.selected_count > 0 && ctx->cost.entities[0] != 0) {
    sacrifice_card(world, ctx->cost.entities[0]);
  }
}

void azk01_112_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t source_zone =
      ecs_get_target(world, ctx->runtime.source_card, EcsChildOf, 0);
  if (source_zone != gs->zones[owner_num].garden &&
      source_zone != gs->zones[owner_num].alley) {
    return;
  }

  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  bool has_remaining_garden_entity = false;

  for (int32_t i = 0; i < garden_cards.count; ++i) {
    ecs_entity_t garden_card = garden_cards.ids[i];
    bool selected_as_cost = false;

    for (uint8_t j = 0; j < ctx->cost.selected_count; ++j) {
      if (ctx->cost.entities[j] == garden_card) {
        selected_as_cost = true;
        break;
      }
    }

    if (!selected_as_cost) {
      has_remaining_garden_entity = true;
      break;
    }
  }

  if (!has_remaining_garden_entity) {
    apply_charge_grant(world, ctx->runtime.source_card, TAG_GRANT_TICK_NONE,
                       -1);
  }
}
