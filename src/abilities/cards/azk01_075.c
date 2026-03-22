#include "abilities/cards/azk01_075.h"

#include "abilities/core/ability_context.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_075_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);

  int beanz_count = 0;
  for (int32_t i = 0; i < garden_cards.count; ++i) {
    if (has_subtype(world, garden_cards.ids[i], ecs_id(TSubtype_Beanz))) {
      beanz_count++;
    }
  }

  return beanz_count >= 2;
}

bool azk01_075_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0 || !has_subtype(world, target, ecs_id(TSubtype_Beanz))) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  if (ecs_get_target(world, target, EcsChildOf, 0) !=
      gs->zones[owner_num].garden) {
    return false;
  }

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  if (ctx == NULL) {
    return false;
  }

  for (uint8_t i = 0; i < ctx->cost.selected_count; ++i) {
    if (ctx->cost.entities[i] == target) {
      return false;
    }
  }

  return true;
}

void azk01_075_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  for (uint8_t i = 0; i < ctx->cost.selected_count; ++i) {
    ecs_entity_t target = ctx->cost.entities[i];
    if (target != 0) {
      sacrifice_card(world, target);
    }
  }
}

void azk01_075_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t card = ctx->runtime.source_card;
  apply_attack_modifier(world, card, card, 2, true);
  apply_health_modifier(world, card, card, 2, true);
}
