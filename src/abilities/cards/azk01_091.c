#include "abilities/cards/azk01_091.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_091_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);

  if (ecs_get_target(world, card, EcsChildOf, 0) != gs->zones[owner_num].garden) {
    return false;
  }

  for (uint8_t player_num = 0; player_num < MAX_PLAYERS_PER_MATCH; ++player_num) {
    ecs_entities_t garden_cards =
        ecs_get_ordered_children(world, gs->zones[player_num].garden);
    if (garden_cards.count > 0) {
      return true;
    }
  }

  return false;
}

bool azk01_091_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  return parent == gs->zones[0].garden || parent == gs->zones[1].garden;
}

void azk01_091_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  sacrifice_card(world, ctx->runtime.source_card);
}

void azk01_091_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0) {
    return;
  }

  ecs_entity_t target = ctx->effect.entities[0];
  if (target != 0) {
    apply_attack_modifier(world, target, ctx->runtime.source_card, -1, true);
  }
}
