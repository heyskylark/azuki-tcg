#include "abilities/cards/azk01_008.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/entity_util.h"
#include "utils/player_util.h"

static bool is_valid_assassin_target(ecs_world_t *world, ecs_entity_t target) {
  if (!is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, target, IKZCost);
  return cost != NULL && cost->ikz_cost <= 3;
}

bool azk01_008_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entities_t cards =
      ecs_get_ordered_children(world, gs->zones[opponent_num].garden);

  for (int32_t i = 0; i < cards.count; i++) {
    if (is_valid_assassin_target(world, cards.ids[i])) {
      return true;
    }
  }

  return false;
}

bool azk01_008_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0 || !is_valid_assassin_target(world, target)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  return parent == gs->zones[opponent_num].garden;
}

void azk01_008_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  discard_equipped_weapon_cards(world, ctx->runtime.source_card);
  sacrifice_card(world, ctx->runtime.source_card);
}

void azk01_008_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect.entities[0];
  if (target == 0) {
    return;
  }

  discard_equipped_weapon_cards(world, target);
  sacrifice_card(world, target);
}
