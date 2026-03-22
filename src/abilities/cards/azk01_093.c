#include "abilities/cards/azk01_093.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

static bool is_valid_bottom_deck_target(ecs_world_t *world, ecs_entity_t target) {
  if (!is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, target, IKZCost);
  return cost != NULL && cost->ikz_cost <= 4;
}

bool azk01_093_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  if (ecs_get_target(world, card, EcsChildOf, 0) != gs->zones[owner_num].garden) {
    return false;
  }

  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[opponent_num].garden);
  for (int32_t i = 0; i < garden_cards.count; ++i) {
    if (is_valid_bottom_deck_target(world, garden_cards.ids[i])) {
      return true;
    }
  }

  return false;
}

bool azk01_093_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0 || !is_valid_bottom_deck_target(world, target)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[opponent_num].garden;
}

void azk01_093_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0) {
    return;
  }

  ecs_entity_t target = ctx->effect.entities[0];
  ecs_entity_t owner = ecs_get_target(world, target, Rel_OwnedBy, 0);
  if (target != 0 && owner != 0) {
    add_card_to_bottom_of_deck(world, owner, target);
  }
}
