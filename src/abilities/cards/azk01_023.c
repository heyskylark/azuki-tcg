#include "abilities/cards/azk01_023.h"

#include "components/components.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

bool azk01_023_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  if (parent != gs->zones[owner_num].garden) {
    return false;
  }

  ecs_entities_t hand_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].hand);
  return hand_cards.count <= 2;
}

void azk01_023_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  draw_cards_with_deckout_check(world, ctx->runtime.owner, 1, NULL);
}
