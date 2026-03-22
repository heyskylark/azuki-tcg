#include "abilities/cards/azk01_068.h"

#include "components/components.h"
#include "utils/deck_utils.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"

bool azk01_068_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);

  if (ecs_get_target(world, card, EcsChildOf, 0) !=
      gs->zones[owner_num].alley) {
    return false;
  }

  ecs_entities_t deck_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].deck);
  return deck_cards.count > 0;
}

bool azk01_068_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[owner_num].hand;
}

void azk01_068_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  draw_cards_with_deckout_check(world, ctx->runtime.owner, 1, NULL);
}

void azk01_068_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0) {
    return;
  }

  ecs_entity_t to_discard = ctx->effect.entities[0];
  if (to_discard != 0) {
    discard_card(world, to_discard);
  }
}
