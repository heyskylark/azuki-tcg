#include "abilities/cards/azk01_071.h"

#include "components/components.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

bool azk01_071_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  return parent == gs->zones[owner_num].garden ||
         parent == gs->zones[owner_num].alley;
}

void azk01_071_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  add_card_to_bottom_of_deck(world, ctx->runtime.owner, ctx->runtime.source_card);
}

void azk01_071_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  draw_cards_with_deckout_check(world, ctx->runtime.owner, 1, NULL);
}
