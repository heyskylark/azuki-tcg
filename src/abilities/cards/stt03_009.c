#include "abilities/cards/stt03_009.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

bool stt03_009_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_ordered_children(world, gs->zones[owner_num].ikz_pile).count > 0;
}

void stt03_009_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t moved[1] = {0};
  if (!move_cards_to_zone(world, gs->zones[owner_num].ikz_pile,
                          gs->zones[owner_num].ikz_area, 1, moved) ||
      moved[0] == 0) {
    return;
  }

  tap_card(world, moved[0]);
}
