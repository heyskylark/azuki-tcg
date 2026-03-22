#include "abilities/cards/azk01_011.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/entity_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

bool azk01_011_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  if (card == 0 || owner == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (!gs) {
    return false;
  }

  uint8_t owner_num = get_player_number(world, owner);
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);

  return parent == gs->zones[owner_num].garden;
}

void azk01_011_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t card = ctx->runtime.source_card;
  if (card == 0) {
    return;
  }

  if (!is_card_tapped(world, card)) {
    discard_equipped_weapon_cards(world, card);
    discard_card(world, card);
  }
}
