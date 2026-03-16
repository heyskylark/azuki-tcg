#include "abilities/cards/stt01_003.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

// STT01-003 "Crate Rat Kurobo": On Play; Put 3 cards from the top of your deck
// into your discard pile. If you have no weapon cards in your discard pile
// when you activate this ability, put 5 cards instead.

bool stt01_003_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;

  // Always valid - no cost to pay, effect is self-contained
  return true;
}

void stt01_003_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  uint8_t player_num = get_player_number(world, ctx->owner);
  const GameState *gs = ecs_singleton_get(world, GameState);

  ecs_entity_t discard_zone = gs->zones[player_num].discard;

  // Check if there are any weapons in the discard pile
  int weapon_count = count_weapons_in_zone(world, discard_zone);
  int mill_count = (weapon_count == 0) ? 5 : 3;

  mill_cards_with_deckout_check(world, ctx->owner, mill_count, NULL);
}
