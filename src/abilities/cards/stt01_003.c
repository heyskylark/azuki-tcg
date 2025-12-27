#include "abilities/cards/stt01_003.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"

// STT01-003 "Crate Rat Kurobo": On Play; You may put 3 cards from the top of
// your deck into your discard pile. If you have no weapon cards in your
// discard pile, put 5 cards instead.

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

  ecs_entity_t deck_zone = gs->zones[player_num].deck;
  ecs_entity_t discard_zone = gs->zones[player_num].discard;

  // Check if there are any weapons in the discard pile
  int weapon_count = count_weapons_in_zone(world, discard_zone);
  int mill_count = (weapon_count == 0) ? 5 : 3;

  // Get cards in deck (top of deck = end of array)
  ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck_zone);

  // Mill up to mill_count cards (or all remaining if deck is smaller)
  int to_mill = mill_count;
  if (to_mill > deck_cards.count) {
    to_mill = deck_cards.count;
  }

  // Mill from top of deck (last elements in the array)
  for (int i = 0; i < to_mill; i++) {
    ecs_entity_t card = deck_cards.ids[deck_cards.count - 1 - i];
    discard_card(world, card);
  }
}
