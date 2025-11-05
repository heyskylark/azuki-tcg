#ifndef AZUKI_UTILS_CARD_UTILS_H
#define AZUKI_UTILS_CARD_UTILS_H

#include <flecs.h>
#include "generated/card_defs.h"

bool is_card_type(ecs_world_t *world, ecs_entity_t card, CardType type);
void discard_card(ecs_world_t *world, ecs_entity_t card);
void untap_all_cards_in_zone(ecs_world_t *world, ecs_entity_t zone);
/**
  Inserts a card into a zone at a given index.
  If the given zone is not a garden or alley, or the index is out of bounds, this function will assert.
  If the given card owner is different from the zone owner, this function will assert.
  The card will be added to the zone as a child and the ZoneIndex component will be set to the given index.
  If the given index is already occupied and the zone is full, the card will be discarded. Otherwise, this function will assert.
*/
int insert_card_into_zone_index(ecs_world_t *world, ecs_entity_t card, ecs_entity_t zone, int index);

#endif
