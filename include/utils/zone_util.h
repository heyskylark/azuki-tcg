#ifndef AZUKI_UTILS_ZONE_UTIL_H
#define AZUKI_UTILS_ZONE_UTIL_H

#include <flecs.h>
#include "generated/card_defs.h"

typedef enum {
  ZONE_GARDEN = 0,
  ZONE_ALLEY = 1
} ZonePlacementType;

void untap_all_cards_in_zone(ecs_world_t *world, ecs_entity_t zone);
/**
  Inserts a card into a zone at a given index.
  If the given zone is not a garden or alley, or the index is out of bounds, this function will assert.
  If the given card owner is different from the zone owner, this function will assert.
  The card will be added to the zone as a child and the ZoneIndex component will be set to the given index.
  If the given index is already occupied and the zone is full, the card will be discarded. Otherwise, this function will assert.
*/
int summon_card_into_zone_index(
  ecs_world_t *world,
  ecs_entity_t card,
  ecs_entity_t player,
  ZonePlacementType placement_type,
  int index
);

int gate_card_into_garden(
  ecs_world_t *world,
  ecs_entity_t player,
  int alley_index,
  int garden_index
);

#endif