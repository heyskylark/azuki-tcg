#include "utils/card_utils.h"
#include "generated/card_defs.h"
#include <stdio.h>

void untap_all_cards_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t card = cards.ids[i];
    ecs_set(world, card, TapState, { .tapped = false });
  }

  printf("Untapped cards in zone %s: %d\n", ecs_get_name(world, zone), cards.count);
}