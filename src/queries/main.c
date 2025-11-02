#include "queries/main.h"
#include "queries/card_zone.h"

void init_all_queries(ecs_world_t *world) {
  init_card_zone_queries(world);
}
