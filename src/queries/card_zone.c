#include <flecs.h>
#include <limits.h>
#include "queries/card_zone.h"
#include "components.h"

void init_card_zone_queries(ecs_world_t *world) {
  (void)world;
}

bool get_top_card_in_zone(
  ecs_world_t *world,
  ecs_entity_t zone,
  ecs_entity_t *out_card,
  int *out_count
) {
  int total = 0;
  int best_index = INT_MAX;
  ecs_entity_t best_card = 0;

  ecs_iter_t it = ecs_each_id(world, ecs_pair(Rel_InZone, zone));
  while (ecs_each_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      ecs_entity_t entity = it.entities[i];
      const ZoneIndex *zone_index = ecs_get(world, entity, ZoneIndex);
      if (!zone_index) {
        continue;
      }
      total++;
      if (zone_index->value < best_index) {
        best_index = zone_index->value;
        best_card = entity;
      }
    }
  }
  ecs_iter_fini(&it);

  if (out_count) {
    *out_count = total;
  }
  if (out_card) {
    *out_card = best_card;
  }

  return best_card != 0;
}
