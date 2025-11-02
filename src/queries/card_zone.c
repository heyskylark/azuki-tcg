#include <flecs.h>
#include "queries/card_zone.h"
#include "components.h"

static int compare_zone_index_desc(
  ecs_entity_t e1,
  const void *v1,
  ecs_entity_t e2,
  const void *v2
) {
  const ZoneIndex *zi1 = v1;
  const ZoneIndex *zi2 = v2;

  return zi1->value - zi2->value;
}

static ecs_query_t *cards_owned_by_player_in_zone;

void init_card_zone_queries(ecs_world_t *world) {
  cards_owned_by_player_in_zone = ecs_query(world, {
    .terms = {
      { ecs_id(BaseStats) },
      { ecs_id(TapState) },
      { ecs_id(Element) },
      { ecs_id(GatePoints) },
      { ecs_id(IKZCost) },
      { ecs_id(ZoneIndex) },
      {
        .first.id = ecs_id(Rel_OwnedBy),
        .second = {
          .name = "$Player",
        }
      },
      {
        .first.id = ecs_id(Rel_InZone),
        .second = {
          .name = "$Zone",
        }
      },
    },
    .order_by = ecs_id(ZoneIndex),
    .order_by_callback = compare_zone_index_desc,
  });
}

ecs_iter_t get_cards_owned_by_player_in_zone(
  ecs_world_t *world,
  ecs_entity_t player,
  ecs_entity_t zone
) {
  ecs_assert(cards_owned_by_player_in_zone != NULL, ECS_INVALID_PARAMETER, "cards_owned_by_player_in_zone query not initialized");

  int32_t player_var = ecs_query_find_var(cards_owned_by_player_in_zone, "Player");
  int32_t zone_var = ecs_query_find_var(cards_owned_by_player_in_zone, "Zone");
  
  ecs_iter_t it = ecs_query_iter(world, cards_owned_by_player_in_zone);
  ecs_iter_set_var(&it, player_var, player);
  ecs_iter_set_var(&it, zone_var, zone);

  return it;
}
