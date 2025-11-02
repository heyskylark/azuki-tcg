#ifndef AZUKI_ECS_QUERIES_CARD_ZONE_H
#define AZUKI_ECS_QUERIES_CARD_ZONE_H

#include <stdbool.h>
#include <flecs.h>

void init_card_zone_queries(ecs_world_t *world);
bool get_top_card_in_zone(ecs_world_t *world, ecs_entity_t zone, ecs_entity_t *out_card, int *out_count);

#endif
