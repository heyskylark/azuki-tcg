#ifndef AZUKI_ECS_QUERIES_CARD_ZONE_H
#define AZUKI_ECS_QUERIES_CARD_ZONE_H

#include <flecs.h>

void init_card_zone_queries(ecs_world_t *world);
ecs_iter_t get_cards_owned_by_player_in_zone(ecs_world_t *world, ecs_entity_t player, ecs_entity_t zone);

#endif
