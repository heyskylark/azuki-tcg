#ifndef AZUKI_UTILS_DECK_UTILS_H
#define AZUKI_UTILS_DECK_UTILS_H

#include <flecs.h>  

void shuffle_deck(ecs_world_t *world, ecs_entity_t deck_zone);
bool draw_cards(ecs_world_t *world, ecs_entity_t from_zone, ecs_entity_t to_zone, int draw_count, ecs_entity_t *out_cards);

#endif
