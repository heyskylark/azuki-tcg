#ifndef AZUKI_UTILS_DECK_UTILS_H
#define AZUKI_UTILS_DECK_UTILS_H

#include <flecs.h>  

void shuffle_deck(ecs_world_t *world, ecs_entity_t deck_zone);
bool move_cards_to_zone(ecs_world_t *world, ecs_entity_t from_zone, ecs_entity_t to_zone, int draw_count, ecs_entity_t *out_cards);

/**
 * Draw cards from deck to hand with deck-out check after each draw.
 * If deck becomes empty after any draw, sets winner to opponent and returns false.
 * @param world The ECS world
 * @param player The player entity who is drawing
 * @param draw_count Number of cards to draw
 * @param out_cards Optional array to store drawn card entities (must be at least draw_count size)
 * @return true if all cards were drawn successfully, false if deck-out occurred
 */
bool draw_cards_with_deckout_check(ecs_world_t *world, ecs_entity_t player, int draw_count, ecs_entity_t *out_cards);

#endif
