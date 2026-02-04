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

/**
 * Move top N cards from deck to selection zone for examination.
 * Does NOT trigger deck-out (examining cards is not drawing).
 * @param world The ECS world
 * @param player The player entity examining cards
 * @param count Number of cards to look at
 * @param out_cards Array to store the card entities (must be at least count size)
 * @return Number of cards actually moved (may be less if deck has fewer cards)
 */
int look_at_top_n_cards(ecs_world_t *world, ecs_entity_t player, int count, ecs_entity_t *out_cards);

/**
 * Add a card to the bottom of a player's deck.
 * @param world The ECS world
 * @param player The player whose deck to modify
 * @param card The card to add to the bottom
 */
void add_card_to_bottom_of_deck(ecs_world_t *world, ecs_entity_t player, ecs_entity_t card);

/**
 * Move a card from the selection zone to the player's hand.
 * @param world The ECS world
 * @param card The card to move to hand
 */
void move_selection_to_hand(ecs_world_t *world, ecs_entity_t card);

/**
 * Move a card from the selection zone to the bottom of the deck.
 * @param world The ECS world
 * @param player The player whose deck to modify
 * @param card The card to bottom deck
 */
void move_selection_to_deck_bottom(ecs_world_t *world, ecs_entity_t player, ecs_entity_t card);

/**
 * Queue a deck reorder so it can be processed after deferred ops flush.
 * @return true if queued, false if queue is full.
 */
bool azk_queue_deck_reorder(ecs_world_t *world, ecs_entity_t deck, ecs_entity_t card);

/**
 * Check for pending deck reorder operations.
 */
bool azk_has_pending_deck_reorders(ecs_world_t *world);

/**
 * Process any queued deck reorders.
 */
void azk_process_deck_reorder_queue(ecs_world_t *world);

#endif
