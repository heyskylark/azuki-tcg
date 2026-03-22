#ifndef AZUKI_UTILS_DECK_UTILS_H
#define AZUKI_UTILS_DECK_UTILS_H

#include <flecs.h>
#include "generated/card_defs.h"

typedef enum {
  AZK_DEBUG_DRAW_OK = 0,
  AZK_DEBUG_DRAW_CARD_NOT_FOUND,
  AZK_DEBUG_DRAW_INVALID_PLAYER,
  AZK_DEBUG_DRAW_INVALID_STATE,
} AzkDebugDrawResult;

void shuffle_deck(ecs_world_t *world, ecs_entity_t deck_zone);
bool move_cards_to_zone(ecs_world_t *world, ecs_entity_t from_zone,
                        ecs_entity_t to_zone, int draw_count,
                        ecs_entity_t *out_cards);

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
 * Draw a specific card from the player's current deck into hand.
 * Only succeeds if a matching card instance is still present in the deck.
 */
AzkDebugDrawResult azk_debug_draw_card_from_deck(ecs_world_t *world,
                                                 ecs_entity_t player,
                                                 CardDefId card_def_id);

/**
 * Mill cards from deck to discard pile with deck-out check after the mill.
 * If the deck becomes empty after milling, sets winner to opponent and returns
 * false.
 * @param world The ECS world
 * @param player The player entity whose deck is being milled
 * @param mill_count Number of cards to mill
 * @param out_cards Optional array to store milled card entities (must be at
 * least mill_count size)
 * @return true if the deck still has cards remaining after milling, false if
 * deck-out occurred
 */
bool mill_cards_with_deckout_check(ecs_world_t *world, ecs_entity_t player,
                                   int mill_count, ecs_entity_t *out_cards);

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
 * Add a card to the top of a player's deck.
 * @param world The ECS world
 * @param player The player whose deck to modify
 * @param card The card to add to the top
 */
void add_card_to_top_of_deck(ecs_world_t *world, ecs_entity_t player,
                             ecs_entity_t card);

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
 * Move a card from the selection zone to the top of the deck.
 * @param world The ECS world
 * @param player The player whose deck to modify
 * @param card The card to top deck
 */
void move_selection_to_deck_top(ecs_world_t *world, ecs_entity_t player,
                                ecs_entity_t card);

/**
 * Queue a deck reorder so it can be processed after deferred ops flush.
 * @return true if queued, false if queue is full.
 */
bool azk_queue_deck_reorder(ecs_world_t *world, ecs_entity_t deck, ecs_entity_t card);

/**
 * Queue a deck reorder that moves the card to the top of the deck.
 * @return true if queued, false if queue is full.
 */
bool azk_queue_deck_reorder_to_top(ecs_world_t *world, ecs_entity_t deck,
                                   ecs_entity_t card);

/**
 * Check for pending deck reorder operations.
 */
bool azk_has_pending_deck_reorders(ecs_world_t *world);

/**
 * Process any queued deck reorders.
 */
void azk_process_deck_reorder_queue(ecs_world_t *world);

#endif
