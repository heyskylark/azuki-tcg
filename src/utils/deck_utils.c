#include "utils/deck_utils.h"

#include <stdint.h>
#include <string.h>

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"

static inline uint32_t deck_next_rand(uint32_t *state) {
  uint32_t x = *state;
  if (x == 0) {
    x = 0x9E3779B9u; // ensure non-zero state for xorshift
  }
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

static bool set_deck_out_loss(ecs_world_t *world, GameState *gs,
                              uint8_t player_num) {
  gs->winner = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
  gs->phase = PHASE_END_MATCH;
  ecs_singleton_modified(world, GameState);
  azk_log_game_ended(world, gs->winner, GLOG_END_DECK_OUT);
  return false;
}

void shuffle_deck(ecs_world_t *world, ecs_entity_t deck_zone) {
  ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck_zone);
  int32_t count = deck_cards.count;

  if (count <= 1) {
    return;
  }

  GameState *state = ecs_singleton_get_mut(world, GameState);
  ecs_assert(state != NULL, ECS_INVALID_PARAMETER,
             "GameState singleton missing");

  ecs_entity_t *shuffled = ecs_os_malloc_n(ecs_entity_t, count);
  ecs_assert(shuffled != NULL, ECS_OUT_OF_MEMORY,
             "Failed to allocate shuffle buffer");

  memcpy(shuffled, deck_cards.ids, (size_t)count * sizeof(ecs_entity_t));

  uint32_t rng_state = state->rng_state;
  for (int32_t i = count - 1; i > 0; --i) {
    uint32_t roll = deck_next_rand(&rng_state);
    int32_t j = (int32_t)(roll % (uint32_t)(i + 1));

    ecs_entity_t temp = shuffled[i];
    shuffled[i] = shuffled[j];
    shuffled[j] = temp;
  }

  ecs_set_child_order(world, deck_zone, shuffled, count);

  state->rng_state = rng_state;
  ecs_singleton_modified(world, GameState);

  ecs_os_free(shuffled);

  // Log deck shuffle (get player from deck zone owner)
  ecs_entity_t owner = ecs_get_target(world, deck_zone, Rel_OwnedBy, 0);
  if (owner) {
    uint8_t player_num = get_player_number(world, owner);
    // Default to GAME_START reason - caller can override for specific contexts
    azk_log_deck_shuffled(world, player_num, GLOG_SHUFFLE_GAME_START);
  }
}

bool move_cards_to_zone(ecs_world_t *world, ecs_entity_t from_zone,
                        ecs_entity_t to_zone, int draw_count,
                        ecs_entity_t *out_cards) {
  ecs_entities_t cards = ecs_get_ordered_children(world, from_zone);
  int32_t count = cards.count;

  if (draw_count <= 0) {
    return true;
  }

  int32_t to_draw = draw_count;
  if (to_draw > count) {
    to_draw = count;
  }

  GameLogZone from_log_zone = azk_zone_entity_to_log_zone(world, from_zone);
  GameLogZone to_log_zone = azk_zone_entity_to_log_zone(world, to_zone);

  // Get initial to_zone count for computing to_index (cards appended at end)
  int32_t to_zone_count = ecs_get_ordered_children(world, to_zone).count;

  for (int32_t index = 0; index < to_draw; index++) {
    // Cards are moved from end of source zone, so from_index = count - 1 - index
    int8_t from_index = (int8_t)(count - 1 - index);
    ecs_entity_t card = cards.ids[from_index];
    ecs_add_pair(world, card, EcsChildOf, to_zone);

    // Log zone movement (to_index = original count + cards already moved)
    azk_log_card_zone_moved(world, card, from_log_zone, from_index, to_log_zone,
                            (int8_t)(to_zone_count + index));

    if (out_cards) {
      out_cards[index] = card;
    }
  }

  if (out_cards) {
    for (int32_t index = to_draw; index < draw_count; index++) {
      out_cards[index] = 0;
    }
  }

  return to_draw == draw_count;
}

static const CardId *get_card_id_component(ecs_world_t *world,
                                           ecs_entity_t card) {
  const CardId *card_id = ecs_get(world, card, CardId);
  if (card_id != NULL) {
    return card_id;
  }

  ecs_entity_t prefab = ecs_get_target(world, card, EcsIsA, 0);
  if (prefab == 0) {
    return NULL;
  }

  return ecs_get(world, prefab, CardId);
}

bool draw_cards_with_deckout_check(ecs_world_t *world, ecs_entity_t player,
                                   int draw_count, ecs_entity_t *out_cards) {
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  uint8_t player_num = get_player_number(world, player);
  ecs_entity_t deck = gs->zones[player_num].deck;
  ecs_entity_t hand = gs->zones[player_num].hand;

  // Get deck snapshot once to avoid Flecs deferred operation issues
  // (calling move_cards_to_zone repeatedly would re-fetch stale children)
  ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck);
  int32_t deck_count = deck_cards.count;

  // Get initial hand count for computing indices (cards are appended to end)
  int32_t hand_index = ecs_get_ordered_children(world, hand).count;

  for (int i = 0; i < draw_count; i++) {
    if (deck_count == 0) {
      // Couldn't draw (deck was already empty)
      return set_deck_out_loss(world, gs, player_num);
    }

    // Move the top card (last in ordered list) to hand
    ecs_entity_t card = deck_cards.ids[deck_count - 1 - i];
    int8_t from_index = (int8_t)(deck_count - 1 - i);
    ecs_add_pair(world, card, EcsChildOf, hand);

    // Capture the draw source index before the deferred move; the hand index is
    // finalized post-commit.
    azk_log_card_zone_moved(world, card, GLOG_ZONE_DECK, from_index,
                            GLOG_ZONE_HAND, (int8_t)(hand_index + i));

    if (out_cards) {
      out_cards[i] = card;
    }

    // Check if deck is now empty after this draw
    // (deck_count - i - 1 = remaining cards after this draw)
    int remaining = deck_count - i - 1;
    if (remaining == 0) {
      return set_deck_out_loss(world, gs, player_num);
    }
  }
  return true;
}

AzkDebugDrawResult azk_debug_draw_card_from_deck(ecs_world_t *world,
                                                 ecs_entity_t player,
                                                 CardDefId card_def_id) {
  if (world == NULL || player == 0) {
    return AZK_DEBUG_DRAW_INVALID_PLAYER;
  }

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  if (gs == NULL || gs->winner >= 0 || gs->phase == PHASE_END_MATCH) {
    return AZK_DEBUG_DRAW_INVALID_STATE;
  }

  uint8_t player_num = get_player_number(world, player);
  ecs_entity_t deck = gs->zones[player_num].deck;
  ecs_entity_t hand = gs->zones[player_num].hand;
  if (deck == 0 || hand == 0) {
    return AZK_DEBUG_DRAW_INVALID_STATE;
  }

  ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck);
  int32_t deck_count = deck_cards.count;
  int32_t hand_index = ecs_get_ordered_children(world, hand).count;

  for (int32_t index = deck_count - 1; index >= 0; --index) {
    ecs_entity_t card = deck_cards.ids[index];
    const CardId *card_id = get_card_id_component(world, card);
    if (card_id == NULL || card_id->id != card_def_id) {
      continue;
    }

    ecs_add_pair(world, card, EcsChildOf, hand);
    azk_log_card_zone_moved(world, card, GLOG_ZONE_DECK, (int8_t)index,
                            GLOG_ZONE_HAND, (int8_t)hand_index);

    if (deck_count == 1) {
      bool draw_ok = set_deck_out_loss(world, gs, player_num);
      (void)draw_ok;
    }

    return AZK_DEBUG_DRAW_OK;
  }

  return AZK_DEBUG_DRAW_CARD_NOT_FOUND;
}

bool mill_cards_with_deckout_check(ecs_world_t *world, ecs_entity_t player,
                                   int mill_count, ecs_entity_t *out_cards) {
  if (mill_count <= 0) {
    return true;
  }

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  uint8_t player_num = get_player_number(world, player);
  ecs_entity_t deck = gs->zones[player_num].deck;

  ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck);
  int32_t deck_count = deck_cards.count;
  int32_t to_mill = mill_count;
  if (to_mill > deck_count) {
    to_mill = deck_count;
  }

  for (int32_t i = 0; i < to_mill; i++) {
    ecs_entity_t card = deck_cards.ids[deck_count - 1 - i];
    discard_card(world, card);
    if (out_cards) {
      out_cards[i] = card;
    }
  }

  if (out_cards) {
    for (int32_t i = to_mill; i < mill_count; i++) {
      out_cards[i] = 0;
    }
  }

  if (deck_count > 0 && to_mill == deck_count) {
    return set_deck_out_loss(world, gs, player_num);
  }

  return true;
}

int look_at_top_n_cards(ecs_world_t *world, ecs_entity_t player, int count,
                        ecs_entity_t *out_cards) {
  if (count <= 0 || !out_cards) {
    return 0;
  }

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  uint8_t player_num = get_player_number(world, player);
  ecs_entity_t deck = gs->zones[player_num].deck;
  ecs_entity_t selection = gs->zones[player_num].selection;

  ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck);
  int32_t deck_count = deck_cards.count;

  int to_move = count;
  if (to_move > deck_count) {
    to_move = deck_count;
  }

  // Move top N cards from deck (end of array) to selection zone
  for (int i = 0; i < to_move; i++) {
    ecs_entity_t card = deck_cards.ids[deck_count - 1 - i];
    int8_t from_index = (int8_t)(deck_count - 1 - i);
    ecs_add_pair(world, card, EcsChildOf, selection);
    out_cards[i] = card;
    // Capture the deck source index before the deferred move; the selection
    // index is finalized post-commit.
    azk_log_card_zone_moved(world, card, GLOG_ZONE_DECK, from_index,
                            GLOG_ZONE_SELECTION, (int8_t)i);
  }

  // Zero out remaining slots if fewer cards available
  for (int i = to_move; i < count; i++) {
    out_cards[i] = 0;
  }

  return to_move;
}

void add_card_to_bottom_of_deck(ecs_world_t *world, ecs_entity_t player,
                                ecs_entity_t card) {
  // Capture source zone and index before moving
  ecs_entity_t from_zone_entity = ecs_get_target(world, card, EcsChildOf, 0);
  GameLogZone from_zone = azk_zone_entity_to_log_zone(world, from_zone_entity);
  int8_t from_index =
      azk_get_card_index_in_zone(world, card, from_zone_entity);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  uint8_t player_num = get_player_number(world, player);
  ecs_entity_t deck = gs->zones[player_num].deck;

  // Move the card to the deck zone (may be deferred)
  ecs_add_pair(world, card, EcsChildOf, deck);

  // Defer reordering until after deferred ops flush
  if (!azk_queue_deck_reorder(world, deck, card)) {
    cli_render_logf("[Deck] Reorder queue full - bottom deck reorder skipped");
  }

  // Log zone movement (card ends up at index 0 = bottom of deck)
  azk_log_card_zone_moved(world, card, from_zone, from_index, GLOG_ZONE_DECK,
                          0);
}

void add_card_to_top_of_deck(ecs_world_t *world, ecs_entity_t player,
                             ecs_entity_t card) {
  ecs_entity_t from_zone_entity = ecs_get_target(world, card, EcsChildOf, 0);
  GameLogZone from_zone = azk_zone_entity_to_log_zone(world, from_zone_entity);
  int8_t from_index =
      azk_get_card_index_in_zone(world, card, from_zone_entity);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  uint8_t player_num = get_player_number(world, player);
  ecs_entity_t deck = gs->zones[player_num].deck;

  ecs_add_pair(world, card, EcsChildOf, deck);

  if (!azk_queue_deck_reorder_to_top(world, deck, card)) {
    cli_render_logf("[Deck] Reorder queue full - top deck reorder skipped");
  }

  ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck);
  int8_t to_index = (int8_t)deck_cards.count;
  azk_log_card_zone_moved(world, card, from_zone, from_index, GLOG_ZONE_DECK,
                          to_index);
}

void move_selection_to_hand(ecs_world_t *world, ecs_entity_t card) {
  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  ecs_assert(owner != 0, ECS_INVALID_PARAMETER, "Card has no owner");

  const PlayerNumber *player_number = ecs_get(world, owner, PlayerNumber);
  ecs_assert(player_number != NULL, ECS_INVALID_PARAMETER,
             "PlayerNumber not found");

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t hand_zone = gs->zones[player_number->player_number].hand;
  ecs_entity_t selection_zone = gs->zones[player_number->player_number].selection;

  // Capture the selection slot before the card leaves the source zone.
  int8_t from_index = azk_get_card_index_in_zone(world, card, selection_zone);

  // Include already-logged hand moves in this action batch so repeated
  // selection-to-hand moves under deferred ops keep their append order.
  int32_t hand_index = azk_get_effective_hand_count(
      world, hand_zone, player_number->player_number);

  // Move from selection to hand
  ecs_add_pair(world, card, EcsChildOf, hand_zone);

  // Log zone movement. The final hand index is resolved after commit.
  azk_log_card_zone_moved(world, card, GLOG_ZONE_SELECTION, from_index,
                          GLOG_ZONE_HAND, (int8_t)hand_index);
}

void move_selection_to_deck_bottom(ecs_world_t *world, ecs_entity_t player,
                                   ecs_entity_t card) {
  // Just delegate to add_card_to_bottom_of_deck
  add_card_to_bottom_of_deck(world, player, card);
}

void move_selection_to_deck_top(ecs_world_t *world, ecs_entity_t player,
                                ecs_entity_t card) {
  add_card_to_top_of_deck(world, player, card);
}

static bool azk_queue_deck_reorder_internal(ecs_world_t *world,
                                            ecs_entity_t deck,
                                            ecs_entity_t card, bool to_top) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World pointer is null");

  DeckReorderQueue *queue = ecs_singleton_get_mut(world, DeckReorderQueue);
  ecs_assert(queue != NULL, ECS_INVALID_PARAMETER,
             "DeckReorderQueue singleton missing");

  if (queue->count >= MAX_DECK_REORDER_QUEUE) {
    return false;
  }

  queue->entries[queue->count++] = (PendingDeckReorder){
      .deck = deck,
      .card = card,
      .to_top = to_top,
  };

  ecs_singleton_modified(world, DeckReorderQueue);
  return true;
}

bool azk_queue_deck_reorder(ecs_world_t *world, ecs_entity_t deck,
                            ecs_entity_t card) {
  return azk_queue_deck_reorder_internal(world, deck, card, false);
}

bool azk_queue_deck_reorder_to_top(ecs_world_t *world, ecs_entity_t deck,
                                   ecs_entity_t card) {
  return azk_queue_deck_reorder_internal(world, deck, card, true);
}

bool azk_has_pending_deck_reorders(ecs_world_t *world) {
  const DeckReorderQueue *queue = ecs_singleton_get(world, DeckReorderQueue);
  return queue && queue->count > 0;
}

void azk_process_deck_reorder_queue(ecs_world_t *world) {
  DeckReorderQueue *queue = ecs_singleton_get_mut(world, DeckReorderQueue);
  if (!queue || queue->count == 0) {
    return;
  }

  for (uint8_t i = 0; i < queue->count; i++) {
    ecs_entity_t deck = queue->entries[i].deck;
    ecs_entity_t card = queue->entries[i].card;

    if (deck == 0 || card == 0) {
      continue;
    }

    ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
    if (parent != deck) {
      cli_render_logf("[Deck] Reorder skipped - card not in deck");
      continue;
    }

    ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck);
    int32_t count = deck_cards.count;
    if (count <= 1) {
      continue;
    }

    ecs_entity_t *new_order = ecs_os_malloc_n(ecs_entity_t, count);
    ecs_assert(new_order != NULL, ECS_OUT_OF_MEMORY,
               "Failed to allocate reorder buffer");

    bool found = false;
    int32_t dest = 0;
    if (queue->entries[i].to_top) {
      for (int32_t j = 0; j < count; j++) {
        if (deck_cards.ids[j] == card) {
          found = true;
          continue;
        }
        if (dest < count - 1) {
          new_order[dest++] = deck_cards.ids[j];
        }
      }
      new_order[dest++] = card;
    } else {
      new_order[dest++] = card;
      for (int32_t j = 0; j < count; j++) {
        if (deck_cards.ids[j] == card) {
          found = true;
          continue;
        }
        if (dest < count) {
          new_order[dest++] = deck_cards.ids[j];
        }
      }
    }

    if (found && dest == count) {
      ecs_set_child_order(world, deck, new_order, count);
    } else {
      cli_render_logf("[Deck] Reorder skipped - card missing from deck list");
    }

    ecs_os_free(new_order);
  }

  queue->count = 0;
  ecs_singleton_modified(world, DeckReorderQueue);
}
