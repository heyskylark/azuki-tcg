#include "utils/deck_utils.h"

#include <stdint.h>
#include <string.h>

#include "components/components.h"
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

void shuffle_deck(ecs_world_t *world, ecs_entity_t deck_zone) {
  ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck_zone);
  int32_t count = deck_cards.count;

  if (count <= 1) {
    return;
  }

  GameState *state = ecs_singleton_get_mut(world, GameState);
  ecs_assert(state != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");

  ecs_entity_t *shuffled = ecs_os_malloc_n(ecs_entity_t, count);
  ecs_assert(shuffled != NULL, ECS_OUT_OF_MEMORY, "Failed to allocate shuffle buffer");

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
}


bool move_cards_to_zone(ecs_world_t *world, ecs_entity_t from_zone, ecs_entity_t to_zone, int draw_count, ecs_entity_t *out_cards) {
  ecs_entities_t cards = ecs_get_ordered_children(world, from_zone);
  int32_t count = cards.count;

  if (draw_count <= 0) {
    return true;
  }

  int32_t to_draw = draw_count;
  if (to_draw > count) {
    to_draw = count;
  }

  for (int32_t index = 0; index < to_draw; index++) {
    ecs_entity_t card = cards.ids[count - 1 - index];
    ecs_add_pair(world, card, EcsChildOf, to_zone);

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

bool draw_cards_with_deckout_check(ecs_world_t *world, ecs_entity_t player, int draw_count, ecs_entity_t *out_cards) {
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  uint8_t player_num = get_player_number(world, player);
  ecs_entity_t deck = gs->zones[player_num].deck;
  ecs_entity_t hand = gs->zones[player_num].hand;

  // Get deck snapshot once to avoid Flecs deferred operation issues
  // (calling move_cards_to_zone repeatedly would re-fetch stale children)
  ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck);
  int32_t deck_count = deck_cards.count;

  for (int i = 0; i < draw_count; i++) {
    if (deck_count == 0) {
      // Couldn't draw (deck was already empty)
      gs->winner = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
      gs->phase = PHASE_END_MATCH;
      ecs_singleton_modified(world, GameState);
      return false;
    }

    // Move the top card (last in ordered list) to hand
    ecs_entity_t card = deck_cards.ids[deck_count - 1 - i];
    ecs_add_pair(world, card, EcsChildOf, hand);

    if (out_cards) {
      out_cards[i] = card;
    }

    // Check if deck is now empty after this draw
    // (deck_count - i - 1 = remaining cards after this draw)
    int remaining = deck_count - i - 1;
    if (remaining == 0) {
      gs->winner = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
      gs->phase = PHASE_END_MATCH;
      ecs_singleton_modified(world, GameState);
      return false;  // Player loses due to deck-out
    }
  }
  return true;
}
