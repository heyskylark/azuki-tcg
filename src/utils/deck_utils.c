#include "utils/deck_utils.h"

#include <stdint.h>
#include <string.h>

#include "components.h"

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

  uint32_t rng_state = state->seed;
  for (int32_t i = count - 1; i > 0; --i) {
    uint32_t roll = deck_next_rand(&rng_state);
    int32_t j = (int32_t)(roll % (uint32_t)(i + 1));

    ecs_entity_t temp = shuffled[i];
    shuffled[i] = shuffled[j];
    shuffled[j] = temp;
  }

  ecs_set_child_order(world, deck_zone, shuffled, count);

  state->seed = rng_state;
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
