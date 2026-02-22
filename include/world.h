#ifndef AZUKI_ECS_WORLD_H
#define AZUKI_ECS_WORLD_H

#include "generated/card_defs.h"
#include "constants/game.h"
#include "components/components.h"

typedef enum {
  RAIZEN = 0,
  SHAO = 1
} DeckType;

typedef struct {
  CardDefId card_id;
  int card_count;
} CardInfo;

ecs_world_t* azk_world_init(uint32_t seed);
ecs_world_t* azk_world_init_with_starting_player(uint32_t seed,
                                                 int8_t starting_player_index);
ecs_world_t* azk_world_init_with_decks(
  uint32_t seed,
  const CardInfo *player0_deck,
  size_t player0_deck_count,
  const CardInfo *player1_deck,
  size_t player1_deck_count
);
void azk_world_fini(ecs_world_t *world);

/**
 * Initialize a player's deck with a predefined deck type.
 * @return true on success, false on critical validation failure (missing leader/gate/etc)
 */
bool init_player_deck(ecs_world_t *world, ecs_entity_t player, DeckType deck_type, PlayerZones *zones);

/**
 * Initialize a player's deck with custom card data.
 * @return true on success, false on critical validation failure (missing leader/gate/etc)
 */
bool init_player_deck_custom(ecs_world_t *world, ecs_entity_t player, const CardInfo *cards, size_t card_count, PlayerZones *zones);

#endif
