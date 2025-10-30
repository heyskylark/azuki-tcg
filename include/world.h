#ifndef AZUKI_ECS_WORLD_H
#define AZUKI_ECS_WORLD_H

#include "generated/card_defs.h"
#include "constants/game.h"
#include "components.h"

typedef enum {
  RAIZEN = 0,
  SHAO = 1
} DeckType;

typedef struct {
  CardDefId card_id;
  int card_count;
} CardInfo;

ecs_world_t* azk_world_init(uint32_t seed);
void azk_world_fini(ecs_world_t *world);
void init_player_deck(ecs_world_t *world, ecs_entity_t player, DeckType deck_type, PlayerZones *zones);

#endif
