#ifndef AZUKI_ECS_WORLD_H
#define AZUKI_ECS_WORLD_H

#include "constants/game.h"
#include "components.h"

typedef enum {
  RAIZEN = 0,
  SHAO = 1
} DeckType;

ecs_world_t* azk_world_init(uint32_t seed);
void azk_world_fini(ecs_world_t *world);

#endif