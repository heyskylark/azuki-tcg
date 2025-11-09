#ifndef AZUKI_UTILS_COMBAT_UTIL_H
#define AZUKI_UTILS_COMBAT_UTIL_H

#include <flecs.h>
#include "components.h"

/**
 * Handles attack from attacker_index to defender_index.
 * attacker_index and defender_index of 5 is the leader
*/
int attack(
  ecs_world_t *world,
  ecs_entity_t attacking_player,
  uint8_t garden_attacker_index,
  uint8_t defender_index
);
void resolve_combat(ecs_world_t *world);

#endif