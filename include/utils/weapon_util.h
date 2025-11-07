#ifndef AZUKI_UTILS_WEAPON_UTIL_H
#define AZUKI_UTILS_WEAPON_UTIL_H

#include <flecs.h>

int attach_weapon_from_hand(
  ecs_world_t *world,
  ecs_entity_t player,
  int hand_index,
  int entity_index,
  bool use_ikz_token
);

#endif