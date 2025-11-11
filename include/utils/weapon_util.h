#ifndef AZUKI_UTILS_WEAPON_UTIL_H
#define AZUKI_UTILS_WEAPON_UTIL_H

#include <flecs.h>
#include "validation/action_intents.h"

int attach_weapon_from_hand(
  ecs_world_t *world,
  const AttachWeaponIntent *intent
);

#endif
