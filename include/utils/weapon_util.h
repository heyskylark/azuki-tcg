#ifndef AZUKI_UTILS_WEAPON_UTIL_H
#define AZUKI_UTILS_WEAPON_UTIL_H

#include <flecs.h>
#include "validation/action_intents.h"

int attach_weapon_from_hand(
  ecs_world_t *world,
  const AttachWeaponIntent *intent
);

// Apply a weapon's attack bonus to a target entity and emit a stat change log.
// Returns false if target stats are missing.
bool apply_weapon_attack_bonus(
  ecs_world_t *world,
  ecs_entity_t target_card,
  int8_t weapon_atk
);

#endif
