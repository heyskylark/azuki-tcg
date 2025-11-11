#ifndef AZUKI_UTILS_COMBAT_UTIL_H
#define AZUKI_UTILS_COMBAT_UTIL_H

#include <flecs.h>
#include "components.h"
#include "validation/action_intents.h"

int attack(
  ecs_world_t *world,
  const AttackIntent *intent
);
void resolve_combat(ecs_world_t *world);

#endif
