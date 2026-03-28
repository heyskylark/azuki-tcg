#ifndef AZUKI_UTILS_COMBAT_UTIL_H
#define AZUKI_UTILS_COMBAT_UTIL_H

#include <flecs.h>
#include "components/components.h"
#include "validation/action_intents.h"

int attack(
  ecs_world_t *world,
  const AttackIntent *intent
);
bool azk_queue_current_defender_when_attacked(ecs_world_t *world);
bool azk_transition_to_combat_resolve(ecs_world_t *world);
void resolve_combat(ecs_world_t *world);

#endif
