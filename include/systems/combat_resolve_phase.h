#ifndef AZUKI_SYSTEMS_COMBAT_RESOLVE_PHASE_H
#define AZUKI_SYSTEMS_COMBAT_RESOLVE_PHASE_H

#include <flecs.h>
#include "components.h"

void HandleCombatResolution(ecs_iter_t *it);
void init_combat_resolve_phase_system(ecs_world_t *world);

#endif