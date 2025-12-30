#ifndef AZUKI_ECS_SYSTEMS_ABILITY_RESOLUTION_PHASE_H
#define AZUKI_ECS_SYSTEMS_ABILITY_RESOLUTION_PHASE_H

#include "components/components.h"

void HandleAbilityResolution(ecs_iter_t *it);
void init_ability_resolution_phase_system(ecs_world_t *world);

#endif
