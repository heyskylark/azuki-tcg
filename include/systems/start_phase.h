#ifndef AZUKI_ECS_SYSTEMS_START_PHASE_H
#define AZUKI_ECS_SYSTEMS_START_PHASE_H

#include "components/components.h"

void StartPhase(ecs_iter_t *it);
void init_start_phase_system(ecs_world_t *world);
void run_start_phase_system(ecs_world_t *world);

#endif