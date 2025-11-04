#ifndef AZUKI_ECS_SYSTEMS_PHASE_MANAGEMENT_H
#define AZUKI_ECS_SYSTEMS_PHASE_MANAGEMENT_H

#include <flecs.h>

void PhaseManagement(ecs_iter_t *it);
void init_phase_management_system(ecs_world_t *world);
void run_phase_management_system(ecs_world_t *world);

#endif
