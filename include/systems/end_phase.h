#ifndef AZUKI_ECS_SYSTEMS_END_PHASE_H
#define AZUKI_ECS_SYSTEMS_END_PHASE_H

#include <flecs.h>

void HandleEndPhase(ecs_iter_t *it);
void init_end_phase_system(ecs_world_t *world);

#endif