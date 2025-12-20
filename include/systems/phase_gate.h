#ifndef AZUKI_ECS_SYSTEMS_PHASE_GATE_H
#define AZUKI_ECS_SYSTEMS_PHASE_GATE_H

#include "components/components.h"

void PhaseGate(ecs_iter_t *it);
void init_phase_gate_system(ecs_world_t *world);
void run_phase_gate_system(ecs_world_t *world);

#endif
