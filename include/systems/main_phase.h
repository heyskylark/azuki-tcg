#ifndef AZUKI_ECS_SYSTEMS_MAIN_PHASE_H
#define AZUKI_ECS_SYSTEMS_MAIN_PHASE_H

#include "components.h"

void HandleMainAction(ecs_iter_t *it);
void init_main_phase_system(ecs_world_t *world);

#endif