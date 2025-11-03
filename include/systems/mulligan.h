#ifndef AZUKI_ECS_SYSTEMS_MULLIGAN_H
#define AZUKI_ECS_SYSTEMS_MULLIGAN_H

#include "components.h"

void HandleMulliganAction(ecs_iter_t *it);
void init_mulligan_system(ecs_world_t *world);

#endif