#ifndef AZUKI_ECS_SYSTEMS_MAIN_ACTION_H
#define AZUKI_ECS_SYSTEMS_MAIN_ACTION_H

#include "components.h"

void HandleMainAction(ecs_iter_t *it);
void init_main_action_system(ecs_world_t *world);

#endif