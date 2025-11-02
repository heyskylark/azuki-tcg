#ifndef AZUKI_ECS_SYSTEMS_ECONOMY_H
#define AZUKI_ECS_SYSTEMS_ECONOMY_H

#include "components.h"

void DrawCard(ecs_iter_t *it);
void GrantIKZ(ecs_iter_t *it);
void init_economy_systems(ecs_world_t *world);

#endif