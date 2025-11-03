#ifndef AZUKI_ECS_SYSTEMS_CARD_STATE_H
#define AZUKI_ECS_SYSTEMS_CARD_STATE_H

#include "components.h"

void UntapAllCards(ecs_iter_t *it);
void init_card_state_systems(ecs_world_t *world);

#endif