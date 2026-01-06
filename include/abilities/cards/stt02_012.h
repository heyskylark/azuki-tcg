#ifndef AZUKI_ABILITIES_STT02_012_H
#define AZUKI_ABILITIES_STT02_012_H

#include <flecs.h>

// STT02-012: If the number of entities in your garden is 2 or more than
// the number of entities in your opponent's garden, this card has +1 attack
// and +1 health.

void stt02_012_init_passive_observers(ecs_world_t *world, ecs_entity_t card);
void stt02_012_cleanup_passive_observers(ecs_world_t *world, ecs_entity_t card);

#endif
