#ifndef AZUKI_ABILITIES_AZK01_019_H
#define AZUKI_ABILITIES_AZK01_019_H

#include <flecs.h>

// AZK01-019 "Jay": If you only have (Normal) entities in your Garden, this
// card has +2 health.

void azk01_019_init_passive_observers(ecs_world_t *world, ecs_entity_t card);
void azk01_019_cleanup_passive_observers(ecs_world_t *world, ecs_entity_t card);

#endif // AZUKI_ABILITIES_AZK01_019_H
