#ifndef AZUKI_ABILITIES_AZK01_010_H
#define AZUKI_ABILITIES_AZK01_010_H

#include <flecs.h>

// AZK01-010 "JD": If you only have (Normal) entities in your Garden, this
// card has +2 attack.

void azk01_010_init_passive_observers(ecs_world_t *world, ecs_entity_t card);
void azk01_010_cleanup_passive_observers(ecs_world_t *world, ecs_entity_t card);

#endif // AZUKI_ABILITIES_AZK01_010_H
