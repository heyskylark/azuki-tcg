#ifndef AZUKI_ABILITIES_CARDS_AZK01_043_H
#define AZUKI_ABILITIES_CARDS_AZK01_043_H

#include <flecs.h>

void azk01_043_init_passive_observers(ecs_world_t *world,
                                      ecs_entity_t ability_entity);
void azk01_043_cleanup_passive_observers(ecs_world_t *world,
                                         ecs_entity_t ability_entity);

#endif
