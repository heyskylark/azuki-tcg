#ifndef AZUKI_ABILITIES_CARDS_AZK01_095_H
#define AZUKI_ABILITIES_CARDS_AZK01_095_H

#include <flecs.h>

void azk01_095_init_passive_observers(ecs_world_t *world,
                                      ecs_entity_t ability_entity);
void azk01_095_cleanup_passive_observers(ecs_world_t *world,
                                         ecs_entity_t ability_entity);

#endif
