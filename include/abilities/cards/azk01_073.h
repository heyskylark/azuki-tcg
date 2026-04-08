#ifndef AZUKI_ABILITIES_AZK01_073_H
#define AZUKI_ABILITIES_AZK01_073_H

#include "abilities/ability_registry.h"

void azk01_073_init_passive_observers(ecs_world_t *world,
                                      ecs_entity_t ability_entity);
void azk01_073_cleanup_passive_observers(ecs_world_t *world,
                                         ecs_entity_t ability_entity);

#endif // AZUKI_ABILITIES_AZK01_073_H
