#ifndef AZUKI_ECS_UTILS_ENTITY_UTIL_H
#define AZUKI_ECS_UTILS_ENTITY_UTIL_H

#include <flecs.h>

void reset_entity_health(ecs_world_t *world, ecs_entity_t entity);
void discard_equipped_weapon_cards(ecs_world_t *world, ecs_entity_t entity);

#endif