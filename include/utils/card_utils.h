#ifndef AZUKI_UTILS_CARD_UTILS_H
#define AZUKI_UTILS_CARD_UTILS_H

#include <flecs.h>
#include "generated/card_defs.h"

bool is_card_type(ecs_world_t *world, ecs_entity_t card, CardType type);
void discard_card(ecs_world_t *world, ecs_entity_t card);
void set_card_to_tapped(ecs_world_t *world, ecs_entity_t card);
void set_card_to_cooldown(ecs_world_t *world, ecs_entity_t card);
bool is_card_tapped(ecs_world_t *world, ecs_entity_t card);
bool is_card_cooldown(ecs_world_t *world, ecs_entity_t card);

#endif
