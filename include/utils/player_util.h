#ifndef AZUKI_UTILS_PLAYER_UTIL_H
#define AZUKI_UTILS_PLAYER_UTIL_H

#include <flecs.h>
#include "components/components.h"

uint8_t get_player_number(ecs_world_t *world, ecs_entity_t player);

/**
 * Check if the defending player has any response spells they can play.
 * Returns true if defender has at least one response spell in hand with enough
 * IKZ to cast.
 */
bool defender_can_respond(ecs_world_t *world, const GameState *gs,
                          uint8_t defender_index);

#endif