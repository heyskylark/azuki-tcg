#ifndef AZUKI_UTILS_STATUS_UTIL_H
#define AZUKI_UTILS_STATUS_UTIL_H

#include <flecs.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * Apply frozen status to an entity with a duration.
 * @param world The ECS world
 * @param entity The entity to freeze
 * @param duration Number of turns (-1 for permanent, decrements each turn start)
 */
void apply_frozen(ecs_world_t *world, ecs_entity_t entity, int8_t duration);

/**
 * Remove frozen status from an entity.
 * @param world The ECS world
 * @param entity The entity to unfreeze
 */
void remove_frozen(ecs_world_t *world, ecs_entity_t entity);

/**
 * Check if an entity is frozen.
 * @param world The ECS world
 * @param entity The entity to check
 * @return true if the entity has the Frozen tag
 */
bool is_frozen(ecs_world_t *world, ecs_entity_t entity);

/**
 * Tick down status effect durations for a player's cards.
 * Call at start of turn BEFORE untap.
 * Removes expired status effects when duration reaches 0.
 * @param world The ECS world
 * @param player_index The player index (0 or 1)
 */
void tick_status_effects_for_player(ecs_world_t *world, uint8_t player_index);

#endif
