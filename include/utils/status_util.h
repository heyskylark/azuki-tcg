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
 * Apply effect immunity status to an entity with a duration.
 * Effect immune entities cannot take damage from card effects (spells,
 * abilities, weapons) but can still take combat damage from attacks.
 * @param world The ECS world
 * @param entity The entity to make immune
 * @param duration Number of turns (-1 for permanent, decrements each turn
 * start)
 */
void apply_effect_immune(ecs_world_t *world, ecs_entity_t entity,
                         int8_t duration);

/**
 * Remove effect immunity status from an entity.
 * @param world The ECS world
 * @param entity The entity to remove immunity from
 */
void remove_effect_immune(ecs_world_t *world, ecs_entity_t entity);

/**
 * Check if an entity is effect immune.
 * @param world The ECS world
 * @param entity The entity to check
 * @return true if the entity has the EffectImmune tag
 */
bool is_effect_immune(ecs_world_t *world, ecs_entity_t entity);

/**
 * Tick down status effect durations for a player's cards.
 * Call at start of turn BEFORE untap.
 * Removes expired status effects when duration reaches 0.
 * @param world The ECS world
 * @param player_index The player index (0 or 1)
 */
void tick_status_effects_for_player(ecs_world_t *world, uint8_t player_index);

/**
 * Apply an attack modifier to an entity (stacks additively with existing modifiers).
 * @param world The ECS world
 * @param entity The entity to modify
 * @param modifier The attack modifier value (negative for debuff, positive for buff)
 * @param expires_eot If true, the modifier expires at end of current turn
 */
void apply_attack_modifier(ecs_world_t *world, ecs_entity_t entity,
                           int8_t modifier, bool expires_eot);

/**
 * Remove all attack modifiers from an entity, restoring its attack to base + weapons.
 * @param world The ECS world
 * @param entity The entity to restore
 */
void remove_attack_modifier(ecs_world_t *world, ecs_entity_t entity);

/**
 * Expire all end-of-turn attack modifiers in a zone.
 * Call at end of turn to clean up temporary attack modifiers.
 * @param world The ECS world
 * @param zone The zone entity to process
 */
void expire_eot_attack_modifiers_in_zone(ecs_world_t *world, ecs_entity_t zone);

#endif
