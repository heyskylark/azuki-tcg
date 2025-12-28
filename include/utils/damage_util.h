#ifndef AZUKI_UTILS_DAMAGE_UTIL_H
#define AZUKI_UTILS_DAMAGE_UTIL_H

#include <flecs.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * Deal damage from a card effect (spell, ability, weapon) to an entity.
 * This checks for EffectImmune status and prevents damage if the target
 * is immune. Combat damage (from attacks) should NOT use this function.
 *
 * @param world The ECS world
 * @param target The entity to deal damage to
 * @param damage Amount of damage to deal
 * @return true if damage was dealt, false if blocked by EffectImmune
 */
bool deal_effect_damage(ecs_world_t *world, ecs_entity_t target, int8_t damage);

#endif
