#ifndef AZUKI_UTILS_DAMAGE_UTIL_H
#define AZUKI_UTILS_DAMAGE_UTIL_H

#include <flecs.h>
#include <stdbool.h>
#include <stdint.h>

#include "components/abilities.h"
#include "components/components.h"

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
bool deal_effect_damage_from_source(ecs_world_t *world, ecs_entity_t source,
                                    ecs_entity_t target, int8_t damage);
bool deal_effect_damage_from_source_no_redirect(ecs_world_t *world,
                                                ecs_entity_t source,
                                                ecs_entity_t target,
                                                int8_t damage);
void azk_record_damage_event(ecs_world_t *world, ecs_entity_t source,
                             ecs_entity_t target, int8_t actual_damage,
                             bool from_effect);
bool azk_damage_tracker_is_current_turn(ecs_world_t *world,
                                        const DamageTracker *tracker);
const DamageTracker *azk_get_current_turn_damage_tracker(ecs_world_t *world,
                                                         ecs_entity_t entity);
bool azk_enqueue_pending_damage_redirect(ecs_world_t *world, ecs_entity_t source,
                                         ecs_entity_t original_target,
                                         ecs_entity_t owner, int8_t damage);
bool azk_has_pending_damage_redirect_for_target(ecs_world_t *world,
                                                ecs_entity_t target);
bool azk_consume_pending_damage_redirect(ecs_world_t *world,
                                         ecs_entity_t target,
                                         PendingDamageRedirect *out_redirect);

#endif
