#ifndef AZUKI_UTILS_STATUS_UTIL_H
#define AZUKI_UTILS_STATUS_UTIL_H

#include "components/abilities.h"
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
 * Apply shocked status to an entity with a duration.
 * Shocked cards do not untap during their next untap step.
 * @param world The ECS world
 * @param entity The entity to shock
 * @param duration Number of turns remaining before Shocked is removed
 */
void apply_shocked(ecs_world_t *world, ecs_entity_t entity, int8_t duration);

/**
 * Remove shocked status from an entity.
 * @param world The ECS world
 * @param entity The entity to clear
 */
void remove_shocked(ecs_world_t *world, ecs_entity_t entity);

/**
 * Check if an entity is shocked.
 * @param world The ECS world
 * @param entity The entity to inspect
 * @return true if the entity currently has Shocked
 */
bool is_shocked(ecs_world_t *world, ecs_entity_t entity);

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

int8_t get_total_carapace_value(ecs_world_t *world, ecs_entity_t entity);
void apply_carapace_modifier(ecs_world_t *world, ecs_entity_t entity,
                             ecs_entity_t source, int8_t amount,
                             bool expires_eot);
void remove_all_carapace_modifiers(ecs_world_t *world, ecs_entity_t entity);
void expire_eot_carapace_modifiers_in_zone(ecs_world_t *world, ecs_entity_t zone);

/**
 * Apply a tracked tag grant to an entity.
 * Use TAG_GRANT_TICK_NONE with remaining_ticks = -1 for permanent non-innate grants.
 * @param world The ECS world
 * @param entity The entity to modify
 * @param tag The ECS tag to grant
 * @param tick_phase Which phase advances the expiry timer
 * @param remaining_ticks Number of matching ticks before removal (-1 for permanent)
 */
bool apply_timed_tag_grant(ecs_world_t *world, ecs_entity_t entity, ecs_id_t tag,
                           TagGrantTickPhase tick_phase,
                           int8_t remaining_ticks);

/**
 * Grant Charge using the tracked tag-grant system.
 * This works for both permanent and timed Charge grants and preserves innate Charge.
 * @param world The ECS world
 * @param entity The entity to grant Charge to
 * @param tick_phase Which phase advances the expiry timer
 * @param remaining_ticks Number of matching ticks before removal (-1 for permanent)
 */
void apply_charge_grant(ecs_world_t *world, ecs_entity_t entity,
                        TagGrantTickPhase tick_phase, int8_t remaining_ticks);

/**
 * Clear temporary statuses and tracked non-innate tag grants from a card.
 * Use this when a card changes zones and should lose transient state.
 * Permanent prefab keywords are preserved.
 * @param world The ECS world
 * @param entity The card to clean up
 */
void clear_card_temporary_state(ecs_world_t *world, ecs_entity_t entity);

/**
 * Tick down status effect durations for a player's cards.
 * Call at start of turn BEFORE untap.
 * Removes expired status effects and timed tag grants when duration reaches 0.
 * @param world The ECS world
 * @param player_index The player index (0 or 1)
 */
void tick_status_effects_for_player(ecs_world_t *world, uint8_t player_index);

/**
 * Tick end-of-turn tracked tag grants for a player's in-play cards.
 * @param world The ECS world
 * @param player_index The player index (0 or 1)
 */
void tick_end_of_turn_effects_for_player(ecs_world_t *world,
                                         uint8_t player_index);

/**
 * Recalculate an entity's attack from base stats, weapons, and all buff pairs.
 * Call this after adding/removing weapons or AttackBuff pairs.
 * @param world The ECS world
 * @param entity The entity to recalculate attack for
 */
void recalculate_attack_from_buffs(ecs_world_t *world, ecs_entity_t entity);

/**
 * Apply an attack modifier to an entity using relationship-based buffs.
 * Creates an (AttackBuff, source) pair on the target entity.
 * @param world The ECS world
 * @param entity The entity to modify
 * @param source The source entity applying the buff (e.g., the card causing the effect)
 * @param modifier The attack modifier value (negative for debuff, positive for buff)
 * @param expires_eot If true, the modifier expires at end of current turn
 */
void apply_attack_modifier(ecs_world_t *world, ecs_entity_t entity,
                           ecs_entity_t source, int8_t modifier, bool expires_eot);

/**
 * Remove an attack modifier from a specific source.
 * @param world The ECS world
 * @param entity The entity to remove the modifier from
 * @param source The source entity whose buff should be removed
 */
void remove_attack_modifier(ecs_world_t *world, ecs_entity_t entity,
                            ecs_entity_t source);

/**
 * Remove all attack modifiers from an entity.
 * @param world The ECS world
 * @param entity The entity to remove all modifiers from
 */
void remove_all_attack_modifiers(ecs_world_t *world, ecs_entity_t entity);

/**
 * Expire all end-of-turn attack modifiers in a zone.
 * Call at end of turn to clean up temporary attack modifiers.
 * @param world The ECS world
 * @param zone The zone entity to process
 */
void expire_eot_attack_modifiers_in_zone(ecs_world_t *world, ecs_entity_t zone);

/**
 * Recalculate an entity's health from base stats and all HealthBuff pairs.
 * Call this after adding/removing HealthBuff pairs.
 * @param world The ECS world
 * @param entity The entity to recalculate health for
 */
void recalculate_health_from_buffs(ecs_world_t *world, ecs_entity_t entity);

/**
 * Apply a health modifier to an entity using relationship-based buffs.
 * Creates a (HealthBuff, source) pair on the target entity.
 * @param world The ECS world
 * @param entity The entity to modify
 * @param source The source entity applying the buff
 * @param modifier The health modifier value (negative for debuff, positive for buff)
 * @param expires_eot If true, the modifier expires at end of current turn
 */
void apply_health_modifier(ecs_world_t *world, ecs_entity_t entity,
                           ecs_entity_t source, int8_t modifier, bool expires_eot);

/**
 * Remove a health modifier from a specific source.
 * WARNING: If this reduces cur_hp to 0 or below, the entity should be killed.
 * @param world The ECS world
 * @param entity The entity to remove the modifier from
 * @param source The source entity whose buff should be removed
 * @return true if the entity's cur_hp is now <= 0 (death should be triggered)
 */
bool remove_health_modifier(ecs_world_t *world, ecs_entity_t entity,
                            ecs_entity_t source);

/**
 * Remove all health modifiers from an entity.
 * @param world The ECS world
 * @param entity The entity to remove all modifiers from
 * @return true if the entity's cur_hp is now <= 0 (death should be triggered)
 */
bool remove_all_health_modifiers(ecs_world_t *world, ecs_entity_t entity);

/**
 * Expire all end-of-turn health modifiers in a zone.
 * Call at end of turn to clean up temporary health modifiers.
 * @param world The ECS world
 * @param zone The zone entity to process
 */
void expire_eot_health_modifiers_in_zone(ecs_world_t *world, ecs_entity_t zone);

/**
 * Sum all incoming combat damage modifiers on an entity.
 * Negative values reduce damage taken, positive values increase it.
 * @param world The ECS world
 * @param entity The entity to inspect
 * @return Total incoming combat damage modifier
 */
int16_t get_total_incoming_combat_damage_modifier(ecs_world_t *world,
                                                  ecs_entity_t entity);

/**
 * Sum all outgoing combat damage modifiers on an entity.
 * Negative values reduce damage dealt, positive values increase it.
 * @param world The ECS world
 * @param entity The entity to inspect
 * @return Total outgoing combat damage modifier
 */
int16_t get_total_outgoing_combat_damage_modifier(ecs_world_t *world,
                                                  ecs_entity_t entity);

/**
 * Apply a combat damage modifier to an entity using relationship pairs.
 * Creates a (CombatDamageModifier, source) pair on the target entity.
 * @param world The ECS world
 * @param entity The entity to modify
 * @param source The source entity applying the modifier
 * @param incoming_modifier Modifier applied to damage this entity takes in combat
 * @param outgoing_modifier Modifier applied to damage this entity deals in combat
 * @param expires_eot If true, the modifier expires at end of current turn
 */
void apply_combat_damage_modifier(ecs_world_t *world, ecs_entity_t entity,
                                  ecs_entity_t source,
                                  int8_t incoming_modifier,
                                  int8_t outgoing_modifier,
                                  bool expires_eot);

/**
 * Remove a combat damage modifier from a specific source.
 * @param world The ECS world
 * @param entity The entity to update
 * @param source The source entity whose modifier should be removed
 * @return true if a modifier was removed
 */
bool remove_combat_damage_modifier(ecs_world_t *world, ecs_entity_t entity,
                                   ecs_entity_t source);

/**
 * Remove all combat damage modifiers from an entity.
 * @param world The ECS world
 * @param entity The entity to clear
 */
void remove_all_combat_damage_modifiers(ecs_world_t *world,
                                        ecs_entity_t entity);

/**
 * Expire all end-of-turn combat damage modifiers in a zone.
 * @param world The ECS world
 * @param zone The zone entity to process
 */
void expire_eot_combat_damage_modifiers_in_zone(ecs_world_t *world,
                                                ecs_entity_t zone);

/**
 * Queue a passive buff update for deferred processing.
 * Use this from observers where writes are deferred and not immediately visible.
 * The buff will be applied/removed on the next processing cycle.
 * @param world The ECS world
 * @param entity The entity to apply/remove buff from
 * @param source The source entity for the buff pair
 * @param atk_modifier The attack modifier (0 if not changing)
 * @param hp_modifier The health modifier (0 if not changing)
 * @param is_removal True to remove the buff, false to apply it
 */
void azk_queue_passive_buff_update(ecs_world_t *world, ecs_entity_t entity,
                                   ecs_entity_t source, int8_t atk_modifier,
                                   int8_t hp_modifier, bool is_removal);

/**
 * Check if there are pending passive buff updates.
 * @param world The ECS world
 * @return true if there are queued updates
 */
bool azk_has_pending_passive_buffs(ecs_world_t *world);

/**
 * Process all pending passive buff updates.
 * Call this from the game loop after deferred operations are flushed.
 * @param world The ECS world
 */
void azk_process_passive_buff_queue(ecs_world_t *world);

#endif
