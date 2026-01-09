#ifndef AZUKI_UTILS_GAME_LOG_UTIL_H
#define AZUKI_UTILS_GAME_LOG_UTIL_H

#include <flecs.h>
#include <stdbool.h>

#include "components/game_log.h"

/**
 * Clear all logs - call at start of action processing.
 */
void azk_clear_game_logs(ecs_world_t *world);

/**
 * Get current log count.
 */
uint8_t azk_get_game_log_count(ecs_world_t *world);

/**
 * Get logs array (for serialization).
 * Returns pointer to logs array and sets out_count to the number of logs.
 */
const GameStateLog *azk_get_game_logs(ecs_world_t *world, uint8_t *out_count);

/**
 * Build GameLogCardRef from entity.
 * Extracts player, card_def_id, zone, and zone_index from entity.
 */
GameLogCardRef azk_make_card_ref(ecs_world_t *world, ecs_entity_t card);

/**
 * Convert zone entity to GameLogZone enum.
 * Checks zone tags on the entity to determine zone type.
 */
GameLogZone azk_zone_entity_to_log_zone(ecs_world_t *world,
                                        ecs_entity_t zone_entity);

/**
 * Build card metadata from entity (for zone moved logs).
 */
GameLogCardMetadata azk_make_card_metadata(ecs_world_t *world,
                                           ecs_entity_t card);

/* ========== Zone Movement Logs ========== */

/**
 * Log a card moving between zones.
 */
void azk_log_card_zone_moved(ecs_world_t *world, ecs_entity_t card,
                             GameLogZone from_zone, int8_t from_index,
                             GameLogZone to_zone, int8_t to_index);

/**
 * Log a card moving between zones with explicit card info.
 * Use when card entity may have already changed zones.
 */
void azk_log_card_zone_moved_ex(ecs_world_t *world, uint8_t player,
                                CardDefId card_def_id, GameLogZone from_zone,
                                int8_t from_index, GameLogZone to_zone,
                                int8_t to_index,
                                const GameLogCardMetadata *metadata);

/* ========== Tap State Logs ========== */

/**
 * Log a card's tap state changing.
 */
void azk_log_card_tap_state_changed(ecs_world_t *world, ecs_entity_t card,
                                    GameLogTapState new_state);

/* ========== Stat Change Logs ========== */

/**
 * Log a card's stats changing.
 */
void azk_log_card_stat_change(ecs_world_t *world, ecs_entity_t card,
                              int8_t atk_delta, int8_t hp_delta, int8_t new_atk,
                              int8_t new_hp);

/* ========== Status Effect Logs ========== */

/**
 * Log a status effect being applied to a card.
 */
void azk_log_status_effect_applied(ecs_world_t *world, ecs_entity_t card,
                                   GameLogStatusEffect effect, int8_t duration);

/**
 * Log a status effect expiring from a card.
 */
void azk_log_status_effect_expired(ecs_world_t *world, ecs_entity_t card,
                                   GameLogStatusEffect effect);

/* ========== Combat Logs ========== */

/**
 * Log combat being declared (attacker and target).
 */
void azk_log_combat_declared(ecs_world_t *world, ecs_entity_t attacker,
                             ecs_entity_t target);

/**
 * Log a defender intercepting an attack.
 */
void azk_log_defender_declared(ecs_world_t *world, ecs_entity_t defender);

/**
 * Log combat damage being dealt.
 */
void azk_log_combat_damage(ecs_world_t *world, ecs_entity_t attacker,
                           ecs_entity_t defender, int8_t attacker_dmg_dealt,
                           int8_t attacker_dmg_taken, int8_t defender_dmg_dealt,
                           int8_t defender_dmg_taken);

/**
 * Log an entity dying.
 */
void azk_log_entity_died(ecs_world_t *world, ecs_entity_t card,
                         GameLogDeathCause cause);

/* ========== Ability Logs ========== */

/**
 * Log an effect being queued (ability trigger).
 */
void azk_log_effect_queued(ecs_world_t *world, ecs_entity_t card,
                           uint8_t ability_index, uint8_t trigger_tag);

/**
 * Log an effect being enabled/executed.
 */
void azk_log_effect_enabled(ecs_world_t *world, ecs_entity_t card,
                            uint8_t ability_index);

/* ========== Game Flow Logs ========== */

/**
 * Log a deck being shuffled.
 */
void azk_log_deck_shuffled(ecs_world_t *world, uint8_t player,
                           GameLogShuffleReason reason);

/**
 * Log a turn starting.
 */
void azk_log_turn_started(ecs_world_t *world, uint8_t player,
                          uint16_t turn_number);

/**
 * Log a turn ending.
 */
void azk_log_turn_ended(ecs_world_t *world, uint8_t player,
                        uint16_t turn_number);

/**
 * Log the game ending.
 */
void azk_log_game_ended(ecs_world_t *world, int8_t winner,
                        GameLogEndReason reason);

/* ========== String Conversion Utilities ========== */

/**
 * Get string representation of log type (for debugging).
 */
const char *azk_log_type_to_string(GameLogType type);

/**
 * Get string representation of zone (for debugging).
 */
const char *azk_log_zone_to_string(GameLogZone zone);

#endif /* AZUKI_UTILS_GAME_LOG_UTIL_H */
