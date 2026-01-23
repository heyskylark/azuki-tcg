#ifndef AZUKI_COMPONENTS_GAME_LOG_H
#define AZUKI_COMPONENTS_GAME_LOG_H

#include <flecs.h>
#include <stdbool.h>
#include <stdint.h>

#include "generated/card_defs.h"

#define MAX_GAME_STATE_LOGS 64
#define MAX_ATTACHED_WEAPONS_LOG 4

/**
 * Game log types matching web-service.md specification.
 */
typedef enum {
  GLOG_NONE = 0,
  GLOG_CARD_ZONE_MOVED,        // Card moved between zones
  GLOG_CARD_STAT_CHANGE,       // Attack/health delta
  GLOG_CARD_TAP_STATE_CHANGED, // Tapped, untapped, or cooldown
  GLOG_STATUS_EFFECT_APPLIED,  // Frozen/Shocked/EffectImmune added
  GLOG_STATUS_EFFECT_EXPIRED,  // Status effect removed
  GLOG_COMBAT_DECLARED,        // Attacker + target
  GLOG_DEFENDER_DECLARED,      // Intercepting entity
  GLOG_COMBAT_DAMAGE,          // Damage dealt to each combatant
  GLOG_ENTITY_DIED,            // Entity HP <= 0, going to discard
  GLOG_EFFECT_QUEUED,          // Triggered ability enters queue
  GLOG_CARD_EFFECT_ENABLED,    // Ability actually fires/resolves
  GLOG_DECK_SHUFFLED,          // Deck order randomized
  GLOG_TURN_STARTED,           // Turn begins for a player
  GLOG_TURN_ENDED,             // Turn ends
  GLOG_GAME_ENDED,             // Winner determined
} GameLogType;

/**
 * Zone type for game logs.
 */
typedef enum {
  GLOG_ZONE_NONE = 0,
  GLOG_ZONE_DECK,
  GLOG_ZONE_HAND,
  GLOG_ZONE_LEADER,
  GLOG_ZONE_GATE,
  GLOG_ZONE_GARDEN,
  GLOG_ZONE_ALLEY,
  GLOG_ZONE_IKZ_PILE,
  GLOG_ZONE_IKZ_AREA,
  GLOG_ZONE_DISCARD,
  GLOG_ZONE_SELECTION,
  GLOG_ZONE_EQUIPPED, // Weapon attached to an entity
} GameLogZone;

/**
 * Tap state for game logs.
 */
typedef enum {
  GLOG_TAP_UNTAPPED = 0,
  GLOG_TAP_TAPPED,
  GLOG_TAP_COOLDOWN,
} GameLogTapState;

/**
 * Status effect type for game logs.
 */
typedef enum {
  GLOG_STATUS_FROZEN = 0,
  GLOG_STATUS_SHOCKED,
  GLOG_STATUS_EFFECT_IMMUNE,
} GameLogStatusEffect;

/**
 * Death cause for entity died logs.
 */
typedef enum {
  GLOG_DEATH_COMBAT = 0,
  GLOG_DEATH_ABILITY,
  GLOG_DEATH_EFFECT,
} GameLogDeathCause;

/**
 * Reason for deck shuffle.
 */
typedef enum {
  GLOG_SHUFFLE_MULLIGAN = 0,
  GLOG_SHUFFLE_EFFECT,
  GLOG_SHUFFLE_GAME_START,
} GameLogShuffleReason;

/**
 * Reason for game end.
 */
typedef enum {
  GLOG_END_LEADER_DEFEATED = 0,
  GLOG_END_DECK_OUT,
  GLOG_END_CONCEDE,
} GameLogEndReason;

/**
 * Card reference - identifies a card by player, definition, zone, and index.
 */
typedef struct {
  uint8_t player;           // 0 or 1
  CardDefId card_def_id;    // Card definition ID
  GameLogZone zone;         // Current zone
  int8_t zone_index;        // -1 if not applicable (e.g., hand, deck)
} GameLogCardRef;

/**
 * Metadata for CARD_ZONE_MOVED - captures card state after move.
 */
typedef struct {
  int8_t cur_atk;
  int8_t cur_hp;
  bool tapped;
  bool cooldown;
  bool has_charge;
  bool has_defender;
  bool has_infiltrate;
  bool is_frozen;
  bool is_effect_immune;
  CardDefId attached_weapons[MAX_ATTACHED_WEAPONS_LOG];
  uint8_t weapon_count;
} GameLogCardMetadata;

/* Log-specific data structures */

typedef struct {
  GameLogCardRef card;
  GameLogZone from_zone;
  int8_t from_index;
  GameLogZone to_zone;
  int8_t to_index;
  GameLogCardMetadata metadata;
} GameLogZoneMoved;

typedef struct {
  GameLogCardRef card;
  int8_t atk_delta;
  int8_t hp_delta;
  int8_t new_atk;
  int8_t new_hp;
} GameLogStatChange;

typedef struct {
  GameLogCardRef card;
  GameLogTapState new_state;
} GameLogTapStateChanged;

typedef struct {
  GameLogCardRef card;
  GameLogStatusEffect effect;
  int8_t duration; // -1 for permanent
} GameLogStatusApplied;

typedef struct {
  GameLogCardRef card;
  GameLogStatusEffect effect;
} GameLogStatusExpired;

typedef struct {
  GameLogCardRef attacker;
  GameLogCardRef target;
} GameLogCombatDeclared;

typedef struct {
  GameLogCardRef defender;
} GameLogDefenderDeclared;

typedef struct {
  GameLogCardRef attacker;
  GameLogCardRef defender;
  int8_t attacker_damage_dealt;
  int8_t attacker_damage_taken;
  int8_t defender_damage_dealt;
  int8_t defender_damage_taken;
} GameLogCombatDamage;

typedef struct {
  GameLogCardRef card;
  GameLogDeathCause cause;
} GameLogEntityDied;

typedef struct {
  GameLogCardRef card;
  uint8_t ability_index;
  uint8_t trigger_tag; // Timing tag index (AOnPlay, etc.)
} GameLogEffectQueued;

typedef struct {
  GameLogCardRef card;
  uint8_t ability_index;
} GameLogEffectEnabled;

typedef struct {
  uint8_t player;
  GameLogShuffleReason reason;
} GameLogDeckShuffled;

typedef struct {
  uint8_t player;
  uint16_t turn_number;
} GameLogTurnStarted;

typedef struct {
  uint8_t player;
  uint16_t turn_number;
} GameLogTurnEnded;

typedef struct {
  int8_t winner; // 0, 1, or 2 for draw
  GameLogEndReason reason;
} GameLogGameEnded;

/**
 * Main log entry - tagged union containing log-type-specific data.
 */
typedef struct {
  GameLogType type;
  union {
    GameLogZoneMoved zone_moved;
    GameLogTapStateChanged tap_changed;
    GameLogStatChange stat_change;
    GameLogStatusApplied status_applied;
    GameLogStatusExpired status_expired;
    GameLogCombatDeclared combat_declared;
    GameLogDefenderDeclared defender_declared;
    GameLogCombatDamage combat_damage;
    GameLogEntityDied entity_died;
    GameLogEffectQueued effect_queued;
    GameLogEffectEnabled effect_enabled;
    GameLogDeckShuffled deck_shuffled;
    GameLogTurnStarted turn_started;
    GameLogTurnEnded turn_ended;
    GameLogGameEnded game_ended;
  } data;
} GameStateLog;

/**
 * ECS Singleton - stores logs for current action batch.
 */
typedef struct {
  GameStateLog logs[MAX_GAME_STATE_LOGS];
  uint8_t count;
  uint16_t turn_number; // Current turn for context
} GameStateLogContext;

extern ECS_COMPONENT_DECLARE(GameStateLogContext);

void azk_register_game_log_components(ecs_world_t *world);

#endif /* AZUKI_COMPONENTS_GAME_LOG_H */
