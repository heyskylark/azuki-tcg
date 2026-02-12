#ifndef AZUKI_ECS_COMPONENTS_H
#define AZUKI_ECS_COMPONENTS_H

#include <flecs.h>
#include <stdbool.h>
#include <stdint.h>

#include "constants/game.h"

typedef enum {
  PHASE_PREGAME_MULLIGAN = 0,
  PHASE_START_OF_TURN = 1,
  PHASE_MAIN = 2,
  PHASE_RESPONSE_WINDOW = 3,
  PHASE_COMBAT_RESOLVE = 4,
  PHASE_END_TURN_ACTION = 5,
  PHASE_END_TURN = 6,
  PHASE_END_MATCH = 7,
  PHASE_COUNT = PHASE_END_MATCH + 1
} Phase;

typedef enum {
  ABILITY_PHASE_NONE = 0,
  ABILITY_PHASE_CONFIRMATION = 1,
  ABILITY_PHASE_COST_SELECTION = 2,
  ABILITY_PHASE_EFFECT_SELECTION = 3,
  ABILITY_PHASE_SELECTION_PICK = 4, // Pick from selection zone
  ABILITY_PHASE_BOTTOM_DECK = 5,    // Order cards to bottom of deck
} AbilityPhase;

typedef enum {
  ACT_NOOP = 0,
  ACT_PLAY_ENTITY_TO_GARDEN = 1,
  ACT_PLAY_ENTITY_TO_ALLEY = 2,
  ACT_ATTACK = 6,
  ACT_ATTACH_WEAPON_FROM_HAND = 7,
  ACT_PLAY_SPELL_FROM_HAND = 8,
  ACT_DECLARE_DEFENDER = 9,
  ACT_GATE_PORTAL = 10,
  ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY = 11,
  ACT_ACTIVATE_ALLEY_ABILITY = 12,
  ACT_SELECT_COST_TARGET = 13,
  ACT_SELECT_EFFECT_TARGET = 14,
  ACT_CONFIRM_ABILITY = 16,
  ACT_SELECT_FROM_SELECTION = 18, // Pick card from selection zone
  ACT_BOTTOM_DECK_CARD = 19,      // Bottom deck a single card from selection
  ACT_BOTTOM_DECK_ALL = 20,       // Bottom deck all remaining cards in order
  ACT_SELECT_TO_ALLEY = 21,       // Select card from selection zone to alley
  ACT_SELECT_TO_EQUIP = 22,       // Select weapon from selection to equip to entity
  ACT_MULLIGAN_SHUFFLE = 23 // Must always be highest for AZK_ACTION_TYPE_COUNT
} ActionType;

#define AZK_ACTION_TYPE_COUNT (ACT_MULLIGAN_SHUFFLE + 1)

typedef struct {
  ecs_entity_t player;
  ActionType type;
  int subaction_1;
  int subaction_2;
  int subaction_3;
} UserAction;

typedef struct {
  UserAction user_action;
  bool invalid_action;
  UserAction user_action_history[MAX_USER_ACTION_HISTORY_SIZE];
  uint32_t history_size;
  uint32_t history_head;
} ActionContext;

typedef struct {
  ecs_entity_t deck, hand, leader, gate, garden, alley, ikz_pile, ikz_area,
      discard, selection;
  uint16_t deck_size, hand_size, leader_size, gate_size, garden_size,
      alley_size, ikz_pile_size, ikz_area_size, discard_size, selection_size;
} PlayerZones;

typedef struct {
  ecs_entity_t attacking_card;
  ecs_entity_t defender_card;
  bool defender_intercepted;
} CombatState;

typedef struct {
  AbilityPhase phase;
  ecs_entity_t source_card;
  ecs_entity_t owner;
  bool is_optional;
  uint8_t cost_min, effect_min;
  uint8_t cost_expected, effect_expected;
  uint8_t cost_filled, effect_filled;
  ecs_entity_t cost_targets[MAX_ABILITY_SELECTION];
  ecs_entity_t effect_targets[MAX_ABILITY_SELECTION];

  // Selection zone tracking for reveal/examine effects
  uint8_t selection_count;    // Cards currently in selection zone
  uint8_t selection_picked;   // Cards picked from selection zone
  uint8_t selection_pick_max; // Max cards to pick (e.g., 1 for "up to 1")
  ecs_entity_t selection_cards[MAX_SELECTION_ZONE_SIZE];
} AbilityContext;

typedef struct {
  uint32_t seed;
  uint32_t rng_state;
  int8_t active_player_index;
  Phase phase;
  uint8_t response_window;
  int8_t winner; // -1 if no winner, 0 if player 0, 1 if player 1, 2 if draw
  uint16_t turn_number; // Current turn number (increments at start of each turn)
  ecs_entity_t players[MAX_PLAYERS_PER_MATCH];
  PlayerZones zones[MAX_PLAYERS_PER_MATCH];
  CombatState combat_state;
  uint8_t entities_played_garden_this_turn[MAX_PLAYERS_PER_MATCH];
  uint8_t entities_played_alley_this_turn[MAX_PLAYERS_PER_MATCH];
  uint8_t entities_returned_to_hand_this_turn[MAX_PLAYERS_PER_MATCH];
} GameState;
typedef struct {
  uint8_t player_number;
} PlayerNumber;
typedef struct {
  uint8_t pid;
} PlayerId;
typedef struct {
  ecs_entity_t ikz_token;
} IKZToken;
/*
ZoneIndex must exist (even with ordered children) because cards can be placed in
gaps in the zone.
*/
typedef struct {
  uint8_t index;
} ZoneIndex;

/* Triggered effect queue for deferred ability processing */
typedef struct {
  ecs_entity_t source_card;
  ecs_entity_t owner;
  uint8_t timing_tag; // Index into timing tag array (AOnPlay, etc.)
} PendingTriggeredEffect;

#define MAX_TRIGGERED_EFFECT_QUEUE 8

typedef struct {
  PendingTriggeredEffect effects[MAX_TRIGGERED_EFFECT_QUEUE];
  uint8_t count;
} TriggeredEffectQueue;

/* Passive buff queue for deferred passive ability processing */
typedef struct {
  ecs_entity_t entity;   // Entity to apply buff to
  ecs_entity_t source;   // Source of the buff (for pair tracking)
  int8_t atk_modifier;   // Attack modifier to apply (0 if not changing)
  int8_t hp_modifier;    // Health modifier to apply (0 if not changing)
  bool is_removal;       // True if this is a removal request
} PendingPassiveBuff;

#define MAX_PASSIVE_BUFF_QUEUE 8

typedef struct {
  PendingPassiveBuff buffs[MAX_PASSIVE_BUFF_QUEUE];
  uint8_t count;
} PassiveBuffQueue;

/* Deck reorder queue for deferred child ordering fixes */
typedef struct {
  ecs_entity_t deck;
  ecs_entity_t card;
} PendingDeckReorder;

#define MAX_DECK_REORDER_QUEUE 16

typedef struct {
  PendingDeckReorder entries[MAX_DECK_REORDER_QUEUE];
  uint8_t count;
} DeckReorderQueue;

typedef struct {
  ecs_entity_t pipeline_mulligan;
  ecs_entity_t pipeline_start_of_turn;
  ecs_entity_t pipeline_main;
  ecs_entity_t pipeline_response;
  ecs_entity_t pipeline_combat;
  ecs_entity_t pipeline_end_turn;
  ecs_entity_t pipeline_end_match;
  ecs_entity_t pipeline_ability;
  ecs_entity_t phase_gate_system;
  ecs_entity_t current_pipeline;
} PhaseGateCache;

extern ECS_COMPONENT_DECLARE(ActionContext);
extern ECS_COMPONENT_DECLARE(AbilityContext);
extern ECS_COMPONENT_DECLARE(GameState);
extern ECS_COMPONENT_DECLARE(PlayerNumber);
extern ECS_COMPONENT_DECLARE(PlayerId);
extern ECS_COMPONENT_DECLARE(IKZToken);
extern ECS_COMPONENT_DECLARE(ZoneIndex);
extern ECS_COMPONENT_DECLARE(TriggeredEffectQueue);
extern ECS_COMPONENT_DECLARE(PassiveBuffQueue);
extern ECS_COMPONENT_DECLARE(DeckReorderQueue);
extern ECS_COMPONENT_DECLARE(PhaseGateCache);

/* Relationship Entities */
extern ECS_ENTITY_DECLARE(Rel_OwnedBy);

/* Board Zone Tags */
extern ECS_TAG_DECLARE(ZDeck);
extern ECS_TAG_DECLARE(ZHand);
extern ECS_TAG_DECLARE(ZLeader);
extern ECS_TAG_DECLARE(ZGate);
extern ECS_TAG_DECLARE(ZGarden);
extern ECS_TAG_DECLARE(ZAlley);
extern ECS_TAG_DECLARE(ZIKZPileTag);
extern ECS_TAG_DECLARE(ZIKZAreaTag);
extern ECS_TAG_DECLARE(ZDiscard);
extern ECS_TAG_DECLARE(ZSelection);

/* System Phase Tags */
extern ECS_TAG_DECLARE(TMulligan);
extern ECS_TAG_DECLARE(TStartOfTurn);
extern ECS_TAG_DECLARE(TMain);
extern ECS_TAG_DECLARE(TResponseWindow);
extern ECS_TAG_DECLARE(TCombatResolve);
extern ECS_TAG_DECLARE(TEndTurnAction);
extern ECS_TAG_DECLARE(TEndTurn);
extern ECS_TAG_DECLARE(TEndMatch);
extern ECS_TAG_DECLARE(TAbilityResolution);

void azk_register_components(ecs_world_t *world);

#endif
