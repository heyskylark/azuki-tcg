#ifndef AZUKI_ECS_COMPONENTS_H
#define AZUKI_ECS_COMPONENTS_H

#include <stdbool.h>
#include <stdint.h>
#include <flecs.h>

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
  ACT_MULLIGAN_SHUFFLE = 17  // Must always be highest for AZK_ACTION_TYPE_COUNT
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
  ecs_entity_t deck, hand, leader, gate, garden, alley, ikz_pile, ikz_area, discard;
  uint16_t deck_size, hand_size, leader_size, gate_size, garden_size, alley_size, ikz_pile_size, ikz_area_size, discard_size;
} PlayerZones;

typedef struct {
  ecs_entity_t attacking_card;
  ecs_entity_t defender_card;
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
} AbilityContext;

typedef struct { 
  uint32_t seed;
  uint32_t rng_state;
  int8_t active_player_index;
  Phase phase;
  uint8_t response_window;
  int8_t winner; // -1 if no winner, 0 if player 0, 1 if player 1, 2 if draw
  ecs_entity_t players[MAX_PLAYERS_PER_MATCH];
  PlayerZones zones[MAX_PLAYERS_PER_MATCH];
  CombatState combat_state;
} GameState;
typedef struct { uint8_t player_number; } PlayerNumber;
typedef struct { uint8_t pid; } PlayerId;
typedef struct { ecs_entity_t ikz_token; } IKZToken;
/*
ZoneIndex must exist (even with ordered children) because cards can be placed in gaps in the zone.
*/
typedef struct { uint8_t index; } ZoneIndex;

extern ECS_COMPONENT_DECLARE(ActionContext);
extern ECS_COMPONENT_DECLARE(AbilityContext);
extern ECS_COMPONENT_DECLARE(GameState);
extern ECS_COMPONENT_DECLARE(PlayerNumber);
extern ECS_COMPONENT_DECLARE(PlayerId);
extern ECS_COMPONENT_DECLARE(IKZToken);
extern ECS_COMPONENT_DECLARE(ZoneIndex);

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

/* System Phase Tags */
extern ECS_TAG_DECLARE(TMulligan);
extern ECS_TAG_DECLARE(TStartOfTurn);
extern ECS_TAG_DECLARE(TMain);
extern ECS_TAG_DECLARE(TResponseWindow);
extern ECS_TAG_DECLARE(TCombatResolve);
extern ECS_TAG_DECLARE(TEndTurnAction);
extern ECS_TAG_DECLARE(TEndTurn);
extern ECS_TAG_DECLARE(TEndMatch);

void azk_register_components(ecs_world_t *world);

#endif
