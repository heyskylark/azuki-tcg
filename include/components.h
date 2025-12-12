#ifndef AZUKI_ECS_COMPONENTS_H
#define AZUKI_ECS_COMPONENTS_H

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
  ACT_NOOP = 0,
  ACT_PLAY_ENTITY_TO_GARDEN = 1,
  ACT_PLAY_ENTITY_TO_ALLEY = 2,
  /* 3-5 reserved for weapons/spells once those flows are online */
  ACT_ATTACK = 6,
  ACT_ATTACH_WEAPON_FROM_HAND = 7,
  ACT_DECLARE_DEFENDER = 8,
  ACT_GATE_PORTAL = 9,
  /* 10-11 reserved for gate portal + ability activation */
  ACT_MULLIGAN_SHUFFLE = 12,
  ACT_ACTIVATE_ABILITY = 13,
  ACT_SELECT_COST_TARGET = 14,
  ACT_SELECT_EFFECT_TARGET = 15,
  ACT_ACCEPT_TRIGGERED_ABILITY = 16
} ActionType;

#define AZK_ACTION_TYPE_COUNT (ACT_ACCEPT_TRIGGERED_ABILITY + 1)

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

typedef enum {
  ABILITY_SELECTION_NONE = 0,
  ABILITY_SELECTION_PROMPT = 1,
  ABILITY_SELECTION_COST = 2,
  ABILITY_SELECTION_EFFECT = 3
} AbilitySelectionPhase;

#define AZK_MAX_ABILITY_SELECTIONS 3

typedef struct {
  bool has_pending;
  bool from_trigger;
  bool optional;
  bool awaiting_consent;
  ecs_entity_t source_card;
  ecs_entity_t player;
  uint16_t ability_uid; // card_def_id << 8 | ability_index
  AbilitySelectionPhase phase;
  uint8_t cost_min, effect_min;
  uint8_t cost_expected, cost_filled;
  uint8_t effect_expected, effect_filled;
  ecs_entity_t cost_targets[AZK_MAX_ABILITY_SELECTIONS];
  ecs_entity_t effect_targets[AZK_MAX_ABILITY_SELECTIONS];
  ecs_entity_t ikz_cards[AZK_MAX_IKZ_PAYMENT];
  uint8_t ikz_card_count;
} AbilityContext;

typedef struct {
  uint32_t once_per_turn_mask;
} AbilityUsage;

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
extern ECS_COMPONENT_DECLARE(GameState);
extern ECS_COMPONENT_DECLARE(PlayerNumber);
extern ECS_COMPONENT_DECLARE(PlayerId);
extern ECS_COMPONENT_DECLARE(IKZToken);
extern ECS_COMPONENT_DECLARE(ZoneIndex);
extern ECS_COMPONENT_DECLARE(AbilityContext);
extern ECS_COMPONENT_DECLARE(AbilityUsage);

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
