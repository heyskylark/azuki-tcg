#ifndef AZUKI_ECS_COMPONENTS_H
#define AZUKI_ECS_COMPONENTS_H

#include <stdint.h>
#include <flecs.h>

#include "constants/game.h"

typedef enum {
  PHASE_PREGAME_MULLIGAN = 0,
  PHASE_START_OF_TURN = 1,
  PHASE_MAIN = 2,
  PHASE_COMBAT_DECLARED = 3,
  PHASE_RESPONSE_WINDOW = 4,
  PHASE_COMBAT_RESOLVE = 5,
  PHASE_END_TURN = 6,
  PHASE_END_MATCH = 7
} Phase;

typedef enum {
  ACT_NOOP = 0,
  ACT_PLAY_ENTITY_TO_GARDEN = 1,
  ACT_PLAY_ENTITY_TO_ALLEY = 2,
  /* 3-5 reserved for weapons/spells once those flows are online */
  ACT_ATTACK = 6,
  ACT_DECLARE_DEFENDER = 7,
  /* 8-9 reserved for gate portal + ability activation */
  ACT_END_TURN = 10,
  ACT_MULLIGAN_KEEP = 11,
  ACT_MULLIGAN_SHUFFLE = 12
} ActionType;

typedef struct {
  ecs_entity_t deck, hand, leader, gate, garden, alley, ikz_pile, ikz_area, discard;
  uint16_t deck_size, hand_size, leader_size, gate_size, garden_size, alley_size, ikz_pile_size, ikz_area_size, discard_size;
} PlayerZones;

typedef struct { 
  uint32_t seed;
  int8_t active_player_index;
  Phase phase;
  uint8_t response_window;
  int8_t winner;
  ecs_entity_t players[MAX_PLAYERS_PER_MATCH];
  PlayerZones zones[MAX_PLAYERS_PER_MATCH];
} GameState;
typedef struct { uint8_t player_number; } PlayerNumber;
typedef struct { uint8_t pid; } PlayerId;

extern ECS_COMPONENT_DECLARE(GameState);
extern ECS_COMPONENT_DECLARE(PlayerNumber);
extern ECS_COMPONENT_DECLARE(PlayerId);

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
extern ECS_TAG_DECLARE(TCombatDeclared);
extern ECS_TAG_DECLARE(TResponseWindow);
extern ECS_TAG_DECLARE(TCombatResolve);
extern ECS_TAG_DECLARE(TEndTurn);
extern ECS_TAG_DECLARE(TEndMatch);

void azk_register_components(ecs_world_t *world);

#endif
