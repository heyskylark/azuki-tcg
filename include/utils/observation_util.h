#ifndef AZUKI_UTILS_OBSERVATION_UTIL_H
#define AZUKI_UTILS_OBSERVATION_UTIL_H

#include "components/components.h"
#include "generated/card_defs.h"
#include "validation/action_enumerator.h"

typedef struct {
  Type type;
  CardId id;
  TapState tap_state;
  uint8_t zone_index;
} IKZCardObservationData;

typedef struct {
  Type type;
  CardId id;
  BaseStats base_stats;
  IKZCost ikz_cost;
} WeaponObservationData;

typedef struct {
  Type type;
  CardId id;
  CurStats cur_stats;
  TapState tap_state;
  uint8_t weapon_count;
  WeaponObservationData weapons[MAX_ATTACHED_WEAPONS];
} LeaderCardObservationData;

typedef struct {
  Type type;
  CardId id;
  TapState tap_state;
} GateCardObservationData;

typedef struct {
  Type type;
  CardId id;
  TapState tap_state;
  IKZCost ikz_cost;
  uint8_t zone_index;
  bool has_cur_stats;
  CurStats cur_stats;
  bool has_gate_points;
  GatePoints gate_points;
  uint8_t weapon_count;
  WeaponObservationData weapons[MAX_ATTACHED_WEAPONS];
  bool is_frozen;
  bool is_shocked;
  bool is_effect_immune;
} CardObservationData;

typedef struct {
  LeaderCardObservationData leader;
  GateCardObservationData gate;
  CardObservationData hand[MAX_HAND_SIZE];
  CardObservationData alley[ALLEY_SIZE];
  CardObservationData garden[GARDEN_SIZE];
  CardObservationData discard[MAX_DECK_SIZE];
  CardObservationData selection[MAX_SELECTION_ZONE_SIZE];
  IKZCardObservationData ikz_area[IKZ_AREA_SIZE];
  uint8_t deck_count;
  uint8_t ikz_pile_count;
  uint8_t selection_count;
  bool has_ikz_token;
} MyObservationData;

typedef struct {
  LeaderCardObservationData leader;
  GateCardObservationData gate;
  CardObservationData alley[ALLEY_SIZE];
  CardObservationData garden[GARDEN_SIZE];
  CardObservationData discard[MAX_DECK_SIZE];
  IKZCardObservationData ikz_area[IKZ_AREA_SIZE];
  uint8_t hand_count;
  uint8_t deck_count;
  uint8_t ikz_pile_count;
  bool has_ikz_token;
} OpponentObservationData;

typedef struct {
  bool primary_action_mask[AZK_ACTION_TYPE_COUNT];
  uint16_t legal_action_count;
  uint8_t legal_primary[AZK_MAX_LEGAL_ACTIONS];
  uint8_t legal_sub1[AZK_MAX_LEGAL_ACTIONS];
  uint8_t legal_sub2[AZK_MAX_LEGAL_ACTIONS];
  uint8_t legal_sub3[AZK_MAX_LEGAL_ACTIONS];
} ActionMaskObs;

typedef struct {
  MyObservationData my_observation_data;
  OpponentObservationData opponent_observation_data;
  Phase phase;
  ActionMaskObs action_mask;
} ObservationData;

ObservationData create_observation_data(ecs_world_t *world,
                                        int8_t player_index);
bool is_game_over(ecs_world_t *world);

#endif
