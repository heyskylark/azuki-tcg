#ifndef AZUKI_UTILS_OBSERVATION_UTIL_H
#define AZUKI_UTILS_OBSERVATION_UTIL_H

#include "components.h"
#include "generated/card_defs.h"

typedef struct {
  Type type;
  CardId id;
  TapState tap_state;
} IKZCardObservationData;

typedef struct {
  Type type;
  CardId id;
  CurStats cur_stats;
  TapState tap_state;
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
  bool has_cur_stats;
  CurStats cur_stats;
  bool has_gate_points;
  GatePoints gate_points;
} CardObservationData;

typedef struct {
  LeaderCardObservationData leader;
  GateCardObservationData gate;
  CardObservationData hand[MAX_HAND_SIZE];
  CardObservationData alley[ALLEY_SIZE];
  CardObservationData garden[GARDEN_SIZE];
  IKZCardObservationData ikz_area[IKZ_AREA_SIZE];
  uint8_t ikz_pile_count;
  uint8_t discard_count;
  bool has_ikz_token;
} MyObservationData;

typedef struct {
  LeaderCardObservationData leader;
  GateCardObservationData gate;
  CardObservationData alley[ALLEY_SIZE];
  CardObservationData garden[GARDEN_SIZE];
  IKZCardObservationData ikz_area[IKZ_AREA_SIZE];
  uint8_t hand_count;
  uint8_t ikz_pile_count;
  uint8_t discard_count;
  bool has_ikz_token;
} OpponentObservationData;

typedef struct {
  MyObservationData my_observation_data;
  OpponentObservationData opponent_observation_data;
} ObservationData;

ObservationData create_observation_data(ecs_world_t *world);

#endif