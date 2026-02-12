#ifndef AZUKI_UTILS_TRAINING_OBSERVATION_UTIL_H
#define AZUKI_UTILS_TRAINING_OBSERVATION_UTIL_H

#include "components/components.h"
#include "constants/game.h"
#include "generated/card_defs.h"
#include "validation/action_enumerator.h"

typedef struct {
  int16_t card_def_id; // -1 indicates empty slot
  int8_t cur_atk;
} TrainingWeaponObservationData;

typedef struct {
  int16_t card_def_id; // Always present
  TapState tap_state;
  CurStats cur_stats;
  uint8_t weapon_count;
  TrainingWeaponObservationData weapons[MAX_ATTACHED_WEAPONS];
  bool has_charge;
  bool has_defender;
  bool has_infiltrate;
} TrainingLeaderObservationData;

typedef struct {
  int16_t card_def_id; // Always present
  TapState tap_state;
} TrainingGateObservationData;

typedef struct {
  int16_t card_def_id; // -1 indicates empty slot
  uint8_t zone_index;
} TrainingHandCardObservationData;

typedef struct {
  int16_t card_def_id; // -1 indicates empty slot
  uint8_t zone_index;
} TrainingDiscardCardObservationData;

typedef struct {
  int16_t card_def_id; // -1 indicates empty slot
  TapState tap_state;
  uint8_t zone_index;
  bool has_cur_stats;
  CurStats cur_stats;
  uint8_t weapon_count;
  TrainingWeaponObservationData weapons[MAX_ATTACHED_WEAPONS];
  bool has_charge;
  bool has_defender;
  bool has_infiltrate;
  bool is_frozen;
  bool is_shocked;
  bool is_effect_immune;
} TrainingBoardCardObservationData;

typedef struct {
  int16_t card_def_id; // -1 indicates empty slot
  TapState tap_state;
  uint8_t zone_index;
} TrainingIKZCardObservationData;

typedef struct {
  TrainingLeaderObservationData leader;
  TrainingGateObservationData gate;
  TrainingHandCardObservationData hand[MAX_HAND_SIZE];
  TrainingBoardCardObservationData alley[ALLEY_SIZE];
  TrainingBoardCardObservationData garden[GARDEN_SIZE];
  TrainingDiscardCardObservationData discard[MAX_DECK_SIZE];
  TrainingBoardCardObservationData selection[MAX_SELECTION_ZONE_SIZE];
  TrainingIKZCardObservationData ikz_area[IKZ_AREA_SIZE];
  uint8_t hand_count;
  uint8_t deck_count;
  uint8_t ikz_pile_count;
  uint8_t selection_count;
  bool has_ikz_token;
} TrainingMyObservationData;

typedef struct {
  TrainingLeaderObservationData leader;
  TrainingGateObservationData gate;
  TrainingBoardCardObservationData alley[ALLEY_SIZE];
  TrainingBoardCardObservationData garden[GARDEN_SIZE];
  TrainingDiscardCardObservationData discard[MAX_DECK_SIZE];
  TrainingIKZCardObservationData ikz_area[IKZ_AREA_SIZE];
  uint8_t hand_count;
  uint8_t deck_count;
  uint8_t ikz_pile_count;
  bool has_ikz_token;
} TrainingOpponentObservationData;

typedef struct {
  bool primary_action_mask[AZK_ACTION_TYPE_COUNT];
  uint16_t legal_action_count;
  uint8_t legal_primary[AZK_MAX_LEGAL_ACTIONS];
  uint8_t legal_sub1[AZK_MAX_LEGAL_ACTIONS];
  uint8_t legal_sub2[AZK_MAX_LEGAL_ACTIONS];
  uint8_t legal_sub3[AZK_MAX_LEGAL_ACTIONS];
} TrainingActionMaskObs;

typedef struct {
  AbilityPhase phase;
  uint8_t pending_confirmation_count;
  bool has_source_card_def_id;
  int16_t source_card_def_id; // -1 when absent
  uint8_t cost_target_type;
  uint8_t effect_target_type;
  uint8_t selection_count;
  uint8_t selection_picked;
  uint8_t selection_pick_max;
  int8_t active_player_index;
} TrainingAbilityContextObservationData;

typedef struct {
  TrainingMyObservationData my_observation_data;
  TrainingOpponentObservationData opponent_observation_data;
  Phase phase;
  TrainingAbilityContextObservationData ability_context;
  TrainingActionMaskObs action_mask;
} TrainingObservationData;

TrainingObservationData create_training_observation_data(ecs_world_t *world,
                                                         int8_t player_index);
void create_training_observation_data_pair(
    ecs_world_t *world,
    TrainingObservationData out_observations[MAX_PLAYERS_PER_MATCH]);

#endif
