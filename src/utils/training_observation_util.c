#include "utils/training_observation_util.h"

#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "abilities/ability_registry.h"
#include "abilities/ability_system.h"
#include "components/abilities.h"
#include "utils/card_utils.h"
#include "utils/deck_utils.h"
#include "utils/phase_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"
#include "utils/zone_util.h"

typedef struct {
  bool initialized;
  bool enabled;
  uint64_t report_every;
  uint64_t my_calls;
  uint64_t my_total_ns;
  uint64_t leader_gate_ns;
  uint64_t hand_ns;
  uint64_t alley_ns;
  uint64_t garden_ns;
  uint64_t discard_ns;
  uint64_t selection_ns;
  uint64_t ikz_ns;
  uint64_t counts_ns;
  uint64_t pair_calls;
  uint64_t pair_total_ns;
  uint64_t action_mask_calls;
  uint64_t action_mask_ns;
} ObsProfileState;

static ObsProfileState k_obs_profile = {0};

static uint64_t obs_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static bool obs_env_enabled(const char *name) {
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0') {
    return false;
  }
  if (value[0] == '0' && value[1] == '\0') {
    return false;
  }
  return true;
}

static void init_obs_profile_if_needed(void) {
  if (k_obs_profile.initialized) {
    return;
  }

  k_obs_profile.initialized = true;
  k_obs_profile.enabled = obs_env_enabled("AZK_OBS_PROFILE");

  const char *report_every = getenv("AZK_OBS_PROFILE_EVERY");
  if (report_every != NULL && report_every[0] != '\0') {
    char *end_ptr = NULL;
    unsigned long long parsed = strtoull(report_every, &end_ptr, 10);
    if (end_ptr != report_every && parsed > 0ull) {
      k_obs_profile.report_every = (uint64_t)parsed;
    }
  }
  if (k_obs_profile.report_every == 0) {
    k_obs_profile.report_every = 50000;
  }
}

static void maybe_report_obs_profile(void) {
  if (!k_obs_profile.enabled || k_obs_profile.my_calls == 0 ||
      (k_obs_profile.my_calls % k_obs_profile.report_every) != 0) {
    return;
  }

  const double avg_my_us =
      k_obs_profile.my_total_ns / (double)k_obs_profile.my_calls / 1000.0;
  const double avg_pair_us =
      k_obs_profile.pair_calls == 0
          ? 0.0
          : k_obs_profile.pair_total_ns / (double)k_obs_profile.pair_calls /
                1000.0;
  const double avg_mask_us =
      k_obs_profile.action_mask_calls == 0
          ? 0.0
          : k_obs_profile.action_mask_ns /
                (double)k_obs_profile.action_mask_calls / 1000.0;

  fprintf(stderr,
          "[ObsProfile] my_calls=%" PRIu64 " avg_my_us=%.2f "
          "leader_gate_us=%.2f hand_us=%.2f alley_us=%.2f garden_us=%.2f "
          "discard_us=%.2f selection_us=%.2f ikz_us=%.2f counts_us=%.2f "
          "mask_calls=%" PRIu64 " avg_mask_us=%.2f "
          "pair_calls=%" PRIu64 " avg_pair_us=%.2f\n",
          k_obs_profile.my_calls, avg_my_us,
          k_obs_profile.leader_gate_ns / (double)k_obs_profile.my_calls / 1000.0,
          k_obs_profile.hand_ns / (double)k_obs_profile.my_calls / 1000.0,
          k_obs_profile.alley_ns / (double)k_obs_profile.my_calls / 1000.0,
          k_obs_profile.garden_ns / (double)k_obs_profile.my_calls / 1000.0,
          k_obs_profile.discard_ns / (double)k_obs_profile.my_calls / 1000.0,
          k_obs_profile.selection_ns / (double)k_obs_profile.my_calls / 1000.0,
          k_obs_profile.ikz_ns / (double)k_obs_profile.my_calls / 1000.0,
          k_obs_profile.counts_ns / (double)k_obs_profile.my_calls / 1000.0,
          k_obs_profile.action_mask_calls, avg_mask_us, k_obs_profile.pair_calls,
          avg_pair_us);
}

static TrainingWeaponObservationData empty_weapon_observation(void) {
  TrainingWeaponObservationData observation = {0};
  observation.card_def_id = -1;
  return observation;
}

static TrainingHandCardObservationData empty_hand_card_observation(
    uint8_t zone_index) {
  TrainingHandCardObservationData observation = {0};
  observation.card_def_id = -1;
  observation.zone_index = zone_index;
  return observation;
}

static TrainingDiscardCardObservationData empty_discard_card_observation(
    uint8_t zone_index) {
  TrainingDiscardCardObservationData observation = {0};
  observation.card_def_id = -1;
  observation.zone_index = zone_index;
  return observation;
}

static TrainingBoardCardObservationData empty_board_card_observation(
    uint8_t zone_index) {
  TrainingBoardCardObservationData observation = {0};
  observation.card_def_id = -1;
  observation.zone_index = zone_index;
  for (size_t i = 0; i < MAX_ATTACHED_WEAPONS; ++i) {
    observation.weapons[i] = empty_weapon_observation();
  }
  return observation;
}

static TrainingIKZCardObservationData empty_ikz_card_observation(
    uint8_t zone_index) {
  TrainingIKZCardObservationData observation = {0};
  observation.card_def_id = -1;
  observation.zone_index = zone_index;
  return observation;
}

static int16_t card_def_id_or_empty(const CardId *card_id) {
  if (card_id == NULL) {
    return -1;
  }
  return (int16_t)card_id->id;
}

static void reset_legal_actions(TrainingActionMaskObs *action_mask) {
  ecs_assert(action_mask != NULL, ECS_INVALID_PARAMETER, "Output mask is null");
  action_mask->legal_action_count = 0;
  for (size_t i = 0; i < AZK_ACTION_TYPE_COUNT; ++i) {
    action_mask->primary_action_mask[i] = false;
  }
}

static void populate_action_mask_from_set(const AzkActionMaskSet *mask_set,
                                          TrainingActionMaskObs *out_mask) {
  ecs_assert(mask_set != NULL, ECS_INVALID_PARAMETER, "Mask set is null");
  ecs_assert(out_mask != NULL, ECS_INVALID_PARAMETER, "Output mask is null");

  for (size_t action_type = 0; action_type < AZK_ACTION_TYPE_COUNT;
       ++action_type) {
    out_mask->primary_action_mask[action_type] =
        mask_set->head0_mask[action_type] != 0;
  }

  uint16_t legal_action_count = mask_set->legal_action_count;
  if (legal_action_count > AZK_MAX_LEGAL_ACTIONS) {
    legal_action_count = AZK_MAX_LEGAL_ACTIONS;
  }

  for (uint16_t i = 0; i < legal_action_count; ++i) {
    const UserAction *action = &mask_set->legal_actions[i];
    ecs_assert(action->type >= 0 && action->type < AZK_ACTION_TYPE_COUNT,
               ECS_INVALID_PARAMETER, "Action type %d out of bounds",
               action->type);
    out_mask->legal_primary[i] = (uint8_t)action->type;
    out_mask->legal_sub1[i] = (uint8_t)action->subaction_1;
    out_mask->legal_sub2[i] = (uint8_t)action->subaction_2;
    out_mask->legal_sub3[i] = (uint8_t)action->subaction_3;
  }

  out_mask->legal_action_count = legal_action_count;
}

static TrainingActionMaskObs build_training_action_mask(ecs_world_t *world,
                                                        const GameState *gs,
                                                        int8_t player_index) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");

  TrainingActionMaskObs action_mask = {0};
  reset_legal_actions(&action_mask);

  AzkActionMaskSet mask_set = {0};
  bool ok =
      azk_build_action_mask_for_player(world, gs, player_index, &mask_set);
  ecs_assert(ok, ECS_INVALID_OPERATION, "Failed to build action mask");

  populate_action_mask_from_set(&mask_set, &action_mask);
  return action_mask;
}

static bool should_expose_training_action_mask(ecs_world_t *world,
                                               const GameState *gs,
                                               int8_t player_index) {
  if (gs == NULL || gs->winner != -1) {
    return false;
  }

  if (player_index != gs->active_player_index) {
    return false;
  }

  if (!phase_requires_user_action(world, gs->phase)) {
    return false;
  }

  // Keep mask generation aligned with azk_engine_requires_action.
  if (azk_has_pending_deck_reorders(world) ||
      azk_has_pending_passive_buffs(world) ||
      (azk_has_queued_triggered_effects(world) &&
       !azk_is_in_ability_phase(world))) {
    return false;
  }

  // Mirror PhaseGate auto-transitions so we don't expose stale legal actions.
  if (gs->phase == PHASE_MAIN && gs->combat_state.attacking_card != 0 &&
      !azk_has_queued_triggered_effects(world) &&
      !azk_is_in_ability_phase(world)) {
    return false;
  }

  if (gs->phase == PHASE_RESPONSE_WINDOW &&
      !azk_has_queued_triggered_effects(world) &&
      !azk_is_in_ability_phase(world) &&
      !defender_can_respond(world, gs, gs->active_player_index)) {
    return false;
  }

  return true;
}

static TrainingWeaponObservationData get_weapon_observation(
    ecs_world_t *world, ecs_entity_t weapon_card) {
  TrainingWeaponObservationData observation = empty_weapon_observation();
  const CardId *card_id = ecs_get(world, weapon_card, CardId);
  observation.card_def_id = card_def_id_or_empty(card_id);
  if (observation.card_def_id == -1) {
    return observation;
  }
  const CurStats *cur_stats = ecs_get(world, weapon_card, CurStats);
  if (cur_stats != NULL) {
    observation.cur_atk = cur_stats->cur_atk;
  }
  return observation;
}

static uint8_t set_attached_weapon_observations(
    ecs_world_t *world, ecs_entity_t parent_card,
    TrainingWeaponObservationData *weapons) {
  uint8_t weapon_count = 0;
  for (size_t i = 0; i < MAX_ATTACHED_WEAPONS; ++i) {
    weapons[i] = empty_weapon_observation();
  }

  ecs_iter_t it = ecs_children(world, parent_card);
  while (ecs_children_next(&it)) {
    for (int i = 0; i < it.count; ++i) {
      ecs_entity_t child = it.entities[i];
      if (!ecs_has_id(world, child, TWeapon)) {
        continue;
      }
      if (weapon_count < MAX_ATTACHED_WEAPONS) {
        weapons[weapon_count++] = get_weapon_observation(world, child);
      }
    }
  }
  return weapon_count;
}

static TrainingBoardCardObservationData get_board_card_observation(
    ecs_world_t *world, ecs_entity_t card, uint8_t fallback_zone_index,
    bool use_entity_zone_index) {
  TrainingBoardCardObservationData observation =
      empty_board_card_observation(fallback_zone_index);
  const CardId *card_id = ecs_get(world, card, CardId);
  observation.card_def_id = card_def_id_or_empty(card_id);
  if (observation.card_def_id == -1) {
    return observation;
  }

  const TapState *tap_state = ecs_get(world, card, TapState);
  if (tap_state != NULL) {
    observation.tap_state = *tap_state;
  }

  if (use_entity_zone_index) {
    const ZoneIndex *zone_index = ecs_get(world, card, ZoneIndex);
    if (zone_index != NULL) {
      observation.zone_index = zone_index->index;
    }
  }

  const CurStats *cur_stats = ecs_get(world, card, CurStats);
  if (cur_stats != NULL) {
    observation.has_cur_stats = true;
    observation.cur_stats = *cur_stats;
  }

  observation.weapon_count =
      set_attached_weapon_observations(world, card, observation.weapons);

  observation.has_charge = ecs_has(world, card, Charge);
  observation.has_defender = ecs_has(world, card, Defender);
  observation.has_infiltrate = ecs_has(world, card, Infiltrate);
  observation.is_frozen = ecs_has(world, card, Frozen);
  observation.is_shocked = ecs_has(world, card, Shocked);
  observation.is_effect_immune = ecs_has(world, card, EffectImmune);
  return observation;
}

static TrainingHandCardObservationData get_hand_card_observation(
    ecs_world_t *world, ecs_entity_t card, uint8_t fallback_zone_index) {
  TrainingHandCardObservationData observation =
      empty_hand_card_observation(fallback_zone_index);
  const CardId *card_id = ecs_get(world, card, CardId);
  observation.card_def_id = card_def_id_or_empty(card_id);
  return observation;
}

static TrainingDiscardCardObservationData get_discard_card_observation(
    ecs_world_t *world, ecs_entity_t card, uint8_t fallback_zone_index) {
  TrainingDiscardCardObservationData observation =
      empty_discard_card_observation(fallback_zone_index);
  const CardId *card_id = ecs_get(world, card, CardId);
  observation.card_def_id = card_def_id_or_empty(card_id);
  return observation;
}

static TrainingIKZCardObservationData get_ikz_card_observation(
    ecs_world_t *world, ecs_entity_t card, uint8_t fallback_zone_index) {
  TrainingIKZCardObservationData observation =
      empty_ikz_card_observation(fallback_zone_index);
  const CardId *card_id = ecs_get(world, card, CardId);
  observation.card_def_id = card_def_id_or_empty(card_id);
  if (observation.card_def_id == -1) {
    return observation;
  }

  const TapState *tap_state = ecs_get(world, card, TapState);
  if (tap_state != NULL) {
    observation.tap_state = *tap_state;
  }

  const ZoneIndex *zone_index = ecs_get(world, card, ZoneIndex);
  if (zone_index != NULL) {
    observation.zone_index = zone_index->index;
  }
  return observation;
}

static uint8_t get_board_observation_array_for_zone(
    ecs_world_t *world, ecs_entity_t zone,
    TrainingBoardCardObservationData *observation_data, size_t max_count,
    bool use_zone_index) {
  for (size_t i = 0; i < max_count; ++i) {
    observation_data[i] = empty_board_card_observation((uint8_t)i);
  }

  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  size_t count = cards.count;
  if (count > max_count) {
    count = max_count;
  }

  if (!use_zone_index) {
    for (size_t i = 0; i < count; ++i) {
      observation_data[i] = get_board_card_observation(
          world, cards.ids[i], (uint8_t)i, false);
    }
    return (uint8_t)cards.count;
  }

  bool slot_has_card[max_count];
  for (size_t i = 0; i < max_count; ++i) {
    slot_has_card[i] = false;
  }

  size_t fallback_slot = 0;
  for (size_t i = 0; i < count; ++i) {
    ecs_entity_t card = cards.ids[i];
    const ZoneIndex *zone_index = ecs_get(world, card, ZoneIndex);
    size_t target_index = max_count;

    if (zone_index != NULL && zone_index->index < max_count &&
        !slot_has_card[zone_index->index]) {
      target_index = zone_index->index;
    } else {
      while (fallback_slot < max_count && slot_has_card[fallback_slot]) {
        fallback_slot++;
      }
      if (fallback_slot < max_count) {
        target_index = fallback_slot;
        fallback_slot++;
      }
    }

    if (target_index >= max_count) {
      continue;
    }

    observation_data[target_index] =
        get_board_card_observation(world, card, (uint8_t)target_index, false);
    slot_has_card[target_index] = true;
  }
  return (uint8_t)cards.count;
}

static uint8_t get_hand_observation_array_for_zone(
    ecs_world_t *world, ecs_entity_t zone,
    TrainingHandCardObservationData *observation_data, size_t max_count) {
  for (size_t i = 0; i < max_count; ++i) {
    observation_data[i] = empty_hand_card_observation((uint8_t)i);
  }
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  size_t count = cards.count;
  if (count > max_count) {
    count = max_count;
  }
  for (size_t i = 0; i < count; ++i) {
    observation_data[i] =
        get_hand_card_observation(world, cards.ids[i], (uint8_t)i);
  }
  return (uint8_t)cards.count;
}

static void get_discard_observation_array_for_zone(
    ecs_world_t *world, ecs_entity_t zone,
    TrainingDiscardCardObservationData *observation_data, size_t max_count) {
  for (size_t i = 0; i < max_count; ++i) {
    observation_data[i] = empty_discard_card_observation((uint8_t)i);
  }
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  size_t count = cards.count;
  if (count > max_count) {
    count = max_count;
  }
  for (size_t i = 0; i < count; ++i) {
    observation_data[i] =
        get_discard_card_observation(world, cards.ids[i], (uint8_t)i);
  }
}

static void get_ikz_observation_array_for_zone(
    ecs_world_t *world, ecs_entity_t zone,
    TrainingIKZCardObservationData *observation_data, size_t max_count) {
  for (size_t i = 0; i < max_count; ++i) {
    observation_data[i] = empty_ikz_card_observation((uint8_t)i);
  }
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  size_t count = cards.count;
  if (count > max_count) {
    count = max_count;
  }
  for (size_t i = 0; i < count; ++i) {
    observation_data[i] =
        get_ikz_card_observation(world, cards.ids[i], (uint8_t)i);
  }
}

static uint8_t get_zone_card_count(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  return (uint8_t)cards.count;
}

static TrainingLeaderObservationData get_leader_observation(
    ecs_world_t *world, ecs_entity_t leader_zone) {
  TrainingLeaderObservationData observation = {0};
  for (size_t i = 0; i < MAX_ATTACHED_WEAPONS; ++i) {
    observation.weapons[i] = empty_weapon_observation();
  }

  ecs_entities_t leader_cards = ecs_get_ordered_children(world, leader_zone);
  ecs_assert(leader_cards.count == 1, ECS_INVALID_PARAMETER,
             "Leader zone must contain exactly 1 card, got %d",
             leader_cards.count);

  ecs_entity_t leader_card = leader_cards.ids[0];
  const CardId *card_id = ecs_get(world, leader_card, CardId);
  observation.card_def_id = card_def_id_or_empty(card_id);

  const TapState *tap_state = ecs_get(world, leader_card, TapState);
  if (tap_state != NULL) {
    observation.tap_state = *tap_state;
  }

  const CurStats *cur_stats = ecs_get(world, leader_card, CurStats);
  if (cur_stats != NULL) {
    observation.cur_stats = *cur_stats;
  }

  observation.weapon_count =
      set_attached_weapon_observations(world, leader_card, observation.weapons);
  observation.has_charge = ecs_has(world, leader_card, Charge);
  observation.has_defender = ecs_has(world, leader_card, Defender);
  observation.has_infiltrate = ecs_has(world, leader_card, Infiltrate);
  return observation;
}

static TrainingGateObservationData get_gate_observation(ecs_world_t *world,
                                                        ecs_entity_t gate_zone) {
  TrainingGateObservationData observation = {0};
  ecs_entities_t gate_cards = ecs_get_ordered_children(world, gate_zone);
  ecs_assert(gate_cards.count == 1, ECS_INVALID_PARAMETER,
             "Gate zone must contain exactly 1 card, got %d", gate_cards.count);

  ecs_entity_t gate_card = gate_cards.ids[0];
  const CardId *card_id = ecs_get(world, gate_card, CardId);
  observation.card_def_id = card_def_id_or_empty(card_id);

  const TapState *tap_state = ecs_get(world, gate_card, TapState);
  if (tap_state != NULL) {
    observation.tap_state = *tap_state;
  }
  return observation;
}

static void get_selection_from_ability_context(
    ecs_world_t *world, const AbilityContext *ctx,
    TrainingBoardCardObservationData *observation_data, uint8_t *out_count) {
  for (size_t i = 0; i < MAX_SELECTION_ZONE_SIZE; ++i) {
    observation_data[i] = empty_board_card_observation((uint8_t)i);
  }

  int selection_count = ctx->selection_count;
  if (selection_count > MAX_SELECTION_ZONE_SIZE) {
    selection_count = MAX_SELECTION_ZONE_SIZE;
  }

  for (int i = 0; i < selection_count; ++i) {
    ecs_entity_t card = ctx->selection_cards[i];
    if (card != 0) {
      observation_data[i] =
          get_board_card_observation(world, card, (uint8_t)i, false);
    }
  }
  *out_count = (uint8_t)selection_count;
}

static bool player_has_ready_ikz_token(ecs_world_t *world,
                                       ecs_entity_t player) {
  const IKZToken *ikz_token = ecs_get(world, player, IKZToken);
  if (ikz_token == NULL || ikz_token->ikz_token == 0) {
    return false;
  }

  const TapState *tap_state = ecs_get(world, ikz_token->ikz_token, TapState);
  if (tap_state == NULL) {
    return false;
  }

  return tap_state->tapped == 0;
}

static uint8_t get_pending_confirmation_count(ecs_world_t *world,
                                              const GameState *state) {
  if (state == NULL || state->active_player_index < 0 ||
      state->active_player_index >= MAX_PLAYERS_PER_MATCH) {
    return 0;
  }

  uint8_t pending_count = 0;
  ecs_entity_t active_player = state->players[state->active_player_index];

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  if (ctx != NULL && ctx->phase == ABILITY_PHASE_CONFIRMATION &&
      ctx->is_optional && ctx->owner == active_player) {
    pending_count++;
  }

  const TriggeredEffectQueue *queue = ecs_singleton_get(world, TriggeredEffectQueue);
  if (queue == NULL || queue->count == 0) {
    return pending_count;
  }

  for (uint8_t i = 0; i < queue->count; ++i) {
    const PendingTriggeredEffect *effect = &queue->effects[i];
    if (effect->owner != active_player) {
      continue;
    }

    const CardId *card_id = ecs_get(world, effect->source_card, CardId);
    if (card_id == NULL) {
      continue;
    }

    const AbilityDef *def = azk_get_ability_def(card_id->id);
    if (def == NULL || !def->has_ability || !def->is_optional) {
      continue;
    }
    pending_count++;
  }
  return pending_count;
}

static TrainingAbilityContextObservationData build_ability_context_observation(
    ecs_world_t *world, const GameState *gs) {
  TrainingAbilityContextObservationData observation = {0};
  observation.phase = ABILITY_PHASE_NONE;
  observation.source_card_def_id = -1;
  observation.active_player_index = gs != NULL ? gs->active_player_index : -1;

  if (gs != NULL) {
    observation.pending_confirmation_count =
        get_pending_confirmation_count(world, gs);
  }

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  if (ctx == NULL || ctx->phase == ABILITY_PHASE_NONE) {
    return observation;
  }

  observation.phase = ctx->phase;
  observation.selection_count = ctx->selection_count <= MAX_SELECTION_ZONE_SIZE
                                    ? ctx->selection_count
                                    : MAX_SELECTION_ZONE_SIZE;
  observation.selection_picked = ctx->selection_picked <= MAX_SELECTION_ZONE_SIZE
                                     ? ctx->selection_picked
                                     : MAX_SELECTION_ZONE_SIZE;
  observation.selection_pick_max =
      ctx->selection_pick_max <= MAX_SELECTION_ZONE_SIZE
          ? ctx->selection_pick_max
          : MAX_SELECTION_ZONE_SIZE;

  const CardId *card_id = ecs_get(world, ctx->source_card, CardId);
  if (card_id == NULL) {
    return observation;
  }

  observation.has_source_card_def_id = true;
  observation.source_card_def_id = (int16_t)card_id->id;

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (def == NULL) {
    return observation;
  }

  observation.cost_target_type = (uint8_t)def->cost_req.type;
  observation.effect_target_type = (uint8_t)def->effect_req.type;
  return observation;
}

static TrainingMyObservationData build_training_my_observation(
    ecs_world_t *world, const GameState *gs, int8_t player_index) {
  init_obs_profile_if_needed();
  const bool profile_enabled = k_obs_profile.enabled;
  const uint64_t my_start_ns = profile_enabled ? obs_now_ns() : 0;
  uint64_t leader_gate_ns = 0;
  uint64_t hand_ns = 0;
  uint64_t alley_ns = 0;
  uint64_t garden_ns = 0;
  uint64_t discard_ns = 0;
  uint64_t selection_ns = 0;
  uint64_t ikz_ns = 0;
  uint64_t counts_ns = 0;

  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");
  ecs_assert(player_index >= 0 && player_index < MAX_PLAYERS_PER_MATCH,
             ECS_INVALID_PARAMETER, "Player index %d out of bounds",
             player_index);

  const PlayerZones *my_zones = &gs->zones[player_index];
  ecs_entity_t my_player = gs->players[player_index];

  TrainingMyObservationData my_observation = {0};
  uint64_t section_start_ns = profile_enabled ? obs_now_ns() : 0;
  my_observation.leader = get_leader_observation(world, my_zones->leader);
  my_observation.gate = get_gate_observation(world, my_zones->gate);
  if (profile_enabled) {
    leader_gate_ns += obs_now_ns() - section_start_ns;
  }

  section_start_ns = profile_enabled ? obs_now_ns() : 0;
  my_observation.hand_count = get_hand_observation_array_for_zone(
      world, my_zones->hand, my_observation.hand, MAX_HAND_SIZE);
  if (profile_enabled) {
    hand_ns += obs_now_ns() - section_start_ns;
  }

  section_start_ns = profile_enabled ? obs_now_ns() : 0;
  get_board_observation_array_for_zone(world, my_zones->alley,
                                       my_observation.alley, ALLEY_SIZE, true);
  if (profile_enabled) {
    alley_ns += obs_now_ns() - section_start_ns;
  }

  section_start_ns = profile_enabled ? obs_now_ns() : 0;
  get_board_observation_array_for_zone(world, my_zones->garden,
                                       my_observation.garden, GARDEN_SIZE, true);
  if (profile_enabled) {
    garden_ns += obs_now_ns() - section_start_ns;
  }

  section_start_ns = profile_enabled ? obs_now_ns() : 0;
  get_discard_observation_array_for_zone(
      world, my_zones->discard, my_observation.discard, MAX_DECK_SIZE);
  if (profile_enabled) {
    discard_ns += obs_now_ns() - section_start_ns;
  }

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  bool use_ctx_selection = ctx != NULL && ctx->selection_count > 0 &&
                           (ctx->phase == ABILITY_PHASE_SELECTION_PICK ||
                            ctx->phase == ABILITY_PHASE_BOTTOM_DECK);
  section_start_ns = profile_enabled ? obs_now_ns() : 0;
  if (use_ctx_selection) {
    get_selection_from_ability_context(world, ctx, my_observation.selection,
                                       &my_observation.selection_count);
  } else {
    my_observation.selection_count = get_board_observation_array_for_zone(
        world, my_zones->selection, my_observation.selection,
        MAX_SELECTION_ZONE_SIZE, false);
  }
  if (profile_enabled) {
    selection_ns += obs_now_ns() - section_start_ns;
  }

  section_start_ns = profile_enabled ? obs_now_ns() : 0;
  get_ikz_observation_array_for_zone(world, my_zones->ikz_area,
                                     my_observation.ikz_area, IKZ_AREA_SIZE);
  if (profile_enabled) {
    ikz_ns += obs_now_ns() - section_start_ns;
  }

  section_start_ns = profile_enabled ? obs_now_ns() : 0;
  my_observation.deck_count = get_zone_card_count(world, my_zones->deck);
  my_observation.ikz_pile_count = get_zone_card_count(world, my_zones->ikz_pile);
  my_observation.has_ikz_token = player_has_ready_ikz_token(world, my_player);
  if (profile_enabled) {
    counts_ns += obs_now_ns() - section_start_ns;

    k_obs_profile.my_calls++;
    k_obs_profile.my_total_ns += obs_now_ns() - my_start_ns;
    k_obs_profile.leader_gate_ns += leader_gate_ns;
    k_obs_profile.hand_ns += hand_ns;
    k_obs_profile.alley_ns += alley_ns;
    k_obs_profile.garden_ns += garden_ns;
    k_obs_profile.discard_ns += discard_ns;
    k_obs_profile.selection_ns += selection_ns;
    k_obs_profile.ikz_ns += ikz_ns;
    k_obs_profile.counts_ns += counts_ns;
    maybe_report_obs_profile();
  }
  return my_observation;
}

static TrainingOpponentObservationData build_training_opponent_from_my(
    const TrainingMyObservationData *my_observation) {
  ecs_assert(my_observation != NULL, ECS_INVALID_PARAMETER,
             "My observation is null");

  TrainingOpponentObservationData opponent_observation = {0};
  opponent_observation.leader = my_observation->leader;
  opponent_observation.gate = my_observation->gate;
  memcpy(opponent_observation.alley, my_observation->alley,
         sizeof(opponent_observation.alley));
  memcpy(opponent_observation.garden, my_observation->garden,
         sizeof(opponent_observation.garden));
  memcpy(opponent_observation.discard, my_observation->discard,
         sizeof(opponent_observation.discard));
  memcpy(opponent_observation.ikz_area, my_observation->ikz_area,
         sizeof(opponent_observation.ikz_area));
  opponent_observation.hand_count = my_observation->hand_count;
  opponent_observation.deck_count = my_observation->deck_count;
  opponent_observation.ikz_pile_count = my_observation->ikz_pile_count;
  opponent_observation.has_ikz_token = my_observation->has_ikz_token;
  return opponent_observation;
}

TrainingObservationData create_training_observation_data(ecs_world_t *world,
                                                         int8_t player_index) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(player_index >= 0 && player_index < MAX_PLAYERS_PER_MATCH,
             ECS_INVALID_PARAMETER, "Player index %d out of bounds",
             player_index);

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");

  int8_t opponent_player_index = (player_index + 1) % MAX_PLAYERS_PER_MATCH;
  const PlayerZones *my_zones = &gs->zones[player_index];
  const PlayerZones *opponent_zones = &gs->zones[opponent_player_index];

  ecs_entity_t my_player = gs->players[player_index];
  ecs_entity_t opponent_player = gs->players[opponent_player_index];

  TrainingMyObservationData my_observation = {0};
  my_observation.leader = get_leader_observation(world, my_zones->leader);
  my_observation.gate = get_gate_observation(world, my_zones->gate);
  my_observation.hand_count = get_hand_observation_array_for_zone(
      world, my_zones->hand, my_observation.hand, MAX_HAND_SIZE);
  get_board_observation_array_for_zone(world, my_zones->alley,
                                       my_observation.alley, ALLEY_SIZE, true);
  get_board_observation_array_for_zone(world, my_zones->garden,
                                       my_observation.garden, GARDEN_SIZE,
                                       true);
  get_discard_observation_array_for_zone(
      world, my_zones->discard, my_observation.discard, MAX_DECK_SIZE);

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  bool use_ctx_selection = ctx != NULL && ctx->selection_count > 0 &&
                           (ctx->phase == ABILITY_PHASE_SELECTION_PICK ||
                            ctx->phase == ABILITY_PHASE_BOTTOM_DECK);
  if (use_ctx_selection) {
    get_selection_from_ability_context(world, ctx, my_observation.selection,
                                       &my_observation.selection_count);
  } else {
    my_observation.selection_count = get_board_observation_array_for_zone(
        world, my_zones->selection, my_observation.selection,
        MAX_SELECTION_ZONE_SIZE, false);
  }

  get_ikz_observation_array_for_zone(world, my_zones->ikz_area,
                                     my_observation.ikz_area, IKZ_AREA_SIZE);

  my_observation.deck_count = get_zone_card_count(world, my_zones->deck);
  my_observation.ikz_pile_count =
      get_zone_card_count(world, my_zones->ikz_pile);
  my_observation.has_ikz_token = player_has_ready_ikz_token(world, my_player);

  TrainingOpponentObservationData opponent_observation = {0};
  opponent_observation.leader =
      get_leader_observation(world, opponent_zones->leader);
  opponent_observation.gate = get_gate_observation(world, opponent_zones->gate);
  get_board_observation_array_for_zone(
      world, opponent_zones->alley, opponent_observation.alley, ALLEY_SIZE,
      true);
  get_board_observation_array_for_zone(
      world, opponent_zones->garden, opponent_observation.garden, GARDEN_SIZE,
      true);
  get_discard_observation_array_for_zone(
      world, opponent_zones->discard, opponent_observation.discard,
      MAX_DECK_SIZE);
  get_ikz_observation_array_for_zone(world, opponent_zones->ikz_area,
                                     opponent_observation.ikz_area,
                                     IKZ_AREA_SIZE);
  opponent_observation.hand_count =
      get_zone_card_count(world, opponent_zones->hand);
  opponent_observation.deck_count =
      get_zone_card_count(world, opponent_zones->deck);
  opponent_observation.ikz_pile_count =
      get_zone_card_count(world, opponent_zones->ikz_pile);
  opponent_observation.has_ikz_token =
      player_has_ready_ikz_token(world, opponent_player);

  TrainingObservationData observation = {0};
  observation.my_observation_data = my_observation;
  observation.opponent_observation_data = opponent_observation;
  observation.phase = gs->phase;
  observation.ability_context = build_ability_context_observation(world, gs);
  reset_legal_actions(&observation.action_mask);
  if (should_expose_training_action_mask(world, gs, player_index)) {
    observation.action_mask = build_training_action_mask(world, gs, player_index);
  }
  return observation;
}

void create_training_observation_data_pair(
    ecs_world_t *world,
    TrainingObservationData out_observations[MAX_PLAYERS_PER_MATCH]) {
  init_obs_profile_if_needed();
  const bool profile_enabled = k_obs_profile.enabled;
  const uint64_t pair_start_ns = profile_enabled ? obs_now_ns() : 0;

  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(out_observations != NULL, ECS_INVALID_PARAMETER,
             "Output observations array is null");

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");

  TrainingMyObservationData my_observations[MAX_PLAYERS_PER_MATCH];
  for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH;
       ++player_index) {
    my_observations[player_index] =
        build_training_my_observation(world, gs, player_index);
  }

  TrainingAbilityContextObservationData ability_observation =
      build_ability_context_observation(world, gs);

  for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH;
       ++player_index) {
    const int8_t opponent_player_index =
        (player_index + 1) % MAX_PLAYERS_PER_MATCH;

    TrainingObservationData observation = {0};
    observation.my_observation_data = my_observations[player_index];
    observation.opponent_observation_data =
        build_training_opponent_from_my(
            &my_observations[opponent_player_index]);
    observation.phase = gs->phase;
    observation.ability_context = ability_observation;

    // Fast path: only the active player can have legal actions.
    // Keep non-active players' masks empty without invoking the full builder.
    reset_legal_actions(&observation.action_mask);
    if (should_expose_training_action_mask(world, gs, player_index)) {
      const uint64_t mask_start_ns = profile_enabled ? obs_now_ns() : 0;
      observation.action_mask =
          build_training_action_mask(world, gs, player_index);
      if (profile_enabled) {
        k_obs_profile.action_mask_calls++;
        k_obs_profile.action_mask_ns += obs_now_ns() - mask_start_ns;
      }
    }

    out_observations[player_index] = observation;
  }

  if (profile_enabled) {
    k_obs_profile.pair_calls++;
    k_obs_profile.pair_total_ns += obs_now_ns() - pair_start_ns;
    maybe_report_obs_profile();
  }
}
