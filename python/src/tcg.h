#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <limits.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include "azuki/engine.h"
#include "generated/card_defs.h"
#include "abilities/ability_system.h"
#include "utils/deck_utils.h"
#include "utils/status_util.h"

#define PBRS_LEADER_WEIGHT 4.0f
#define PBRS_GARDEN_ATTACK_WEIGHT 0.7f
#define PBRS_UNTAPPED_GARDEN_WEIGHT 0.15f
#define PBRS_UNTAPPED_IKZ_WEIGHT 0.15f

#define PBRS_GARDEN_ATTACK_CAP 10.0f
#define PBRS_UNTAPPED_GARDEN_CAP 5.0f
#define PBRS_UNTAPPED_IKZ_CAP 10.0f

#define PBRS_TIME_DECAY_DEFAULT 0.95f
#define TERMINAL_REWARD 5.0f

#define SHAPED_LEADER_DELTA_WEIGHT 1.25f
#define SHAPED_BOARD_DELTA_WEIGHT 0.35f
#define SHAPED_NOOP_PENALTY 0.02f

#define FLOAT_EPSILON 1e-6f

#define PLAYER_1 1.0f
#define PLAYER_2 -1.0f

#define DONE 1
#define NOT_DONE 0

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float p0_episode_return;
    float p1_episode_return;
    float p0_winrate;
    float p1_winrate;
    float p0_start_rate;
    float p1_start_rate;
    float draw_rate;
    float timeout_truncation_rate;
    float auto_tick_truncation_rate;
    float zero_legal_action_truncation_rate;
    float gameover_terminal_rate;
    float winner_terminal_rate;
    float curriculum_episode_cap;
    float reward_shaping_scale;
    float potential_reward_scale;
    float exploration_reward_scale;
    float completed_episodes;
    float p0_noop_selected_rate;
    float p1_noop_selected_rate;
    float p0_attack_selected_rate;
    float p1_attack_selected_rate;
    float p0_attach_weapon_from_hand_selected_rate;
    float p1_attach_weapon_from_hand_selected_rate;
    float p0_play_spell_from_hand_selected_rate;
    float p1_play_spell_from_hand_selected_rate;
    float p0_activate_garden_or_leader_ability_selected_rate;
    float p1_activate_garden_or_leader_ability_selected_rate;
    float p0_activate_alley_ability_selected_rate;
    float p1_activate_alley_ability_selected_rate;
    float p0_gate_portal_selected_rate;
    float p1_gate_portal_selected_rate;
    float p0_play_entity_to_alley_selected_rate;
    float p1_play_entity_to_alley_selected_rate;
    float p0_play_entity_to_garden_selected_rate;
    float p1_play_entity_to_garden_selected_rate;
    float p0_play_selected_rate;
    float p1_play_selected_rate;
    float p0_ability_selected_rate;
    float p1_ability_selected_rate;
    float p0_target_selected_rate;
    float p1_target_selected_rate;
    float p0_avg_leader_health;
    float p1_avg_leader_health;
    float p0_entity_damage_dealt;
    float p1_entity_damage_dealt;
    float p0_entity_damage_taken;
    float p1_entity_damage_taken;
    float p0_generated_ikz_created;
    float p1_generated_ikz_created;
    float p0_generated_ikz_converted;
    float p1_generated_ikz_converted;
    float p0_generated_ikz_conversion_rate;
    float p1_generated_ikz_conversion_rate;
    float p0_temporary_charge_realized;
    float p1_temporary_charge_realized;
    float p0_temporary_attack_damage_realized;
    float p1_temporary_attack_damage_realized;
    float p0_contextual_response_reserve_opportunities;
    float p1_contextual_response_reserve_opportunities;
    float p0_gate_ability_outcomes;
    float p1_gate_ability_outcomes;
    float p0_leader_ability_outcomes;
    float p1_leader_ability_outcomes;
    float n;
} Log;

#define AZK_REWARD_TELEMETRY_GAMMA 0.99f
#define AZK_REWARD_TURN_BUCKET_COUNT 5

typedef enum {
  AZK_REWARD_TERMINAL_OUTCOME = 0,
  AZK_REWARD_TRUNCATION_TIMEOUT,
  AZK_REWARD_TRUNCATION_LEADER_EDGE,
  AZK_REWARD_TRUNCATION_BOARD_EDGE,
  AZK_REWARD_POTENTIAL_LEADER_HEALTH,
  AZK_REWARD_POTENTIAL_GARDEN_ATTACK,
  AZK_REWARD_POTENTIAL_UNTAPPED_GARDEN,
  AZK_REWARD_POTENTIAL_UNTAPPED_IKZ,
  AZK_REWARD_DIRECT_LEADER_EDGE,
  AZK_REWARD_DIRECT_BOARD_EDGE,
  AZK_REWARD_NOOP_PENALTY,
  AZK_REWARD_PORTAL_GP,
  AZK_REWARD_PORTAL_OUTCOME,
  AZK_REWARD_EARLY_TEMPO,
  AZK_REWARD_DAMAGE_MITIGATION,
  AZK_REWARD_TEMPORARY_CHARGE,
  AZK_REWARD_TEMPORARY_ATTACK,
  AZK_REWARD_ENTITY_DAMAGE_EXCHANGE,
  AZK_REWARD_GENERATED_IKZ_CONVERSION,
  AZK_REWARD_RESPONSE_RESERVE,
  AZK_REWARD_GATE_ABILITY_OUTCOME,
  AZK_REWARD_LEADER_ABILITY_OUTCOME,
  AZK_REWARD_COMPONENT_COUNT
} AzkRewardComponent;

typedef struct {
  float raw_sum;
  float raw_abs_sum;
  float raw_discounted_sum;
  float raw_max_abs;
  uint32_t raw_positive_count;
  uint32_t raw_negative_count;
  float scaled_sum;
  float scaled_abs_sum;
  float scaled_discounted_sum;
  float scaled_max_abs;
  uint32_t scaled_positive_count;
  uint32_t scaled_negative_count;
} AzkRewardComponentStats;

typedef struct {
  float portal_gp;
  float portal_outcome;
  float early_tempo;
  float damage_mitigation;
  float temporary_charge;
  float temporary_attack;
  float response_reserve;
} AzkActionRewardComponents;

typedef struct {
  AzkRewardComponentStats
      overall[MAX_PLAYERS_PER_MATCH][AZK_REWARD_COMPONENT_COUNT];
  AzkRewardComponentStats
      by_action[MAX_PLAYERS_PER_MATCH][AZK_ACTION_TYPE_COUNT]
               [AZK_REWARD_COMPONENT_COUNT];
  AzkRewardComponentStats
      by_turn_bucket[MAX_PLAYERS_PER_MATCH][AZK_REWARD_TURN_BUCKET_COUNT]
                    [AZK_REWARD_COMPONENT_COUNT];
  uint32_t action_step_count[MAX_PLAYERS_PER_MATCH][AZK_ACTION_TYPE_COUNT];
  uint32_t turn_bucket_step_count[MAX_PLAYERS_PER_MATCH]
                                 [AZK_REWARD_TURN_BUCKET_COUNT];
  float raw_shaping_return[MAX_PLAYERS_PER_MATCH];
  float scaled_shaping_return[MAX_PLAYERS_PER_MATCH];
  float terminal_return[MAX_PLAYERS_PER_MATCH];
  float raw_reconstruction_max_abs_error;
  float scaled_reconstruction_max_abs_error;
  float discount;
  float shaping_scale_sum;
  float shaping_scale_min;
  float shaping_scale_max;
  uint32_t shaping_step_count;
  int8_t step_action_type;
  bool enabled;
} AzkRewardTelemetry;

static inline const char* debug_card_code_for_entity(AzkEngine* engine,
                                                     ecs_entity_t entity) {
  if (engine == NULL || entity == 0) {
    return "none";
  }

  const CardId* card_id = ecs_get(engine, entity, CardId);
  if (card_id == NULL || card_id->code == NULL) {
    return "no_card_id";
  }

  return card_id->code;
}

static inline void debug_log_hand_zone(AzkEngine* engine,
                                       ecs_entity_t hand_zone,
                                       const char* label) {
  if (engine == NULL || hand_zone == 0 || label == NULL) {
    return;
  }

  ecs_entities_t hand_cards = ecs_get_ordered_children(engine, hand_zone);
  fprintf(stderr, "[ZeroMask] %s hand_count=%d cards=", label, (int)hand_cards.count);
  for (int32_t i = 0; i < hand_cards.count; ++i) {
    fprintf(stderr, "%s%s", i == 0 ? "" : ",",
            debug_card_code_for_entity(engine, hand_cards.ids[i]));
  }
  fprintf(stderr, "\n");
}

static inline int debug_zero_mask_logging_enabled(void) {
  static int enabled = -1;
  if (enabled < 0) {
    const char* value = getenv("AZK_DEBUG_ZERO_MASK");
    enabled = (value != NULL && value[0] != '\0' && strcmp(value, "0") != 0)
                  ? 1
                  : 0;
  }
  return enabled;
}

// Mirrors include/utils/actions_util.h::AZK_USER_ACTION_VALUE_COUNT tuple layout.
typedef struct {
  int32_t type;
  int32_t subaction_1;
  int32_t subaction_2;
  int32_t subaction_3;
} ActionVector;

typedef struct {
  CardInfo *cards;
  size_t card_count;
} TrainingDeckSpec;

// ---- Native deck-building draft support -------------------------------------
// Packed deck-context block appended to the battle observation for
// deck-building training. Field order/types/alignment MUST mirror
// observation.py's _TrainingDeckContextObservationData exactly.
#define AZK_DECKBUILD_OBS_MAX_CANDIDATES AZK_MAX_LEGAL_ACTIONS
// Real per-element candidate pools are ~80 cards; catalog storage bound.
#define AZK_DRAFT_MAX_CANDIDATES 192
#define AZK_DRAFT_MAX_GATES 16
#define AZK_DRAFT_MAX_POPULATION 512
#define AZK_DRAFT_MAX_LEADERS 8
#define AZK_DECKBUILD_BEHAVIOR_COUNT 20

typedef struct {
  int32_t mode;
  int16_t gate_card_def_id;
  int16_t leader_card_def_id;
  int16_t main_card_def_ids[REQUIRED_DECK_SIZE];
  uint8_t main_count;
  int32_t candidate_count;
  int16_t candidate_card_def_ids[AZK_DECKBUILD_OBS_MAX_CANDIDATES];
  uint8_t candidate_copy_counts[AZK_DECKBUILD_OBS_MAX_CANDIDATES];
} AzkTrainingDeckContextData;

typedef struct {
  TrainingObservationData base;
  AzkTrainingDeckContextData deck_context;
} TrainingObservationDataDeckBuild;

// base is 4-aligned and its size is a multiple of 4, so no padding may appear
// between base and deck_context (the Python ctypes mirror relies on this).
_Static_assert(sizeof(TrainingObservationDataDeckBuild) ==
                   sizeof(TrainingObservationData) +
                       sizeof(AzkTrainingDeckContextData),
               "deck-build observation struct must not introduce padding");

// Draft candidate catalog, provided by Python at vec_init from the same
// build_deck_build_catalog() the legacy wrapper uses, so candidate ordering
// is identical by construction. Process-global: one vec per worker process.
typedef struct {
  int16_t gate_def_ids[AZK_DRAFT_MAX_GATES];
  int gate_count;
  int16_t gate_population[AZK_DRAFT_MAX_POPULATION];
  int population_count;
  int16_t leader_flat[AZK_DRAFT_MAX_GATES * AZK_DRAFT_MAX_LEADERS];
  int leader_offsets[AZK_DRAFT_MAX_GATES + 1];
  int16_t main_flat[AZK_DRAFT_MAX_GATES * AZK_DRAFT_MAX_CANDIDATES];
  int main_offsets[AZK_DRAFT_MAX_GATES + 1];
  int16_t ikz_def_id;
  // Same-element partner per gate slot (-1 if none), used by the per-env
  // draft_same_element_matchup_prob knob to oversample sibling-gate matchups
  // (e.g. Surge vs Stormchain).
  int16_t gate_sibling_def_ids[AZK_DRAFT_MAX_GATES];
  bool loaded;
} AzkDraftCatalog;

static AzkDraftCatalog g_draft_catalog = {0};

typedef struct Client Client;
typedef enum EpisodeEndReason {
  EP_END_REASON_GAMEOVER = 0,
  EP_END_REASON_TIMEOUT_TRUNCATION = 1,
  EP_END_REASON_AUTO_TICK_TRUNCATION = 2,
  EP_END_REASON_ZERO_LEGAL_ACTION_TRUNCATION = 3
} EpisodeEndReason;

typedef struct {
  // Puffer I/O
  TrainingObservationData* observations; // MAX_PLAYERS_PER_MATCH
  ActionVector* actions;         // 1 MAX_PLAYERS_PER_MATCH rows of {type, subaction_1..3}
  float* rewards;                // MAX_PLAYERS_PER_MATCH scalars
  float* terminal_rewards;       // MAX_PLAYERS_PER_MATCH scalars
  float* shaped_rewards;         // MAX_PLAYERS_PER_MATCH scalars
  float* reward_scales;          // {potential, exploration}; NULL uses legacy schedule
  unsigned char* terminals;      // MAX_PLAYERS_PER_MATCH scalars {0,1}
  unsigned char* truncations;    // MAX_PLAYERS_PER_MATCH scalars {0,1}
  Log log;
  Client* client;

  // Game State
  AzkEngine* engine;
  uint32_t seed;
  uint32_t starter_rng_state;
  uint32_t deck_rng_state;
  TrainingDeckSpec *deck_pool;
  size_t deck_pool_count;
  int current_deck_indices[MAX_PLAYERS_PER_MATCH];
  // Optional two-seat prebuilt curriculum. Group offsets index the flat
  // prebuilt_deck_indices array; prebuilt_probability is a caller-owned shared
  // float32[1] read once at each training episode reset.
  int *prebuilt_deck_indices;
  size_t *prebuilt_group_offsets;
  size_t prebuilt_group_count;
  float *prebuilt_probability;
  bool episode_prebuilt;
  int episode_prebuilt_deck_indices[MAX_PLAYERS_PER_MATCH];
  int tick;
  AzkActionMaskSet action_masks[MAX_PLAYERS_PER_MATCH];
  float last_phi[MAX_PLAYERS_PER_MATCH];
  float last_scaled_phi[MAX_PLAYERS_PER_MATCH];
  float last_phi_components[MAX_PLAYERS_PER_MATCH][4];
  float last_scaled_phi_components[MAX_PLAYERS_PER_MATCH][4];
  float episode_returns[MAX_PLAYERS_PER_MATCH];
  float episode_terminal_returns[MAX_PLAYERS_PER_MATCH];
  float episode_shaped_returns[MAX_PLAYERS_PER_MATCH];
  uint64_t completed_episodes;
  int current_episode_cap;
  float time_weight;
  float time_decay;
  bool proper_pbrs;
  float pbrs_gamma;
  bool pbrs_terminal_closure;
  float episode_initial_phi[MAX_PLAYERS_PER_MATCH];
  float episode_initial_scaled_phi[MAX_PLAYERS_PER_MATCH];
  AzkRewardSnapshot last_snapshot;
  bool has_last_snapshot;
  uint32_t episode_action_total[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_action_noop[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_action_attack[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_action_play[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_action_ability[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_action_target[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_action_attach_weapon_from_hand[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_action_play_spell_from_hand[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_action_activate_garden_or_leader_ability[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_action_activate_alley_ability[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_action_gate_portal[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_action_play_entity_to_alley[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_action_play_entity_to_garden[MAX_PLAYERS_PER_MATCH];
  // S12 early-tempo per-turn accounting (per player).
  uint16_t tempo_last_turn[MAX_PLAYERS_PER_MATCH];
  uint8_t tempo_turn_count[MAX_PLAYERS_PER_MATCH];
  // S13-DMG: player declared a defender in the open response window; the
  // next combat resolution is an interception credited to them.
  bool pending_interception[MAX_PLAYERS_PER_MATCH];
  AzkAttackRewardContext pending_attack_reward;
  bool has_pending_attack_reward;
  uint16_t response_reserve_last_rewarded_turn[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_temporary_charge_realized[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_temporary_attack_damage_realized[MAX_PLAYERS_PER_MATCH];
  uint32_t episode_contextual_response_reserve_opportunities[MAX_PLAYERS_PER_MATCH];

  // Deck-building draft state (native path). observations rows are
  // TrainingObservationDataDeckBuild when deck_building is set.
  bool deck_building;
  // When enabled, gates are sampled from the unique catalog and compatible
  // leaders are assigned before the first observation. The actor drafts only
  // the 50 main-deck cards.
  bool draft_uniform_assignment;
  // With this probability an episode replaces P1's sampled gate with the
  // same-element sibling of P0's. 0 leaves the RNG stream bit-identical to
  // builds without the knob.
  float draft_same_element_matchup_prob;
  // S3: with this probability one random seat battles under its sibling gate
  // (deck unchanged) — critic contrast data. See draft_start_battle.
  float draft_cross_gate_replay_prob;
  // Privileged-critic support (A-PRIVCRITIC): when set, critic_privileged
  // self/opponent deck lists carry the DRAFTED pick-order compositions during
  // draft and battle instead of being sanitized. Default off (sanitized).
  bool deck_building_privileged_decks;
  bool draft_active;
  int8_t draft_active_player;
  uint32_t draft_rng_state;
  uint32_t episode_world_seed;
  // Promotion evaluator controls. These are inert in training: the binding
  // only sets them for explicitly scheduled native evaluation games.
  bool evaluation_pause_on_done;
  bool evaluation_forced_gates;
  int16_t evaluation_forced_gate[MAX_PLAYERS_PER_MATCH];
  bool evaluation_forced_leaders;
  int16_t evaluation_forced_leader[MAX_PLAYERS_PER_MATCH];
  int8_t evaluation_reference_seat;
  int16_t evaluation_reference_deck_index;
  int16_t draft_gate[MAX_PLAYERS_PER_MATCH];
  int draft_gate_slot[MAX_PLAYERS_PER_MATCH];
  int16_t draft_original_gate[MAX_PLAYERS_PER_MATCH];
  bool draft_gate_swapped[MAX_PLAYERS_PER_MATCH];
  int16_t draft_leader[MAX_PLAYERS_PER_MATCH];
  int16_t draft_main[MAX_PLAYERS_PER_MATCH][REQUIRED_DECK_SIZE];
  uint8_t draft_main_count[MAX_PLAYERS_PER_MATCH];
  uint8_t draft_copies[MAX_PLAYERS_PER_MATCH][AZK_DRAFT_MAX_CANDIDATES];
  // S4 reference seat: with prob AZK_DRAFT_REF_SEAT_PROB one seat skips the
  // draft and plays env->deck_pool[draft_ref_deck_index] (restricted to
  // AZK_DRAFT_REF_DECK_INDICES when set) — external meta anchor + promotion
  // yardstick. -1 = no reference seat this episode.
  int8_t draft_ref_seat;
  int16_t draft_ref_deck_index;

  // Per-episode export record for Python-side deckbuild metrics/snapshots
  // (drained via binding.vec_drain_deck_records right on the terminal step).
  bool deck_record_valid;
  uint32_t deck_record_seed;
  int16_t deck_record_gate[MAX_PLAYERS_PER_MATCH];
  int16_t deck_record_original_gate[MAX_PLAYERS_PER_MATCH];
  bool deck_record_gate_swapped[MAX_PLAYERS_PER_MATCH];
  int16_t deck_record_leader[MAX_PLAYERS_PER_MATCH];
  int16_t deck_record_main[MAX_PLAYERS_PER_MATCH][REQUIRED_DECK_SIZE];
  float deck_record_win[MAX_PLAYERS_PER_MATCH];
  float deck_record_behavior[MAX_PLAYERS_PER_MATCH][AZK_DECKBUILD_BEHAVIOR_COUNT];
  float deck_record_leader_health[MAX_PLAYERS_PER_MATCH];
  float deck_record_episode_length;
  int8_t deck_record_ref_seat;
  int16_t deck_record_ref_deck_index;
  int8_t deck_record_end_reason;
  int8_t deck_record_starting_player;
  bool deck_record_prebuilt;
  int deck_record_prebuilt_deck_indices[MAX_PLAYERS_PER_MATCH];
  AzkRewardTelemetry reward_telemetry;
} CAzukiTCG;

typedef struct {
  int16_t gate_card_def_id;
  int16_t leader_card_def_id;
  int16_t main_card_def_ids[REQUIRED_DECK_SIZE];
  uint8_t main_count;
} AzkDraftSnapshot;

// Copy a completed native draft without exposing pointers into environment
// storage. A snapshot only exists after the final pick has created the battle
// engine, and remains stable for the lifetime of that battle.
static bool c_draft_snapshot(const CAzukiTCG* env, int player_index,
                             AzkDraftSnapshot* out) {
  if (env == NULL || out == NULL || !env->deck_building ||
      env->draft_active || env->engine == NULL ||
      player_index < 0 || player_index >= MAX_PLAYERS_PER_MATCH ||
      env->draft_main_count[player_index] != REQUIRED_DECK_SIZE) {
    return false;
  }
  out->gate_card_def_id = env->draft_gate[player_index];
  out->leader_card_def_id = env->draft_leader[player_index];
  out->main_count = env->draft_main_count[player_index];
  memcpy(out->main_card_def_ids, env->draft_main[player_index],
         sizeof(out->main_card_def_ids));
  return true;
}

static void draft_begin_episode(CAzukiTCG* env);

// Observation rows are TrainingObservationData for battle-only training and
// TrainingObservationDataDeckBuild (larger stride) for deck-building; every
// per-player access must go through these accessors.
static inline TrainingObservationData* obs_base(CAzukiTCG* env, int player_index) {
  if (env->deck_building) {
    TrainingObservationDataDeckBuild* rows =
        (TrainingObservationDataDeckBuild*)env->observations;
    return &rows[player_index].base;
  }
  return &env->observations[player_index];
}

static inline AzkTrainingDeckContextData* obs_deck_context(CAzukiTCG* env,
                                                           int player_index) {
  TrainingObservationDataDeckBuild* rows =
      (TrainingObservationDataDeckBuild*)env->observations;
  return &rows[player_index].deck_context;
}

static inline void debug_log_zero_mask_state(CAzukiTCG* env,
                                             const char* context) {
  if (env == NULL || env->engine == NULL || context == NULL ||
      !debug_zero_mask_logging_enabled()) {
    return;
  }

  const GameState* gs = azk_engine_game_state(env->engine);
  if (gs == NULL) {
    fprintf(stderr, "[ZeroMask] context=%s missing game state\n", context);
    return;
  }

  const int8_t active_player_index = gs->active_player_index;
  if (active_player_index < 0 ||
      active_player_index >= MAX_PLAYERS_PER_MATCH) {
    fprintf(stderr,
            "[ZeroMask] context=%s invalid active_player_index=%d\n",
            context, (int)active_player_index);
    return;
  }

  const TrainingActionMaskObs* obs_mask =
      &obs_base(env, active_player_index)->action_mask;
  const TrainingAbilityContextObservationData* ability_ctx =
      &obs_base(env, active_player_index)->ability_context;
  const bool requires_action = azk_engine_requires_action(env->engine);
  const AbilityPhase ability_phase = azk_engine_get_ability_phase(env->engine);
  const bool has_deck_reorders = azk_has_pending_deck_reorders(env->engine);
  const bool has_passive_buffs = azk_has_pending_passive_buffs(env->engine);
  const bool has_queued_effects = azk_has_queued_triggered_effects(env->engine);

  AzkActionMaskSet fresh_mask = {0};
  const bool built_fresh_mask = azk_build_action_mask_for_player(
      env->engine, gs, active_player_index, &fresh_mask);

  const ecs_entity_t active_player = gs->players[active_player_index];
  const PlayerNumber* player_number =
      ecs_get(env->engine, active_player, PlayerNumber);

  const DeckReorderQueue* deck_queue =
      ecs_singleton_get(env->engine, DeckReorderQueue);
  const PassiveBuffQueue* passive_queue =
      ecs_singleton_get(env->engine, PassiveBuffQueue);
  const TriggeredEffectQueue* trigger_queue =
      ecs_singleton_get(env->engine, TriggeredEffectQueue);

  fprintf(
      stderr,
      "[ZeroMask] context=%s tick=%d phase=%d ability_phase=%d "
      "active_player_index=%d player_entity=%llu player_number=%d "
      "deck_indices=[%d,%d] obs_legal=%u requires_action=%d "
      "fresh_mask_ok=%d fresh_legal=%u deck_reorders=%d(%u) "
      "passive_buffs=%d(%u) queued_effects=%d(%u) "
      "ability_ctx_phase=%d source_card_def_id=%d effect_target_type=%u "
      "cost_target_type=%u selection_count=%u pending_confirmations=%u\n",
      context,
      env->tick,
      (int)gs->phase,
      (int)ability_phase,
      (int)active_player_index,
      (unsigned long long)active_player,
      player_number != NULL ? (int)player_number->player_number : -1,
      env->current_deck_indices[0],
      env->current_deck_indices[1],
      obs_mask->legal_action_count,
      requires_action ? 1 : 0,
      built_fresh_mask ? 1 : 0,
      fresh_mask.legal_action_count,
      has_deck_reorders ? 1 : 0,
      deck_queue != NULL ? deck_queue->count : 0,
      has_passive_buffs ? 1 : 0,
      passive_queue != NULL ? passive_queue->count : 0,
      has_queued_effects ? 1 : 0,
      trigger_queue != NULL ? trigger_queue->count : 0,
      (int)ability_ctx->phase,
      ability_ctx->has_source_card_def_id
          ? (int)ability_ctx->source_card_def_id
          : -1,
      (unsigned)ability_ctx->effect_target_type,
      (unsigned)ability_ctx->cost_target_type,
      (unsigned)ability_ctx->selection_count,
      (unsigned)ability_ctx->pending_confirmation_count);

  if (deck_queue != NULL && deck_queue->count > 0) {
    for (uint8_t i = 0; i < deck_queue->count; ++i) {
      fprintf(stderr,
              "[ZeroMask] deck_queue[%u] deck=%llu card=%s to_top=%d\n",
              (unsigned)i,
              (unsigned long long)deck_queue->entries[i].deck,
              debug_card_code_for_entity(env->engine,
                                         deck_queue->entries[i].card),
              deck_queue->entries[i].to_top ? 1 : 0);
    }
  }

  if (passive_queue != NULL && passive_queue->count > 0) {
    for (uint8_t i = 0; i < passive_queue->count; ++i) {
      fprintf(stderr,
              "[ZeroMask] passive_queue[%u] entity=%s source=%s atk=%d hp=%d "
              "removal=%d\n",
              (unsigned)i,
              debug_card_code_for_entity(env->engine,
                                         passive_queue->buffs[i].entity),
              debug_card_code_for_entity(env->engine,
                                         passive_queue->buffs[i].source),
              (int)passive_queue->buffs[i].atk_modifier,
              (int)passive_queue->buffs[i].hp_modifier,
              passive_queue->buffs[i].is_removal ? 1 : 0);
    }
  }

  if (trigger_queue != NULL && trigger_queue->count > 0) {
    for (uint8_t i = 0; i < trigger_queue->count; ++i) {
      fprintf(stderr,
              "[ZeroMask] trigger_queue[%u] source=%s owner=%llu timing_tag=%u "
              "action_index=%d\n",
              (unsigned)i,
              debug_card_code_for_entity(env->engine,
                                         trigger_queue->effects[i].source_card),
              (unsigned long long)trigger_queue->effects[i].owner,
              (unsigned)trigger_queue->effects[i].timing_tag,
              (int)trigger_queue->effects[i].action_index);
    }
  }

  debug_log_hand_zone(env->engine, gs->zones[0].hand, "player0");
  debug_log_hand_zone(env->engine, gs->zones[1].hand, "player1");
  debug_log_hand_zone(env->engine, gs->zones[active_player_index].selection,
                      "active_selection");
}

typedef struct {
  int initialized;
  int enabled;
  uint64_t report_every;
  uint64_t step_calls;
  uint64_t total_step_ns;
  uint64_t total_tick_ns;
  uint64_t total_refresh_ns;
  uint64_t total_auto_ticks;
} EnvProfileState;

static EnvProfileState g_env_profile = {0};

static inline uint64_t env_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline int env_flag_enabled(const char *name) {
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0') {
    return 0;
  }
  if (value[0] == '0' && value[1] == '\0') {
    return 0;
  }
  return 1;
}

static inline uint32_t advance_episode_seed(uint32_t seed) {
  uint32_t x = seed;
  if (x == 0u) {
    x = 0x9E3779B9u;
  }
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

static inline uint32_t starter_seed_from_env_seed(uint32_t seed) {
  return seed ^ 0xA511E9B3u;
}

static inline uint32_t deck_seed_from_env_seed(uint32_t seed) {
  return seed ^ 0x6D2B79F5u;
}

static inline int8_t next_starting_player(CAzukiTCG* env) {
  env->starter_rng_state = advance_episode_seed(env->starter_rng_state);
  return (int8_t)(env->starter_rng_state % (uint32_t)MAX_PLAYERS_PER_MATCH);
}

static inline void reset_current_deck_indices(CAzukiTCG* env) {
  for (int player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    env->current_deck_indices[player_index] = -1;
  }
}

static inline size_t next_training_deck_index(CAzukiTCG* env) {
  env->deck_rng_state = advance_episode_seed(env->deck_rng_state);
  return (size_t)(env->deck_rng_state % (uint32_t)env->deck_pool_count);
}

static void free_training_deck_pool(CAzukiTCG* env) {
  if (env == NULL) {
    return;
  }
  if (env->deck_pool != NULL) {
    for (size_t deck_index = 0; deck_index < env->deck_pool_count; ++deck_index) {
      free(env->deck_pool[deck_index].cards);
      env->deck_pool[deck_index].cards = NULL;
      env->deck_pool[deck_index].card_count = 0;
    }
    free(env->deck_pool);
  }
  free(env->prebuilt_deck_indices);
  free(env->prebuilt_group_offsets);
  env->deck_pool = NULL;
  env->deck_pool_count = 0;
  env->prebuilt_deck_indices = NULL;
  env->prebuilt_group_offsets = NULL;
  env->prebuilt_probability = NULL;
  env->episode_prebuilt = false;
  reset_current_deck_indices(env);
}

static AzkEngine *create_env_engine(CAzukiTCG* env, int8_t starting_player) {
  if (env->deck_pool_count == 0 || env->deck_pool == NULL) {
    reset_current_deck_indices(env);
    return azk_engine_create_with_starting_player(env->seed, starting_player);
  }

  const size_t player0_deck_index = next_training_deck_index(env);
  const size_t player1_deck_index = next_training_deck_index(env);
  env->current_deck_indices[0] = (int)player0_deck_index;
  env->current_deck_indices[1] = (int)player1_deck_index;

  const TrainingDeckSpec *player0_deck = &env->deck_pool[player0_deck_index];
  const TrainingDeckSpec *player1_deck = &env->deck_pool[player1_deck_index];
  return azk_engine_create_with_decks_and_starting_player(
      env->seed, starting_player, player0_deck->cards, player0_deck->card_count,
      player1_deck->cards, player1_deck->card_count);
}

static void init_env_profile_if_needed(void) {
  if (g_env_profile.initialized) {
    return;
  }
  g_env_profile.initialized = 1;
  g_env_profile.enabled = env_flag_enabled("AZK_ENV_PROFILE");
  g_env_profile.report_every = 20000;
  const char *report_every = getenv("AZK_ENV_PROFILE_EVERY");
  if (report_every != NULL && report_every[0] != '\0') {
    char *end_ptr = NULL;
    unsigned long long parsed = strtoull(report_every, &end_ptr, 10);
    if (end_ptr != report_every && parsed > 0ull) {
      g_env_profile.report_every = (uint64_t)parsed;
    }
  }
}

static void maybe_report_env_profile(void) {
  if (!g_env_profile.enabled || g_env_profile.step_calls == 0 ||
      (g_env_profile.step_calls % g_env_profile.report_every) != 0) {
    return;
  }

  const double avg_step_us =
      g_env_profile.total_step_ns / (double)g_env_profile.step_calls / 1000.0;
  const double avg_tick_us =
      g_env_profile.total_tick_ns / (double)g_env_profile.step_calls / 1000.0;
  const double avg_refresh_us =
      g_env_profile.total_refresh_ns / (double)g_env_profile.step_calls / 1000.0;
  const double avg_auto_ticks =
      g_env_profile.total_auto_ticks / (double)g_env_profile.step_calls;
  const double tick_share =
      g_env_profile.total_step_ns == 0
          ? 0.0
          : (double)g_env_profile.total_tick_ns /
                (double)g_env_profile.total_step_ns;
  const double refresh_share =
      g_env_profile.total_step_ns == 0
          ? 0.0
          : (double)g_env_profile.total_refresh_ns /
                (double)g_env_profile.total_step_ns;

  fprintf(stderr,
          "[EnvProfile] steps=%" PRIu64
          " avg_step_us=%.2f avg_tick_us=%.2f avg_refresh_us=%.2f"
          " avg_auto_ticks=%.2f tick_share=%.3f refresh_share=%.3f\n",
          g_env_profile.step_calls, avg_step_us, avg_tick_us, avg_refresh_us,
          avg_auto_ticks, tick_share, refresh_share);
}

static inline float clampf(float value, float min_value, float max_value) {
  if (value < min_value) {
    return min_value;
  }
  if (value > max_value) {
    return max_value;
  }
  return value;
}

static inline float safe_delta(float numerator, float denominator) {
  if (fabsf(denominator) <= FLOAT_EPSILON) {
    return 0.0f;
  }
  return numerator / denominator;
}

typedef struct RewardTuningConfig {
  int initialized;
  float leader_delta_weight;
  float board_delta_weight;
  float noop_penalty;
  float potential_leader_health_weight;
  float potential_garden_attack_weight;
  float potential_untapped_garden_weight;
  float untapped_ikz_weight;
  // Annealed portal exposure bonus: on GATE_PORTAL, weight * min(GP,4)/4 of
  // the portaled entity joins base_shaped_reward (rides shaping scale +
  // zero-sum symmetry). 0-GP portals earn nothing. Default off.
  float portal_gp_bonus;
  // S2: outcome-graded variant — same GP scaling but paid ONLY if the gate
  // ability observably resolved (weapon attached / IKZ untapped / card
  // returned / damage dealt / buff-defender-charge landed). Whiffs pay 0.
  // When set (>0) it supersedes portal_gp_bonus.
  float portal_outcome_bonus;
  // S12 early-tempo bonus: flat reward per qualifying DEVELOPMENT action
  // (plays, portal, ability activations, affirmative confirms, attacks)
  // during each player's first N turns, capped per turn (0 = unbounded).
  // Rides the shaping anneal + zero-sum channel. Default off.
  float early_tempo_bonus;
  int early_tempo_cap;
  int early_tempo_turns;
  // Remove generic tempo credit from portal and ability/confirmation actions
  // whose closer outcome signals already receive reward. Default off.
  int early_tempo_dedup_portal_abilities;
  // S13-DMG damage-mitigation bonus: on an intercepted combat, the
  // intercepting player earns w * min(soak, cap)/cap where soak is the
  // realized damage_to_defender (ATK debuffs flow through automatically).
  // Opponent-gated (cannot be farmed unilaterally); rides anneal, zero-sum.
  float dmg_mitigation_bonus;
  int dmg_mitigation_cap;
  // Effective non-leader damage differential. Default off.
  float entity_damage_exchange_per_hp;
  int entity_damage_exchange_step_cap;
  // Credit an effect-recovered/created IKZ only when it is later tapped.
  float generated_ikz_conversion_bonus;
  int generated_ikz_conversion_step_cap;
  // Credit temporary Charge and only the incremental realized EOT ATK damage.
  float temporary_charge_realization_bonus;
  float temporary_attack_realization_per_damage;
  int temporary_attack_realization_damage_cap;
  // Once per opposing turn, credit an actually affordable paid response.
  float contextual_response_reserve_bonus;
  // One fixed bonus per completed nonempty gate/leader effect. Default off.
  float ability_outcome_bonus;
} RewardTuningConfig;

// Pre-action summary for grading a portal's realized effect (S2).
typedef struct {
  int weapon_total;
  int untapped_ikz;
  int hand_count;
  int opp_leader_hp;
  int garden_atk_sum;
  int defender_count;
  int charge_count;
  int portaled_atk;
} PortalOutcomeSnapshot;

static void portal_outcome_capture(const TrainingObservationData* base,
                                   int portaled_atk,
                                   PortalOutcomeSnapshot* out) {
  const TrainingMyObservationData* my = &base->my_observation_data;
  int weapon_total = my->leader.weapon_count;
  int atk_sum = 0;
  int defenders = 0;
  int charges = 0;
  for (int i = 0; i < GARDEN_SIZE; ++i) {
    const TrainingBoardCardObservationData* c = &my->garden[i];
    if (c->card_def_id < 0) {
      continue;
    }
    weapon_total += c->weapon_count;
    atk_sum += c->cur_stats.cur_atk;
    defenders += c->has_defender ? 1 : 0;
    charges += c->has_charge ? 1 : 0;
  }
  int untapped_ikz = 0;
  for (int i = 0; i < IKZ_AREA_SIZE; ++i) {
    if (my->ikz_area[i].card_def_id >= 0 && !my->ikz_area[i].tap_state.tapped) {
      untapped_ikz++;
    }
  }
  out->weapon_total = weapon_total;
  out->untapped_ikz = untapped_ikz;
  out->hand_count = my->hand_count;
  out->opp_leader_hp = base->opponent_observation_data.leader.cur_stats.cur_hp;
  out->garden_atk_sum = atk_sum;
  out->defender_count = defenders;
  out->charge_count = charges;
  out->portaled_atk = portaled_atk;
}

static bool portal_outcome_resolved(const PortalOutcomeSnapshot* pre,
                                    const PortalOutcomeSnapshot* post) {
  if (post->weapon_total > pre->weapon_total) return true;      // Surge/Stormchain
  if (post->untapped_ikz > pre->untapped_ikz) return true;      // Hydromancy
  if (post->hand_count > pre->hand_count) return true;          // EchoedWaves
  if (post->opp_leader_hp < pre->opp_leader_hp) return true;    // Devotion/dmg
  if (post->defender_count > pre->defender_count) return true;  // Stonehaven
  if (post->charge_count > pre->charge_count) return true;      // Rushfire
  // Ragefire ATK buff / Rushfire extra body: garden ATK grew beyond the
  // portaled entity's own contribution.
  if (post->garden_atk_sum - pre->garden_atk_sum > pre->portaled_atk) return true;
  return false;
}

static RewardTuningConfig g_reward_tuning = {0};

typedef struct RewardShapingAnnealConfig {
  int initialized;
  int enabled;
  float initial_scale;
  float final_scale;
  int warmup_episodes;
  int ramp_episodes;
} RewardShapingAnnealConfig;

static RewardShapingAnnealConfig g_reward_shaping_anneal = {0};
static int parse_nonnegative_env_int(const char* name, int default_value);

static float parse_nonnegative_env_float(const char *name, float default_value) {
  const char *raw = getenv(name);
  if (raw == NULL || raw[0] == '\0') {
    return default_value;
  }

  char *endptr = NULL;
  float parsed = strtof(raw, &endptr);
  if (endptr == raw || *endptr != '\0' || !isfinite(parsed) || parsed < 0.0f) {
    fprintf(stderr, "Invalid %s='%s'; using default %.3f\n", name, raw, default_value);
    return default_value;
  }
  return parsed;
}

static void init_reward_tuning_if_needed(void) {
  if (g_reward_tuning.initialized) {
    return;
  }
  g_reward_tuning.initialized = 1;
  g_reward_tuning.leader_delta_weight =
      parse_nonnegative_env_float("AZK_REWARD_LEADER_DELTA_WEIGHT", SHAPED_LEADER_DELTA_WEIGHT);
  g_reward_tuning.board_delta_weight =
      parse_nonnegative_env_float("AZK_REWARD_BOARD_DELTA_WEIGHT", SHAPED_BOARD_DELTA_WEIGHT);
  g_reward_tuning.noop_penalty =
      parse_nonnegative_env_float("AZK_REWARD_NOOP_PENALTY", SHAPED_NOOP_PENALTY);
  g_reward_tuning.potential_leader_health_weight =
      parse_nonnegative_env_float(
          "AZK_REWARD_LEADER_HEALTH_WEIGHT", PBRS_LEADER_WEIGHT);
  g_reward_tuning.potential_garden_attack_weight =
      parse_nonnegative_env_float(
          "AZK_REWARD_GARDEN_ATTACK_WEIGHT", PBRS_GARDEN_ATTACK_WEIGHT);
  g_reward_tuning.potential_untapped_garden_weight =
      parse_nonnegative_env_float(
          "AZK_REWARD_UNTAPPED_GARDEN_WEIGHT", PBRS_UNTAPPED_GARDEN_WEIGHT);
  g_reward_tuning.untapped_ikz_weight =
      parse_nonnegative_env_float(
          "AZK_REWARD_UNTAPPED_IKZ_WEIGHT", PBRS_UNTAPPED_IKZ_WEIGHT);
  g_reward_tuning.portal_gp_bonus =
      parse_nonnegative_env_float("AZK_PORTAL_GP_BONUS", 0.0f);
  g_reward_tuning.portal_outcome_bonus =
      parse_nonnegative_env_float("AZK_PORTAL_OUTCOME_BONUS", 0.0f);
  g_reward_tuning.early_tempo_bonus =
      parse_nonnegative_env_float("AZK_EARLY_TEMPO_BONUS", 0.0f);
  g_reward_tuning.early_tempo_cap =
      (int)parse_nonnegative_env_float("AZK_EARLY_TEMPO_CAP", 4.0f);
  g_reward_tuning.early_tempo_turns =
      (int)parse_nonnegative_env_float("AZK_EARLY_TEMPO_TURNS", 2.0f);
  g_reward_tuning.early_tempo_dedup_portal_abilities =
      env_flag_enabled("AZK_EARLY_TEMPO_DEDUP_PORTAL_ABILITIES") ? 1 : 0;
  g_reward_tuning.dmg_mitigation_bonus =
      parse_nonnegative_env_float("AZK_DMG_MITIGATION_BONUS", 0.0f);
  g_reward_tuning.dmg_mitigation_cap =
      (int)parse_nonnegative_env_float("AZK_DMG_MITIGATION_CAP", 10.0f);
  g_reward_tuning.entity_damage_exchange_per_hp =
      parse_nonnegative_env_float("AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP", 0.0f);
  g_reward_tuning.entity_damage_exchange_step_cap =
      (int)parse_nonnegative_env_float("AZK_ENTITY_DAMAGE_EXCHANGE_STEP_CAP", 6.0f);
  g_reward_tuning.generated_ikz_conversion_bonus =
      parse_nonnegative_env_float("AZK_GENERATED_IKZ_CONVERSION_BONUS", 0.0f);
  g_reward_tuning.generated_ikz_conversion_step_cap =
      (int)parse_nonnegative_env_float("AZK_GENERATED_IKZ_CONVERSION_STEP_CAP", 4.0f);
  g_reward_tuning.temporary_charge_realization_bonus =
      parse_nonnegative_env_float("AZK_TEMP_CHARGE_REALIZATION_BONUS", 0.0f);
  g_reward_tuning.temporary_attack_realization_per_damage =
      parse_nonnegative_env_float("AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE", 0.0f);
  g_reward_tuning.temporary_attack_realization_damage_cap =
      (int)parse_nonnegative_env_float("AZK_TEMP_ATTACK_REALIZATION_DAMAGE_CAP", 4.0f);
  g_reward_tuning.contextual_response_reserve_bonus =
      parse_nonnegative_env_float("AZK_CONTEXTUAL_RESPONSE_RESERVE_BONUS", 0.0f);
  g_reward_tuning.ability_outcome_bonus =
      parse_nonnegative_env_float("AZK_ABILITY_OUTCOME_BONUS", 0.0f);
}

// S12: development actions that count toward the early-tempo bonus. Declining
// an optional ability is ACT_NOOP in the confirmation phase, so
// ACT_CONFIRM_ABILITY is always an affirmative use. Target/cost/selection
// sub-actions and mulligans never count.
static bool early_tempo_qualifying_action(ActionType type) {
  switch (type) {
    case ACT_PLAY_ENTITY_TO_GARDEN:
    case ACT_PLAY_ENTITY_TO_ALLEY:
    case ACT_PLAY_SPELL_FROM_HAND:
    case ACT_ATTACH_WEAPON_FROM_HAND:
    case ACT_GATE_PORTAL:
    case ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY:
    case ACT_ACTIVATE_ALLEY_ABILITY:
    case ACT_CONFIRM_ABILITY:
    case ACT_ATTACK:
      return true;
    default:
      return false;
  }
}

static bool early_tempo_dedup_excluded_action(ActionType type) {
  switch (type) {
    case ACT_GATE_PORTAL:
    case ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY:
    case ACT_ACTIVATE_ALLEY_ABILITY:
    case ACT_CONFIRM_ABILITY:
      return true;
    default:
      return false;
  }
}

static float parse_unit_env_float(const char *name, float default_value) {
  const char *raw = getenv(name);
  if (raw == NULL || raw[0] == '\0') {
    return default_value;
  }

  char *endptr = NULL;
  float parsed = strtof(raw, &endptr);
  if (endptr == raw || *endptr != '\0' || !isfinite(parsed) ||
      parsed < 0.0f || parsed > 1.0f) {
    fprintf(stderr, "Invalid %s='%s'; using default %.3f\n", name, raw, default_value);
    return default_value;
  }
  return parsed;
}

static void init_reward_shaping_anneal_if_needed(void) {
  if (g_reward_shaping_anneal.initialized) {
    return;
  }

  g_reward_shaping_anneal.initialized = 1;
  g_reward_shaping_anneal.enabled = env_flag_enabled("AZK_REWARD_SHAPING_ANNEAL");
  g_reward_shaping_anneal.initial_scale =
      parse_unit_env_float("AZK_REWARD_SHAPING_ANNEAL_INITIAL", 1.0f);
  g_reward_shaping_anneal.final_scale =
      parse_unit_env_float("AZK_REWARD_SHAPING_ANNEAL_FINAL", 0.05f);
  g_reward_shaping_anneal.warmup_episodes =
      parse_nonnegative_env_int("AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES", 2000);
  g_reward_shaping_anneal.ramp_episodes =
      parse_nonnegative_env_int("AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES", 30000);

  if (g_reward_shaping_anneal.final_scale > g_reward_shaping_anneal.initial_scale) {
    fprintf(stderr,
            "AZK_REWARD_SHAPING_ANNEAL_FINAL (%.3f) > INITIAL (%.3f); clamping final to initial\n",
            g_reward_shaping_anneal.final_scale,
            g_reward_shaping_anneal.initial_scale);
    g_reward_shaping_anneal.final_scale = g_reward_shaping_anneal.initial_scale;
  }
}

static float current_reward_shaping_scale(CAzukiTCG* env) {
  init_reward_shaping_anneal_if_needed();
  if (!g_reward_shaping_anneal.enabled) {
    return 1.0f;
  }

  const uint64_t completed = env->completed_episodes;
  const uint64_t warmup = (uint64_t)g_reward_shaping_anneal.warmup_episodes;
  const uint64_t ramp = (uint64_t)g_reward_shaping_anneal.ramp_episodes;
  const float initial_scale = g_reward_shaping_anneal.initial_scale;
  const float final_scale = g_reward_shaping_anneal.final_scale;

  if (completed < warmup) {
    return initial_scale;
  }
  if (ramp == 0) {
    return final_scale;
  }

  const uint64_t elapsed = completed - warmup;
  if (elapsed >= ramp) {
    return final_scale;
  }

  const float fraction = (float)((double)elapsed / (double)ramp);
  return initial_scale + (final_scale - initial_scale) * fraction;
}

static void current_reward_component_scales(
    CAzukiTCG* env, float* potential_scale, float* exploration_scale) {
  if (env->reward_scales != NULL) {
    *potential_scale = clampf(env->reward_scales[0], 0.0f, 1.0f);
    *exploration_scale = clampf(env->reward_scales[1], 0.0f, 1.0f);
    return;
  }
  const float legacy_scale = current_reward_shaping_scale(env);
  *potential_scale = legacy_scale;
  *exploration_scale = legacy_scale;
}


static inline float leader_health_transform(float normalized_hp) {
  const float x = clampf(normalized_hp, 0.0f, 1.0f);
  const float one_minus_x = 1.0f - x;
  const float one_minus_x_sq = one_minus_x * one_minus_x;
  const float one_minus_x_pow4 = one_minus_x_sq * one_minus_x_sq;
  return 0.5f * (x + 1.0f - one_minus_x_pow4);
}

// TODO: Amplify rewards for specific actions in ratio to number of turns elapsed (encourages aggressive play)
static float compute_phi_for_player(const AzkRewardSnapshot* snapshot, int8_t player_index) {
  const int8_t opponent_index = (player_index + 1) % MAX_PLAYERS_PER_MATCH;
  const float leader_term = g_reward_tuning.potential_leader_health_weight * (
    leader_health_transform(snapshot->leader_health_ratio[player_index]) -
    leader_health_transform(snapshot->leader_health_ratio[opponent_index])
  );
  const float attack_term = g_reward_tuning.potential_garden_attack_weight * safe_delta(
    snapshot->garden_attack_sum[player_index] - snapshot->garden_attack_sum[opponent_index],
    PBRS_GARDEN_ATTACK_CAP
  );
  const float untapped_garden_term =
      g_reward_tuning.potential_untapped_garden_weight * safe_delta(
    snapshot->untapped_garden_count[player_index] - snapshot->untapped_garden_count[opponent_index],
    PBRS_UNTAPPED_GARDEN_CAP
  );
  const float untapped_ikz_term = g_reward_tuning.untapped_ikz_weight * safe_delta(
    snapshot->untapped_ikz_count[player_index] - snapshot->untapped_ikz_count[opponent_index],
    PBRS_UNTAPPED_IKZ_CAP
  );

  const float phi_input = leader_term + attack_term + untapped_garden_term + untapped_ikz_term;
  return tanhf(phi_input);
}

static bool compute_phi_values(CAzukiTCG* env, float out_phi[MAX_PLAYERS_PER_MATCH]) {
  AzkRewardSnapshot snapshot;
  if (!azk_engine_reward_snapshot(env->engine, &snapshot)) {
    return false;
  }

  for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    out_phi[player_index] = compute_phi_for_player(&snapshot, player_index);
  }

  return true;
}
static void compute_phi_component_values(
    const AzkRewardSnapshot* snapshot, int8_t player_index, float out[4]) {
  const int8_t opponent_index = (player_index + 1) % MAX_PLAYERS_PER_MATCH;
  const float linear[4] = {
      g_reward_tuning.potential_leader_health_weight *
          (leader_health_transform(snapshot->leader_health_ratio[player_index]) -
           leader_health_transform(snapshot->leader_health_ratio[opponent_index])),
      g_reward_tuning.potential_garden_attack_weight *
          safe_delta(snapshot->garden_attack_sum[player_index] -
                         snapshot->garden_attack_sum[opponent_index],
                     PBRS_GARDEN_ATTACK_CAP),
      g_reward_tuning.potential_untapped_garden_weight *
          safe_delta(snapshot->untapped_garden_count[player_index] -
                         snapshot->untapped_garden_count[opponent_index],
                     PBRS_UNTAPPED_GARDEN_CAP),
      g_reward_tuning.untapped_ikz_weight *
          safe_delta(snapshot->untapped_ikz_count[player_index] -
                         snapshot->untapped_ikz_count[opponent_index],
                     PBRS_UNTAPPED_IKZ_CAP),
  };
  // Allocate tanh(sum(linear)) symmetrically and exactly across its linear
  // inputs. The final residual removes floating-point summation drift.
  const float input = linear[0] + linear[1] + linear[2] + linear[3];
  const float phi = tanhf(input);
  const float factor =
      fabsf(input) > FLOAT_EPSILON ? phi / input : 1.0f;
  float allocated = 0.0f;
  for (int index = 0; index < 4; ++index) {
    out[index] = linear[index] * factor;
    allocated += out[index];
  }
  out[3] += phi - allocated;
}

static inline int reward_turn_bucket(uint16_t turn_number) {
  if (turn_number <= 2) {
    return 0;
  }
  if (turn_number <= 4) {
    return 1;
  }
  if (turn_number <= 8) {
    return 2;
  }
  if (turn_number <= 16) {
    return 3;
  }
  return 4;
}

static inline void set_zero_sum_reward_component(
    float components[MAX_PLAYERS_PER_MATCH][AZK_REWARD_COMPONENT_COUNT],
    int acting_player, int opponent, AzkRewardComponent component,
    float value) {
  components[acting_player][component] = value;
  components[opponent][component] = -value;
}

static void reward_component_stats_add(
    AzkRewardComponentStats* stats, float raw, float scaled, float discount) {
  stats->raw_sum += raw;
  stats->raw_abs_sum += fabsf(raw);
  stats->raw_discounted_sum += discount * raw;
  stats->raw_max_abs = fmaxf(stats->raw_max_abs, fabsf(raw));
  stats->raw_positive_count += raw > 0.0f ? 1u : 0u;
  stats->raw_negative_count += raw < 0.0f ? 1u : 0u;
  stats->scaled_sum += scaled;
  stats->scaled_abs_sum += fabsf(scaled);
  stats->scaled_discounted_sum += discount * scaled;
  stats->scaled_max_abs = fmaxf(stats->scaled_max_abs, fabsf(scaled));
  stats->scaled_positive_count += scaled > 0.0f ? 1u : 0u;
  stats->scaled_negative_count += scaled < 0.0f ? 1u : 0u;
}

static void record_reward_telemetry_step(
    CAzukiTCG* env,
    const float raw_components[MAX_PLAYERS_PER_MATCH]
                              [AZK_REWARD_COMPONENT_COUNT],
    const float scaled_components[MAX_PLAYERS_PER_MATCH]
                                 [AZK_REWARD_COMPONENT_COUNT],
    float potential_scale, bool shaping_step,
    const float expected_raw[MAX_PLAYERS_PER_MATCH],
    const float expected_scaled[MAX_PLAYERS_PER_MATCH],
    bool finalize_step) {
  AzkRewardTelemetry* telemetry = &env->reward_telemetry;
  if (!telemetry->enabled) {
    return;
  }

  const GameState* game_state =
      env->engine != NULL ? azk_engine_game_state(env->engine) : NULL;
  const int turn_bucket =
      reward_turn_bucket(game_state != NULL ? game_state->turn_number : 0);
  const int action_type = telemetry->step_action_type;
  float raw_reconstructed[MAX_PLAYERS_PER_MATCH] = {0.0f};
  float scaled_reconstructed[MAX_PLAYERS_PER_MATCH] = {0.0f};

  for (int player = 0; player < MAX_PLAYERS_PER_MATCH; ++player) {
    for (int component = 0; component < AZK_REWARD_COMPONENT_COUNT;
         ++component) {
      const float raw = raw_components[player][component];
      const float scaled = scaled_components[player][component];
      raw_reconstructed[player] += raw;
      scaled_reconstructed[player] += scaled;
      if (raw == 0.0f && scaled == 0.0f) {
        continue;
      }
      reward_component_stats_add(
          &telemetry->overall[player][component], raw, scaled,
          telemetry->discount);
      if (action_type >= 0 && action_type < AZK_ACTION_TYPE_COUNT) {
        reward_component_stats_add(
            &telemetry->by_action[player][action_type][component], raw, scaled,
            telemetry->discount);
      }
      reward_component_stats_add(
          &telemetry->by_turn_bucket[player][turn_bucket][component], raw,
          scaled, telemetry->discount);
    }
    if (finalize_step) {
      if (action_type >= 0 && action_type < AZK_ACTION_TYPE_COUNT) {
        telemetry->action_step_count[player][action_type]++;
      }
      telemetry->turn_bucket_step_count[player][turn_bucket]++;
    }
    if (shaping_step) {
      telemetry->raw_shaping_return[player] += expected_raw[player];
      telemetry->scaled_shaping_return[player] += expected_scaled[player];
    } else {
      telemetry->terminal_return[player] += expected_scaled[player];
    }
    telemetry->raw_reconstruction_max_abs_error =
        fmaxf(telemetry->raw_reconstruction_max_abs_error,
              fabsf(raw_reconstructed[player] - expected_raw[player]));
    telemetry->scaled_reconstruction_max_abs_error =
        fmaxf(telemetry->scaled_reconstruction_max_abs_error,
              fabsf(scaled_reconstructed[player] - expected_scaled[player]));
  }

  if (shaping_step) {
    telemetry->shaping_scale_sum += potential_scale;
    telemetry->shaping_scale_min =
        fminf(telemetry->shaping_scale_min, potential_scale);
    telemetry->shaping_scale_max =
        fmaxf(telemetry->shaping_scale_max, potential_scale);
    telemetry->shaping_step_count++;
  }
  if (finalize_step) {
    telemetry->discount *=
        env->proper_pbrs ? env->pbrs_gamma : AZK_REWARD_TELEMETRY_GAMMA;
    telemetry->step_action_type = -1;
  }
}

static void reset_reward_tracking(CAzukiTCG* env) {
  init_reward_tuning_if_needed();
  init_reward_shaping_anneal_if_needed();
  const bool reward_telemetry_enabled = env->reward_telemetry.enabled;
  memset(&env->reward_telemetry, 0, sizeof(env->reward_telemetry));
  env->reward_telemetry.enabled = reward_telemetry_enabled;
  env->reward_telemetry.discount = 1.0f;
  env->reward_telemetry.shaping_scale_min = INFINITY;
  env->reward_telemetry.step_action_type = -1;
  memset(env->last_phi_components, 0, sizeof(env->last_phi_components));
  memset(env->last_scaled_phi_components, 0,
         sizeof(env->last_scaled_phi_components));
  env->time_weight = 1.0f;
  env->time_decay = PBRS_TIME_DECAY_DEFAULT;
  for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    env->episode_returns[player_index] = 0.0f;
    env->episode_terminal_returns[player_index] = 0.0f;
    env->episode_shaped_returns[player_index] = 0.0f;
    env->episode_action_total[player_index] = 0;
    env->episode_action_noop[player_index] = 0;
    env->episode_action_attack[player_index] = 0;
    env->episode_action_play[player_index] = 0;
    env->episode_action_ability[player_index] = 0;
    env->episode_action_target[player_index] = 0;
    env->episode_action_attach_weapon_from_hand[player_index] = 0;
    env->episode_action_play_spell_from_hand[player_index] = 0;
    env->episode_action_activate_garden_or_leader_ability[player_index] = 0;
    env->episode_action_activate_alley_ability[player_index] = 0;
    env->episode_action_gate_portal[player_index] = 0;
    env->episode_action_play_entity_to_alley[player_index] = 0;
    env->episode_action_play_entity_to_garden[player_index] = 0;
    env->tempo_last_turn[player_index] = 0;
    env->tempo_turn_count[player_index] = 0;
    env->pending_interception[player_index] = false;
    env->response_reserve_last_rewarded_turn[player_index] = UINT16_MAX;
    env->episode_temporary_charge_realized[player_index] = 0;
    env->episode_temporary_attack_damage_realized[player_index] = 0;
    env->episode_contextual_response_reserve_opportunities[player_index] = 0;
  }
  env->pending_attack_reward = (AzkAttackRewardContext){0};
  env->has_pending_attack_reward = false;
  AzkRewardSnapshot snapshot = {0};
  env->has_last_snapshot = false;
  if (env->engine != NULL &&
      azk_engine_reward_snapshot(env->engine, &snapshot)) {
    env->last_snapshot = snapshot;
    env->has_last_snapshot = true;
    if (reward_telemetry_enabled) {
      for (int8_t player_index = 0;
           player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
        compute_phi_component_values(
            &snapshot, player_index, env->last_phi_components[player_index]);
      }
    }
  }

  float potential_scale = 1.0f;
  float exploration_scale = 1.0f;
  current_reward_component_scales(
      env, &potential_scale, &exploration_scale);
  float phi_values[MAX_PLAYERS_PER_MATCH] = {0.0f};
  const bool have_phi =
      env->engine != NULL && compute_phi_values(env, phi_values);
  for (int8_t player_index = 0;
       player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    const float phi = have_phi ? phi_values[player_index] : 0.0f;
    env->last_phi[player_index] = phi;
    env->last_scaled_phi[player_index] = potential_scale * phi;
    env->episode_initial_phi[player_index] = phi;
    env->episode_initial_scaled_phi[player_index] = potential_scale * phi;
    for (int component = 0; component < 4; ++component) {
      env->last_scaled_phi_components[player_index][component] =
          potential_scale *
          env->last_phi_components[player_index][component];
    }
  }
}
static void adopt_current_potential_without_reward(CAzukiTCG* env) {
  float potential_scale = 1.0f;
  float exploration_scale = 1.0f;
  current_reward_component_scales(
      env, &potential_scale, &exploration_scale);
  float phi_values[MAX_PLAYERS_PER_MATCH] = {0.0f};
  if (!compute_phi_values(env, phi_values)) {
    fprintf(stderr, "Failed to initialize battle potential\n");
    abort();
  }
  AzkRewardSnapshot snapshot = {0};
  if (!azk_engine_reward_snapshot(env->engine, &snapshot)) {
    fprintf(stderr, "Failed to initialize battle reward snapshot\n");
    abort();
  }
  env->last_snapshot = snapshot;
  env->has_last_snapshot = true;
  for (int player = 0; player < MAX_PLAYERS_PER_MATCH; ++player) {
    env->last_phi[player] = phi_values[player];
    env->last_scaled_phi[player] = potential_scale * phi_values[player];
    if (env->reward_telemetry.enabled) {
      compute_phi_component_values(
          &snapshot, player, env->last_phi_components[player]);
      for (int component = 0; component < 4; ++component) {
        env->last_scaled_phi_components[player][component] =
            potential_scale * env->last_phi_components[player][component];
      }
    }
  }
}

static inline void zero_step_reward_components(CAzukiTCG* env) {
  for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    env->rewards[player_index] = 0.0f;
    if (env->terminal_rewards != NULL) {
      env->terminal_rewards[player_index] = 0.0f;
    }
    if (env->shaped_rewards != NULL) {
      env->shaped_rewards[player_index] = 0.0f;
    }
  }
}
static void record_zero_pbrs_step(CAzukiTCG* env) {
  float potential_scale = 1.0f;
  float exploration_scale = 1.0f;
  current_reward_component_scales(
      env, &potential_scale, &exploration_scale);
  const float components[MAX_PLAYERS_PER_MATCH]
                        [AZK_REWARD_COMPONENT_COUNT] = {{0}};
  const float rewards[MAX_PLAYERS_PER_MATCH] = {0.0f};
  record_reward_telemetry_step(
      env, components, components, potential_scale, true,
      rewards, rewards, true);
}

static void apply_pbrs_terminal_closure(CAzukiTCG* env) {
  if (!env->proper_pbrs || !env->pbrs_terminal_closure) {
    return;
  }

  float potential_scale = 1.0f;
  float exploration_scale = 1.0f;
  current_reward_component_scales(
      env, &potential_scale, &exploration_scale);
  float raw_closure[MAX_PLAYERS_PER_MATCH] = {0.0f};
  float scaled_closure[MAX_PLAYERS_PER_MATCH] = {0.0f};
  float raw_components[MAX_PLAYERS_PER_MATCH]
                      [AZK_REWARD_COMPONENT_COUNT] = {{0}};
  float scaled_components[MAX_PLAYERS_PER_MATCH]
                         [AZK_REWARD_COMPONENT_COUNT] = {{0}};
  for (int8_t player_index = 0;
       player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    raw_closure[player_index] = -env->last_phi[player_index];
    scaled_closure[player_index] = -env->last_scaled_phi[player_index];
    env->rewards[player_index] += scaled_closure[player_index];
    if (env->shaped_rewards != NULL) {
      env->shaped_rewards[player_index] += scaled_closure[player_index];
    }

    float raw_component_sum = 0.0f;
    float scaled_component_sum = 0.0f;
    for (int component = 0; component < 4; ++component) {
      const int reward_component =
          AZK_REWARD_POTENTIAL_LEADER_HEALTH + component;
      const float raw_contribution =
          -env->last_phi_components[player_index][component];
      const float scaled_contribution =
          -env->last_scaled_phi_components[player_index][component];
      raw_components[player_index][reward_component] = raw_contribution;
      scaled_components[player_index][reward_component] = scaled_contribution;
      raw_component_sum += raw_contribution;
      scaled_component_sum += scaled_contribution;
    }
    raw_components[player_index][AZK_REWARD_POTENTIAL_UNTAPPED_IKZ] +=
        raw_closure[player_index] - raw_component_sum;
    scaled_components[player_index][AZK_REWARD_POTENTIAL_UNTAPPED_IKZ] +=
        scaled_closure[player_index] - scaled_component_sum;
  }
  record_reward_telemetry_step(
      env, raw_components, scaled_components, potential_scale, true,
      raw_closure, scaled_closure, false);
  memset(env->last_phi, 0, sizeof(env->last_phi));
  memset(env->last_scaled_phi, 0, sizeof(env->last_scaled_phi));
  memset(env->last_phi_components, 0, sizeof(env->last_phi_components));
  memset(env->last_scaled_phi_components, 0,
         sizeof(env->last_scaled_phi_components));
}

static void ability_outcome_reward_terms(
    const CAzukiTCG* env, const AzkRewardSnapshot* snapshot, int player,
    float* gate_term, float* leader_term) {
  *gate_term = 0.0f;
  *leader_term = 0.0f;
  if (!env->has_last_snapshot) {
    return;
  }
  const int opponent = 1 - player;
  *gate_term = g_reward_tuning.ability_outcome_bonus *
      ((snapshot->gate_ability_outcomes[player] -
        env->last_snapshot.gate_ability_outcomes[player]) -
       (snapshot->gate_ability_outcomes[opponent] -
        env->last_snapshot.gate_ability_outcomes[opponent]));
  *leader_term = g_reward_tuning.ability_outcome_bonus *
      ((snapshot->leader_ability_outcomes[player] -
        env->last_snapshot.leader_ability_outcomes[player]) -
       (snapshot->leader_ability_outcomes[opponent] -
        env->last_snapshot.leader_ability_outcomes[opponent]));
}

static void apply_terminal_ability_outcomes(CAzukiTCG* env) {
  AzkRewardSnapshot snapshot;
  if (g_reward_tuning.ability_outcome_bonus == 0.0f ||
      !azk_engine_reward_snapshot(env->engine, &snapshot)) {
    return;
  }
  float gate_term, leader_term;
  ability_outcome_reward_terms(env, &snapshot, 0, &gate_term, &leader_term);
  env->last_snapshot = snapshot;
  env->has_last_snapshot = true;
  if (gate_term == 0.0f && leader_term == 0.0f) {
    return;
  }
  float potential_scale, exploration_scale;
  current_reward_component_scales(env, &potential_scale, &exploration_scale);
  const float raw[MAX_PLAYERS_PER_MATCH] = {
      gate_term + leader_term, -(gate_term + leader_term)};
  float scaled[MAX_PLAYERS_PER_MATCH];
  float components[MAX_PLAYERS_PER_MATCH][AZK_REWARD_COMPONENT_COUNT] = {{0}};
  float scaled_components[MAX_PLAYERS_PER_MATCH][AZK_REWARD_COMPONENT_COUNT] = {{0}};
  set_zero_sum_reward_component(
      components, 0, 1, AZK_REWARD_GATE_ABILITY_OUTCOME, gate_term);
  set_zero_sum_reward_component(
      components, 0, 1, AZK_REWARD_LEADER_ABILITY_OUTCOME, leader_term);
  for (int player = 0; player < MAX_PLAYERS_PER_MATCH; ++player) {
    scaled[player] = exploration_scale * raw[player];
    env->rewards[player] += scaled[player];
    if (env->shaped_rewards != NULL) {
      env->shaped_rewards[player] += scaled[player];
    }
    scaled_components[player][AZK_REWARD_GATE_ABILITY_OUTCOME] =
        exploration_scale * components[player][AZK_REWARD_GATE_ABILITY_OUTCOME];
    scaled_components[player][AZK_REWARD_LEADER_ABILITY_OUTCOME] =
        exploration_scale * components[player][AZK_REWARD_LEADER_ABILITY_OUTCOME];
  }
  record_reward_telemetry_step(
      env, components, scaled_components, potential_scale, true,
      raw, scaled, false);
}

static void apply_terminal_rewards(CAzukiTCG* env) {
  const GameState* game_state = azk_engine_game_state(env->engine);
  if (game_state == NULL) {
    fprintf(stderr, "No game state available when applying terminal rewards\n");
    abort();
  }

  if (game_state->winner == 0) {
    env->rewards[0] = TERMINAL_REWARD;
    env->rewards[1] = -TERMINAL_REWARD;
  } else if (game_state->winner == 1) {
    env->rewards[0] = -TERMINAL_REWARD;
    env->rewards[1] = TERMINAL_REWARD;
  } else {
    env->rewards[0] = 0.0f;
    env->rewards[1] = 0.0f;
  }
  float terminal_reward[MAX_PLAYERS_PER_MATCH] = {
      env->rewards[0], env->rewards[1]};
  if (env->terminal_rewards != NULL) {
    env->terminal_rewards[0] = terminal_reward[0];
    env->terminal_rewards[1] = terminal_reward[1];
  }
  if (env->shaped_rewards != NULL) {
    env->shaped_rewards[0] = 0.0f;
    env->shaped_rewards[1] = 0.0f;
  }
  float components[MAX_PLAYERS_PER_MATCH][AZK_REWARD_COMPONENT_COUNT] = {{0}};
  components[0][AZK_REWARD_TERMINAL_OUTCOME] = terminal_reward[0];
  components[1][AZK_REWARD_TERMINAL_OUTCOME] = terminal_reward[1];
  apply_terminal_ability_outcomes(env);
  apply_pbrs_terminal_closure(env);
  record_reward_telemetry_step(
      env, components, components, 1.0f, false,
      terminal_reward, terminal_reward, true);
}

static void apply_truncation_rewards(CAzukiTCG* env, EpisodeEndReason reason) {
  (void)reason;
  // A truncation is a trace boundary, not an MDP terminal. If no action was
  // taken (forced evaluation or a pre-action zero-mask guard), there is no
  // state transition to reward. Action-driven truncations apply their normal
  // nonterminal shaping before this function is called.
  zero_step_reward_components(env);
}

static void accumulate_step_rewards(CAzukiTCG* env) {
  for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    env->episode_returns[player_index] += env->rewards[player_index];
    env->episode_terminal_returns[player_index] +=
        env->terminal_rewards != NULL ? env->terminal_rewards[player_index] : 0.0f;
    env->episode_shaped_returns[player_index] +=
        env->shaped_rewards != NULL ? env->shaped_rewards[player_index] : 0.0f;
  }
}

static void deckbuild_fill_export_record(CAzukiTCG* env);

static void record_episode_stats(CAzukiTCG* env, EpisodeEndReason reason) {
  float potential_scale = 1.0f;
  float exploration_scale = 1.0f;
  current_reward_component_scales(
      env, &potential_scale, &exploration_scale);
  AzkRewardSnapshot snapshot = {0};
  if (!azk_engine_reward_snapshot(env->engine, &snapshot)) {
    fprintf(stderr, "Failed to collect reward snapshot for episode stats\n");
  }

  const GameState* game_state = azk_engine_game_state(env->engine);
  if (game_state == NULL) {
    fprintf(stderr, "No game state available when recording episode stats\n");
    return;
  }

  env->log.n += 1.0f;
  env->completed_episodes += 1;
  env->log.completed_episodes += (float)env->completed_episodes;
  env->log.episode_length += (float)env->tick;
  env->log.episode_return += env->episode_returns[0];
  env->log.p0_episode_return += env->episode_returns[0];
  env->log.p1_episode_return += env->episode_returns[1];
  env->log.curriculum_episode_cap += (float)env->current_episode_cap;
  env->log.reward_shaping_scale += potential_scale;
  env->log.potential_reward_scale += potential_scale;
  env->log.exploration_reward_scale += exploration_scale;
  if (reason == EP_END_REASON_GAMEOVER) {
    env->log.gameover_terminal_rate += 1.0f;
  } else if (reason == EP_END_REASON_TIMEOUT_TRUNCATION) {
    env->log.timeout_truncation_rate += 1.0f;
  } else if (reason == EP_END_REASON_AUTO_TICK_TRUNCATION) {
    env->log.auto_tick_truncation_rate += 1.0f;
  } else if (reason == EP_END_REASON_ZERO_LEGAL_ACTION_TRUNCATION) {
    env->log.zero_legal_action_truncation_rate += 1.0f;
  }
  if (game_state->winner == 0) {
    env->log.p0_winrate += 1.0f;
    env->log.winner_terminal_rate += 1.0f;
  } else if (game_state->winner == 1) {
    env->log.p1_winrate += 1.0f;
    env->log.winner_terminal_rate += 1.0f;
  } else {
    env->log.draw_rate += 1.0f;
  }
  if (game_state->starting_player_index == 0) {
    env->log.p0_start_rate += 1.0f;
  } else if (game_state->starting_player_index == 1) {
    env->log.p1_start_rate += 1.0f;
  }

  env->log.p0_avg_leader_health += snapshot.leader_health_ratio[0];
  env->log.p1_avg_leader_health += snapshot.leader_health_ratio[1];
  env->log.p0_entity_damage_dealt += snapshot.entity_damage_taken[1];
  env->log.p1_entity_damage_dealt += snapshot.entity_damage_taken[0];
  env->log.p0_entity_damage_taken += snapshot.entity_damage_taken[0];
  env->log.p1_entity_damage_taken += snapshot.entity_damage_taken[1];
  env->log.p0_generated_ikz_created += snapshot.generated_ikz_created[0];
  env->log.p1_generated_ikz_created += snapshot.generated_ikz_created[1];
  env->log.p0_generated_ikz_converted += snapshot.generated_ikz_converted[0];
  env->log.p1_generated_ikz_converted += snapshot.generated_ikz_converted[1];
  env->log.p0_generated_ikz_conversion_rate +=
      safe_delta(snapshot.generated_ikz_converted[0],
                 snapshot.generated_ikz_created[0]);
  env->log.p1_generated_ikz_conversion_rate +=
      safe_delta(snapshot.generated_ikz_converted[1],
                 snapshot.generated_ikz_created[1]);
  env->log.p0_temporary_charge_realized +=
      (float)env->episode_temporary_charge_realized[0];
  env->log.p1_temporary_charge_realized +=
      (float)env->episode_temporary_charge_realized[1];
  env->log.p0_temporary_attack_damage_realized +=
      (float)env->episode_temporary_attack_damage_realized[0];
  env->log.p1_temporary_attack_damage_realized +=
      (float)env->episode_temporary_attack_damage_realized[1];
  env->log.p0_contextual_response_reserve_opportunities +=
      (float)env->episode_contextual_response_reserve_opportunities[0];
  env->log.p1_contextual_response_reserve_opportunities +=
      (float)env->episode_contextual_response_reserve_opportunities[1];
  env->log.p0_gate_ability_outcomes += snapshot.gate_ability_outcomes[0];
  env->log.p1_gate_ability_outcomes += snapshot.gate_ability_outcomes[1];
  env->log.p0_leader_ability_outcomes += snapshot.leader_ability_outcomes[0];
  env->log.p1_leader_ability_outcomes += snapshot.leader_ability_outcomes[1];

  for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    const float total = (float)env->episode_action_total[player_index];
    const float inv_total = total > 0.0f ? 1.0f / total : 0.0f;
    const float noop_rate = (float)env->episode_action_noop[player_index] * inv_total;
    const float attack_rate = (float)env->episode_action_attack[player_index] * inv_total;
    const float play_rate = (float)env->episode_action_play[player_index] * inv_total;
    const float ability_rate = (float)env->episode_action_ability[player_index] * inv_total;
    const float target_rate = (float)env->episode_action_target[player_index] * inv_total;
    const float attach_weapon_from_hand_rate =
        (float)env->episode_action_attach_weapon_from_hand[player_index] * inv_total;
    const float play_spell_from_hand_rate =
        (float)env->episode_action_play_spell_from_hand[player_index] * inv_total;
    const float activate_garden_or_leader_ability_rate =
        (float)env->episode_action_activate_garden_or_leader_ability[player_index] * inv_total;
    const float activate_alley_ability_rate =
        (float)env->episode_action_activate_alley_ability[player_index] * inv_total;
    const float gate_portal_rate =
        (float)env->episode_action_gate_portal[player_index] * inv_total;
    const float play_entity_to_alley_rate =
        (float)env->episode_action_play_entity_to_alley[player_index] * inv_total;
    const float play_entity_to_garden_rate =
        (float)env->episode_action_play_entity_to_garden[player_index] * inv_total;
    if (player_index == 0) {
      env->log.p0_noop_selected_rate += noop_rate;
      env->log.p0_attack_selected_rate += attack_rate;
      env->log.p0_attach_weapon_from_hand_selected_rate += attach_weapon_from_hand_rate;
      env->log.p0_play_spell_from_hand_selected_rate += play_spell_from_hand_rate;
      env->log.p0_activate_garden_or_leader_ability_selected_rate +=
          activate_garden_or_leader_ability_rate;
      env->log.p0_activate_alley_ability_selected_rate += activate_alley_ability_rate;
      env->log.p0_gate_portal_selected_rate += gate_portal_rate;
      env->log.p0_play_entity_to_alley_selected_rate += play_entity_to_alley_rate;
      env->log.p0_play_entity_to_garden_selected_rate += play_entity_to_garden_rate;
      env->log.p0_play_selected_rate += play_rate;
      env->log.p0_ability_selected_rate += ability_rate;
      env->log.p0_target_selected_rate += target_rate;
    } else {
      env->log.p1_noop_selected_rate += noop_rate;
      env->log.p1_attack_selected_rate += attack_rate;
      env->log.p1_attach_weapon_from_hand_selected_rate += attach_weapon_from_hand_rate;
      env->log.p1_play_spell_from_hand_selected_rate += play_spell_from_hand_rate;
      env->log.p1_activate_garden_or_leader_ability_selected_rate +=
          activate_garden_or_leader_ability_rate;
      env->log.p1_activate_alley_ability_selected_rate += activate_alley_ability_rate;
      env->log.p1_gate_portal_selected_rate += gate_portal_rate;
      env->log.p1_play_entity_to_alley_selected_rate += play_entity_to_alley_rate;
      env->log.p1_play_entity_to_garden_selected_rate += play_entity_to_garden_rate;
      env->log.p1_play_selected_rate += play_rate;
      env->log.p1_ability_selected_rate += ability_rate;
      env->log.p1_target_selected_rate += target_rate;
    }
  }

  if (env->deck_building) {
    env->deck_record_end_reason = (int8_t)reason;
    env->deck_record_starting_player = game_state->starting_player_index;
    deckbuild_fill_export_record(env);
  }
}

static void record_action_choice(CAzukiTCG* env, int8_t player_index, ActionType type) {
  if (player_index < 0 || player_index >= MAX_PLAYERS_PER_MATCH) {
    return;
  }
  env->episode_action_total[player_index] += 1;
  if (type == ACT_NOOP) {
    env->episode_action_noop[player_index] += 1;
  } else if (type == ACT_ATTACK) {
    env->episode_action_attack[player_index] += 1;
  } else if (type == ACT_ATTACH_WEAPON_FROM_HAND) {
    env->episode_action_attach_weapon_from_hand[player_index] += 1;
  } else if (type == ACT_PLAY_SPELL_FROM_HAND) {
    env->episode_action_play_spell_from_hand[player_index] += 1;
  } else if (type == ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY) {
    env->episode_action_activate_garden_or_leader_ability[player_index] += 1;
  } else if (type == ACT_ACTIVATE_ALLEY_ABILITY) {
    env->episode_action_activate_alley_ability[player_index] += 1;
  } else if (type == ACT_GATE_PORTAL) {
    env->episode_action_gate_portal[player_index] += 1;
  } else if (type == ACT_PLAY_ENTITY_TO_ALLEY) {
    env->episode_action_play_entity_to_alley[player_index] += 1;
  } else if (type == ACT_PLAY_ENTITY_TO_GARDEN) {
    env->episode_action_play_entity_to_garden[player_index] += 1;
  }

  if (type == ACT_PLAY_ENTITY_TO_GARDEN || type == ACT_PLAY_ENTITY_TO_ALLEY ||
      type == ACT_PLAY_SPELL_FROM_HAND || type == ACT_ATTACH_WEAPON_FROM_HAND ||
      type == ACT_GATE_PORTAL) {
    env->episode_action_play[player_index] += 1;
  } else if (type == ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY ||
             type == ACT_ACTIVATE_ALLEY_ABILITY || type == ACT_CONFIRM_ABILITY) {
    env->episode_action_ability[player_index] += 1;
  } else if (type == ACT_SELECT_COST_TARGET || type == ACT_SELECT_EFFECT_TARGET ||
             type == ACT_SELECT_FROM_SELECTION || type == ACT_SELECT_TO_ALLEY ||
             type == ACT_SELECT_TO_EQUIP || type == ACT_BOTTOM_DECK_CARD ||
             type == ACT_BOTTOM_DECK_ALL || type == ACT_DECLARE_DEFENDER ||
             type == ACT_MULLIGAN_SHUFFLE) {
    env->episode_action_target[player_index] += 1;
  }
}

static bool refreshed_mask_has_ikz_response(CAzukiTCG* env,
                                            int8_t player_index) {
  if (player_index < 0 || player_index >= MAX_PLAYERS_PER_MATCH) {
    return false;
  }
  const TrainingActionMaskObs *mask =
      &obs_base(env, player_index)->action_mask;
  const GameState *gs = azk_engine_game_state(env->engine);
  if (gs == NULL) {
    return false;
  }

  for (uint16_t i = 0; i < mask->legal_action_count; ++i) {
    const ActionType type = (ActionType)mask->legal_primary[i];
    if (type != ACT_PLAY_ENTITY_TO_GARDEN &&
        type != ACT_PLAY_ENTITY_TO_ALLEY &&
        type != ACT_ATTACH_WEAPON_FROM_HAND &&
        type != ACT_PLAY_SPELL_FROM_HAND &&
        type != ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY) {
      continue;
    }
    const UserAction action = {
        .player = gs->players[player_index],
        .type = type,
        .subaction_1 = (int)mask->legal_sub1[i],
        .subaction_2 = (int)mask->legal_sub2[i],
        .subaction_3 = (int)mask->legal_sub3[i],
    };
    if (azk_engine_legal_action_spends_ikz(env->engine, player_index,
                                           &action)) {
      return true;
    }
  }
  return false;
}

static void apply_shaped_rewards(
    CAzukiTCG* env, int8_t acting_player_index, ActionType selected_type,
    bool noop_had_alternatives, float action_bonus,
    const AzkActionRewardComponents* action_components) {
  if (acting_player_index < 0 || acting_player_index >= MAX_PLAYERS_PER_MATCH) {
    fprintf(stderr, "Invalid acting player index %d when applying shaped rewards\n", acting_player_index);
    abort();
  }

  float phi_values[MAX_PLAYERS_PER_MATCH] = {0.0f};
  if (!compute_phi_values(env, phi_values)) {
    fprintf(stderr, "Failed to compute phi values when applying shaped rewards\n");
    abort();
  }
  float phi_components[MAX_PLAYERS_PER_MATCH][4] = {{0}};
  bool have_phi_components = false;

  float potential_scale = 1.0f;
  float exploration_scale = 1.0f;
  current_reward_component_scales(
      env, &potential_scale, &exploration_scale);
  const int8_t opponent_index =
      (acting_player_index + 1) % MAX_PLAYERS_PER_MATCH;
  const float potential_term =
      env->proper_pbrs
          ? env->pbrs_gamma * phi_values[acting_player_index] -
                env->last_phi[acting_player_index]
          : env->time_weight *
                (phi_values[acting_player_index] -
                 env->last_phi[acting_player_index]);
  const float scaled_potential_term =
      env->proper_pbrs
          ? env->pbrs_gamma *
                    (potential_scale * phi_values[acting_player_index]) -
                env->last_scaled_phi[acting_player_index]
          : potential_scale * potential_term;
  float leader_delta_term = 0.0f;
  float board_delta_term = 0.0f;
  float entity_damage_exchange_term = 0.0f;
  float generated_ikz_conversion_term = 0.0f;
  float gate_ability_term = 0.0f;
  float leader_ability_term = 0.0f;
  AzkRewardSnapshot snapshot = {0};
  const bool have_snapshot = azk_engine_reward_snapshot(env->engine, &snapshot);
  if (have_snapshot) {
    ability_outcome_reward_terms(
        env, &snapshot, acting_player_index,
        &gate_ability_term, &leader_ability_term);
    if (env->has_last_snapshot) {
      const float prev_leader_edge = env->last_snapshot.leader_health_ratio[acting_player_index] -
                                     env->last_snapshot.leader_health_ratio[opponent_index];
      const float curr_leader_edge = snapshot.leader_health_ratio[acting_player_index] -
                                     snapshot.leader_health_ratio[opponent_index];
      leader_delta_term = g_reward_tuning.leader_delta_weight *
                          (curr_leader_edge - prev_leader_edge);

      const float prev_board_edge = safe_delta(
          env->last_snapshot.garden_attack_sum[acting_player_index] -
              env->last_snapshot.garden_attack_sum[opponent_index],
          PBRS_GARDEN_ATTACK_CAP);
      const float curr_board_edge = safe_delta(
          snapshot.garden_attack_sum[acting_player_index] -
              snapshot.garden_attack_sum[opponent_index],
          PBRS_GARDEN_ATTACK_CAP);
      board_delta_term = g_reward_tuning.board_delta_weight *
                         (curr_board_edge - prev_board_edge);

      const float own_entity_damage_delta =
          snapshot.entity_damage_taken[acting_player_index] -
          env->last_snapshot.entity_damage_taken[acting_player_index];
      const float opponent_entity_damage_delta =
          snapshot.entity_damage_taken[opponent_index] -
          env->last_snapshot.entity_damage_taken[opponent_index];
      const int entity_cap = g_reward_tuning.entity_damage_exchange_step_cap;
      const float entity_exchange =
          opponent_entity_damage_delta - own_entity_damage_delta;
      entity_damage_exchange_term =
          g_reward_tuning.entity_damage_exchange_per_hp *
          (entity_cap > 0
               ? clampf(entity_exchange, -(float)entity_cap, (float)entity_cap)
               : entity_exchange);

      const float own_conversion_delta =
          snapshot.generated_ikz_converted[acting_player_index] -
          env->last_snapshot.generated_ikz_converted[acting_player_index];
      const float opponent_conversion_delta =
          snapshot.generated_ikz_converted[opponent_index] -
          env->last_snapshot.generated_ikz_converted[opponent_index];
      const int conversion_cap =
          g_reward_tuning.generated_ikz_conversion_step_cap;
      const float conversion_edge =
          own_conversion_delta - opponent_conversion_delta;
      generated_ikz_conversion_term =
          g_reward_tuning.generated_ikz_conversion_bonus *
          (conversion_cap > 0
               ? clampf(conversion_edge, -(float)conversion_cap,
                        (float)conversion_cap)
               : conversion_edge);
    }
    env->last_snapshot = snapshot;
    env->has_last_snapshot = true;
    if (env->reward_telemetry.enabled) {
      for (int8_t player_index = 0;
           player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
        compute_phi_component_values(
            &snapshot, player_index, phi_components[player_index]);
      }
      have_phi_components = true;
    }
  }

  float noop_penalty = 0.0f;
  if (selected_type == ACT_NOOP && noop_had_alternatives) {
    noop_penalty = g_reward_tuning.noop_penalty;
  }

  const float potential_reward =
      potential_term + leader_delta_term + board_delta_term;
  const float exploration_reward =
      entity_damage_exchange_term + generated_ikz_conversion_term +
      gate_ability_term + leader_ability_term - noop_penalty + action_bonus;
  const float base_shaped_reward = potential_reward + exploration_reward;
  const float shaped_reward =
      scaled_potential_term +
      potential_scale * (leader_delta_term + board_delta_term) +
      exploration_scale * exploration_reward;
  env->rewards[acting_player_index] = shaped_reward;
  env->rewards[opponent_index] = -shaped_reward;
  if (env->terminal_rewards != NULL) {
    env->terminal_rewards[acting_player_index] = 0.0f;
    env->terminal_rewards[opponent_index] = 0.0f;
  }
  if (env->shaped_rewards != NULL) {
    env->shaped_rewards[acting_player_index] = shaped_reward;
    env->shaped_rewards[opponent_index] = -shaped_reward;
  }

  if (env->reward_telemetry.enabled) {
    float components[MAX_PLAYERS_PER_MATCH][AZK_REWARD_COMPONENT_COUNT] = {{0}};
    float scaled_components[MAX_PLAYERS_PER_MATCH]
                           [AZK_REWARD_COMPONENT_COUNT] = {{0}};
    float potential_sum = 0.0f;
    for (int component = 0; component < 4; ++component) {
      const float contribution =
          have_phi_components
              ? (env->proper_pbrs
                     ? env->pbrs_gamma *
                               phi_components[acting_player_index][component] -
                           env->last_phi_components[acting_player_index][component]
                     : env->time_weight *
                           (phi_components[acting_player_index][component] -
                            env->last_phi_components[acting_player_index][component]))
              : 0.0f;
      const int reward_component =
          AZK_REWARD_POTENTIAL_LEADER_HEALTH + component;
      components[acting_player_index][reward_component] = contribution;
      components[opponent_index][reward_component] = -contribution;
      potential_sum += contribution;
    }
    const float potential_total = potential_term;
    const float potential_residual = potential_total - potential_sum;
    components[acting_player_index][AZK_REWARD_POTENTIAL_UNTAPPED_IKZ] +=
        potential_residual;
    components[opponent_index][AZK_REWARD_POTENTIAL_UNTAPPED_IKZ] -=
        potential_residual;

    set_zero_sum_reward_component(
        components, acting_player_index, opponent_index,
        AZK_REWARD_DIRECT_LEADER_EDGE, leader_delta_term);
    set_zero_sum_reward_component(
        components, acting_player_index, opponent_index,
        AZK_REWARD_DIRECT_BOARD_EDGE, board_delta_term);
    set_zero_sum_reward_component(
        components, acting_player_index, opponent_index,
        AZK_REWARD_NOOP_PENALTY, -noop_penalty);
    set_zero_sum_reward_component(
        components, acting_player_index, opponent_index,
        AZK_REWARD_ENTITY_DAMAGE_EXCHANGE, entity_damage_exchange_term);
    set_zero_sum_reward_component(
        components, acting_player_index, opponent_index,
        AZK_REWARD_GENERATED_IKZ_CONVERSION,
        generated_ikz_conversion_term);
    set_zero_sum_reward_component(
        components, acting_player_index, opponent_index,
        AZK_REWARD_GATE_ABILITY_OUTCOME, gate_ability_term);
    set_zero_sum_reward_component(
        components, acting_player_index, opponent_index,
        AZK_REWARD_LEADER_ABILITY_OUTCOME, leader_ability_term);
    if (action_components != NULL) {
      set_zero_sum_reward_component(
          components, acting_player_index, opponent_index,
          AZK_REWARD_PORTAL_GP, action_components->portal_gp);
      set_zero_sum_reward_component(
          components, acting_player_index, opponent_index,
          AZK_REWARD_PORTAL_OUTCOME, action_components->portal_outcome);
      set_zero_sum_reward_component(
          components, acting_player_index, opponent_index,
          AZK_REWARD_EARLY_TEMPO, action_components->early_tempo);
      set_zero_sum_reward_component(
          components, acting_player_index, opponent_index,
          AZK_REWARD_DAMAGE_MITIGATION, action_components->damage_mitigation);
      set_zero_sum_reward_component(
          components, acting_player_index, opponent_index,
          AZK_REWARD_TEMPORARY_CHARGE, action_components->temporary_charge);
      set_zero_sum_reward_component(
          components, acting_player_index, opponent_index,
          AZK_REWARD_TEMPORARY_ATTACK, action_components->temporary_attack);
      set_zero_sum_reward_component(
          components, acting_player_index, opponent_index,
          AZK_REWARD_RESPONSE_RESERVE, action_components->response_reserve);
    }
    for (int player = 0; player < MAX_PLAYERS_PER_MATCH; ++player) {
      for (int component = 0; component < AZK_REWARD_COMPONENT_COUNT;
           ++component) {
        const bool potential_component =
            component >= AZK_REWARD_POTENTIAL_LEADER_HEALTH &&
            component <= AZK_REWARD_DIRECT_BOARD_EDGE;
        scaled_components[player][component] =
            (potential_component ? potential_scale : exploration_scale) *
            components[player][component];
      }
      for (int component = 0; component < 4; ++component) {
        const int reward_component =
            AZK_REWARD_POTENTIAL_LEADER_HEALTH + component;
        scaled_components[player][reward_component] =
            have_phi_components
                ? (env->proper_pbrs
                       ? env->pbrs_gamma * potential_scale *
                                 phi_components[player][component] -
                             env->last_scaled_phi_components[player][component]
                       : potential_scale *
                             env->time_weight *
                             (phi_components[player][component] -
                              env->last_phi_components[player][component]))
                : 0.0f;
      }
      float scaled_pbrs_sum = 0.0f;
      for (int component = 0; component < 4; ++component) {
        scaled_pbrs_sum +=
            scaled_components[player]
                             [AZK_REWARD_POTENTIAL_LEADER_HEALTH + component];
      }
      scaled_components[player][AZK_REWARD_POTENTIAL_UNTAPPED_IKZ] +=
          (player == acting_player_index ? scaled_potential_term
                                         : -scaled_potential_term) -
          scaled_pbrs_sum;
    }
    float expected_raw[MAX_PLAYERS_PER_MATCH] = {0.0f};
    expected_raw[acting_player_index] = base_shaped_reward;
    expected_raw[opponent_index] = -base_shaped_reward;
    record_reward_telemetry_step(
        env, components, scaled_components, potential_scale, true,
        expected_raw, env->rewards, true);
  }

  for (int8_t player_index = 0;
       player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    env->last_phi[player_index] = phi_values[player_index];
    env->last_scaled_phi[player_index] =
        potential_scale * phi_values[player_index];
    if (env->reward_telemetry.enabled && have_phi_components) {
      memcpy(env->last_phi_components[player_index],
             phi_components[player_index],
             sizeof(env->last_phi_components[player_index]));
      for (int component = 0; component < 4; ++component) {
        env->last_scaled_phi_components[player_index][component] =
            potential_scale * phi_components[player_index][component];
      }
    }
  }
  if (!env->proper_pbrs) {
    env->time_weight *= env->time_decay;
  }
}

static int max_episode_ticks_limit(void) {
  // Optional cap for periodic episode boundaries/logging.
  // Default is disabled; set AZK_MAX_TICKS_PER_EPISODE to a positive integer
  // to force truncation at that many environment ticks.
  static int cached = INT_MIN;
  if (cached != INT_MIN) {
    return cached;
  }

  const int default_limit = 0;
  cached = default_limit;
  const char* raw = getenv("AZK_MAX_TICKS_PER_EPISODE");
  if (raw == NULL || raw[0] == '\0') {
    return cached;
  }

  char* endptr = NULL;
  long parsed = strtol(raw, &endptr, 10);
  if (endptr == raw || *endptr != '\0' || parsed < 0 || parsed > INT_MAX) {
    fprintf(stderr,
            "Invalid AZK_MAX_TICKS_PER_EPISODE='%s'; using default %d\n",
            raw, default_limit);
    cached = default_limit;
    return cached;
  }

  cached = (int)parsed;
  return cached;
}

typedef struct EpisodeCapCurriculumConfig {
  int initialized;
  int enabled;
  int initial_cap;
  int final_cap;
  int warmup_episodes;
  int ramp_episodes;
  int long_episode_every;
  int long_episode_cap;
} EpisodeCapCurriculumConfig;

static EpisodeCapCurriculumConfig g_episode_cap_curriculum = {0};

static int parse_nonnegative_env_int(const char* name, int default_value) {
  const char* raw = getenv(name);
  if (raw == NULL || raw[0] == '\0') {
    return default_value;
  }

  char* endptr = NULL;
  long parsed = strtol(raw, &endptr, 10);
  if (endptr == raw || *endptr != '\0' || parsed < 0 || parsed > INT_MAX) {
    fprintf(stderr, "Invalid %s='%s'; using default %d\n", name, raw, default_value);
    return default_value;
  }
  return (int)parsed;
}

static uint64_t initial_completed_episodes_offset(void) {
  static int initialized = 0;
  static uint64_t cached = 0;
  if (initialized) {
    return cached;
  }
  initialized = 1;

  const char* raw = getenv("AZK_RESUME_COMPLETED_EPISODES");
  if (raw == NULL || raw[0] == '\0') {
    return cached;
  }

  char* endptr = NULL;
  unsigned long long parsed = strtoull(raw, &endptr, 10);
  if (endptr == raw || *endptr != '\0') {
    fprintf(stderr,
            "Invalid AZK_RESUME_COMPLETED_EPISODES='%s'; using default %" PRIu64 "\n",
            raw, cached);
    return cached;
  }
  cached = (uint64_t)parsed;
  return cached;
}

static void init_episode_cap_curriculum_if_needed(void) {
  if (g_episode_cap_curriculum.initialized) {
    return;
  }
  g_episode_cap_curriculum.initialized = 1;
  g_episode_cap_curriculum.enabled = env_flag_enabled("AZK_MAX_TICKS_CURRICULUM");

  const int base_cap = max_episode_ticks_limit();
  if (!g_episode_cap_curriculum.enabled) {
    g_episode_cap_curriculum.initial_cap = base_cap;
    g_episode_cap_curriculum.final_cap = base_cap;
    g_episode_cap_curriculum.warmup_episodes = 0;
    g_episode_cap_curriculum.ramp_episodes = 0;
    g_episode_cap_curriculum.long_episode_every = 0;
    g_episode_cap_curriculum.long_episode_cap = base_cap;
    return;
  }

  const int default_final_cap = base_cap > 0 ? base_cap : 1000;
  int default_initial_cap = default_final_cap;
  if (default_final_cap > 300) {
    default_initial_cap = 300;
  }

  g_episode_cap_curriculum.initial_cap =
      parse_nonnegative_env_int("AZK_MAX_TICKS_CURRICULUM_INITIAL", default_initial_cap);
  g_episode_cap_curriculum.final_cap =
      parse_nonnegative_env_int("AZK_MAX_TICKS_CURRICULUM_FINAL", default_final_cap);
  g_episode_cap_curriculum.warmup_episodes =
      parse_nonnegative_env_int("AZK_MAX_TICKS_CURRICULUM_WARMUP_EPISODES", 0);
  g_episode_cap_curriculum.ramp_episodes =
      parse_nonnegative_env_int("AZK_MAX_TICKS_CURRICULUM_RAMP_EPISODES", 3000);
  g_episode_cap_curriculum.long_episode_every =
      parse_nonnegative_env_int("AZK_MAX_TICKS_CURRICULUM_LONG_EPISODE_EVERY", 8);
  int default_long_episode_cap = g_episode_cap_curriculum.final_cap + 400;
  if (default_long_episode_cap < 1600) {
    default_long_episode_cap = 1600;
  }
  g_episode_cap_curriculum.long_episode_cap =
      parse_nonnegative_env_int("AZK_MAX_TICKS_CURRICULUM_LONG_EPISODE_CAP",
                                default_long_episode_cap);

  if (g_episode_cap_curriculum.initial_cap <= 0 ||
      g_episode_cap_curriculum.final_cap <= 0) {
    fprintf(stderr,
            "Episode cap curriculum requires positive caps; disabling (initial=%d final=%d)\n",
            g_episode_cap_curriculum.initial_cap, g_episode_cap_curriculum.final_cap);
    g_episode_cap_curriculum.enabled = 0;
    g_episode_cap_curriculum.initial_cap = base_cap;
    g_episode_cap_curriculum.final_cap = base_cap;
    g_episode_cap_curriculum.warmup_episodes = 0;
    g_episode_cap_curriculum.ramp_episodes = 0;
    g_episode_cap_curriculum.long_episode_every = 0;
    g_episode_cap_curriculum.long_episode_cap = base_cap;
    return;
  }

  if (g_episode_cap_curriculum.long_episode_cap <= 0) {
    g_episode_cap_curriculum.long_episode_cap = g_episode_cap_curriculum.final_cap;
  }
}

static int current_episode_ticks_limit(CAzukiTCG* env) {
  init_episode_cap_curriculum_if_needed();
  if (!g_episode_cap_curriculum.enabled) {
    return max_episode_ticks_limit();
  }

  const uint64_t completed = env->completed_episodes;
  const uint64_t warmup = (uint64_t)g_episode_cap_curriculum.warmup_episodes;
  const uint64_t ramp = (uint64_t)g_episode_cap_curriculum.ramp_episodes;
  const int initial_cap = g_episode_cap_curriculum.initial_cap;
  const int final_cap = g_episode_cap_curriculum.final_cap;

  if (completed < warmup) {
    if (g_episode_cap_curriculum.long_episode_every > 0 &&
        (completed % (uint64_t)g_episode_cap_curriculum.long_episode_every) == 0) {
      return g_episode_cap_curriculum.long_episode_cap;
    }
    return initial_cap;
  }
  if (ramp == 0) {
    return final_cap;
  }

  const uint64_t elapsed = completed - warmup;
  if (elapsed >= ramp) {
    if (g_episode_cap_curriculum.long_episode_every > 0 &&
        (completed % (uint64_t)g_episode_cap_curriculum.long_episode_every) == 0) {
      return g_episode_cap_curriculum.long_episode_cap;
    }
    return final_cap;
  }

  const double fraction = (double)elapsed / (double)ramp;
  const double interpolated =
      (double)initial_cap + ((double)final_cap - (double)initial_cap) * fraction;
  int cap = (int)llround(interpolated);
  if (cap <= 0) {
    cap = 1;
  }
  if (g_episode_cap_curriculum.long_episode_every > 0 &&
      (completed % (uint64_t)g_episode_cap_curriculum.long_episode_every) == 0) {
    return g_episode_cap_curriculum.long_episode_cap;
  }
  return cap;
}

static int max_auto_ticks_per_step_limit(void) {
  // Guard against rare infinite/no-progress engine auto-tick loops within one
  // env step. Set AZK_MAX_AUTO_TICKS_PER_STEP=0 to disable.
  static int cached = INT_MIN;
  if (cached != INT_MIN) {
    return cached;
  }

  const int default_limit = 20000;
  cached = default_limit;
  const char* raw = getenv("AZK_MAX_AUTO_TICKS_PER_STEP");
  if (raw == NULL || raw[0] == '\0') {
    return cached;
  }

  char* endptr = NULL;
  long parsed = strtol(raw, &endptr, 10);
  if (endptr == raw || *endptr != '\0' || parsed < 0 || parsed > INT_MAX) {
    fprintf(stderr,
            "Invalid AZK_MAX_AUTO_TICKS_PER_STEP='%s'; using default %d\n",
            raw, default_limit);
    cached = default_limit;
    return cached;
  }

  cached = (int)parsed;
  return cached;
}

void init(CAzukiTCG* env) {
  env->starter_rng_state = starter_seed_from_env_seed(env->seed);
  env->deck_rng_state = deck_seed_from_env_seed(env->seed);
  reset_current_deck_indices(env);
  env->tick = 0;
  env->completed_episodes = initial_completed_episodes_offset();
  env->current_episode_cap = current_episode_ticks_limit(env);
  if (env->deck_building) {
    if (!g_draft_catalog.loaded) {
      fprintf(stderr, "deck_building env requires a draft catalog\n");
      abort();
    }
    env->draft_rng_state = env->seed ^ 0x9E3779B9u;
    env->engine = NULL;
    env->deck_record_valid = false;
    draft_begin_episode(env);
    return;
  }
  const int8_t starting_player = next_starting_player(env);
  env->engine = create_env_engine(env, starting_player);
}

static inline int8_t tcg_active_player_index(CAzukiTCG* env) {
  if (env == NULL || env->engine == NULL) {
    return -1;
  }

  if (azk_engine_is_game_over(env->engine)) {
    return -1;
  }

  const GameState* game_state = azk_engine_game_state(env->engine);
  if (game_state == NULL) {
    return -1;
  }

  const int8_t active_player_index = game_state->active_player_index;
  if (active_player_index < 0 || active_player_index >= MAX_PLAYERS_PER_MATCH) {
    return -1;
  }

  return active_player_index;
}

// ---- Draft-phase implementation (deck-building native path) ----------------
static void refresh_observations(CAzukiTCG* env);
static void reset_reward_tracking(CAzukiTCG* env);

static void azk_fill_empty_weapons(TrainingWeaponObservationData* weapons) {
  for (int w = 0; w < MAX_ATTACHED_WEAPONS; ++w) {
    weapons[w].card_def_id = -1;
  }
}

static void azk_fill_empty_board_cards(TrainingBoardCardObservationData* cards,
                                       int count) {
  for (int i = 0; i < count; ++i) {
    cards[i].card_def_id = -1;
    cards[i].zone_index = (uint8_t)i;
    azk_fill_empty_weapons(cards[i].weapons);
  }
}

// Mirrors deck_building.empty_training_observation(): all card ids -1 with
// per-slot zone indices, ability/combat sentinel ids -1, everything else 0.
static void azk_fill_empty_battle_observation(TrainingObservationData* obs) {
  memset(obs, 0, sizeof(*obs));

  obs->my_observation_data.leader.card_def_id = -1;
  azk_fill_empty_weapons(obs->my_observation_data.leader.weapons);
  obs->my_observation_data.gate.card_def_id = -1;
  obs->opponent_observation_data.leader.card_def_id = -1;
  azk_fill_empty_weapons(obs->opponent_observation_data.leader.weapons);
  obs->opponent_observation_data.gate.card_def_id = -1;

  for (int i = 0; i < MAX_HAND_SIZE; ++i) {
    obs->my_observation_data.hand[i].card_def_id = -1;
    obs->my_observation_data.hand[i].zone_index = (uint8_t)i;
    obs->critic_privileged.opponent_hand[i].card_def_id = -1;
    obs->critic_privileged.opponent_hand[i].zone_index = (uint8_t)i;
  }
  azk_fill_empty_board_cards(obs->my_observation_data.alley, ALLEY_SIZE);
  azk_fill_empty_board_cards(obs->my_observation_data.garden, GARDEN_SIZE);
  azk_fill_empty_board_cards(obs->my_observation_data.selection,
                             MAX_SELECTION_ZONE_SIZE);
  azk_fill_empty_board_cards(obs->opponent_observation_data.alley, ALLEY_SIZE);
  azk_fill_empty_board_cards(obs->opponent_observation_data.garden, GARDEN_SIZE);
  for (int i = 0; i < MAX_DECK_SIZE; ++i) {
    obs->my_observation_data.discard[i].card_def_id = -1;
    obs->my_observation_data.discard[i].zone_index = (uint8_t)i;
    obs->opponent_observation_data.discard[i].card_def_id = -1;
    obs->opponent_observation_data.discard[i].zone_index = (uint8_t)i;
    obs->critic_privileged.self_deck[i].card_def_id = -1;
    obs->critic_privileged.self_deck[i].zone_index = (uint8_t)i;
    obs->critic_privileged.opponent_deck[i].card_def_id = -1;
    obs->critic_privileged.opponent_deck[i].zone_index = (uint8_t)i;
  }
  for (int i = 0; i < IKZ_AREA_SIZE; ++i) {
    obs->my_observation_data.ikz_area[i].card_def_id = -1;
    obs->my_observation_data.ikz_area[i].zone_index = (uint8_t)i;
    obs->opponent_observation_data.ikz_area[i].card_def_id = -1;
    obs->opponent_observation_data.ikz_area[i].zone_index = (uint8_t)i;
  }

  obs->ability_context.source_card_def_id = -1;
  obs->ability_context.active_player_index = -1;
  obs->combat_context.attacker_card_def_id = -1;
  obs->combat_context.target_card_def_id = -1;
}

static int draft_gate_slot_for(int16_t gate_def_id) {
  for (int i = 0; i < g_draft_catalog.gate_count; ++i) {
    if (g_draft_catalog.gate_def_ids[i] == gate_def_id) {
      return i;
    }
  }
  return -1;
}

static bool draft_player_complete(const CAzukiTCG* env, int player_index) {
  return env->draft_leader[player_index] >= 0 &&
         env->draft_main_count[player_index] >= REQUIRED_DECK_SIZE;
}

// deck_context.mode reflects the PLAYER'S OWN progress: 0 battle/complete,
// 1 picking leader, 2 picking mains (a player who finishes early shows 0
// while the episode is still drafting — legacy wrapper semantics).
static int32_t draft_player_mode(const CAzukiTCG* env, int player_index) {
  if (draft_player_complete(env, player_index)) {
    return 0;
  }
  return env->draft_leader[player_index] < 0 ? 1 : 2;
}

// Test hook: AZK_DEBUG_FORCE_GATE_DEF_IDS="p0_def_id,p1_def_id" pins gates
// for parity tests against the Python wrapper.
static bool draft_forced_gates(const CAzukiTCG* env,
                               int16_t out[MAX_PLAYERS_PER_MATCH]) {
  if (env->evaluation_forced_gates) {
    out[0] = env->evaluation_forced_gate[0];
    out[1] = env->evaluation_forced_gate[1];
    return true;
  }
  // Re-read every call (once per episode): tests toggle this within a process.
  const char* raw = getenv("AZK_DEBUG_FORCE_GATE_DEF_IDS");
  if (raw != NULL && raw[0] != '\0') {
    int a = -1, b = -1;
    if (sscanf(raw, "%d,%d", &a, &b) == 2) {
      out[0] = (int16_t)a;
      out[1] = (int16_t)b;
      return true;
    }
  }
  return false;
}

// Evaluation/debug hook for the no-leader-row lifecycle. Assigned leaders are
// still validated against each final gate before use.
static bool draft_forced_leaders(const CAzukiTCG* env,
                                 int16_t out[MAX_PLAYERS_PER_MATCH]) {
  if (env->evaluation_forced_leaders) {
    out[0] = env->evaluation_forced_leader[0];
    out[1] = env->evaluation_forced_leader[1];
    return true;
  }
  const char* raw = getenv("AZK_DEBUG_FORCE_LEADER_DEF_IDS");
  if (raw != NULL && raw[0] != '\0') {
    int a = -1, b = -1;
    if (sscanf(raw, "%d,%d", &a, &b) == 2) {
      out[0] = (int16_t)a;
      out[1] = (int16_t)b;
      return true;
    }
  }
  return false;
}

static bool draft_leader_valid_for_slot(int slot, int16_t leader_def_id) {
  if (slot < 0 || slot >= g_draft_catalog.gate_count) {
    return false;
  }
  const int begin = g_draft_catalog.leader_offsets[slot];
  const int end = g_draft_catalog.leader_offsets[slot + 1];
  for (int i = begin; i < end; ++i) {
    if (g_draft_catalog.leader_flat[i] == leader_def_id) {
      return true;
    }
  }
  return false;
}

static int16_t draft_sample_assigned_leader(CAzukiTCG* env, int slot) {
  const int begin = g_draft_catalog.leader_offsets[slot];
  const int end = g_draft_catalog.leader_offsets[slot + 1];
  const int count = end - begin;
  if (count <= 0) {
    fprintf(stderr, "Draft gate slot %d has no compatible leaders\n", slot);
    abort();
  }
  env->draft_rng_state = advance_episode_seed(env->draft_rng_state);
  return g_draft_catalog
      .leader_flat[begin + env->draft_rng_state % (uint32_t)count];
}

static int draft_active_candidates(const CAzukiTCG* env, int player_index,
                                   int16_t* out_ids, uint8_t* out_copies,
                                   int max_out) {
  const int slot = env->draft_gate_slot[player_index];
  if (env->draft_leader[player_index] < 0) {
    const int begin = g_draft_catalog.leader_offsets[slot];
    const int end = g_draft_catalog.leader_offsets[slot + 1];
    int n = 0;
    for (int i = begin; i < end && n < max_out; ++i) {
      out_ids[n] = g_draft_catalog.leader_flat[i];
      out_copies[n] = 0;
      n++;
    }
    return n;
  }
  const int begin = g_draft_catalog.main_offsets[slot];
  const int end = g_draft_catalog.main_offsets[slot + 1];
  int n = 0;
  for (int i = begin; i < end && n < max_out; ++i) {
    const int local = i - begin;
    const uint8_t copies = env->draft_copies[player_index][local];
    if (copies >= 4) {
      continue;  // maxed-out cards drop out and the list re-indexes
    }
    out_ids[n] = g_draft_catalog.main_flat[i];
    out_copies[n] = copies;
    n++;
  }
  return n;
}

// Sanitized by default (composition hidden); with deck_building_privileged_decks
// the DRAFTED pick-order lists are exposed to the critic-only block: own picks
// in self_deck, opponent picks-so-far in opponent_deck (during draft this is
// exactly the information a matchup-aware pick baseline needs).
static void fill_privileged_deck_lists(CAzukiTCG* env, int player_index,
                                       TrainingObservationData* base) {
  const int opponent_index = 1 - player_index;
  for (int i = 0; i < MAX_DECK_SIZE; ++i) {
    int16_t self_id = -1;
    int16_t opp_id = -1;
    if (env->deck_building_privileged_decks) {
      if (i < (int)env->draft_main_count[player_index]) {
        self_id = env->draft_main[player_index][i];
      }
      if (i < (int)env->draft_main_count[opponent_index]) {
        opp_id = env->draft_main[opponent_index][i];
      }
    }
    base->critic_privileged.self_deck[i].card_def_id = self_id;
    base->critic_privileged.self_deck[i].zone_index = (uint8_t)i;
    base->critic_privileged.opponent_deck[i].card_def_id = opp_id;
    base->critic_privileged.opponent_deck[i].zone_index = (uint8_t)i;
  }
}

static void fill_draft_observations(CAzukiTCG* env) {
  for (int player_index = 0; player_index < MAX_PLAYERS_PER_MATCH;
       ++player_index) {
    TrainingObservationData* base = obs_base(env, player_index);
    azk_fill_empty_battle_observation(base);
    fill_privileged_deck_lists(env, player_index, base);

    AzkTrainingDeckContextData* dc = obs_deck_context(env, player_index);
    memset(dc, 0, sizeof(*dc));
    dc->mode = draft_player_mode(env, player_index);
    dc->gate_card_def_id = env->draft_gate[player_index];
    dc->leader_card_def_id = env->draft_leader[player_index];
    for (int i = 0; i < REQUIRED_DECK_SIZE; ++i) {
      dc->main_card_def_ids[i] = env->draft_main[player_index][i];
    }
    dc->main_count = env->draft_main_count[player_index];
    for (int i = 0; i < AZK_DECKBUILD_OBS_MAX_CANDIDATES; ++i) {
      dc->candidate_card_def_ids[i] = -1;
    }

    const bool is_active = player_index == env->draft_active_player &&
                           !draft_player_complete(env, player_index);
    if (!is_active) {
      continue;
    }

    int16_t cand_ids[AZK_DRAFT_MAX_CANDIDATES];
    uint8_t cand_copies[AZK_DRAFT_MAX_CANDIDATES];
    const int n = draft_active_candidates(env, player_index, cand_ids,
                                          cand_copies, AZK_DRAFT_MAX_CANDIDATES);
    if (n <= 0 || n > 255) {
      fprintf(stderr,
              "Draft candidate count %d out of range for player %d\n", n,
              player_index);
      abort();
    }
    for (int i = 0; i < n; ++i) {
      dc->candidate_card_def_ids[i] = cand_ids[i];
      dc->candidate_copy_counts[i] = cand_copies[i];
    }
    dc->candidate_count = n;

    TrainingActionMaskObs* mask = &base->action_mask;
    mask->primary_action_mask[3] = true;  // ActionType.DECK_PICK_CARD
    mask->legal_action_count = (uint16_t)n;
    for (int i = 0; i < n; ++i) {
      mask->legal_primary[i] = 3;
      mask->legal_sub1[i] = (uint8_t)i;
    }
  }
}

// Battle steps in deck-building mode keep a deck_context (mode 0, full
// pick-order log, no candidates) and hide the privileged deck lists exactly
// like the legacy wrapper's _copy_with_sanitized_privileged_decks.
static void deckbuild_postprocess_battle_observations(CAzukiTCG* env) {
  for (int player_index = 0; player_index < MAX_PLAYERS_PER_MATCH;
       ++player_index) {
    TrainingObservationData* base = obs_base(env, player_index);
    AzkTrainingDeckContextData* dc = obs_deck_context(env, player_index);
    memset(dc, 0, sizeof(*dc));
    dc->mode = 0;
    dc->gate_card_def_id = env->draft_gate[player_index];
    dc->leader_card_def_id = env->draft_leader[player_index];
    for (int i = 0; i < REQUIRED_DECK_SIZE; ++i) {
      dc->main_card_def_ids[i] = env->draft_main[player_index][i];
    }
    dc->main_count = env->draft_main_count[player_index];
    for (int i = 0; i < AZK_DECKBUILD_OBS_MAX_CANDIDATES; ++i) {
      dc->candidate_card_def_ids[i] = -1;
    }

    fill_privileged_deck_lists(env, player_index, base);

    if (base->action_mask.primary_action_mask[3]) {
      fprintf(stderr,
              "Battle action mask unexpectedly exposes DECK_PICK_CARD\n");
      abort();
    }
  }
}

// S4: roll the reference-seat lottery. Returns a deck_pool index or -1.
// Env vars are re-read per episode (mirrors draft_forced_gates); prob 0 /
// unset leaves the RNG stream bit-identical to builds without the knob.
static int draft_sample_ref_deck(CAzukiTCG* env) {
  const char* prob_text = getenv("AZK_DRAFT_REF_SEAT_PROB");
  if (prob_text == NULL || prob_text[0] == '\0') {
    return -1;
  }
  const float prob = strtof(prob_text, NULL);
  if (prob <= 0.0f || env->deck_pool == NULL || env->deck_pool_count == 0) {
    return -1;
  }
  env->draft_rng_state = advance_episode_seed(env->draft_rng_state);
  if (prob < 1.0f &&
      (double)env->draft_rng_state >= (double)prob * 4294967296.0) {
    return -1;
  }
  int16_t allowed[64];
  int allowed_count = 0;
  const char* idx_text = getenv("AZK_DRAFT_REF_DECK_INDICES");
  if (idx_text != NULL && idx_text[0] != '\0') {
    const char* cursor = idx_text;
    while (*cursor != '\0' && allowed_count < 64) {
      char* end = NULL;
      const long value = strtol(cursor, &end, 10);
      if (end == cursor) {
        break;
      }
      if (value >= 0 && (size_t)value < env->deck_pool_count) {
        allowed[allowed_count++] = (int16_t)value;
      }
      cursor = (*end == ',') ? end + 1 : end;
    }
  }
  env->draft_rng_state = advance_episode_seed(env->draft_rng_state);
  if (allowed_count > 0) {
    return allowed[env->draft_rng_state % (uint32_t)allowed_count];
  }
  return (int)(env->draft_rng_state % (uint32_t)env->deck_pool_count);
}

static int16_t draft_spec_gate(const TrainingDeckSpec* spec) {
  for (size_t i = 0; i < spec->card_count; ++i) {
    const CardDef* def = azk_card_def_from_id((CardDefId)spec->cards[i].card_id);
    if (def != NULL && def->type == CARD_TYPE_GATE) {
      return (int16_t)spec->cards[i].card_id;
    }
  }
  return -1;
}

// Maps a pool deck onto a COMPLETED draft state for `seat` (validates before
// mutating: one leader, one catalog gate, exactly REQUIRED_DECK_SIZE mains;
// IKZ entries ignored). draft_copies is filled best-effort for the draft
// observations only — reference decks may contain cards outside the gate's
// draftable pool, so draft_start_battle uses the spec directly for this seat.
static bool draft_prefill_from_spec(CAzukiTCG* env, int seat,
                                    const TrainingDeckSpec* spec) {
  int16_t leader = -1;
  int16_t gate = -1;
  int gate_slot = -1;
  int mains = 0;
  for (size_t i = 0; i < spec->card_count; ++i) {
    const CardDef* def = azk_card_def_from_id((CardDefId)spec->cards[i].card_id);
    if (def == NULL) {
      return false;
    }
    switch (def->type) {
      case CARD_TYPE_LEADER:
        leader = (int16_t)spec->cards[i].card_id;
        break;
      case CARD_TYPE_GATE:
        gate = (int16_t)spec->cards[i].card_id;
        gate_slot = draft_gate_slot_for(gate);
        break;
      case CARD_TYPE_IKZ:
        break;
      default:
        mains += spec->cards[i].card_count;
        break;
    }
  }
  if (leader < 0 || gate < 0 || gate_slot < 0 || mains != REQUIRED_DECK_SIZE) {
    return false;
  }
  env->draft_gate[seat] = gate;
  env->draft_gate_slot[seat] = gate_slot;
  env->draft_leader[seat] = leader;
  env->draft_main_count[seat] = 0;
  memset(env->draft_copies[seat], 0, sizeof(env->draft_copies[seat]));
  const int begin = g_draft_catalog.main_offsets[gate_slot];
  const int end = g_draft_catalog.main_offsets[gate_slot + 1];
  for (size_t i = 0; i < spec->card_count; ++i) {
    const CardDef* def = azk_card_def_from_id((CardDefId)spec->cards[i].card_id);
    const int type = def->type;
    if (type == CARD_TYPE_LEADER || type == CARD_TYPE_GATE ||
        type == CARD_TYPE_IKZ) {
      continue;
    }
    for (int c = 0; c < spec->cards[i].card_count &&
                    env->draft_main_count[seat] < REQUIRED_DECK_SIZE;
         ++c) {
      env->draft_main[seat][env->draft_main_count[seat]] =
          (int16_t)spec->cards[i].card_id;
      env->draft_main_count[seat] += 1;
    }
    for (int j = begin; j < end; ++j) {
      if (g_draft_catalog.main_flat[j] == (int16_t)spec->cards[i].card_id) {
        env->draft_copies[seat][j - begin] = (uint8_t)spec->cards[i].card_count;
        break;
      }
    }
  }
  return true;
}

static void draft_start_battle(CAzukiTCG* env);

static bool draft_select_prebuilt_decks(CAzukiTCG* env) {
  if (env->evaluation_pause_on_done ||
      env->prebuilt_probability == NULL ||
      env->prebuilt_group_count == 0 ||
      env->prebuilt_group_offsets == NULL ||
      env->prebuilt_deck_indices == NULL) {
    return false;
  }
  const float probability = env->prebuilt_probability[0];
  if (!(probability > 0.0f)) {
    return false;
  }
  env->draft_rng_state = advance_episode_seed(env->draft_rng_state);
  if (probability < 1.0f &&
      (double)env->draft_rng_state >=
          (double)probability * 4294967296.0) {
    return false;
  }
  for (int seat = 0; seat < MAX_PLAYERS_PER_MATCH; ++seat) {
    env->draft_rng_state = advance_episode_seed(env->draft_rng_state);
    const size_t group =
        (size_t)(env->draft_rng_state % (uint32_t)env->prebuilt_group_count);
    const size_t begin = env->prebuilt_group_offsets[group];
    const size_t end = env->prebuilt_group_offsets[group + 1];
    env->draft_rng_state = advance_episode_seed(env->draft_rng_state);
    const size_t selected =
        begin + (size_t)(env->draft_rng_state % (uint32_t)(end - begin));
    const int deck_index = env->prebuilt_deck_indices[selected];
    if (deck_index < 0 || (size_t)deck_index >= env->deck_pool_count ||
        !draft_prefill_from_spec(
            env, seat, &env->deck_pool[(size_t)deck_index])) {
      fprintf(stderr, "Invalid validated prebuilt deck index %d\n", deck_index);
      abort();
    }
    env->episode_prebuilt_deck_indices[seat] = deck_index;
    env->current_deck_indices[seat] = deck_index;
  }
  env->episode_prebuilt = true;
  return true;
}

static void draft_begin_episode(CAzukiTCG* env) {
  env->draft_active = true;
  env->draft_active_player = 0;
  env->draft_ref_seat = -1;
  env->draft_ref_deck_index = -1;
  env->episode_prebuilt = false;
  reset_current_deck_indices(env);
  for (int seat = 0; seat < MAX_PLAYERS_PER_MATCH; ++seat) {
    env->episode_prebuilt_deck_indices[seat] = -1;
  }
  env->draft_rng_state = advance_episode_seed(env->draft_rng_state);
  env->episode_world_seed = env->draft_rng_state;
  if (draft_select_prebuilt_decks(env)) {
    for (int seat = 0; seat < MAX_PLAYERS_PER_MATCH; ++seat) {
      env->draft_original_gate[seat] = env->draft_gate[seat];
      env->draft_gate_swapped[seat] = false;
    }
    draft_start_battle(env);
    return;
  }
  int16_t forced[MAX_PLAYERS_PER_MATCH];
  const bool use_forced = draft_forced_gates(env, forced);
  int16_t forced_leaders[MAX_PLAYERS_PER_MATCH];
  const bool use_forced_leaders =
      env->draft_uniform_assignment &&
      draft_forced_leaders(env, forced_leaders);
  int16_t gates[MAX_PLAYERS_PER_MATCH];
  for (int player_index = 0; player_index < MAX_PLAYERS_PER_MATCH;
       ++player_index) {
    if (use_forced) {
      gates[player_index] = forced[player_index];
    } else {
      env->draft_rng_state = advance_episode_seed(env->draft_rng_state);
      if (env->draft_uniform_assignment) {
        gates[player_index] =
            g_draft_catalog.gate_def_ids[env->draft_rng_state %
                                         (uint32_t)g_draft_catalog.gate_count];
      } else {
        gates[player_index] =
            g_draft_catalog.gate_population[env->draft_rng_state %
                                            (uint32_t)g_draft_catalog
                                                .population_count];
      }
    }
  }
  // S4 reference seat: replace one seat's sampled gate with the reference
  // deck's own gate before the sibling roll (so a ref seat at slot 0 can
  // still be sibling-paired against the drafter).
  int ref_deck = -1;
  int ref_seat = -1;
  if (env->evaluation_reference_deck_index >= 0) {
    ref_deck = (int)env->evaluation_reference_deck_index;
    ref_seat = (int)env->evaluation_reference_seat;
  } else if (!use_forced) {
    ref_deck = draft_sample_ref_deck(env);
    if (ref_deck >= 0) {
      env->draft_rng_state = advance_episode_seed(env->draft_rng_state);
      ref_seat = (int)(env->draft_rng_state & 1u);
    }
  }
  if (ref_deck >= 0) {
    const int16_t ref_gate = draft_spec_gate(&env->deck_pool[ref_deck]);
    if (ref_gate >= 0 && draft_gate_slot_for(ref_gate) >= 0) {
      gates[ref_seat] = ref_gate;
    } else {
      ref_deck = -1;
      ref_seat = -1;
    }
  }
  const float sibling_prob = env->draft_same_element_matchup_prob;
  if (!use_forced && sibling_prob > 0.0f && ref_seat != 1) {
    env->draft_rng_state = advance_episode_seed(env->draft_rng_state);
    const bool hit = sibling_prob >= 1.0f ||
                     (double)env->draft_rng_state <
                         (double)sibling_prob * 4294967296.0;
    if (hit) {
      const int slot0 = draft_gate_slot_for(gates[0]);
      const int16_t sibling =
          slot0 >= 0 ? g_draft_catalog.gate_sibling_def_ids[slot0] : -1;
      if (sibling >= 0) {
        gates[1] = sibling;
      }
    }
  }
  for (int player_index = 0; player_index < MAX_PLAYERS_PER_MATCH;
       ++player_index) {
    const int16_t gate = gates[player_index];
    const int slot = draft_gate_slot_for(gate);
    if (slot < 0) {
      fprintf(stderr, "Sampled gate def id %d missing from draft catalog\n",
              (int)gate);
      abort();
    }
    env->draft_gate[player_index] = gate;
    env->draft_gate_slot[player_index] = slot;
    env->draft_leader[player_index] = -1;
    if (env->draft_uniform_assignment && player_index != ref_seat) {
      if (use_forced_leaders) {
        const int16_t leader = forced_leaders[player_index];
        if (!draft_leader_valid_for_slot(slot, leader)) {
          fprintf(stderr,
                  "Forced leader def id %d is incompatible with gate def id %d\n",
                  (int)leader, (int)gate);
          abort();
        }
        env->draft_leader[player_index] = leader;
      } else {
        env->draft_leader[player_index] =
            draft_sample_assigned_leader(env, slot);
      }
    }
    env->draft_main_count[player_index] = 0;
    for (int i = 0; i < REQUIRED_DECK_SIZE; ++i) {
      env->draft_main[player_index][i] = -1;
    }
    memset(env->draft_copies[player_index], 0,
           sizeof(env->draft_copies[player_index]));
  }
  if (ref_deck >= 0 &&
      draft_prefill_from_spec(env, ref_seat, &env->deck_pool[ref_deck])) {
    env->draft_ref_seat = (int8_t)ref_seat;
    env->draft_ref_deck_index = (int16_t)ref_deck;
    env->draft_active_player = (int8_t)(1 - ref_seat);
  }
  for (int player_index = 0; player_index < MAX_PLAYERS_PER_MATCH;
       ++player_index) {
    env->draft_original_gate[player_index] = env->draft_gate[player_index];
    env->draft_gate_swapped[player_index] = false;
  }
  fill_draft_observations(env);
  // Draft observations have zero potential. Start episode accounting here so
  // the final pick can carry the discounted zero->battle potential transition.
  reset_reward_tracking(env);
}

static void draft_apply_pick(CAzukiTCG* env, int player_index, int32_t sub1) {
  int16_t cand_ids[AZK_DRAFT_MAX_CANDIDATES];
  uint8_t cand_copies[AZK_DRAFT_MAX_CANDIDATES];
  const int n = draft_active_candidates(env, player_index, cand_ids,
                                        cand_copies, AZK_DRAFT_MAX_CANDIDATES);
  if (sub1 < 0 || sub1 >= n) {
    fprintf(stderr, "Draft pick index %d out of range (0..%d)\n", (int)sub1,
            n - 1);
    abort();
  }
  const int16_t picked = cand_ids[sub1];
  if (env->draft_leader[player_index] < 0) {
    env->draft_leader[player_index] = picked;
    return;
  }
  // Map the picked def id back to its position in the full per-gate list to
  // bump its copy counter.
  const int slot = env->draft_gate_slot[player_index];
  const int begin = g_draft_catalog.main_offsets[slot];
  const int end = g_draft_catalog.main_offsets[slot + 1];
  int local = -1;
  for (int i = begin; i < end; ++i) {
    if (g_draft_catalog.main_flat[i] == picked) {
      local = i - begin;
      break;
    }
  }
  if (local < 0) {
    fprintf(stderr, "Picked card %d missing from catalog list\n", (int)picked);
    abort();
  }
  env->draft_main[player_index][env->draft_main_count[player_index]] = picked;
  env->draft_main_count[player_index] += 1;
  env->draft_copies[player_index][local] += 1;
}

// Deck spec order matches the wrapper: leader, gate, unique mains ascending
// by card_def_id (catalog order IS ascending), IKZ x10. Returns entry count.
static size_t draft_assemble_deck(const CAzukiTCG* env, int player_index,
                                  CardInfo* out, size_t out_capacity) {
  size_t n = 0;
  out[n].card_id = (CardDefId)env->draft_leader[player_index];
  out[n].card_count = 1;
  n++;
  out[n].card_id = (CardDefId)env->draft_gate[player_index];
  out[n].card_count = 1;
  n++;
  const int slot = env->draft_gate_slot[player_index];
  const int begin = g_draft_catalog.main_offsets[slot];
  const int end = g_draft_catalog.main_offsets[slot + 1];
  for (int i = begin; i < end; ++i) {
    const uint8_t copies = env->draft_copies[player_index][i - begin];
    if (copies == 0) {
      continue;
    }
    if (n >= out_capacity) {
      fprintf(stderr, "Draft deck spec overflow\n");
      abort();
    }
    out[n].card_id = (CardDefId)g_draft_catalog.main_flat[i];
    out[n].card_count = (int)copies;
    n++;
  }
  if (n + 1 > out_capacity) {
    fprintf(stderr, "Draft deck spec overflow (ikz)\n");
    abort();
  }
  out[n].card_id = (CardDefId)g_draft_catalog.ikz_def_id;
  out[n].card_count = REQUIRED_IKZ_PILE_SIZE;
  n++;
  return n;
}

static void draft_start_battle(CAzukiTCG* env) {
  // Supplied decks must remain exact. Ordinary drafted episodes retain the
  // optional sibling-gate counterfactual replay.
  if (!env->episode_prebuilt && env->draft_cross_gate_replay_prob > 0.0f) {
    env->draft_rng_state = advance_episode_seed(env->draft_rng_state);
    const bool hit = env->draft_cross_gate_replay_prob >= 1.0f ||
                     (double)env->draft_rng_state <
                         (double)env->draft_cross_gate_replay_prob * 4294967296.0;
    if (hit) {
      env->draft_rng_state = advance_episode_seed(env->draft_rng_state);
      const int seat = (int)(env->draft_rng_state & 1u);
      if (seat != (int)env->draft_ref_seat) {
        const int slot = env->draft_gate_slot[seat];
        const int16_t sibling =
            slot >= 0 ? g_draft_catalog.gate_sibling_def_ids[slot] : -1;
        if (sibling >= 0) {
          const int sib_slot = draft_gate_slot_for(sibling);
          if (sib_slot >= 0) {
            env->draft_gate[seat] = sibling;
            env->draft_gate_slot[seat] = sib_slot;
            env->draft_gate_swapped[seat] = true;
          }
        }
      }
    }
  }

  CardInfo deck0[2 + REQUIRED_DECK_SIZE + 1];
  CardInfo deck1[2 + REQUIRED_DECK_SIZE + 1];
  const size_t n0 =
      draft_assemble_deck(env, 0, deck0, sizeof(deck0) / sizeof(deck0[0]));
  const size_t n1 =
      draft_assemble_deck(env, 1, deck1, sizeof(deck1) / sizeof(deck1[0]));
  const CardInfo* spec0 = deck0;
  const CardInfo* spec1 = deck1;
  size_t count0 = n0;
  size_t count1 = n1;
  if (env->episode_prebuilt) {
    spec0 = env->deck_pool[env->episode_prebuilt_deck_indices[0]].cards;
    count0 =
        env->deck_pool[env->episode_prebuilt_deck_indices[0]].card_count;
    spec1 = env->deck_pool[env->episode_prebuilt_deck_indices[1]].cards;
    count1 =
        env->deck_pool[env->episode_prebuilt_deck_indices[1]].card_count;
  } else if (env->draft_ref_seat == 0) {
    spec0 = env->deck_pool[env->draft_ref_deck_index].cards;
    count0 = env->deck_pool[env->draft_ref_deck_index].card_count;
  } else if (env->draft_ref_seat == 1) {
    spec1 = env->deck_pool[env->draft_ref_deck_index].cards;
    count1 = env->deck_pool[env->draft_ref_deck_index].card_count;
  }

  const int8_t starting_player = next_starting_player(env);
  azk_engine_destroy(env->engine);
  env->engine = azk_engine_create_with_decks_and_starting_player(
      env->episode_world_seed, starting_player, spec0, count0, spec1, count1);
  if (env->engine == NULL) {
    const char* error_message = azk_engine_get_last_error();
    fprintf(stderr, "Failed to create engine from drafted decks: %s\n",
            error_message != NULL ? error_message : "unknown error");
    abort();
  }
  env->draft_active = false;
  env->tick = 0;
  env->current_episode_cap = current_episode_ticks_limit(env);
  refresh_observations(env);
  if (env->episode_prebuilt) {
    // The supplied game opens directly in battle. Initial potential is the
    // baseline, never a fabricated final-pick transition.
    zero_step_reward_components(env);
    reset_reward_tracking(env);
  } else if (env->proper_pbrs) {
    apply_shaped_rewards(
        env, env->draft_active_player, ACT_NOOP, false, 0.0f, NULL);
    accumulate_step_rewards(env);
  } else {
    adopt_current_potential_without_reward(env);
  }
}

// Fills the per-episode export record consumed by Python for deckbuild
// metrics + snapshots. Behavior order: attack, spell, weapon, portal,
// play_entity, noop, ability_garden_or_leader, ability_alley,
// play_entity_to_garden, play_entity_to_alley, target, response opportunities,
// temporary Charge realized, temporary-ATK damage realized, generated IKZ
// created/converted, entity damage dealt/taken.
static void deckbuild_fill_export_record(CAzukiTCG* env) {
  const GameState* game_state = azk_engine_game_state(env->engine);
  AzkRewardSnapshot snapshot = {0};
  const bool have_snapshot = azk_engine_reward_snapshot(env->engine, &snapshot);

  env->deck_record_seed = env->episode_world_seed;
  env->deck_record_episode_length = (float)env->tick;
  env->deck_record_prebuilt = env->episode_prebuilt;
  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; ++p) {
    env->deck_record_prebuilt_deck_indices[p] =
        env->episode_prebuilt ? env->episode_prebuilt_deck_indices[p] : -1;
    env->deck_record_gate[p] = env->draft_gate[p];
    env->deck_record_original_gate[p] = env->draft_original_gate[p];
    env->deck_record_gate_swapped[p] = env->draft_gate_swapped[p];
    env->deck_record_leader[p] = env->draft_leader[p];
    for (int i = 0; i < REQUIRED_DECK_SIZE; ++i) {
      env->deck_record_main[p][i] = env->draft_main[p][i];
    }
    env->deck_record_win[p] =
        (game_state != NULL && game_state->winner == p) ? 1.0f : 0.0f;
    const float total = (float)env->episode_action_total[p];
    const float inv_total = total > 0.0f ? 1.0f / total : 0.0f;
    env->deck_record_behavior[p][0] =
        (float)env->episode_action_attack[p] * inv_total;
    env->deck_record_behavior[p][1] =
        (float)env->episode_action_play_spell_from_hand[p] * inv_total;
    env->deck_record_behavior[p][2] =
        (float)env->episode_action_attach_weapon_from_hand[p] * inv_total;
    env->deck_record_behavior[p][3] =
        (float)env->episode_action_gate_portal[p] * inv_total;
    env->deck_record_behavior[p][4] =
        (float)env->episode_action_play[p] * inv_total;
    env->deck_record_behavior[p][5] =
        (float)env->episode_action_noop[p] * inv_total;
    env->deck_record_behavior[p][6] =
        (float)env->episode_action_activate_garden_or_leader_ability[p] *
        inv_total;
    env->deck_record_behavior[p][7] =
        (float)env->episode_action_activate_alley_ability[p] * inv_total;
    env->deck_record_behavior[p][8] =
        (float)env->episode_action_play_entity_to_garden[p] * inv_total;
    env->deck_record_behavior[p][9] =
        (float)env->episode_action_play_entity_to_alley[p] * inv_total;
    env->deck_record_behavior[p][10] =
        (float)env->episode_action_target[p] * inv_total;
    env->deck_record_behavior[p][11] =
        (float)env->episode_contextual_response_reserve_opportunities[p];
    env->deck_record_behavior[p][12] =
        (float)env->episode_temporary_charge_realized[p];
    env->deck_record_behavior[p][13] =
        (float)env->episode_temporary_attack_damage_realized[p];
    env->deck_record_behavior[p][14] =
        have_snapshot ? snapshot.generated_ikz_created[p] : 0.0f;
    env->deck_record_behavior[p][15] =
        have_snapshot ? snapshot.generated_ikz_converted[p] : 0.0f;
    env->deck_record_behavior[p][16] =
        have_snapshot ? snapshot.entity_damage_taken[1 - p] : 0.0f;
    env->deck_record_behavior[p][17] =
        have_snapshot ? snapshot.entity_damage_taken[p] : 0.0f;
    env->deck_record_behavior[p][18] =
        have_snapshot ? snapshot.gate_ability_outcomes[p] : 0.0f;
    env->deck_record_behavior[p][19] =
        have_snapshot ? snapshot.leader_ability_outcomes[p] : 0.0f;
    env->deck_record_leader_health[p] =
        have_snapshot ? snapshot.leader_health_ratio[p] : 0.0f;
  }
  env->deck_record_ref_seat = env->draft_ref_seat;
  env->deck_record_ref_deck_index = env->draft_ref_deck_index;
  env->deck_record_valid = true;
}

static void refresh_observations(CAzukiTCG* env) {
  static int refresh_mode = -1;
  if (refresh_mode < 0) {
    const char *mode = getenv("AZK_OBS_REFRESH_MODE");
    if (mode != NULL && strcmp(mode, "legacy") == 0) {
      refresh_mode = 0;
    } else {
      refresh_mode = 1;
    }
  }

  if (env->deck_building) {
    // Deck-building rows have a larger stride; the engine fill functions
    // assume a contiguous TrainingObservationData pair, so fill a local pair
    // and copy into each row's base struct.
    TrainingObservationData pair[MAX_PLAYERS_PER_MATCH];
    if (refresh_mode == 0) {
      for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH;
           ++player_index) {
        if (!azk_engine_observe_training(env->engine, player_index,
                                         &pair[player_index])) {
          fprintf(stderr,
                  "Failed to refresh training observation for player %d\n",
                  player_index);
          abort();
        }
      }
    } else if (!azk_engine_observe_training_all(env->engine, pair)) {
      fprintf(stderr,
              "Failed to refresh training observations for all players\n");
      abort();
    }
    for (int player_index = 0; player_index < MAX_PLAYERS_PER_MATCH;
         ++player_index) {
      *obs_base(env, player_index) = pair[player_index];
    }
    deckbuild_postprocess_battle_observations(env);
    return;
  }

  if (refresh_mode == 0) {
    for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH;
         ++player_index) {
      const bool ok = azk_engine_observe_training(
          env->engine, player_index, &env->observations[player_index]);
      if (!ok) {
        fprintf(stderr,
                "Failed to refresh training observation for player %d\n",
                player_index);
        abort();
      }
    }
    return;
  }

  const bool ok =
      azk_engine_observe_training_all(env->engine, env->observations);
  if (!ok) {
    fprintf(stderr, "Failed to refresh training observations for all players\n");
    abort();
  }
}

void c_reset(CAzukiTCG* env) {
  env->tick = 0;
  env->terminals[0] = NOT_DONE;
  env->terminals[1] = NOT_DONE;
  env->truncations[0] = NOT_DONE;
  env->truncations[1] = NOT_DONE;
  zero_step_reward_components(env);
  env->current_episode_cap = current_episode_ticks_limit(env);

  if (env->deck_building) {
    // New episode = new draft; the battle engine is created once both
    // players finish drafting (draft_start_battle).
    azk_engine_destroy(env->engine);
    env->engine = NULL;
    draft_begin_episode(env);
    return;
  }

  const int8_t starting_player = next_starting_player(env);
  azk_engine_destroy(env->engine);
  env->engine = create_env_engine(env, starting_player);
  refresh_observations(env);
  {
    const int8_t active_player_index = tcg_active_player_index(env);
    if (active_player_index >= 0 &&
        obs_base(env, active_player_index)->action_mask.legal_action_count ==
            0) {
      debug_log_zero_mask_state(env, "reset");
    }
  }
  reset_reward_tracking(env);
}

void c_reset_with_decks(CAzukiTCG* env,
                        const CardInfo *player0_deck,
                        size_t player0_deck_count,
                        const CardInfo *player1_deck,
                        size_t player1_deck_count) {
  const int8_t starting_player = next_starting_player(env);
  env->tick = 0;
  env->terminals[0] = NOT_DONE;
  env->terminals[1] = NOT_DONE;
  env->truncations[0] = NOT_DONE;
  env->truncations[1] = NOT_DONE;
  zero_step_reward_components(env);
  env->current_episode_cap = current_episode_ticks_limit(env);
  reset_current_deck_indices(env);

  azk_engine_destroy(env->engine);
  env->engine = azk_engine_create_with_decks_and_starting_player(
      env->seed, starting_player, player0_deck, player0_deck_count,
      player1_deck, player1_deck_count);
  if (env->engine == NULL) {
    const char *error_message = azk_engine_get_last_error();
    fprintf(stderr,
            "Failed to reset Azuki engine with explicit decks: %s\n",
            error_message != NULL ? error_message : "unknown error");
    abort();
  }
  refresh_observations(env);
  {
    const int8_t active_player_index = tcg_active_player_index(env);
    if (active_player_index >= 0 &&
        obs_base(env, active_player_index)->action_mask.legal_action_count ==
            0) {
      debug_log_zero_mask_state(env, "reset_with_decks");
    }
  }
  reset_reward_tracking(env);
}

// One draft step: the active drafter's pick is validated against the fresh
// candidate list and applied; turn order strictly alternates to the other
// player while they are incomplete. Draft potential is zero and draft steps
// do not consume the battle tick budget. The final pick creates the engine and
// emits the zero->initial-battle PBRS transition with the first observation.
static void c_step_draft(CAzukiTCG* env) {
  zero_step_reward_components(env);
  const int player_index = env->draft_active_player;
  const ActionVector action = env->actions[player_index];
  if (action.type != 3 || action.subaction_2 != 0 || action.subaction_3 != 0) {
    fprintf(stderr,
            "Invalid draft action [%d,%d,%d,%d] for player %d "
            "(expected DECK_PICK_CARD with sub2=sub3=0)\n",
            action.type, action.subaction_1, action.subaction_2,
            action.subaction_3, player_index);
    abort();
  }
  draft_apply_pick(env, player_index, action.subaction_1);

  const int other = 1 - player_index;
  if (!draft_player_complete(env, other)) {
    env->draft_active_player = (int8_t)other;
  } else if (!draft_player_complete(env, player_index)) {
    env->draft_active_player = (int8_t)player_index;
  } else {
    draft_start_battle(env);
    return;
  }
  fill_draft_observations(env);
  if (env->proper_pbrs) {
    record_zero_pbrs_step(env);
  }
}

void c_step(CAzukiTCG* env) {
  if (env->deck_building && env->draft_active) {
    c_step_draft(env);
    return;
  }

  init_env_profile_if_needed();
  const uint64_t step_start_ns =
      g_env_profile.enabled ? env_now_ns() : 0;
  uint64_t tick_total_ns = 0;
  uint64_t refresh_total_ns = 0;
  uint64_t auto_tick_count = 0;

  env->tick++;
  zero_step_reward_components(env);

  const int8_t active_player_index = tcg_active_player_index(env);
  if (active_player_index < 0) {
    fprintf(stderr, "No active player available when stepping environment\n");
    abort();
  }

  const TrainingActionMaskObs *current_action_mask =
      &obs_base(env, active_player_index)->action_mask;
  if (current_action_mask->legal_action_count == 0) {
    debug_log_zero_mask_state(env, "step");
    fprintf(
        stderr,
        "Zero legal actions detected at tick %d for active player %d; "
        "deck_indices=[%d,%d], forcing truncation\n",
        env->tick,
        active_player_index,
        env->current_deck_indices[0],
        env->current_deck_indices[1]);
    apply_truncation_rewards(env, EP_END_REASON_ZERO_LEGAL_ACTION_TRUNCATION);
    accumulate_step_rewards(env);
    env->truncations[0] = DONE;
    env->truncations[1] = DONE;
    record_episode_stats(env, EP_END_REASON_ZERO_LEGAL_ACTION_TRUNCATION);
    if (g_env_profile.enabled) {
      const uint64_t step_elapsed_ns = env_now_ns() - step_start_ns;
      g_env_profile.step_calls++;
      g_env_profile.total_step_ns += step_elapsed_ns;
      maybe_report_env_profile();
    }
    return;
  }

  const ActionVector action = env->actions[active_player_index];
  const int values[AZK_USER_ACTION_VALUE_COUNT] = {
    action.type,
    action.subaction_1,
    action.subaction_2,
    action.subaction_3
  };

  UserAction parsed_action;
  if (!azk_engine_parse_action_values(env->engine, values, &parsed_action)) {
    fprintf(
      stderr,
      "Invalid action encoding: [%d, %d, %d, %d]\n",
      action.type,
      action.subaction_1,
      action.subaction_2,
      action.subaction_3
    );
    abort();
  }

  if (parsed_action.type == ACT_ATTACK &&
      (g_reward_tuning.temporary_charge_realization_bonus > 0.0f ||
       g_reward_tuning.temporary_attack_realization_per_damage > 0.0f)) {
    AzkAttackRewardContext attack_context = {0};
    const bool captured = azk_engine_attack_reward_context(
        env->engine, &parsed_action, &attack_context);
    env->pending_attack_reward = attack_context;
    env->has_pending_attack_reward = captured && attack_context.valid;
  }

  if (env->reward_telemetry.enabled) {
    env->reward_telemetry.step_action_type = (int8_t)parsed_action.type;
  }
  const bool is_valid = azk_engine_submit_action(env->engine, &parsed_action);
  if (!is_valid) {
    fprintf(
      stderr,
      "Rejected action: [%d, %d, %d, %d]\n",
      action.type,
      action.subaction_1,
      action.subaction_2,
      action.subaction_3
    );
    abort();
  }
  record_action_choice(env, active_player_index, parsed_action.type);
  if (parsed_action.type == ACT_DECLARE_DEFENDER) {
    env->pending_interception[active_player_index] = true;
  }
  const bool noop_had_alternatives =
      (parsed_action.type == ACT_NOOP) &&
      (current_action_mask->legal_action_count > 1);

  // S12 early-tempo bonus: flat credit per qualifying development action in
  // the actor's first N turns (global turn_number <= 2N), capped per turn.
  float early_tempo_bonus = 0.0f;
  if (g_reward_tuning.early_tempo_bonus > 0.0f &&
      early_tempo_qualifying_action(parsed_action.type) &&
      (!g_reward_tuning.early_tempo_dedup_portal_abilities ||
       !early_tempo_dedup_excluded_action(parsed_action.type))) {
    const GameState* tempo_gs = azk_engine_game_state(env->engine);
    if (tempo_gs != NULL &&
        (int)tempo_gs->turn_number <= 2 * g_reward_tuning.early_tempo_turns) {
      if (env->tempo_last_turn[active_player_index] != tempo_gs->turn_number) {
        env->tempo_last_turn[active_player_index] = tempo_gs->turn_number;
        env->tempo_turn_count[active_player_index] = 0;
      }
      const int cap = g_reward_tuning.early_tempo_cap;
      if (cap <= 0 || env->tempo_turn_count[active_player_index] < cap) {
        env->tempo_turn_count[active_player_index]++;
        early_tempo_bonus = g_reward_tuning.early_tempo_bonus;
      }
    }
  }

  // S13-DMG: snapshot pre-action combat result; a post-tick change with
  // defender_intercepted set means an intercepted combat resolved during
  // this (the responder's) step.
  LastCombatResult pre_combat = {0};
  bool have_pre_combat = false;
  if (g_reward_tuning.dmg_mitigation_bonus > 0.0f ||
      g_reward_tuning.temporary_charge_realization_bonus > 0.0f ||
      g_reward_tuning.temporary_attack_realization_per_damage > 0.0f) {
    const GameState* pre_gs = azk_engine_game_state(env->engine);
    if (pre_gs != NULL) {
      pre_combat = pre_gs->last_combat;
      have_pre_combat = true;
    }
  }

  // Portal-GP bonus: read the portaled entity from the pre-action alley obs
  // (sub1 = alley slot); the tick below moves it. Outcome-graded mode (S2)
  // additionally snapshots pre-action state and only pays if the gate
  // ability resolves (graded after the tick loop, post refresh).
  float portal_gp_bonus = 0.0f;
  bool portal_outcome_pending = false;
  PortalOutcomeSnapshot portal_pre = {0};
  const bool outcome_mode = g_reward_tuning.portal_outcome_bonus > 0.0f;
  if (parsed_action.type == ACT_GATE_PORTAL &&
      (outcome_mode || g_reward_tuning.portal_gp_bonus > 0.0f)) {
    const int alley_slot = (int)parsed_action.subaction_1;
    if (alley_slot >= 0 && alley_slot < ALLEY_SIZE) {
      const TrainingObservationData* pre_base = obs_base(env, active_player_index);
      const TrainingBoardCardObservationData* portaled =
          &pre_base->my_observation_data.alley[alley_slot];
      if (portaled->card_def_id >= 0) {
        const CardDef* def =
            azk_card_def_from_id((CardDefId)portaled->card_def_id);
        if (def != NULL && def->has_gate_points) {
          uint8_t gp = def->gate_points.gate_points;
          if (gp > 4) {
            gp = 4;
          }
          const float weight = outcome_mode
                                   ? g_reward_tuning.portal_outcome_bonus
                                   : g_reward_tuning.portal_gp_bonus;
          portal_gp_bonus = weight * (float)gp / 4.0f;
          if (outcome_mode && portal_gp_bonus > 0.0f) {
            portal_outcome_capture(pre_base, (int)portaled->cur_stats.cur_atk,
                                   &portal_pre);
            portal_outcome_pending = true;
          }
        }
      }
    }
  }

  // Some sub-actions do not require a user action
  // We should progress those until a user action is required (or the game ends)
  bool forced_auto_tick_truncation = false;
  const int max_auto_ticks_per_step = max_auto_ticks_per_step_limit();
  do {
    const uint64_t tick_start_ns =
        g_env_profile.enabled ? env_now_ns() : 0;
    azk_engine_tick(env->engine);
    auto_tick_count++;
    if (g_env_profile.enabled) {
      tick_total_ns += env_now_ns() - tick_start_ns;
    }

    if (max_auto_ticks_per_step > 0 &&
        auto_tick_count >= (uint64_t)max_auto_ticks_per_step &&
        !azk_engine_requires_action(env->engine) &&
        !azk_engine_is_game_over(env->engine)) {
      forced_auto_tick_truncation = true;
      fprintf(stderr,
              "Auto-tick guard hit at tick=%d (auto_ticks=%" PRIu64
              "); forcing truncation\n",
              env->tick, auto_tick_count);
      break;
    }

    if (azk_engine_was_prev_action_invalid(env->engine)) {
        const TrainingActionMaskObs *active_action_mask =
            &obs_base(env, active_player_index)->action_mask;
        bool action_in_mask = false;
        for (uint16_t i = 0; i < active_action_mask->legal_action_count; ++i) {
          if (active_action_mask->legal_primary[i] == (uint8_t)action.type &&
              active_action_mask->legal_sub1[i] ==
                  (uint8_t)action.subaction_1 &&
              active_action_mask->legal_sub2[i] ==
                  (uint8_t)action.subaction_2 &&
              active_action_mask->legal_sub3[i] ==
                  (uint8_t)action.subaction_3) {
            action_in_mask = true;
            break;
          }
        }

        const GameState *debug_gs = azk_engine_game_state(env->engine);
        AbilityPhase debug_ability_phase = azk_engine_get_ability_phase(env->engine);
        const TrainingAbilityContextObservationData *ability_context =
            &obs_base(env, active_player_index)->ability_context;
        const int ability_ctx_source_card_def_id =
            ability_context->has_source_card_def_id
                ? (int)ability_context->source_card_def_id
                : -1;
        bool action_in_fresh_mask = false;
        uint16_t fresh_legal_action_count = 0;
        if (debug_gs != NULL) {
          AzkActionMaskSet fresh_mask = {0};
          bool fresh_ok = azk_build_action_mask_for_player(
              env->engine, debug_gs, active_player_index, &fresh_mask);
          if (fresh_ok) {
            fresh_legal_action_count = fresh_mask.legal_action_count;
            for (uint16_t i = 0; i < fresh_mask.legal_action_count; ++i) {
              const UserAction *fresh = &fresh_mask.legal_actions[i];
              if (fresh->type == action.type &&
                  fresh->subaction_1 == action.subaction_1 &&
                  fresh->subaction_2 == action.subaction_2 &&
                  fresh->subaction_3 == action.subaction_3) {
                action_in_fresh_mask = true;
                break;
              }
            }
          }
        }

        fprintf(
          stderr,
          "Invalid action detected at tick %d in phase %d for active player %d: "
          "deck_indices=[%d,%d], "
          "[%d, %d, %d, %d], action_in_mask=%d, legal_action_count=%u, "
          "action_in_fresh_mask=%d, fresh_legal_action_count=%u, "
          "ability_phase=%d, ability_ctx_phase=%d, "
          "ability_ctx_source_card_def_id=%d, ability_ctx_effect_target_type=%u, "
          "ability_ctx_cost_target_type=%u\n",
          env->tick,
          (int)obs_base(env, active_player_index)->phase,
          active_player_index,
          env->current_deck_indices[0],
          env->current_deck_indices[1],
          action.type,
          action.subaction_1,
          action.subaction_2,
          action.subaction_3,
          action_in_mask ? 1 : 0,
          active_action_mask->legal_action_count,
          action_in_fresh_mask ? 1 : 0,
          fresh_legal_action_count,
          (int)debug_ability_phase,
          (int)ability_context->phase,
          ability_ctx_source_card_def_id,
          (unsigned)ability_context->effect_target_type,
          (unsigned)ability_context->cost_target_type
        ); 

        if (!action_in_mask) {
          const uint16_t debug_limit =
              active_action_mask->legal_action_count < 24
                  ? active_action_mask->legal_action_count
                  : 24;
          for (uint16_t i = 0; i < debug_limit; ++i) {
            fprintf(
                stderr,
                "  legal[%u]=[%u,%u,%u,%u]\n",
                i,
                active_action_mask->legal_primary[i],
                active_action_mask->legal_sub1[i],
                active_action_mask->legal_sub2[i],
                active_action_mask->legal_sub3[i]);
          }
        }

        const TrainingMyObservationData* my_observation_data =
            &obs_base(env, active_player_index)->my_observation_data;
        int hand_card_count = 0;
        for (int i = 0; i < MAX_HAND_SIZE; ++i) {
          if (my_observation_data->hand[i].card_def_id >= 0) {
            hand_card_count++;
          }
        }

        int untapped_ikz_card_count = 0;
        for (int i = 0; i < IKZ_AREA_SIZE; ++i) {
          const TrainingIKZCardObservationData* ikz_card =
              &my_observation_data->ikz_area[i];
          if (ikz_card->card_def_id >= 0 && !ikz_card->tap_state.tapped) {
            untapped_ikz_card_count++;
          }
        }

        bool occupied_garden_zones[GARDEN_SIZE] = {false};
        int occupied_garden_slot_count = 0;
        for (int i = 0; i < GARDEN_SIZE; ++i) {
          if (my_observation_data->garden[i].card_def_id >= 0) {
            occupied_garden_zones[i] = true;
            occupied_garden_slot_count++;
          }
        }

        fprintf(
            stderr,
            "Observation data, hand cards %d, occupied garden slots %d/%d "
            "[%d, %d, %d, %d, %d], untapped ikz cards %d\n",
            hand_card_count, occupied_garden_slot_count, GARDEN_SIZE,
            occupied_garden_zones[0], occupied_garden_zones[1],
            occupied_garden_zones[2], occupied_garden_zones[3],
            occupied_garden_zones[4], untapped_ikz_card_count);

        // Repro handle for the underlying engine/mask desync bug.
        fprintf(stderr,
                "Invalid-action truncation: episode_seed=%u gates=[%d,%d] "
                "tick=%d\n",
                env->episode_world_seed, (int)env->draft_gate[0],
                (int)env->draft_gate[1], env->tick);
        fflush(stderr);
        // An abort() here turns one bad episode into a dead worker and a
        // deadlocked vecenv (combo45 hung 2h at 8.8M steps on one hit).
        // Default: truncate the episode like the zero-legal-action guard and
        // keep training; opt back into aborting for parity/debug work.
        const char* abort_raw = getenv("AZK_INVALID_ACTION_ABORT");
        if (abort_raw != NULL && abort_raw[0] == '1') {
          abort();
        }
        refresh_observations(env);
        apply_truncation_rewards(env, EP_END_REASON_ZERO_LEGAL_ACTION_TRUNCATION);
        accumulate_step_rewards(env);
        env->truncations[0] = DONE;
        env->truncations[1] = DONE;
        record_episode_stats(env, EP_END_REASON_ZERO_LEGAL_ACTION_TRUNCATION);
        if (g_env_profile.enabled) {
          const uint64_t step_elapsed_ns = env_now_ns() - step_start_ns;
          g_env_profile.step_calls++;
          g_env_profile.total_step_ns += step_elapsed_ns;
          maybe_report_env_profile();
        }
        return;
    }
  } while (!azk_engine_requires_action(env->engine) && !azk_engine_is_game_over(env->engine));

  if (azk_engine_is_game_over(env->engine)) {
    env->terminals[0] = DONE;
    env->terminals[1] = DONE;
  }

  const uint64_t refresh_start_ns =
      g_env_profile.enabled ? env_now_ns() : 0;
  refresh_observations(env);
  if (g_env_profile.enabled) {
    refresh_total_ns += env_now_ns() - refresh_start_ns;
  }

  if (forced_auto_tick_truncation) {
    apply_shaped_rewards(
        env, active_player_index, parsed_action.type, noop_had_alternatives,
        0.0f, NULL);
    accumulate_step_rewards(env);
    env->truncations[0] = DONE;
    env->truncations[1] = DONE;
    record_episode_stats(env, EP_END_REASON_AUTO_TICK_TRUNCATION);
    if (g_env_profile.enabled) {
      const uint64_t step_elapsed_ns = env_now_ns() - step_start_ns;
      g_env_profile.step_calls++;
      g_env_profile.total_step_ns += step_elapsed_ns;
      g_env_profile.total_tick_ns += tick_total_ns;
      g_env_profile.total_refresh_ns += refresh_total_ns;
      g_env_profile.total_auto_ticks += auto_tick_count;
      maybe_report_env_profile();
    }
    return;
  }

  float contextual_response_reserve_adjustment = 0.0f;
  const GameState* reward_post_gs = azk_engine_game_state(env->engine);
  if (g_reward_tuning.contextual_response_reserve_bonus > 0.0f &&
      parsed_action.type == ACT_ATTACK && reward_post_gs != NULL &&
      reward_post_gs->phase == PHASE_RESPONSE_WINDOW) {
    const int8_t defender_index = reward_post_gs->active_player_index;
    if (defender_index >= 0 && defender_index < MAX_PLAYERS_PER_MATCH &&
        env->response_reserve_last_rewarded_turn[defender_index] !=
            reward_post_gs->turn_number &&
        refreshed_mask_has_ikz_response(env, defender_index)) {
      env->response_reserve_last_rewarded_turn[defender_index] =
          reward_post_gs->turn_number;
      env->episode_contextual_response_reserve_opportunities[defender_index]++;
      contextual_response_reserve_adjustment =
          defender_index == active_player_index
              ? g_reward_tuning.contextual_response_reserve_bonus
              : -g_reward_tuning.contextual_response_reserve_bonus;
    }
  }

  if (azk_engine_is_game_over(env->engine)) {
    apply_terminal_rewards(env);
    accumulate_step_rewards(env);
    record_episode_stats(env, EP_END_REASON_GAMEOVER);
    if (g_env_profile.enabled) {
      const uint64_t step_elapsed_ns = env_now_ns() - step_start_ns;
      g_env_profile.step_calls++;
      g_env_profile.total_step_ns += step_elapsed_ns;
      g_env_profile.total_tick_ns += tick_total_ns;
      g_env_profile.total_refresh_ns += refresh_total_ns;
      g_env_profile.total_auto_ticks += auto_tick_count;
      maybe_report_env_profile();
    }
    return;
  }

  // S2: grade the pending portal bonus against the realized post-tick state
  // (observations were refreshed above); whiffed abilities pay nothing.
  if (portal_outcome_pending) {
    PortalOutcomeSnapshot portal_post;
    portal_outcome_capture(obs_base(env, active_player_index),
                           portal_pre.portaled_atk, &portal_post);
    if (!portal_outcome_resolved(&portal_pre, &portal_post)) {
      portal_gp_bonus = 0.0f;
    }
  }

  // S13-DMG: an intercepted combat resolved during this step — credit the
  // responder (the acting player) with the realized soak.
  float dmg_mitigation_bonus = 0.0f;
  float temporary_effect_adjustment = 0.0f;
  float temporary_charge_adjustment = 0.0f;
  float temporary_attack_adjustment = 0.0f;
  if (have_pre_combat) {
    const GameState* post_gs = azk_engine_game_state(env->engine);
    if (post_gs != NULL &&
        memcmp(&post_gs->last_combat, &pre_combat, sizeof(LastCombatResult)) != 0) {
      if (env->pending_interception[active_player_index]) {
        int soak = (int)post_gs->last_combat.damage_to_defender;
        if (soak > 0) {
          const int cap = g_reward_tuning.dmg_mitigation_cap > 0
                              ? g_reward_tuning.dmg_mitigation_cap
                              : 10;
          if (soak > cap) {
            soak = cap;
          }
          dmg_mitigation_bonus =
              g_reward_tuning.dmg_mitigation_bonus * (float)soak / (float)cap;
        }
      }
      env->pending_interception[0] = false;
      env->pending_interception[1] = false;

      if (env->has_pending_attack_reward &&
          post_gs->last_combat.attacker ==
              env->pending_attack_reward.attacker) {
        const int owner_index = env->pending_attack_reward.player_index;
        int effective_damage = (int)post_gs->last_combat.damage_to_defender;
        const int defender_hp_before =
            (int)post_gs->last_combat.defender_hp_before;
        if (effective_damage < 0) {
          effective_damage = 0;
        }
        if (effective_damage > defender_hp_before) {
          effective_damage = defender_hp_before;
        }

        float owner_bonus = 0.0f;
        float owner_charge_bonus = 0.0f;
        float owner_attack_bonus = 0.0f;
        if (effective_damage > 0 &&
            env->pending_attack_reward.temporary_charge) {
          owner_charge_bonus =
              g_reward_tuning.temporary_charge_realization_bonus;
          owner_bonus += owner_charge_bonus;
          if (owner_index >= 0 && owner_index < MAX_PLAYERS_PER_MATCH) {
            env->episode_temporary_charge_realized[owner_index]++;
          }
        }

        int incremental_damage = 0;
        const int temporary_attack_bonus =
            (int)env->pending_attack_reward.positive_eot_attack_bonus;
        if (effective_damage > 0 && temporary_attack_bonus > 0) {
          int damage_without_bonus =
              (int)post_gs->last_combat.damage_to_defender -
              temporary_attack_bonus;
          if (damage_without_bonus < 0) {
            damage_without_bonus = 0;
          }
          if (damage_without_bonus > defender_hp_before) {
            damage_without_bonus = defender_hp_before;
          }
          incremental_damage = effective_damage - damage_without_bonus;
          if (incremental_damage < 0) {
            incremental_damage = 0;
          }
          const int damage_cap =
              g_reward_tuning.temporary_attack_realization_damage_cap;
          if (damage_cap > 0 && incremental_damage > damage_cap) {
            incremental_damage = damage_cap;
          }
          owner_attack_bonus =
              g_reward_tuning.temporary_attack_realization_per_damage *
              (float)incremental_damage;
          owner_bonus += owner_attack_bonus;
          if (owner_index >= 0 && owner_index < MAX_PLAYERS_PER_MATCH) {
            env->episode_temporary_attack_damage_realized[owner_index] +=
                (uint32_t)incremental_damage;
          }
        }

        temporary_effect_adjustment =
            owner_index == active_player_index ? owner_bonus : -owner_bonus;
        temporary_charge_adjustment =
            owner_index == active_player_index
                ? owner_charge_bonus
                : -owner_charge_bonus;
        temporary_attack_adjustment =
            owner_index == active_player_index
                ? owner_attack_bonus
                : -owner_attack_bonus;
        env->pending_attack_reward = (AzkAttackRewardContext){0};
        env->has_pending_attack_reward = false;
      }
    }
    if (post_gs != NULL && env->has_pending_attack_reward &&
        post_gs->combat_state.attacking_card == 0 &&
        post_gs->phase != PHASE_RESPONSE_WINDOW &&
        post_gs->phase != PHASE_COMBAT_RESOLVE) {
      env->pending_attack_reward = (AzkAttackRewardContext){0};
      env->has_pending_attack_reward = false;
    }
  }

  const int max_ticks = current_episode_ticks_limit(env);
  env->current_episode_cap = max_ticks;
  const bool timeout_truncation =
      max_ticks > 0 && env->tick >= max_ticks;

  const AzkActionRewardComponents reward_components = {
      .portal_gp = outcome_mode ? 0.0f : portal_gp_bonus,
      .portal_outcome = outcome_mode ? portal_gp_bonus : 0.0f,
      .early_tempo = early_tempo_bonus,
      .damage_mitigation = dmg_mitigation_bonus,
      .temporary_charge = temporary_charge_adjustment,
      .temporary_attack = temporary_attack_adjustment,
      .response_reserve = contextual_response_reserve_adjustment,
  };
  apply_shaped_rewards(
      env, active_player_index, parsed_action.type, noop_had_alternatives,
      portal_gp_bonus + early_tempo_bonus + dmg_mitigation_bonus +
          temporary_effect_adjustment +
          contextual_response_reserve_adjustment,
      &reward_components);
  accumulate_step_rewards(env);
  if (timeout_truncation) {
    env->truncations[0] = DONE;
    env->truncations[1] = DONE;
    record_episode_stats(env, EP_END_REASON_TIMEOUT_TRUNCATION);
  }
  if (g_env_profile.enabled) {
    const uint64_t step_elapsed_ns = env_now_ns() - step_start_ns;
    g_env_profile.step_calls++;
    g_env_profile.total_step_ns += step_elapsed_ns;
    g_env_profile.total_tick_ns += tick_total_ns;
    g_env_profile.total_refresh_ns += refresh_total_ns;
    g_env_profile.total_auto_ticks += auto_tick_count;
    maybe_report_env_profile();
  }
}

void c_close(CAzukiTCG* env) {
  azk_engine_destroy(env->engine);
  free_training_deck_pool(env);
}

typedef struct {
  char *data;
  size_t len;
  size_t cap;
} RenderBuffer;

static bool renderbuf_reserve(RenderBuffer *buf, size_t extra) {
  const size_t needed = buf->len + extra + 1; // +1 for null terminator
  if (needed <= buf->cap) {
    return true;
  }
  size_t new_cap = buf->cap ? buf->cap * 2 : 1024;
  if (new_cap < needed) {
    new_cap = needed;
  }
  char *new_data = (char *)realloc(buf->data, new_cap);
  if (!new_data) {
    return false;
  }
  buf->data = new_data;
  buf->cap = new_cap;
  return true;
}

static bool renderbuf_appendf(RenderBuffer *buf, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  va_list args_copy;
  va_copy(args_copy, args);
  const int needed = vsnprintf(NULL, 0, fmt, args_copy);
  va_end(args_copy);
  if (needed < 0) {
    va_end(args);
    return false;
  }
  if (!renderbuf_reserve(buf, (size_t)needed)) {
    va_end(args);
    return false;
  }
  vsnprintf(buf->data + buf->len, buf->cap - buf->len, fmt, args);
  buf->len += (size_t)needed;
  va_end(args);
  return true;
}

static const char *phase_to_string(Phase phase) {
  switch (phase) {
    case PHASE_PREGAME_MULLIGAN:
      return "Pregame Mulligan";
    case PHASE_START_OF_TURN:
      return "Start of Turn";
    case PHASE_MAIN:
      return "Main";
    case PHASE_RESPONSE_WINDOW:
      return "Response Window";
    case PHASE_COMBAT_RESOLVE:
      return "Combat Resolve";
    case PHASE_END_TURN_ACTION:
      return "End Turn Action";
    case PHASE_END_TURN:
      return "End Turn";
    case PHASE_END_MATCH:
      return "End Match";
    default:
      return "Unknown";
  }
}

#define CARD_BOX_WIDTH 18
#define CARD_BOX_HEIGHT 6
#define CARD_BOX_TEXT_WIDTH (CARD_BOX_WIDTH - 2)
#define CARD_BOX_TEXT_CAPACITY (CARD_BOX_TEXT_WIDTH + 1)
#define BOARD_DEFAULT_TOTAL_WIDTH 120
#define BOARD_SECTION_INDENT 2
#define BOARD_CONTENT_INDENT 2

typedef struct {
  char lines[CARD_BOX_HEIGHT][CARD_BOX_WIDTH + 1];
} BoxLines;

typedef struct {
  char **lines;
  size_t count;
  size_t cap;
  size_t width;
} ColumnRender;

static size_t min_board_column_width(void) {
  // Enough room for labels and at least one card box with indent.
  return (size_t)(CARD_BOX_WIDTH + BOARD_SECTION_INDENT + BOARD_CONTENT_INDENT + 4);
}

static const char *card_type_to_string(CardType type) {
  switch (type) {
    case CARD_TYPE_LEADER:
      return "Leader";
    case CARD_TYPE_GATE:
      return "Gate";
    case CARD_TYPE_ENTITY:
      return "Entity";
    case CARD_TYPE_WEAPON:
      return "Weapon";
    case CARD_TYPE_SPELL:
      return "Spell";
    case CARD_TYPE_IKZ:
      return "IKZ";
    case CARD_TYPE_EXTRA_IKZ:
      return "Extra IKZ";
    default:
      return "Unknown";
  }
}

static const char *card_code_or_placeholder(const CardId *id) {
  if (!id || !id->code || id->code[0] == '\0') {
    return "<?>"; 
  }
  return id->code;
}

static char render_tap_char(const TapState *tap_state) {
  if (!tap_state) {
    return '-';
  }
  return tap_state->tapped ? 'T' : 'U';
}

static char render_cooldown_char(const TapState *tap_state) {
  if (!tap_state) {
    return '-';
  }
  return tap_state->cooldown ? 'C' : 'R';
}

static size_t card_observation_count(const CardObservationData *cards, size_t max_count) {
  size_t count = 0;
  for (size_t i = 0; i < max_count; ++i) {
    if (cards[i].id.code) {
      count++;
    }
  }
  return count;
}

static size_t ikz_observation_count(const IKZCardObservationData *cards, size_t max_count) {
  size_t count = 0;
  for (size_t i = 0; i < max_count; ++i) {
    if (cards[i].id.code) {
      count++;
    }
  }
  return count;
}

static void init_box_lines(BoxLines *box) {
  if (!box) {
    return;
  }
  for (int row = 0; row < CARD_BOX_HEIGHT; ++row) {
    for (int col = 0; col < CARD_BOX_WIDTH; ++col) {
      box->lines[row][col] = ' ';
    }
    box->lines[row][CARD_BOX_WIDTH] = '\0';
  }

  for (int col = 0; col < CARD_BOX_WIDTH; ++col) {
    box->lines[0][col] = (col == 0) ? '+' : (col == CARD_BOX_WIDTH - 1) ? '+' : '-';
    box->lines[CARD_BOX_HEIGHT - 1][col] = (col == 0) ? '+' : (col == CARD_BOX_WIDTH - 1) ? '+' : '-';
  }

  for (int row = 1; row < CARD_BOX_HEIGHT - 1; ++row) {
    box->lines[row][0] = '|';
    box->lines[row][CARD_BOX_WIDTH - 1] = '|';
  }
}

static void set_box_text(BoxLines *box, int inner_row, const char *text) {
  if (!box || inner_row < 0 || inner_row >= CARD_BOX_HEIGHT - 2) {
    return;
  }
  char buffer[CARD_BOX_TEXT_CAPACITY];
  if (text) {
    snprintf(buffer, sizeof buffer, "%s", text);
  } else {
    buffer[0] = '\0';
  }
  size_t copy_len = strnlen(buffer, CARD_BOX_TEXT_WIDTH);
  memcpy(&box->lines[inner_row + 1][1], buffer, copy_len);
}

static void build_standard_card_box(const CardObservationData *card, BoxLines *box) {
  init_box_lines(box);
  if (!card) {
    return;
  }

  char line[CARD_BOX_TEXT_CAPACITY];
  snprintf(line, sizeof line, "%s", card_code_or_placeholder(&card->id));
  set_box_text(box, 0, line);

  snprintf(line, sizeof line, "Type: %s", card_type_to_string(card->type.value));
  set_box_text(box, 1, line);

  if (card->has_cur_stats) {
    snprintf(line, sizeof line, "ATK:%d HP:%d", card->cur_stats.cur_atk, card->cur_stats.cur_hp);
  } else if (card->has_gate_points) {
    snprintf(line, sizeof line, "GP: %u", card->gate_points.gate_points);
  } else if (card->ikz_cost.ikz_cost) {
    snprintf(line, sizeof line, "IKZ Cost: %d", card->ikz_cost.ikz_cost);
  } else {
    snprintf(line, sizeof line, "Stats: --");
  }
  set_box_text(box, 2, line);

  const char tap = render_tap_char(&card->tap_state);
  const char cooldown = render_cooldown_char(&card->tap_state);
  if (card->ikz_cost.ikz_cost) {
    snprintf(line, sizeof line, "T/C:%c/%c IKZ:%d", tap, cooldown, card->ikz_cost.ikz_cost);
  } else {
    snprintf(line, sizeof line, "T/C:%c/%c", tap, cooldown);
  }
  set_box_text(box, 3, line);
}

static void build_ikz_card_box(const IKZCardObservationData *card, BoxLines *box) {
  init_box_lines(box);
  if (!card) {
    return;
  }

  char line[CARD_BOX_TEXT_CAPACITY];
  snprintf(line, sizeof line, "%s", card_code_or_placeholder(&card->id));
  set_box_text(box, 0, line);

  snprintf(line, sizeof line, "Type: %s", card_type_to_string(card->type.value));
  set_box_text(box, 1, line);

  const char tap = render_tap_char(&card->tap_state);
  const char cooldown = render_cooldown_char(&card->tap_state);
  snprintf(line, sizeof line, "T/C:%c/%c", tap, cooldown);
  set_box_text(box, 2, line);
}

static bool column_render_reserve(ColumnRender *col, size_t extra) {
  if (!col) {
    return false;
  }
  size_t needed = col->count + extra;
  if (needed <= col->cap) {
    return true;
  }
  size_t new_cap = col->cap ? col->cap * 2 : 32;
  while (new_cap < needed) {
    new_cap *= 2;
  }
  char **new_lines = (char **)realloc(col->lines, new_cap * sizeof *new_lines);
  if (!new_lines) {
    return false;
  }
  col->lines = new_lines;
  col->cap = new_cap;
  return true;
}

static void column_render_init(ColumnRender *col, size_t width) {
  if (!col) {
    return;
  }
  col->lines = NULL;
  col->count = 0;
  col->cap = 0;
  col->width = width;
}

static void column_render_free(ColumnRender *col) {
  if (!col) {
    return;
  }
  for (size_t i = 0; i < col->count; ++i) {
    free(col->lines[i]);
  }
  free(col->lines);
  col->lines = NULL;
  col->count = 0;
  col->cap = 0;
  col->width = 0;
}

static bool column_render_push_line(ColumnRender *col, const char *text) {
  if (!col) {
    return false;
  }
  if (!column_render_reserve(col, 1)) {
    return false;
  }
  char *line = (char *)malloc(col->width + 1);
  if (!line) {
    return false;
  }
  memset(line, ' ', col->width);
  if (text) {
    size_t len = strnlen(text, col->width);
    memcpy(line, text, len);
  }
  line[col->width] = '\0';
  col->lines[col->count++] = line;
  return true;
}

static bool column_render_push_blank(ColumnRender *col) {
  return column_render_push_line(col, "");
}

static bool column_render_pushf_with_indent(ColumnRender *col, size_t indent, const char *fmt, ...) {
  char buffer[256];
  va_list args;
  va_start(args, fmt);
  int written = vsnprintf(buffer, sizeof buffer, fmt, args);
  va_end(args);
  if (written < 0) {
    return false;
  }
  buffer[sizeof buffer - 1] = '\0';

  char line[320];
  size_t prefix = indent < sizeof line ? indent : sizeof line - 1;
  memset(line, ' ', prefix);
  size_t available = sizeof line - prefix - 1;
  size_t copy_len = strnlen(buffer, sizeof buffer);
  if (copy_len > available) {
    copy_len = available;
  }
  memcpy(line + prefix, buffer, copy_len);
  line[prefix + copy_len] = '\0';
  return column_render_push_line(col, line);
}

static int compute_card_columns_for_width(int column_width) {
  int available_width = column_width - 4;
  int per_card = CARD_BOX_WIDTH + 1;
  int cols = (available_width + 1) / per_card;
  if (cols < 1) {
    cols = 1;
  }
  return cols;
}

static bool append_box_rows(ColumnRender *col, const BoxLines *boxes, size_t box_count, size_t cols, size_t indent) {
  if (!col || !boxes || cols == 0) {
    return false;
  }
  size_t index = 0;
  while (index < box_count) {
    size_t row_count = cols;
    if (row_count > box_count - index) {
      row_count = box_count - index;
    }
    for (int line = 0; line < CARD_BOX_HEIGHT; ++line) {
      char row_buffer[256];
      size_t pos = 0;
      if (indent > 0) {
        size_t pad = indent < sizeof row_buffer ? indent : sizeof row_buffer - 1;
        memset(row_buffer, ' ', pad);
        pos = pad;
      }
      for (size_t b = 0; b < row_count; ++b) {
        const char *src = boxes[index + b].lines[line];
        size_t src_len = strnlen(src, CARD_BOX_WIDTH);
        if (pos + src_len >= sizeof row_buffer) {
          src_len = sizeof row_buffer - pos - 1;
        }
        memcpy(row_buffer + pos, src, src_len);
        pos += src_len;
        if (b + 1 < row_count && pos < sizeof row_buffer - 1) {
          row_buffer[pos++] = ' ';
        }
      }
      if (pos >= sizeof row_buffer) {
        pos = sizeof row_buffer - 1;
      }
      row_buffer[pos] = '\0';
      if (!column_render_push_line(col, row_buffer)) {
        return false;
      }
    }
    index += row_count;
    if (index < box_count) {
      if (!column_render_push_blank(col)) {
        return false;
      }
    }
  }
  return true;
}

static void build_leader_box(const LeaderCardObservationData *leader, BoxLines *out_box) {
  CardObservationData as_card = {0};
  if (leader) {
    as_card.type = leader->type;
    as_card.id = leader->id;
    as_card.tap_state = leader->tap_state;
    as_card.cur_stats = leader->cur_stats;
    as_card.has_cur_stats = true;
  }
  build_standard_card_box(&as_card, out_box);
}

static void build_gate_box(const GateCardObservationData *gate, BoxLines *out_box) {
  CardObservationData as_card = {0};
  if (gate) {
    as_card.type = gate->type;
    as_card.id = gate->id;
    as_card.tap_state = gate->tap_state;
    as_card.has_gate_points = false;
    as_card.has_cur_stats = false;
    as_card.ikz_cost.ikz_cost = 0;
  }
  build_standard_card_box(&as_card, out_box);
}

static bool render_leader_gate_section(ColumnRender *col, const LeaderCardObservationData *leader, const GateCardObservationData *gate) {
  if (!column_render_pushf_with_indent(col, BOARD_SECTION_INDENT, "Leader & Gate")) {
    return false;
  }

  BoxLines boxes[2];
  build_leader_box(leader, &boxes[0]);
  build_gate_box(gate, &boxes[1]);

  size_t cols = 2;
  size_t required_width = BOARD_CONTENT_INDENT + CARD_BOX_WIDTH * 2 + 1;
  if (required_width > col->width) {
    cols = 1;
  }
  if (!append_box_rows(col, boxes, 2, cols, BOARD_CONTENT_INDENT)) {
    return false;
  }

  return column_render_push_blank(col);
}

static bool render_card_grid_section(ColumnRender *col, const char *label, const CardObservationData *cards, size_t max_count, bool use_zone_index, size_t cols) {
  if (!column_render_pushf_with_indent(col, BOARD_SECTION_INDENT, "%s", label)) {
    return false;
  }

  size_t count = card_observation_count(cards, max_count);
  if (!use_zone_index && count == 0) {
    if (!column_render_pushf_with_indent(col, BOARD_SECTION_INDENT + 2, "(empty)")) {
      return false;
    }
    return column_render_push_blank(col);
  }

  size_t total_slots = use_zone_index ? max_count : count;
  if (total_slots == 0) {
    total_slots = 0;
  }

  BoxLines boxes[MAX_HAND_SIZE];
  bool slot_has_card[MAX_HAND_SIZE];
  for (size_t i = 0; i < MAX_HAND_SIZE; ++i) {
    slot_has_card[i] = false;
  }

  if (use_zone_index) {
    for (size_t i = 0; i < max_count && i < MAX_HAND_SIZE; ++i) {
      build_standard_card_box(NULL, &boxes[i]);
    }
    size_t fallback_slot = 0;
    for (size_t i = 0; i < max_count; ++i) {
      const CardObservationData *card = &cards[i];
      if (!card->id.code) {
        continue;
      }
      size_t target_index = max_count;
      if (card->zone_index < max_count && !slot_has_card[card->zone_index]) {
        target_index = card->zone_index;
      } else {
        while (fallback_slot < max_count && slot_has_card[fallback_slot]) {
          fallback_slot++;
        }
        if (fallback_slot < max_count) {
          target_index = fallback_slot;
          fallback_slot++;
        }
      }
      if (target_index < max_count && target_index < MAX_HAND_SIZE) {
        build_standard_card_box(card, &boxes[target_index]);
        slot_has_card[target_index] = true;
      }
    }
    total_slots = max_count;
  } else {
    for (size_t i = 0; i < count && i < MAX_HAND_SIZE; ++i) {
      build_standard_card_box(&cards[i], &boxes[i]);
    }
    total_slots = count;
  }

  if (total_slots > MAX_HAND_SIZE) {
    total_slots = MAX_HAND_SIZE;
  }

  if (total_slots > 0) {
    if (!append_box_rows(col, boxes, total_slots, cols, BOARD_CONTENT_INDENT)) {
      return false;
    }
  }

  return column_render_push_blank(col);
}

static bool render_ikz_grid_section(ColumnRender *col, const char *label, const IKZCardObservationData *cards, size_t max_count, size_t cols) {
  if (!column_render_pushf_with_indent(col, BOARD_SECTION_INDENT, "%s", label)) {
    return false;
  }

  size_t count = ikz_observation_count(cards, max_count);
  if (count == 0) {
    if (!column_render_pushf_with_indent(col, BOARD_SECTION_INDENT + 2, "(empty)")) {
      return false;
    }
    return column_render_push_blank(col);
  }

  BoxLines boxes[IKZ_AREA_SIZE];
  size_t total = count > IKZ_AREA_SIZE ? IKZ_AREA_SIZE : count;
  for (size_t i = 0; i < total; ++i) {
    build_ikz_card_box(&cards[i], &boxes[i]);
  }

  if (!append_box_rows(col, boxes, total, cols, BOARD_CONTENT_INDENT)) {
    return false;
  }
  return column_render_push_blank(col);
}

static bool render_opponent_info_section(ColumnRender *col, const OpponentObservationData *opponent) {
  if (!column_render_pushf_with_indent(col, BOARD_SECTION_INDENT, "Info:")) {
    return false;
  }
  size_t discard_size = card_observation_count(opponent->discard, MAX_DECK_SIZE);
  if (!column_render_pushf_with_indent(col, BOARD_SECTION_INDENT + 2, "Hand: %u  IKZ Pile: %u  Discard: %zu", opponent->hand_count, opponent->ikz_pile_count, discard_size)) {
    return false;
  }
  if (!column_render_pushf_with_indent(col, BOARD_SECTION_INDENT + 2, "IKZ Token: %s", opponent->has_ikz_token ? "Yes" : "No")) {
    return false;
  }
  return column_render_push_blank(col);
}

static bool render_my_info_section(ColumnRender *col, const MyObservationData *mine) {
  if (!column_render_pushf_with_indent(col, BOARD_SECTION_INDENT, "Info:")) {
    return false;
  }
  size_t hand_count = card_observation_count(mine->hand, MAX_HAND_SIZE);
  size_t discard_size = card_observation_count(mine->discard, MAX_DECK_SIZE);
  if (!column_render_pushf_with_indent(col, BOARD_SECTION_INDENT + 2, "Hand: %zu  IKZ Pile: %u  Discard: %zu", hand_count, mine->ikz_pile_count, discard_size)) {
    return false;
  }
  if (!column_render_pushf_with_indent(col, BOARD_SECTION_INDENT + 2, "IKZ Token: %s", mine->has_ikz_token ? "Yes" : "No")) {
    return false;
  }
  return column_render_push_blank(col);
}

static bool render_my_board_column(ColumnRender *col, const MyObservationData *mine, size_t cols) {
  if (!column_render_push_line(col, "Your Board")) {
    return false;
  }
  if (!render_leader_gate_section(col, &mine->leader, &mine->gate)) {
    return false;
  }
  if (!render_card_grid_section(col, "Garden", mine->garden, GARDEN_SIZE, true, cols)) {
    return false;
  }
  if (!render_card_grid_section(col, "Alley", mine->alley, ALLEY_SIZE, true, cols)) {
    return false;
  }
  if (!render_card_grid_section(col, "Hand", mine->hand, MAX_HAND_SIZE, false, cols)) {
    return false;
  }
  if (!render_ikz_grid_section(col, "IKZ Area", mine->ikz_area, IKZ_AREA_SIZE, cols)) {
    return false;
  }
  return render_my_info_section(col, mine);
}

static bool render_opponent_board_column(ColumnRender *col, const OpponentObservationData *opponent, size_t cols) {
  if (!column_render_push_line(col, "Opponent Board")) {
    return false;
  }
  if (!render_leader_gate_section(col, &opponent->leader, &opponent->gate)) {
    return false;
  }
  if (!render_card_grid_section(col, "Garden", opponent->garden, GARDEN_SIZE, true, cols)) {
    return false;
  }
  if (!render_card_grid_section(col, "Alley", opponent->alley, ALLEY_SIZE, true, cols)) {
    return false;
  }
  if (!render_ikz_grid_section(col, "IKZ Area", opponent->ikz_area, IKZ_AREA_SIZE, cols)) {
    return false;
  }
  return render_opponent_info_section(col, opponent);
}

static int detect_terminal_width(void) {
  struct winsize ws = {0};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
    return ws.ws_col;
  }
  const char *columns_env = getenv("COLUMNS");
  if (columns_env) {
    char *endptr = NULL;
    long val = strtol(columns_env, &endptr, 10);
    if (endptr != columns_env && val > 0) {
      return (int)val;
    }
  }
  return 0;
}

static bool render_board_two_columns(RenderBuffer *buf, const ObservationData *observation, size_t column_width) {
  if (!buf || !observation) {
    return false;
  }

  ColumnRender left;
  ColumnRender right;
  column_render_init(&left, column_width);
  column_render_init(&right, column_width);
  const size_t card_cols = (size_t)compute_card_columns_for_width((int)column_width);

  bool ok = render_my_board_column(&left, &observation->my_observation_data, card_cols) &&
            render_opponent_board_column(&right, &observation->opponent_observation_data, card_cols);

  if (!ok) {
    column_render_free(&left);
    column_render_free(&right);
    return false;
  }

  size_t max_rows = left.count > right.count ? left.count : right.count;
  char *empty = (char *)malloc(column_width + 1);
  if (!empty) {
    column_render_free(&left);
    column_render_free(&right);
    return false;
  }
  memset(empty, ' ', column_width);
  empty[column_width] = '\0';

  for (size_t row = 0; row < max_rows; ++row) {
    const char *lhs = row < left.count ? left.lines[row] : empty;
    const char *rhs = row < right.count ? right.lines[row] : empty;
    if (!renderbuf_appendf(buf, "%s  %s\n", lhs, rhs)) {
      ok = false;
      break;
    }
  }

  free(empty);
  column_render_free(&left);
  column_render_free(&right);
  return ok;
}

static bool render_board_single_column(RenderBuffer *buf, const ObservationData *observation, size_t column_width) {
  if (!buf || !observation) {
    return false;
  }

  ColumnRender col;
  column_render_init(&col, column_width);
  const size_t card_cols = (size_t)compute_card_columns_for_width((int)column_width);

  bool ok = render_opponent_board_column(&col, &observation->opponent_observation_data, card_cols) &&
            render_my_board_column(&col, &observation->my_observation_data, card_cols);

  if (!ok) {
    column_render_free(&col);
    return false;
  }

  for (size_t row = 0; row < col.count; ++row) {
    if (!renderbuf_appendf(buf, "%s\n", col.lines[row])) {
      ok = false;
      break;
    }
  }

  column_render_free(&col);
  return ok;
}

static bool render_board(RenderBuffer *buf, const ObservationData *observation) {
  int term_width = detect_terminal_width();
  size_t total_width = term_width > 0 ? (size_t)term_width : (size_t)BOARD_DEFAULT_TOTAL_WIDTH;
  size_t min_col = min_board_column_width();

  // Try two columns if there's enough room; otherwise fall back to a single column stacked view.
  if (total_width >= (min_col * 2 + 2)) {
    size_t column_width = (total_width - 2) / 2;
    if (column_width < min_col) {
      column_width = min_col;
    }
    return render_board_two_columns(buf, observation, column_width);
  }

  size_t single_width = total_width > min_col ? total_width : min_col;
  return render_board_single_column(buf, observation, single_width);
}

char* c_render(CAzukiTCG* env) {
  if (!env || !env->engine) {
    return NULL;
  }

  const GameState* gs = azk_engine_game_state(env->engine);
  if (!gs) {
    return NULL;
  }

  int active_player_index = gs->active_player_index;
  if (active_player_index < 0 || active_player_index >= MAX_PLAYERS_PER_MATCH) {
    active_player_index = 0;
  }
  ObservationData observation_data = {0};
  bool observed =
      azk_engine_observe(env->engine, active_player_index, &observation_data);
  if (!observed) {
    return NULL;
  }
  const ObservationData *observation = &observation_data;

  RenderBuffer buf = {0};
  if (!renderbuf_reserve(&buf, 16384)) {
    return NULL;
  }

  if (!renderbuf_appendf(&buf, "Phase: %s\n", phase_to_string(gs->phase))) {
    free(buf.data);
    return NULL;
  }
  if (!renderbuf_appendf(&buf, "Active Player: %d\n", gs->active_player_index)) {
    free(buf.data);
    return NULL;
  }
  if (!renderbuf_appendf(&buf, "Response Window: %u\n", gs->response_window)) {
    free(buf.data);
    return NULL;
  }
  if (gs->winner >= 0) {
    if (!renderbuf_appendf(&buf, "Winner: Player %d\n", gs->winner)) {
      free(buf.data);
      return NULL;
    }
  } else {
    if (!renderbuf_appendf(&buf, "Winner: (in progress)\n")) {
      free(buf.data);
      return NULL;
    }
  }

  if (!renderbuf_appendf(&buf, "\n")) {
    free(buf.data);
    return NULL;
  }

  if (!render_board(&buf, observation)) {
    free(buf.data);
    return NULL;
  }

  if (!renderbuf_reserve(&buf, 0)) {
    free(buf.data);
    return NULL;
  }
  buf.data[buf.len] = '\0';
  return buf.data;
}
