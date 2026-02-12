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

#define PBRS_LEADER_WEIGHT 4.0f
#define PBRS_GARDEN_ATTACK_WEIGHT 0.7f
#define PBRS_UNTAPPED_GARDEN_WEIGHT 0.15f
#define PBRS_UNTAPPED_IKZ_WEIGHT 0.15f

#define PBRS_GARDEN_ATTACK_CAP 10.0f
#define PBRS_UNTAPPED_GARDEN_CAP 5.0f
#define PBRS_UNTAPPED_IKZ_CAP 10.0f

#define PBRS_TIME_DECAY_DEFAULT 0.95f
#define TERMINAL_REWARD 5.0f
#define TRUNCATION_TIMEOUT_PENALTY 0.35f
#define TRUNCATION_AUTO_TICK_PENALTY 0.60f
#define TRUNCATION_LEADER_EDGE_WEIGHT 1.25f
#define TRUNCATION_BOARD_EDGE_WEIGHT 0.45f

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
    float draw_rate;
    float timeout_truncation_rate;
    float auto_tick_truncation_rate;
    float gameover_terminal_rate;
    float winner_terminal_rate;
    float curriculum_episode_cap;
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
    float n;
} Log;

// Mirrors include/utils/actions_util.h::AZK_USER_ACTION_VALUE_COUNT tuple layout.
typedef struct {
  int32_t type;
  int32_t subaction_1;
  int32_t subaction_2;
  int32_t subaction_3;
} ActionVector;

typedef struct Client Client;
typedef enum EpisodeEndReason {
  EP_END_REASON_GAMEOVER = 0,
  EP_END_REASON_TIMEOUT_TRUNCATION = 1,
  EP_END_REASON_AUTO_TICK_TRUNCATION = 2
} EpisodeEndReason;

typedef struct {
  // Puffer I/O
  TrainingObservationData* observations; // MAX_PLAYERS_PER_MATCH
  ActionVector* actions;         // 1 MAX_PLAYERS_PER_MATCH rows of {type, subaction_1..3}
  float* rewards;                // MAX_PLAYERS_PER_MATCH scalars
  unsigned char* terminals;      // MAX_PLAYERS_PER_MATCH scalars {0,1}
  unsigned char* truncations;    // MAX_PLAYERS_PER_MATCH scalars {0,1}
  Log log;
  Client* client;

  // Game State
  AzkEngine* engine;
  uint32_t seed;
  int tick;
  AzkActionMaskSet action_masks[MAX_PLAYERS_PER_MATCH];
  float last_phi[MAX_PLAYERS_PER_MATCH];
  float episode_returns[MAX_PLAYERS_PER_MATCH];
  uint64_t completed_episodes;
  int current_episode_cap;
  float time_weight;
  float time_decay;
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
} CAzukiTCG;

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
  float truncation_board_edge_weight;
} RewardTuningConfig;

static RewardTuningConfig g_reward_tuning = {0};

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
  g_reward_tuning.truncation_board_edge_weight =
      parse_nonnegative_env_float("AZK_TRUNCATION_BOARD_EDGE_WEIGHT", TRUNCATION_BOARD_EDGE_WEIGHT);
}

static inline float board_edge_from_snapshot(const AzkRewardSnapshot *snapshot) {
  const float attack_edge = safe_delta(
      snapshot->garden_attack_sum[0] - snapshot->garden_attack_sum[1],
      PBRS_GARDEN_ATTACK_CAP);
  const float untapped_edge = safe_delta(
      snapshot->untapped_garden_count[0] - snapshot->untapped_garden_count[1],
      PBRS_UNTAPPED_GARDEN_CAP);
  const float ikz_edge = safe_delta(
      snapshot->untapped_ikz_count[0] - snapshot->untapped_ikz_count[1],
      PBRS_UNTAPPED_IKZ_CAP);
  return 0.6f * attack_edge + 0.3f * untapped_edge + 0.1f * ikz_edge;
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
  const float leader_term = PBRS_LEADER_WEIGHT * (
    leader_health_transform(snapshot->leader_health_ratio[player_index]) -
    leader_health_transform(snapshot->leader_health_ratio[opponent_index])
  );
  const float attack_term = PBRS_GARDEN_ATTACK_WEIGHT * safe_delta(
    snapshot->garden_attack_sum[player_index] - snapshot->garden_attack_sum[opponent_index],
    PBRS_GARDEN_ATTACK_CAP
  );
  const float untapped_garden_term = PBRS_UNTAPPED_GARDEN_WEIGHT * safe_delta(
    snapshot->untapped_garden_count[player_index] - snapshot->untapped_garden_count[opponent_index],
    PBRS_UNTAPPED_GARDEN_CAP
  );
  const float untapped_ikz_term = PBRS_UNTAPPED_IKZ_WEIGHT * safe_delta(
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

static void reset_reward_tracking(CAzukiTCG* env) {
  init_reward_tuning_if_needed();
  env->time_weight = 1.0f;
  env->time_decay = PBRS_TIME_DECAY_DEFAULT;
  for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    env->episode_returns[player_index] = 0.0f;
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
  }
  AzkRewardSnapshot snapshot = {0};
  env->has_last_snapshot = false;
  if (azk_engine_reward_snapshot(env->engine, &snapshot)) {
    env->last_snapshot = snapshot;
    env->has_last_snapshot = true;
  }

  float phi_values[MAX_PLAYERS_PER_MATCH] = {0.0f};
  if (compute_phi_values(env, phi_values)) {
    for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
      env->last_phi[player_index] = phi_values[player_index];
    }
  } else {
    for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
      env->last_phi[player_index] = 0.0f;
    }
  }
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
}

static void apply_truncation_rewards(CAzukiTCG* env, EpisodeEndReason reason) {
  float timeout_penalty = TRUNCATION_TIMEOUT_PENALTY;
  if (reason == EP_END_REASON_AUTO_TICK_TRUNCATION) {
    timeout_penalty = TRUNCATION_AUTO_TICK_PENALTY;
  }

  float leader_edge_term = 0.0f;
  float board_edge_term = 0.0f;
  AzkRewardSnapshot snapshot = {0};
  if (azk_engine_reward_snapshot(env->engine, &snapshot)) {
    const float p0_health = leader_health_transform(snapshot.leader_health_ratio[0]);
    const float p1_health = leader_health_transform(snapshot.leader_health_ratio[1]);
    leader_edge_term = TRUNCATION_LEADER_EDGE_WEIGHT * (p0_health - p1_health);
    board_edge_term = g_reward_tuning.truncation_board_edge_weight *
                      board_edge_from_snapshot(&snapshot);
  }

  env->rewards[0] = leader_edge_term + board_edge_term - timeout_penalty;
  env->rewards[1] = -leader_edge_term - board_edge_term - timeout_penalty;
}

static void accumulate_step_rewards(CAzukiTCG* env) {
  for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    env->episode_returns[player_index] += env->rewards[player_index];
  }
}

static void record_episode_stats(CAzukiTCG* env, EpisodeEndReason reason) {
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
  env->log.episode_length += (float)env->tick;
  env->log.episode_return += env->episode_returns[0];
  env->log.p0_episode_return += env->episode_returns[0];
  env->log.p1_episode_return += env->episode_returns[1];
  env->log.curriculum_episode_cap += (float)env->current_episode_cap;
  if (reason == EP_END_REASON_GAMEOVER) {
    env->log.gameover_terminal_rate += 1.0f;
  } else if (reason == EP_END_REASON_TIMEOUT_TRUNCATION) {
    env->log.timeout_truncation_rate += 1.0f;
  } else if (reason == EP_END_REASON_AUTO_TICK_TRUNCATION) {
    env->log.auto_tick_truncation_rate += 1.0f;
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

  env->log.p0_avg_leader_health += snapshot.leader_health_ratio[0];
  env->log.p1_avg_leader_health += snapshot.leader_health_ratio[1];

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

static void apply_shaped_rewards(
    CAzukiTCG* env, int8_t acting_player_index, ActionType selected_type,
    bool noop_had_alternatives) {
  if (acting_player_index < 0 || acting_player_index >= MAX_PLAYERS_PER_MATCH) {
    fprintf(stderr, "Invalid acting player index %d when applying shaped rewards\n", acting_player_index);
    abort();
  }

  float phi_values[MAX_PLAYERS_PER_MATCH] = {0.0f};
  if (!compute_phi_values(env, phi_values)) {
    fprintf(stderr, "Failed to compute phi values when applying shaped rewards\n");
    abort();
  }

  const int8_t opponent_index = (acting_player_index + 1) % MAX_PLAYERS_PER_MATCH;
  const float phi_delta = phi_values[acting_player_index] - env->last_phi[acting_player_index];
  float leader_delta_term = 0.0f;
  float board_delta_term = 0.0f;
  AzkRewardSnapshot snapshot = {0};
  if (azk_engine_reward_snapshot(env->engine, &snapshot)) {
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
    }
    env->last_snapshot = snapshot;
    env->has_last_snapshot = true;
  }

  float noop_penalty = 0.0f;
  if (selected_type == ACT_NOOP && noop_had_alternatives) {
    noop_penalty = g_reward_tuning.noop_penalty;
  }

  const float shaped_reward = env->time_weight * phi_delta + leader_delta_term +
                              board_delta_term - noop_penalty;
  env->rewards[acting_player_index] = shaped_reward;
  env->rewards[opponent_index] = -shaped_reward;

  for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    env->last_phi[player_index] = phi_values[player_index];
  }
  env->time_weight *= env->time_decay;
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
  env->engine = azk_engine_create(env->seed);
  env->tick = 0;
  env->completed_episodes = 0;
  env->current_episode_cap = max_episode_ticks_limit();
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
  env->rewards[0] = 0.0f;
  env->rewards[1] = 0.0f;
  env->current_episode_cap = current_episode_ticks_limit(env);

  azk_engine_destroy(env->engine);
  env->engine = azk_engine_create(env->seed);
  refresh_observations(env);
  reset_reward_tracking(env);
}

void c_step(CAzukiTCG* env) {
  init_env_profile_if_needed();
  const uint64_t step_start_ns =
      g_env_profile.enabled ? env_now_ns() : 0;
  uint64_t tick_total_ns = 0;
  uint64_t refresh_total_ns = 0;
  uint64_t auto_tick_count = 0;

  env->tick++;
  env->rewards[0] = 0.0f;
  env->rewards[1] = 0.0f;

  const int8_t active_player_index = tcg_active_player_index(env);
  if (active_player_index < 0) {
    fprintf(stderr, "No active player available when stepping environment\n");
    abort();
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
  const bool noop_had_alternatives =
      (parsed_action.type == ACT_NOOP) &&
      (env->observations[active_player_index].action_mask.legal_action_count > 1);

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
            &env->observations[active_player_index].action_mask;
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
          "[%d, %d, %d, %d], action_in_mask=%d, legal_action_count=%u, "
          "action_in_fresh_mask=%d, fresh_legal_action_count=%u, "
          "ability_phase=%d\n",
          env->tick,
          env->observations[0].phase,
          active_player_index,
          action.type,
          action.subaction_1,
          action.subaction_2,
          action.subaction_3,
          action_in_mask ? 1 : 0,
          active_action_mask->legal_action_count,
          action_in_fresh_mask ? 1 : 0,
          fresh_legal_action_count,
          (int)debug_ability_phase
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
            &env->observations[0].my_observation_data;
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

        fflush(stderr);
        abort();
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
    apply_truncation_rewards(env, EP_END_REASON_AUTO_TICK_TRUNCATION);
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

  const int max_ticks = current_episode_ticks_limit(env);
  env->current_episode_cap = max_ticks;
  if (max_ticks > 0 && env->tick >= max_ticks) {
    apply_truncation_rewards(env, EP_END_REASON_TIMEOUT_TRUNCATION);
    accumulate_step_rewards(env);
    env->truncations[0] = DONE;
    env->truncations[1] = DONE;
    record_episode_stats(env, EP_END_REASON_TIMEOUT_TRUNCATION);
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

  apply_shaped_rewards(env, active_player_index, parsed_action.type, noop_had_alternatives);
  accumulate_step_rewards(env);
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
  size_t copy_len = strnlen(buffer, sizeof line - prefix - 1);
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
