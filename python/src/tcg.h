#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "azuki/engine.h"

#define PBRS_LEADER_WEIGHT 4.0f
#define PBRS_GARDEN_ATTACK_WEIGHT 0.7f
#define PBRS_UNTAPPED_GARDEN_WEIGHT 0.4f
#define PBRS_UNTAPPED_IKZ_WEIGHT 0.2f

#define PBRS_GARDEN_ATTACK_CAP 10.0f
#define PBRS_UNTAPPED_GARDEN_CAP 5.0f
#define PBRS_UNTAPPED_IKZ_CAP 10.0f

#define PBRS_TIME_DECAY_DEFAULT 0.95f
#define TERMINAL_REWARD 5.0f

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
    float p1_winrate;
    float p2_winrate;
    float draw_rate;
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
typedef struct {
  // Puffer I/O
  ObservationData* observations; // MAX_PLAYERS_PER_MATCH
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
  float time_weight;
  float time_decay;
} CAzukiTCG;

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

static inline float leader_health_transform(float normalized_hp) {
  const float x = clampf(normalized_hp, 0.0f, 1.0f);
  const float one_minus_x = 1.0f - x;
  const float one_minus_x_sq = one_minus_x * one_minus_x;
  const float one_minus_x_pow4 = one_minus_x_sq * one_minus_x_sq;
  return 0.5f * (x + 1.0f - one_minus_x_pow4);
}

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
  env->time_weight = 1.0f;
  env->time_decay = PBRS_TIME_DECAY_DEFAULT;
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

static void apply_shaped_rewards(CAzukiTCG* env, int8_t acting_player_index) {
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
  const float shaped_reward = env->time_weight * phi_delta;
  env->rewards[acting_player_index] = shaped_reward;
  env->rewards[opponent_index] = -shaped_reward;

  for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    env->last_phi[player_index] = phi_values[player_index];
  }
  env->time_weight *= env->time_decay;
}

void init(CAzukiTCG* env) {
  env->engine = azk_engine_create(env->seed);
  env->tick = 0;
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

static void refresh_action_masks(CAzukiTCG* env) {
  for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    azk_engine_build_action_mask(env->engine, player_index, &env->action_masks[player_index]);
  }
}

static void refresh_observations(CAzukiTCG* env) {
  for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    azk_engine_observe(env->engine, player_index, &env->observations[player_index]);
  }
  refresh_action_masks(env);
}

void c_reset(CAzukiTCG* env) {
  env->tick = 0;
  env->terminals[0] = NOT_DONE;
  env->terminals[1] = NOT_DONE;
  env->truncations[0] = NOT_DONE;
  env->truncations[1] = NOT_DONE;
  env->rewards[0] = 0.0f;
  env->rewards[1] = 0.0f;

  azk_engine_destroy(env->engine);
  env->engine = azk_engine_create(env->seed);
  refresh_observations(env);
  reset_reward_tracking(env);
}

void c_step(CAzukiTCG* env) {
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

  bool checked_action_tick = false;

  // Some sub-actions do not require a user action
  // We should progress those until a user action is required (or the game ends)
  do {
    azk_engine_tick(env->engine);

    if (!checked_action_tick) {
      checked_action_tick = true;
      if (azk_engine_was_prev_action_invalid(env->engine)) {
        fprintf(stderr, "Invalid action detected at tick %d\n", env->tick);
        abort();
      }
    }
  } while (!azk_engine_requires_action(env->engine) && !azk_engine_is_game_over(env->engine));

  if (azk_engine_is_game_over(env->engine)) {
    env->terminals[0] = DONE;
    env->terminals[1] = DONE;
  }

  refresh_observations(env);

  if (azk_engine_is_game_over(env->engine)) {
    apply_terminal_rewards(env);
    return;
  }

  apply_shaped_rewards(env, active_player_index);
}

void c_close(CAzukiTCG* env) {
  azk_engine_destroy(env->engine);
}

void c_render(CAzukiTCG* env) {
}
