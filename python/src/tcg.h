#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "azuki/engine.h"

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
} CAzukiTCG;

void init(CAzukiTCG* env) {
  env->engine = azk_engine_create(env->seed);
  env->tick = 0;
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
}

void c_step(CAzukiTCG* env) {
  env->tick++;

  const ActionVector action = env->actions[0];
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

  // Some sub-actions do not require a user action
  // We should progress those until a user action is required (or the game ends)
  do {
    azk_engine_tick(env->engine);
  } while (!azk_engine_requires_action(env->engine) && !azk_engine_is_game_over(env->engine));

  if (azk_engine_is_game_over(env->engine)) {
    env->terminals[0] = DONE;
    env->terminals[1] = DONE;
  }

  refresh_observations(env);
}

void c_close(CAzukiTCG* env) {
  azk_engine_destroy(env->engine);
}

void c_render(CAzukiTCG* env) {
}
