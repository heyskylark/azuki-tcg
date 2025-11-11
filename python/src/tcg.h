#include <math.h>
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

typedef struct Client Client;
typedef struct {
  // Puffer I/O
  ObservationData* observations; // MAX_PLAYERS_PER_MATCH
  int* actions;                  // ACTION_TYPE, subaction_1, subaction_2, subaction_3 // TODO: implement proper action space
  float* rewards;                // MAX_PLAYERS_PER_MATCH scalars
  unsigned char* terminals;      // MAX_PLAYERS_PER_MATCH scalars {0,1}
  unsigned char* truncations;    // MAX_PLAYERS_PER_MATCH scalars {0,1}
  Log log;
  Client* client;

  // Game State
  AzkEngine* engine;
  uint32_t seed;
  int tick;
} CAzukiTCG;

void init(CAzukiTCG* env) {
  env->engine = azk_engine_create(env->seed);
  env->tick = 0;
}

static void refresh_observations(CAzukiTCG* env) {
  for (int8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    azk_engine_observe(env->engine, player_index, &env->observations[player_index]);
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

  azk_engine_destroy(env->engine);
  env->engine = azk_engine_create(env->seed);
  refresh_observations(env);
}

void c_step(CAzukiTCG* env) {
  env->tick++;

  // TODO: implement
  refresh_observations(env);
}

void c_close(CAzukiTCG* env) {
}

void c_render(CAzukiTCG* env) {
}
