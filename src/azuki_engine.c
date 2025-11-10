#include "azuki/engine.h"

#include "systems/phase_gate.h"
#include "utils/phase_utils.h"
#include "world.h"

AzkEngine *azk_engine_create(uint32_t seed) {
  return azk_world_init(seed);
}

void azk_engine_destroy(AzkEngine *engine) {
  if (!engine) {
    return;
  }

  azk_world_fini(engine);
}

const GameState *azk_engine_game_state(const AzkEngine *engine) {
  if (!engine) {
    return NULL;
  }

  return ecs_singleton_get((const ecs_world_t *)engine, GameState);
}

bool azk_engine_observe(AzkEngine *engine, ObservationData *out_observation) {
  if (!engine || !out_observation) {
    return false;
  }

  *out_observation = create_observation_data(engine);
  return true;
}

bool azk_engine_requires_action(AzkEngine *engine) {
  if (!engine) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(engine, GameState);
  return gs && phase_requires_user_action(gs->phase);
}

bool azk_engine_is_game_over(AzkEngine *engine) {
  if (!engine) {
    return true;
  }

  return is_game_over(engine);
}

bool azk_engine_parse_action_values(
  AzkEngine *engine,
  const int values[AZK_USER_ACTION_VALUE_COUNT],
  UserAction *out_action
) {
  if (!engine) {
    return false;
  }

  return azk_parse_user_action_values(engine, values, out_action);
}

bool azk_engine_submit_action(AzkEngine *engine, const UserAction *action) {
  if (!engine || !action) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(engine, GameState);
  if (!gs || !verify_user_action_player(gs, action)) {
    return false;
  }

  azk_store_user_action(engine, action);
  return true;
}

void azk_engine_tick(AzkEngine *engine) {
  if (!engine) {
    return;
  }

  run_phase_gate_system(engine);
  ecs_progress(engine, 0);
}
