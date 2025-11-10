#ifndef AZUKI_PUBLIC_ENGINE_H
#define AZUKI_PUBLIC_ENGINE_H

#include <stdbool.h>
#include <stdint.h>

#include "components.h"
#include "utils/actions_util.h"
#include "utils/observation_util.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ecs_world_t;
typedef struct ecs_world_t AzkEngine;

/**
 * Create a fresh Azuki game world seeded with the provided value.
 */
AzkEngine *azk_engine_create(uint32_t seed);

/**
 * Destroy a previously created Azuki engine instance.
 */
void azk_engine_destroy(AzkEngine *engine);

/**
 * Returns a read-only pointer to the current GameState singleton.
 */
const GameState *azk_engine_game_state(const AzkEngine *engine);

/**
 * Snapshot the current ObservationData into the provided output struct.
 */
bool azk_engine_observe(AzkEngine *engine, ObservationData *out_observation);

/**
 * Returns true when the engine is waiting for a user/agent action.
 */
bool azk_engine_requires_action(AzkEngine *engine);

/**
 * Returns true when the current match is over.
 */
bool azk_engine_is_game_over(AzkEngine *engine);

/**
 * Convenience wrapper around azk_parse_user_action_values.
 */
bool azk_engine_parse_action_values(
  AzkEngine *engine,
  const int values[AZK_USER_ACTION_VALUE_COUNT],
  UserAction *out_action
);

/**
 * Validate and queue an action for consumption by gameplay systems.
 */
bool azk_engine_submit_action(AzkEngine *engine, const UserAction *action);

/**
 * Advance the simulation by running the phase gate and Flecs progression.
 */
void azk_engine_tick(AzkEngine *engine);

#ifdef __cplusplus
}
#endif

#endif
