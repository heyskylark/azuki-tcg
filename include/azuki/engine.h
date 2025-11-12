#ifndef AZUKI_PUBLIC_ENGINE_H
#define AZUKI_PUBLIC_ENGINE_H

#include <stdbool.h>
#include <stdint.h>

#include "components.h"
#include "utils/actions_util.h"
#include "utils/observation_util.h"
#include "validation/action_enumerator.h"

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
 * Snapshot the current ObservationData for the requested player perspective into the provided output struct.
 */
bool azk_engine_observe(AzkEngine *engine, int8_t player_index, ObservationData *out_observation);

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

/**
 * Build the legal action mask for the requested player.
 */
bool azk_engine_build_action_mask(
  AzkEngine *engine,
  int8_t player_index,
  AzkActionMaskSet *out_mask
);

typedef struct {
  float leader_health_ratio[MAX_PLAYERS_PER_MATCH];
  float garden_attack_sum[MAX_PLAYERS_PER_MATCH];
  float untapped_garden_count[MAX_PLAYERS_PER_MATCH];
  float untapped_ikz_count[MAX_PLAYERS_PER_MATCH];
} AzkRewardSnapshot;

bool azk_engine_reward_snapshot(AzkEngine *engine, AzkRewardSnapshot *out_snapshot);

#ifdef __cplusplus
}
#endif

#endif
