#ifndef AZUKI_PUBLIC_ENGINE_H
#define AZUKI_PUBLIC_ENGINE_H

#include <stdbool.h>
#include <stdint.h>

#include "components/components.h"
#include "utils/actions_util.h"
#include "utils/observation_util.h"
#include "validation/action_enumerator.h"
#include "world.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ecs_world_t;
typedef struct ecs_world_t AzkEngine;

/**
 * Create a fresh Azuki game world seeded with the provided value.
 * Uses randomly selected decks for both players.
 */
AzkEngine *azk_engine_create(uint32_t seed);

/**
 * Create a fresh Azuki game world with custom decks.
 * Each deck is an array of CardInfo structs specifying card_id and count.
 * Requires including world.h for CardInfo type definition.
 */
AzkEngine *azk_engine_create_with_decks(
  uint32_t seed,
  const CardInfo *player0_deck,
  size_t player0_deck_count,
  const CardInfo *player1_deck,
  size_t player1_deck_count
);

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
 * Returns true if the previous action was invalid.
 */
bool azk_engine_was_prev_action_invalid(AzkEngine *engine);

typedef struct {
  float leader_health_ratio[MAX_PLAYERS_PER_MATCH];
  float garden_attack_sum[MAX_PLAYERS_PER_MATCH];
  float untapped_garden_count[MAX_PLAYERS_PER_MATCH];
  float untapped_ikz_count[MAX_PLAYERS_PER_MATCH];
} AzkRewardSnapshot;

bool azk_engine_reward_snapshot(AzkEngine *engine, AzkRewardSnapshot *out_snapshot);

/**
 * Get the last error message from engine operations.
 * Returns NULL if no error has occurred.
 * The returned string is valid until the next engine operation.
 */
const char *azk_engine_get_last_error(void);

/**
 * Clear the last error message.
 */
void azk_engine_clear_last_error(void);

/**
 * Get the current ability sub-phase.
 * Returns ABILITY_PHASE_NONE when not in an ability resolution flow.
 */
AbilityPhase azk_engine_get_ability_phase(const AzkEngine *engine);

#ifdef __cplusplus
}
#endif

#endif
