#include "utils/phase_utils.h"

#include "abilities/ability_system.h"
#include "components/components.h"

static const bool PHASE_REQUIRES_USER_ACTION[PHASE_COUNT] = {
    [PHASE_PREGAME_MULLIGAN] = true,
    [PHASE_START_OF_TURN] = false,
    [PHASE_MAIN] = true,
    [PHASE_RESPONSE_WINDOW] = true,
    [PHASE_COMBAT_RESOLVE] = false,
    [PHASE_END_TURN_ACTION] = true,
    [PHASE_END_TURN] = false,
    [PHASE_END_MATCH] = false};

bool phase_requires_user_action(ecs_world_t *world, Phase phase) {
  // If there's a queued triggered effect AND no ability is currently active,
  // no user input needed (we'll auto-process the queue this iteration).
  // If an ability IS active, we still need user input for the current ability;
  // the queued effects will wait until the current ability completes.
  if (azk_has_queued_triggered_effects(world) && !azk_is_in_ability_phase(world)) {
    return false;
  }

  // Standard phase-based lookup
  return PHASE_REQUIRES_USER_ACTION[phase];
}
