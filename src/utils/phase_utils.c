#include "utils/phase_utils.h"
#include "components.h"

// TODO: Temporary, need more intelligent checks later.
static const bool PHASE_REQUIRES_USER_ACTION[PHASE_COUNT] = {
  [PHASE_PREGAME_MULLIGAN] = true,
  [PHASE_START_OF_TURN]    = false,
  [PHASE_MAIN]             = true,
  [PHASE_COMBAT_DECLARED]  = true,
  [PHASE_RESPONSE_WINDOW]  = true,
  [PHASE_COMBAT_RESOLVE]   = false,
  [PHASE_END_TURN]         = true,
  [PHASE_END_MATCH]        = false
};


bool phase_requires_user_action(Phase phase) {
  return PHASE_REQUIRES_USER_ACTION[phase];
}
