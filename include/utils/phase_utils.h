#ifndef AZUKI_UTILS_PHASE_UTILS_H
#define AZUKI_UTILS_PHASE_UTILS_H

#include <flecs.h>
#include "components/components.h"

// Check if the current phase requires user input.
// Returns false if there are queued triggered effects to auto-process.
bool phase_requires_user_action(ecs_world_t *world, Phase phase);

#endif