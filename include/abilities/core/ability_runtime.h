#ifndef AZUKI_ABILITY_RUNTIME_H
#define AZUKI_ABILITY_RUNTIME_H

#include <flecs.h>

#include "components/components.h"

void azk_maybe_transfer_triggered_ability_control(ecs_world_t *world,
                                                  AbilityContext *ctx);

void azk_restore_triggered_ability_control(ecs_world_t *world,
                                           const AbilityContext *ctx);

#endif // AZUKI_ABILITY_RUNTIME_H
