#ifndef AZUKI_ABILITIES_AZK01_079_H
#define AZUKI_ABILITIES_AZK01_079_H

#include "abilities/ability_registry.h"

/* Opponent-chosen modes; `ctx->runtime.owner` is the choosing player. */

/* Mode 1: Gin and Tonika's controller draws 2. */
void azk01_079_apply_draw(ecs_world_t *world, const AbilityContext *ctx);
/* Mode 2: deal 3 damage to the choosing player's leader. */
void azk01_079_apply_damage(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_079_H
