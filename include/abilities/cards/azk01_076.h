#ifndef AZUKI_ABILITIES_AZK01_076_H
#define AZUKI_ABILITIES_AZK01_076_H

#include "abilities/ability_registry.h"

/* Opponent-chosen modes; both benefit Hōren's controller. */

/* Mode 1: this entity gains Charge until end of turn. */
void azk01_076_apply_charge(ecs_world_t *world, const AbilityContext *ctx);
/* Mode 2: heal 2 to Hōren's controller's leader. */
void azk01_076_apply_heal(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_076_H
