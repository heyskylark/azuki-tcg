#ifndef AZUKI_ABILITIES_AZK01_099_H
#define AZUKI_ABILITIES_AZK01_099_H

#include "abilities/ability_registry.h"

/* Owner-chosen modes. */

/* Mode 1: an entity with cost <= 5 in the opponent's Garden becomes Shocked. */
bool azk01_099_validate_shock(ecs_world_t *world, ecs_entity_t card,
                              ecs_entity_t owner);
bool azk01_099_validate_shock_target(ecs_world_t *world, ecs_entity_t card,
                                     ecs_entity_t owner, ecs_entity_t target);
void azk01_099_apply_shock(ecs_world_t *world, const AbilityContext *ctx);
/* Mode 2: this entity gains Charge until end of turn. */
void azk01_099_apply_charge(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_099_H
