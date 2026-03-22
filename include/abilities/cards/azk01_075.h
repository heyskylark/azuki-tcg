#ifndef AZUKI_ABILITIES_AZK01_075_H
#define AZUKI_ABILITIES_AZK01_075_H

#include "abilities/ability_registry.h"

bool azk01_075_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool azk01_075_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target);
void azk01_075_apply_costs(ecs_world_t *world, const AbilityContext *ctx);
void azk01_075_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_075_H
