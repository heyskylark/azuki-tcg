#ifndef AZUKI_ABILITIES_AZK01_016_H
#define AZUKI_ABILITIES_AZK01_016_H

#include "abilities/ability_registry.h"
#include <flecs.h>
#include <stdbool.h>

// AZK01-016 "Sleight of Hand": [Main] Draw 2, then discard 2.

bool azk01_016_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool azk01_016_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void azk01_016_apply_costs(ecs_world_t *world, const AbilityContext *ctx);
void azk01_016_on_cost_paid(ecs_world_t *world, AbilityContext *ctx);
void azk01_016_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_016_H
