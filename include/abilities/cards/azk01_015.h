#ifndef AZUKI_ABILITIES_AZK01_015_H
#define AZUKI_ABILITIES_AZK01_015_H

#include "abilities/ability_registry.h"
#include <flecs.h>
#include <stdbool.h>

// AZK01-015 "Mo": [On Play] If your leader is Water, untap 1 IKZ. If your
// leader is Earth, heal 2 to your leader. If your leader is Lightning, gain
// Charge until the end of the turn. If your leader is Fire, deal 2 damage to
// target leader.

bool azk01_015_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool azk01_015_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void azk01_015_on_cost_paid(ecs_world_t *world, AbilityContext *ctx);
void azk01_015_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_015_H
