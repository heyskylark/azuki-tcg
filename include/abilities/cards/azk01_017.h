#ifndef AZUKI_ABILITIES_AZK01_017_H
#define AZUKI_ABILITIES_AZK01_017_H

#include "abilities/ability_registry.h"
#include <flecs.h>
#include <stdbool.h>

// AZK01-017 "Hook Sword Strike": [Main] Deal up to 1 damage to an entity in
// any Garden and up to 1 damage to a leader.

bool azk01_017_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool azk01_017_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void azk01_017_on_cost_paid(ecs_world_t *world, AbilityContext *ctx);
void azk01_017_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_017_H
