#ifndef AZUKI_ABILITIES_AZK01_008_H
#define AZUKI_ABILITIES_AZK01_008_H

#include "components/components.h"
#include <flecs.h>
#include <stdbool.h>

// AZK01-008 "Rainy Day Assassin": [On Play] You may sacrifice this card:
// Destroy an entity with a cost of 3 or less in your opponent's Garden.

bool azk01_008_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool azk01_008_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void azk01_008_apply_costs(ecs_world_t *world, const AbilityContext *ctx);
void azk01_008_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_008_H
