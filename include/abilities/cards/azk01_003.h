#ifndef AZUKI_ABILITIES_AZK01_003_H
#define AZUKI_ABILITIES_AZK01_003_H

#include "abilities/ability_registry.h"
#include <flecs.h>

// AZK01-003: "[On Play] Look at the top 5 cards of your deck, reveal up to 1
// Black Jade subtype card other than Black Jade Courier and add it to your
// hand, then bottom deck the rest in any order."

bool azk01_003_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void azk01_003_on_cost_paid(ecs_world_t *world, AbilityContext *ctx);
bool azk01_003_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target);
void azk01_003_on_selection_complete(ecs_world_t *world, AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_003_H
