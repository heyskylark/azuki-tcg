#ifndef AZUKI_ABILITIES_AZK01_006_H
#define AZUKI_ABILITIES_AZK01_006_H

#include "components/components.h"
#include <flecs.h>
#include <stdbool.h>

// AZK01-006 "Gus": [When Attacked] You may return this card to your hand.

bool azk01_006_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void azk01_006_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_006_H
