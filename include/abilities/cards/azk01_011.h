#ifndef AZUKI_ABILITIES_AZK01_011_H
#define AZUKI_ABILITIES_AZK01_011_H

#include "components/components.h"
#include <flecs.h>
#include <stdbool.h>

// AZK01-011 "Rooftop Hunter": [Garden Only Ability] If this card is untapped
// at the end of your turn, sacrifice it.

bool azk01_011_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void azk01_011_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_011_H
