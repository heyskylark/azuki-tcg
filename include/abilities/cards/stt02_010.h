#ifndef AZUKI_ABILITY_STT02_010_H
#define AZUKI_ABILITY_STT02_010_H

#include <flecs.h>
#include <stdbool.h>

#include "abilities/ability_registry.h"

// STT02-010: "Garden only; whenever an entity is returned to its owner's hand,
// you may tap this card, then draw 1. (this ability is not affected by
// cooldown)"

bool stt02_010_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void stt02_010_apply_costs(ecs_world_t *world, const AbilityContext *ctx);
void stt02_010_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITY_STT02_010_H
