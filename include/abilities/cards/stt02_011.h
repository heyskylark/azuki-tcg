#ifndef AZUKI_ABILITIES_STT02_011_H
#define AZUKI_ABILITIES_STT02_011_H

#include <flecs.h>
#include "abilities/ability_registry.h"

// STT02-011: "Garden only; Main; You may sacrifice this card: choose an entity
// in your garden; it cannot take damage from card effects until the start of
// your next turn."

bool stt02_011_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool stt02_011_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void stt02_011_apply_costs(ecs_world_t *world, const AbilityContext *ctx);
void stt02_011_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_STT02_011_H
