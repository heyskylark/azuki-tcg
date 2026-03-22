#ifndef AZUKI_ABILITIES_STT03_004_H
#define AZUKI_ABILITIES_STT03_004_H

#include "abilities/ability_registry.h"

bool stt03_004_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void stt03_004_apply_costs(ecs_world_t *world, const AbilityContext *ctx);
void stt03_004_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif
