#ifndef AZUKI_ABILITIES_STT03_011_H
#define AZUKI_ABILITIES_STT03_011_H

#include "abilities/ability_registry.h"

bool stt03_011_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool stt03_011_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void stt03_011_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif
