#ifndef AZUKI_ABILITIES_STT04_012_H
#define AZUKI_ABILITIES_STT04_012_H

#include "abilities/ability_registry.h"

bool stt04_012_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool stt04_012_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void stt04_012_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif
