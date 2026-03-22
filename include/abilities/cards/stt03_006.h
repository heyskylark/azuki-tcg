#ifndef AZUKI_ABILITIES_STT03_006_H
#define AZUKI_ABILITIES_STT03_006_H

#include "abilities/ability_registry.h"

bool stt03_006_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
void stt03_006_on_cost_paid(ecs_world_t *world, AbilityContext *ctx);
bool stt03_006_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void stt03_006_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif
