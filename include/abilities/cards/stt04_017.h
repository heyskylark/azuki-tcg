#ifndef AZUKI_ABILITIES_STT04_017_H
#define AZUKI_ABILITIES_STT04_017_H

#include "abilities/ability_registry.h"

bool stt04_017_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool stt04_017_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target);
bool stt04_017_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
void stt04_017_apply_costs(ecs_world_t *world, const AbilityContext *ctx);
void stt04_017_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif
