#ifndef AZUKI_ABILITIES_STT03_017_H
#define AZUKI_ABILITIES_STT03_017_H

#include "abilities/ability_registry.h"

/* Mode 0: require a nonempty IKZ pile; ramp once before the optional heal choice. */
bool stt03_017_validate_ramp(ecs_world_t *world, ecs_entity_t card,
                             ecs_entity_t owner);
bool stt03_017_validate_heal_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target);
void stt03_017_begin_ramp_effect(ecs_world_t *world, AbilityContext *ctx);
void stt03_017_apply_heal(ecs_world_t *world, const AbilityContext *ctx);
/* Mode 1: draw 1, losing only if that draw cannot be completed. */
void stt03_017_draw(ecs_world_t *world, const AbilityContext *ctx);

#endif
