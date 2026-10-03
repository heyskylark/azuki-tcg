#ifndef AZUKI_ABILITIES_AZK01_083_H
#define AZUKI_ABILITIES_AZK01_083_H

#include "abilities/ability_registry.h"

/* Valid sources: another Neutral card in the owner's Garden that is not a
 * Gurugumi Imitator (copying a copy ability is not supported). */
bool azk01_083_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool azk01_083_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);
/* Copy the target's printed text (abilities + keywords) until end of turn. */
void azk01_083_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_083_H
