#ifndef AZUKI_ABILITIES_STT01_005_H
#define AZUKI_ABILITIES_STT01_005_H

#include <flecs.h>
#include "abilities/ability_registry.h"

// STT01-005: "Main; Alley Only; You may sacrifice this card: Draw 3 cards and discard 2"

bool stt01_005_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner);
bool stt01_005_validate_effect_target(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner, ecs_entity_t target);
void stt01_005_apply_costs(ecs_world_t* world, const AbilityContext* ctx);
void stt01_005_apply_effects(ecs_world_t* world, const AbilityContext* ctx);

#endif // AZUKI_ABILITIES_STT01_005_H
