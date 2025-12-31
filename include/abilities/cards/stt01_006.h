#ifndef AZUKI_ABILITY_STT01_006_H
#define AZUKI_ABILITY_STT01_006_H

#include <flecs.h>
#include <stdbool.h>

#include "components/components.h"

// STT01-006 "Silver Current, Haruhi": [Once/Turn][When Attacking] Deal 1 damage
// to a leader or entity in your opponent's garden.

// Validate if ability can be activated
// Returns true if opponent has at least one valid target (not EffectImmune)
bool stt01_006_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);

// Validate effect target selection
// Returns true if target is opponent's garden entity or leader without EffectImmune
bool stt01_006_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target);

// Apply effect: deal 1 damage to selected target
void stt01_006_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITY_STT01_006_H
