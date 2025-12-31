#ifndef AZUKI_ABILITY_STT01_016_H
#define AZUKI_ABILITY_STT01_016_H

#include <flecs.h>
#include <stdbool.h>

#include "components/components.h"

// STT01-016 "Raizan's Zanbato": [When Attacking] If equipped to a (Raizan)
// card, deal 1 damage to all entities in your opponent's garden.

// Validate if ability can be activated
// Returns true if:
// - Weapon is equipped to a card with Raizan subtype
// - Opponent has at least one non-EffectImmune entity in garden
bool stt01_016_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);

// Apply effect: deal 1 damage to all entities in opponent's garden
void stt01_016_apply_effects(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITY_STT01_016_H
