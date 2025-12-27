#ifndef AZUKI_ABILITIES_STT01_004_H
#define AZUKI_ABILITIES_STT01_004_H

#include "abilities/ability_registry.h"
#include <flecs.h>

// STT01-004: "On Play; You may discard a weapon card: look at the top 5 cards
// of your deck, reveal up to 1 weapon card and add it to your hand, then
// bottom deck the rest in any order"

bool stt01_004_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);
bool stt01_004_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target);
void stt01_004_apply_costs(ecs_world_t *world, const AbilityContext *ctx);
void stt01_004_on_cost_paid(ecs_world_t *world, AbilityContext *ctx);
bool stt01_004_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target);
void stt01_004_on_selection_complete(ecs_world_t *world, AbilityContext *ctx);

#endif // AZUKI_ABILITIES_STT01_004_H
