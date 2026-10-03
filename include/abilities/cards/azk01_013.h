#ifndef AZUKI_ABILITIES_AZK01_013_H
#define AZUKI_ABILITIES_AZK01_013_H

#include "abilities/ability_registry.h"

/* Opponent-chosen modes; `owner` is the choosing player (Gou's opponent). */

/* Mode 1: the chooser sacrifices an entity in their Garden. */
bool azk01_013_validate_sacrifice(ecs_world_t *world, ecs_entity_t card,
                                  ecs_entity_t owner);
bool azk01_013_validate_sacrifice_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target);
void azk01_013_apply_sacrifice(ecs_world_t *world, const AbilityContext *ctx);

/* Mode 2: the chooser discards 2 cards of their choice (all if fewer). */
bool azk01_013_validate_discard(ecs_world_t *world, ecs_entity_t card,
                                ecs_entity_t owner);
bool azk01_013_validate_discard_target(ecs_world_t *world, ecs_entity_t card,
                                       ecs_entity_t owner, ecs_entity_t target);
void azk01_013_begin_discard(ecs_world_t *world, AbilityContext *ctx);
void azk01_013_apply_discard(ecs_world_t *world, const AbilityContext *ctx);

#endif // AZUKI_ABILITIES_AZK01_013_H
