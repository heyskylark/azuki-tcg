#ifndef AZUKI_PASSIVE_RUNTIME_H
#define AZUKI_PASSIVE_RUNTIME_H

#include <flecs.h>
#include <stdbool.h>
#include <stddef.h>

#include "components/abilities.h"

typedef struct {
  bool free_ctx;
  ecs_entity_t attack_buff_source;
  ecs_entity_t health_buff_source;
} PassiveObserverCleanupOptions;

void *azk_alloc_passive_observer_ctx(size_t size);

void azk_init_passive_observer_context(ecs_world_t *world,
                                       ecs_entity_t ability_entity,
                                       void *user_ctx);

bool azk_track_passive_observer(ecs_world_t *world, ecs_entity_t ability_entity,
                                ecs_entity_t observer);

ecs_entity_t azk_create_tracked_passive_observer(
    ecs_world_t *world, ecs_entity_t ability_entity,
    const ecs_observer_desc_t *desc);

void azk_cleanup_passive_observer_context(
    ecs_world_t *world, ecs_entity_t ability_entity,
    const PassiveObserverCleanupOptions *options);

#endif // AZUKI_PASSIVE_RUNTIME_H
