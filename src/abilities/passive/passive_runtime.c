#include "abilities/passive/passive_runtime.h"

#include "components/components.h"
#include "utils/status_util.h"

#include <string.h>

void *azk_alloc_passive_observer_ctx(size_t size) {
  if (size == 0) {
    return NULL;
  }

  void *ctx = ecs_os_malloc(size);
  if (!ctx) {
    return NULL;
  }

  memset(ctx, 0, size);
  return ctx;
}

void azk_init_passive_observer_context(ecs_world_t *world,
                                       ecs_entity_t ability_entity,
                                       void *user_ctx) {
  ecs_set(world, ability_entity, PassiveObserverContext, {.ctx = user_ctx});
}

bool azk_track_passive_observer(ecs_world_t *world, ecs_entity_t ability_entity,
                                ecs_entity_t observer) {
  if (!world || ability_entity == 0 || observer == 0) {
    return false;
  }

  PassiveObserverContext *ctx =
      ecs_get_mut(world, ability_entity, PassiveObserverContext);
  if (!ctx) {
    return false;
  }

  if (ctx->observer_count >= MAX_PASSIVE_OBSERVERS) {
    return false;
  }

  ctx->observers[ctx->observer_count++] = observer;
  ecs_modified(world, ability_entity, PassiveObserverContext);
  return true;
}

ecs_entity_t azk_create_tracked_passive_observer(
    ecs_world_t *world, ecs_entity_t ability_entity,
    const ecs_observer_desc_t *desc) {
  if (!desc) {
    return 0;
  }

  ecs_entity_t observer = ecs_observer_init(world, desc);
  if (observer == 0) {
    return 0;
  }

  if (!azk_track_passive_observer(world, ability_entity, observer)) {
    ecs_delete(world, observer);
    return 0;
  }

  return observer;
}

void azk_cleanup_passive_observer_context(
    ecs_world_t *world, ecs_entity_t ability_entity,
    const PassiveObserverCleanupOptions *options) {
  const PassiveObserverContext *ctx =
      ecs_get(world, ability_entity, PassiveObserverContext);
  if (!ctx) {
    return;
  }

  const PassiveObserverCleanupOptions default_options = {0};
  const PassiveObserverCleanupOptions *cleanup_options =
      options ? options : &default_options;
  ecs_entity_t source_card =
      ecs_get_target(world, ability_entity, Rel_AbilityOf, 0);
  if (source_card == 0) {
    source_card = ability_entity;
  }

  for (uint8_t i = 0; i < ctx->observer_count; ++i) {
    if (ctx->observers[i] != 0) {
      ecs_delete(world, ctx->observers[i]);
    }
  }

  if (cleanup_options->free_ctx && ctx->ctx) {
    ecs_os_free(ctx->ctx);
  }

  if (cleanup_options->attack_buff_source != 0 &&
      ecs_has_pair(world, source_card, ecs_id(AttackBuff),
                   cleanup_options->attack_buff_source)) {
    remove_attack_modifier(world, source_card,
                           cleanup_options->attack_buff_source);
  }

  if (cleanup_options->health_buff_source != 0 &&
      ecs_has_pair(world, source_card, ecs_id(HealthBuff),
                   cleanup_options->health_buff_source)) {
    remove_health_modifier(world, source_card,
                           cleanup_options->health_buff_source);
  }

  ecs_remove(world, ability_entity, PassiveObserverContext);
}
