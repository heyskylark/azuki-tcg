#include "abilities/passive/passive_runtime.h"

#include "components/components.h"
#include "utils/status_util.h"

#include <stdint.h>
#include <string.h>

// Observer contexts are shared between multiple observers whose deletion is
// deferred (e.g. during world fini). A plain free in cleanup leaves a window
// where a still-pending observer fires on freed memory. Contexts are therefore
// refcounted: one reference for the creator (released by cleanup) and one per
// tracked observer (released by flecs via ctx_free when the observer is
// actually destroyed).
typedef struct {
  int32_t refcount;
  char padding[12]; // Keep payload at malloc alignment (16 bytes).
} AzkPassiveCtxHeader;

static AzkPassiveCtxHeader *azk_passive_ctx_header(void *payload) {
  return (AzkPassiveCtxHeader *)((char *)payload - sizeof(AzkPassiveCtxHeader));
}

static void azk_passive_ctx_retain(void *payload) {
  if (!payload) {
    return;
  }
  azk_passive_ctx_header(payload)->refcount += 1;
}

static void azk_passive_ctx_release(void *payload) {
  if (!payload) {
    return;
  }
  AzkPassiveCtxHeader *header = azk_passive_ctx_header(payload);
  header->refcount -= 1;
  if (header->refcount <= 0) {
    ecs_os_free(header);
  }
}

void *azk_alloc_passive_observer_ctx(size_t size) {
  if (size == 0) {
    return NULL;
  }

  AzkPassiveCtxHeader *header =
      ecs_os_malloc((ecs_size_t)(sizeof(AzkPassiveCtxHeader) + size));
  if (!header) {
    return NULL;
  }

  memset(header, 0, sizeof(AzkPassiveCtxHeader) + size);
  header->refcount = 1;
  return (char *)header + sizeof(AzkPassiveCtxHeader);
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

  ecs_observer_desc_t local_desc = *desc;
  bool ctx_ref_held = false;
  if (local_desc.ctx && !local_desc.ctx_free) {
    // Hold a reference per observer so the shared ctx outlives deferred
    // observer deletion; flecs releases it when the observer is destroyed.
    azk_passive_ctx_retain(local_desc.ctx);
    local_desc.ctx_free = azk_passive_ctx_release;
    ctx_ref_held = true;
  }

  ecs_entity_t observer = ecs_observer_init(world, &local_desc);
  if (observer == 0) {
    if (ctx_ref_held) {
      azk_passive_ctx_release(local_desc.ctx);
    }
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
  // During world teardown only release resources; buff bookkeeping touches
  // entities (zones, source cards) that may already be deleted.
  bool quitting = ecs_should_quit(world);

  for (uint8_t i = 0; i < ctx->observer_count; ++i) {
    if (ctx->observers[i] != 0) {
      ecs_delete(world, ctx->observers[i]);
    }
  }

  if (cleanup_options->free_ctx && ctx->ctx) {
    azk_passive_ctx_release(ctx->ctx);
  }

  if (!quitting) {
    ecs_entity_t source_card =
        ecs_get_target(world, ability_entity, Rel_AbilityOf, 0);
    if (source_card == 0) {
      source_card = ability_entity;
    }

    if (ecs_is_valid(world, source_card)) {
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
    }
  }

  ecs_remove(world, ability_entity, PassiveObserverContext);
}
