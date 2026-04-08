#include "abilities/cards/azk01_043.h"

#include "abilities/cards/azk01_095.h"
#include "abilities/passive/passive_runtime.h"
#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/ability_util.h"

typedef struct {
  ecs_entity_t weapon;
} Azk01043ObserverCtx;

static bool is_alley_targeting_weapon(ecs_world_t *world, ecs_entity_t weapon) {
  const CardId *card_id = ecs_get(world, weapon, CardId);
  return card_id != NULL &&
         (card_id->id == CARD_DEF_AZK01_043 ||
          card_id->id == CARD_DEF_AZK01_095);
}

static int count_alley_targeting_weapons(ecs_world_t *world, ecs_entity_t host) {
  int count = 0;
  ecs_iter_t child_it = ecs_children(world, host);
  while (ecs_children_next(&child_it)) {
    for (int i = 0; i < child_it.count; ++i) {
      if (is_alley_targeting_weapon(world, child_it.entities[i])) {
        ++count;
      }
    }
  }

  return count;
}

static void sync_leader_alley_targeting(ecs_world_t *world, ecs_entity_t host,
                                        bool is_removal_event) {
  if (host == 0 || !ecs_has(world, host, TLeader)) {
    return;
  }

  int weapon_count = count_alley_targeting_weapons(world, host);
  if (is_removal_event) {
    --weapon_count;
  }

  const bool should_have_tag = weapon_count > 0;
  const bool has_tag =
      ecs_has(world, host, AttrCanTargetTappedAndUntappedAlley);
  if (should_have_tag == has_tag) {
    return;
  }

  if (should_have_tag) {
    ecs_add(world, host, AttrCanTargetTappedAndUntappedAlley);
  } else {
    ecs_remove(world, host, AttrCanTargetTappedAndUntappedAlley);
  }
}

static void alley_targeting_weapon_observer(ecs_iter_t *it) {
  Azk01043ObserverCtx *ctx = it->ctx;
  if (ctx == NULL || !ecs_is_valid(it->world, ctx->weapon) ||
      !is_alley_targeting_weapon(it->world, ctx->weapon)) {
    return;
  }

  ecs_entity_t host = ecs_get_target(it->world, ctx->weapon, EcsChildOf, 0);
  sync_leader_alley_targeting(it->world, host, it->event == EcsOnRemove);
}

static void init_alley_targeting_weapon_observer(ecs_world_t *world,
                                                 ecs_entity_t ability_entity) {
  ecs_entity_t card = azk_get_ability_source_card(world, ability_entity);
  if (card == 0) {
    return;
  }

  Azk01043ObserverCtx *ctx =
      azk_alloc_passive_observer_ctx(sizeof(Azk01043ObserverCtx));
  if (ctx == NULL) {
    return;
  }

  ctx->weapon = card;
  azk_init_passive_observer_context(world, ability_entity, ctx);

  ecs_entity_t observer = azk_create_tracked_passive_observer(
      world, ability_entity,
      &(ecs_observer_desc_t){
          .query.terms = {
              {.id = ecs_pair(EcsChildOf, EcsWildcard), .src.id = card},
              {.id = TWeapon, .src.id = card},
          },
          .events = {EcsOnAdd, EcsOnRemove},
          .callback = alley_targeting_weapon_observer,
          .ctx = ctx,
      });
  if (observer != 0) {
    return;
  }

  azk_cleanup_passive_observer_context(
      world, ability_entity, &(PassiveObserverCleanupOptions){.free_ctx = true});
}

static void cleanup_alley_targeting_weapon_observer(ecs_world_t *world,
                                                    ecs_entity_t ability_entity) {
  ecs_entity_t card = azk_get_ability_source_card(world, ability_entity);
  ecs_entity_t host =
      card != 0 ? ecs_get_target(world, card, EcsChildOf, 0) : 0;
  if (host != 0) {
    sync_leader_alley_targeting(world, host, true);
  }

  azk_cleanup_passive_observer_context(
      world, ability_entity, &(PassiveObserverCleanupOptions){.free_ctx = true});
}

void azk01_043_init_passive_observers(ecs_world_t *world,
                                      ecs_entity_t ability_entity) {
  init_alley_targeting_weapon_observer(world, ability_entity);
}

void azk01_043_cleanup_passive_observers(ecs_world_t *world,
                                         ecs_entity_t ability_entity) {
  cleanup_alley_targeting_weapon_observer(world, ability_entity);
}

void azk01_095_init_passive_observers(ecs_world_t *world,
                                      ecs_entity_t ability_entity) {
  init_alley_targeting_weapon_observer(world, ability_entity);
}

void azk01_095_cleanup_passive_observers(ecs_world_t *world,
                                         ecs_entity_t ability_entity) {
  cleanup_alley_targeting_weapon_observer(world, ability_entity);
}
