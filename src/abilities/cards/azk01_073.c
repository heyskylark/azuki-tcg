#include "abilities/cards/azk01_073.h"

#include "abilities/passive/passive_runtime.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

typedef struct {
  ecs_entity_t card;
  uint8_t owner_player_num;
} Azk01073ObserverCtx;

static bool owner_garden_is_all_beanz(ecs_world_t *world, ecs_entity_t card,
                                      uint8_t owner_player_num) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t owner_garden = gs->zones[owner_player_num].garden;
  if (ecs_get_target(world, card, EcsChildOf, 0) != owner_garden) {
    return false;
  }

  ecs_entities_t garden_cards = ecs_get_ordered_children(world, owner_garden);
  if (garden_cards.count == 0) {
    return false;
  }

  for (int32_t i = 0; i < garden_cards.count; ++i) {
    ecs_entity_t entity = garden_cards.ids[i];
    if (!ecs_has_id(world, entity, TEntity) ||
        !has_subtype(world, entity, ecs_id(TSubtype_Beanz))) {
      return false;
    }
  }

  return true;
}

static void sync_top_beanz_buff(ecs_world_t *world, ecs_entity_t card,
                                uint8_t owner_player_num) {
  if (!ecs_is_valid(world, card)) {
    return;
  }

  bool active = owner_garden_is_all_beanz(world, card, owner_player_num);
  bool has_buff = ecs_has_pair(world, card, ecs_id(AttackBuff), card) ||
                  ecs_has_pair(world, card, ecs_id(HealthBuff), card);

  if (active && !has_buff) {
    azk_queue_passive_buff_update(world, card, card, 1, 1, false);
  } else if (!active && has_buff) {
    azk_queue_passive_buff_update(world, card, card, 0, 0, true);
  }
}

static void azk01_073_garden_observer(ecs_iter_t *it) {
  Azk01073ObserverCtx *ctx = it->ctx;
  if (ctx == NULL) {
    return;
  }

  sync_top_beanz_buff(it->world, ctx->card, ctx->owner_player_num);
}

void azk01_073_init_passive_observers(ecs_world_t *world, ecs_entity_t card) {
  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  if (owner == 0) {
    return;
  }

  uint8_t owner_player_num = get_player_number(world, owner);
  const GameState *gs = ecs_singleton_get(world, GameState);
  Azk01073ObserverCtx *ctx =
      azk_alloc_passive_observer_ctx(sizeof(Azk01073ObserverCtx));
  if (ctx == NULL) {
    return;
  }

  ctx->card = card;
  ctx->owner_player_num = owner_player_num;
  azk_init_passive_observer_context(world, card, ctx);

  ecs_entity_t observer = azk_create_tracked_passive_observer(
      world, card,
      &(ecs_observer_desc_t){
          .query.terms = {{.id = ecs_pair(EcsChildOf, gs->zones[owner_player_num].garden)},
                          {.id = TEntity}},
          .events = {EcsOnAdd, EcsOnRemove},
          .callback = azk01_073_garden_observer,
          .ctx = ctx,
      });

  if (observer == 0) {
    azk_cleanup_passive_observer_context(
        world, card, &(PassiveObserverCleanupOptions){.free_ctx = true});
  }
}

void azk01_073_cleanup_passive_observers(ecs_world_t *world, ecs_entity_t card) {
  azk_cleanup_passive_observer_context(
      world, card,
      &(PassiveObserverCleanupOptions){
          .free_ctx = true,
          .attack_buff_source = card,
          .health_buff_source = card,
      });
}
