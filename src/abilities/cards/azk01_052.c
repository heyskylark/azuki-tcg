#include "abilities/cards/azk01_052.h"

#include "abilities/passive/passive_runtime.h"
#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"

typedef struct {
  ecs_entity_t card;
  uint8_t owner_player_num;
} Azk01052ObserverCtx;

static int count_entities_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  int count = 0;
  for (int32_t i = 0; i < cards.count; i++) {
    if (ecs_has_id(world, cards.ids[i], TEntity)) {
      count++;
    }
  }
  return count;
}

static void sync_yojin_defender(ecs_world_t *world, ecs_entity_t card,
                                uint8_t owner_player_num,
                                bool is_removal_event,
                                bool player_garden_changed) {
  if (!ecs_is_valid(world, card)) {
    return;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t opponent_num = owner_player_num == 0 ? 1 : 0;
  ecs_entity_t owner_garden = gs->zones[owner_player_num].garden;
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  bool has_defender = ecs_has(world, card, Defender);

  if (parent != owner_garden) {
    if (has_defender) {
      ecs_remove(world, card, Defender);
      azk_log_card_keywords_changed(world, card);
    }
    return;
  }

  int player_count = count_entities_in_zone(world, owner_garden);
  int opponent_count = count_entities_in_zone(world, gs->zones[opponent_num].garden);
  if (is_removal_event) {
    if (player_garden_changed) {
      player_count--;
    } else {
      opponent_count--;
    }
  }

  bool should_have_defender = player_count < opponent_count;
  if (should_have_defender == has_defender) {
    return;
  }

  if (should_have_defender) {
    ecs_add(world, card, Defender);
  } else {
    ecs_remove(world, card, Defender);
  }
  azk_log_card_keywords_changed(world, card);
}

static void azk01_052_player_garden_observer(ecs_iter_t *it) {
  Azk01052ObserverCtx *ctx = it->ctx;
  if (ctx == NULL) {
    return;
  }

  sync_yojin_defender(it->world, ctx->card, ctx->owner_player_num,
                      it->event == EcsOnRemove, true);
}

static void azk01_052_opponent_garden_observer(ecs_iter_t *it) {
  Azk01052ObserverCtx *ctx = it->ctx;
  if (ctx == NULL) {
    return;
  }

  sync_yojin_defender(it->world, ctx->card, ctx->owner_player_num,
                      it->event == EcsOnRemove, false);
}

void azk01_052_init_passive_observers(ecs_world_t *world, ecs_entity_t card) {
  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  if (owner == 0) {
    return;
  }

  uint8_t owner_player_num = get_player_number(world, owner);
  uint8_t opponent_num = owner_player_num == 0 ? 1 : 0;
  const GameState *gs = ecs_singleton_get(world, GameState);
  Azk01052ObserverCtx *ctx =
      azk_alloc_passive_observer_ctx(sizeof(Azk01052ObserverCtx));
  if (ctx == NULL) {
    return;
  }

  ctx->card = card;
  ctx->owner_player_num = owner_player_num;
  azk_init_passive_observer_context(world, card, ctx);

  ecs_entity_t player_obs = azk_create_tracked_passive_observer(
      world, card,
      &(ecs_observer_desc_t){
          .query.terms = {{.id = ecs_pair(EcsChildOf, gs->zones[owner_player_num].garden)},
                          {.id = TEntity}},
          .events = {EcsOnAdd, EcsOnRemove},
          .callback = azk01_052_player_garden_observer,
          .ctx = ctx,
      });
  ecs_entity_t opponent_obs = azk_create_tracked_passive_observer(
      world, card,
      &(ecs_observer_desc_t){
          .query.terms = {{.id = ecs_pair(EcsChildOf, gs->zones[opponent_num].garden)},
                          {.id = TEntity}},
          .events = {EcsOnAdd, EcsOnRemove},
          .callback = azk01_052_opponent_garden_observer,
          .ctx = ctx,
      });

  if (player_obs == 0 || opponent_obs == 0) {
    azk_cleanup_passive_observer_context(
        world, card, &(PassiveObserverCleanupOptions){.free_ctx = true});
  }
}

void azk01_052_cleanup_passive_observers(ecs_world_t *world, ecs_entity_t card) {
  azk_cleanup_passive_observer_context(
      world, card, &(PassiveObserverCleanupOptions){.free_ctx = true});
}
