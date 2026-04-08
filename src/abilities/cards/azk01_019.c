#include "abilities/cards/azk01_019.h"

#include "abilities/passive/passive_runtime.h"
#include "components/abilities.h"
#include "components/components.h"
#include "utils/ability_util.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

typedef struct {
  ecs_entity_t card;
  uint8_t player_num;
} Azk01019ObserverCtx;

static bool owner_garden_has_only_normal_entities(ecs_world_t *world,
                                                  uint8_t player_num) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entities_t cards =
      ecs_get_ordered_children(world, gs->zones[player_num].garden);

  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t entity = cards.ids[i];
    if (!is_card_type(world, entity, CARD_TYPE_ENTITY) ||
        !is_normal_element_card(world, entity)) {
      return false;
    }
  }

  return true;
}

static bool card_is_in_play(ecs_world_t *world, ecs_entity_t card,
                            uint8_t player_num) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  return parent == gs->zones[player_num].garden ||
         parent == gs->zones[player_num].alley;
}

static void update_jay_buff(ecs_world_t *world, ecs_entity_t card,
                            uint8_t player_num) {
  if (!ecs_is_valid(world, card)) {
    return;
  }

  if (!card_is_in_play(world, card, player_num)) {
    azk_queue_passive_buff_update(world, card, card, 0, 0, true);
    return;
  }

  if (owner_garden_has_only_normal_entities(world, player_num)) {
    azk_queue_passive_buff_update(world, card, card, 0, 2, false);
  } else {
    azk_queue_passive_buff_update(world, card, card, 0, 0, true);
  }
}

static void azk01_019_zone_observer(ecs_iter_t *it) {
  Azk01019ObserverCtx *ctx = it->ctx;
  if (!ctx) {
    return;
  }

  update_jay_buff(it->world, ctx->card, ctx->player_num);
}

void azk01_019_init_passive_observers(ecs_world_t *world,
                                      ecs_entity_t ability_entity) {
  ecs_entity_t card = azk_get_ability_source_card(world, ability_entity);
  if (card == 0) {
    return;
  }

  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  if (!owner) {
    return;
  }

  Azk01019ObserverCtx *ctx =
      azk_alloc_passive_observer_ctx(sizeof(Azk01019ObserverCtx));
  if (!ctx) {
    return;
  }

  ctx->card = card;
  ctx->player_num = get_player_number(world, owner);
  azk_init_passive_observer_context(world, ability_entity, ctx);

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t garden = gs->zones[ctx->player_num].garden;
  ecs_entity_t alley = gs->zones[ctx->player_num].alley;

  ecs_entity_t garden_observer = azk_create_tracked_passive_observer(
      world, ability_entity,
      &(ecs_observer_desc_t){
          .query.terms = {{.id = ecs_pair(EcsChildOf, garden)},
                          {.id = ecs_id(CardId)}},
          .events = {EcsOnAdd, EcsOnRemove},
          .callback = azk01_019_zone_observer,
          .ctx = ctx,
      });
  ecs_entity_t alley_observer = azk_create_tracked_passive_observer(
      world, ability_entity,
      &(ecs_observer_desc_t){
          .query.terms = {{.id = ecs_pair(EcsChildOf, alley)},
                          {.id = ecs_id(CardId)}},
          .events = {EcsOnAdd, EcsOnRemove},
          .callback = azk01_019_zone_observer,
          .ctx = ctx,
      });

  if (garden_observer == 0 || alley_observer == 0) {
    azk_cleanup_passive_observer_context(
        world, ability_entity,
        &(PassiveObserverCleanupOptions){
            .free_ctx = true,
        });
    return;
  }

  update_jay_buff(world, card, ctx->player_num);
}

void azk01_019_cleanup_passive_observers(ecs_world_t *world,
                                         ecs_entity_t ability_entity) {
  ecs_entity_t card = azk_get_ability_source_card(world, ability_entity);
  azk_cleanup_passive_observer_context(
      world, ability_entity,
      &(PassiveObserverCleanupOptions){
          .free_ctx = true,
          .health_buff_source = card,
      });
}
