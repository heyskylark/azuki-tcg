#include "abilities/cards/azk01_096.h"

#include "abilities/ability_system.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"

bool azk01_096_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  (void)card;

  if (gs == NULL) {
    return false;
  }

  const uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  ecs_entities_t alley_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].alley);
  return garden_cards.count > 0 && alley_cards.count > 0;
}

bool azk01_096_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[owner_num].garden;
}

bool azk01_096_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[owner_num].alley;
}

void azk01_096_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->cost.selected_count == 0 || ctx->effect.selected_count == 0) {
    return;
  }

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  ecs_entity_t garden_card = ctx->cost.entities[0];
  ecs_entity_t alley_card = ctx->effect.entities[0];
  const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);

  if (gs == NULL || garden_card == 0 || alley_card == 0 ||
      garden_card == alley_card) {
    return;
  }

  if (ecs_get_target(world, garden_card, EcsChildOf, 0) !=
          gs->zones[owner_num].garden ||
      ecs_get_target(world, alley_card, EcsChildOf, 0) !=
          gs->zones[owner_num].alley) {
    return;
  }

  const ZoneIndex *garden_index = ecs_get(world, garden_card, ZoneIndex);
  const ZoneIndex *alley_index = ecs_get(world, alley_card, ZoneIndex);
  if (garden_index == NULL || alley_index == NULL) {
    return;
  }

  const int8_t old_garden_index = (int8_t)garden_index->index;
  const int8_t old_alley_index = (int8_t)alley_index->index;

  azk_log_card_zone_moved(world, garden_card, GLOG_ZONE_GARDEN,
                          old_garden_index, GLOG_ZONE_ALLEY, old_alley_index);
  azk_log_card_zone_moved(world, alley_card, GLOG_ZONE_ALLEY, old_alley_index,
                          GLOG_ZONE_GARDEN, old_garden_index);

  ecs_add_pair(world, garden_card, EcsChildOf, gs->zones[owner_num].alley);
  ecs_set(world, garden_card, ZoneIndex, {.index = (uint8_t)old_alley_index});

  ecs_add_pair(world, alley_card, EcsChildOf, gs->zones[owner_num].garden);
  ecs_set(world, alley_card, ZoneIndex, {.index = (uint8_t)old_garden_index});

  const CardId *alley_card_id = ecs_get(world, alley_card, CardId);
  if (alley_card_id != NULL && alley_card_id->id == CARD_DEF_STT03_013 &&
      !ecs_has(world, alley_card, Taunt)) {
    ecs_add(world, alley_card, Taunt);
    azk_log_card_keywords_changed(world, alley_card);
  }

  if (gs->combat_state.defender_card == garden_card) {
    gs->combat_state.defender_card = alley_card;
  }

  azk_trigger_enter_garden_ability(world, alley_card, ctx->runtime.owner);

  cli_render_logf("[AZK01-096] Swapped selected Garden and Alley entities");
}
